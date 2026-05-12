"""
nse_pipeline.py  /  MC_10Cr_Above_D.py
=======================================
STEP 1 — Download live "Stocks Traded" data from NSE India
          https://www.nseindia.com/market-data/stocks-traded

STEP 2 — Filter EQ-series stocks where Value > ₹10 Crores

STEP 3 — For each filtered stock, download OHLC from yfinance.
          Compute EMA9 on Daily, Weekly and Monthly closes.
          *** Only chart stocks where:
              Daily Close > Daily EMA9
              AND Daily Close > Weekly EMA9 (forward-filled)
              AND Daily Close > Monthly EMA9 (forward-filled)  ***

STEP 4 — Save TradingView-style dark-theme PNG charts for qualifying stocks.
          Chart shows: Candlestick + Daily EMA9 + Weekly EMA9 + Monthly EMA9
                       + 20-bar recent-low + MACD(12,26,9)

Requirements:
    pip install selenium webdriver-manager yfinance pandas openpyxl matplotlib

Usage:
    python nse_pipeline.py                   # headless Chrome (default)
    python nse_pipeline.py --visible         # visible browser window
    python nse_pipeline.py --from-csv FILE   # skip browser, use saved CSV
"""

import sys, os, glob, json, time, shutil, datetime, tempfile
import argparse, traceback, warnings
from datetime import timedelta

import pandas as pd
import numpy as np

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_OK = True
except ImportError:
    SELENIUM_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import yfinance as yf

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  CONFIG  — edit these to change behaviour
# ═══════════════════════════════════════════════════════════════

TRADED_VALUE_MIN_CR = 5      # Step 2 filter: traded value threshold (₹ Cr)

NSE_HOME  = "https://www.nseindia.com"
NSE_PAGE  = "https://www.nseindia.com/market-data/stocks-traded"
NSE_APIS  = [
    "https://www.nseindia.com/api/live-analysis-stocksTraded",
    "https://www.nseindia.com/json/liveAnalysis/stocks-traded.json",
]

PAGE_WAIT     = 30
API_SETTLE    = 10
DOWNLOAD_WAIT = 40

EXCHANGE_SFX      = ".NS"
OUTPUT_DIR        = "NSE_Filtered_Charts"
PERIOD_DAYS       = 365       # daily chart lookback (~250 bars)

EMA_PERIOD        = 9
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
RECENT_LOW_BARS   = 20
CANDLE_BODY_WIDTH = 0.75
CANDLE_WICK_WIDTH = 0.12

# HTF warmup periods (more history → better EMA convergence)
WEEKLY_LOOKBACK_DAYS  = 730    # ~2 years of weekly bars
MONTHLY_LOOKBACK_DAYS = 3650   # ~10 years of monthly bars

# HTF EMA9 line colours
WEEKLY_EMA_COLOR  = "#E040FB"   # bright purple — dashed
MONTHLY_EMA_COLOR = "#00E5FF"   # bright cyan   — dotted

STYLE = {
    "bg":          "#131722",
    "panel_bg":    "#1E222D",
    "grid":        "#2A2E39",
    "up_candle":   "#26A69A",
    "down_candle": "#EF5350",
    "wick_up":     "#26A69A",
    "wick_down":   "#EF5350",
    "ema_color":   "#FF9800",
    "macd_line":   "#2962FF",
    "signal_line": "#FF6D00",
    "hist_up":     "#26A69A",
    "hist_down":   "#EF5350",
    "text":        "#D1D4DC",
    "subtext":     "#787B86",
    "border":      "#2A2E39",
    "recent_low":  "#FFD700",
}

SYNC_XHR = """
var xhr = new XMLHttpRequest();
xhr.open('GET', arguments[0], false);
xhr.setRequestHeader('Accept', 'application/json, text/plain, */*');
xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
xhr.setRequestHeader('Referer',
    'https://www.nseindia.com/market-data/stocks-traded');
try {
    xhr.send(null);
    return {status: xhr.status, body: xhr.responseText};
} catch(e) {
    return {status: -1, body: e.toString()};
}
"""


# ═══════════════════════════════════════════════════════════════
#  STEP 1A — BROWSER SETUP
# ═══════════════════════════════════════════════════════════════

def build_driver(headless: bool, download_dir: str):
    if not SELENIUM_OK:
        print("[ERROR] selenium/webdriver-manager not installed.")
        print("  Fix: pip install selenium webdriver-manager")
        sys.exit(1)

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("prefs", {
        "download.default_directory":   download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade":   True,
        "safebrowsing.enabled":         True,
    })

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=opts)
        print("  ✔  ChromeDriver ready (webdriver-manager)")
        return driver
    except Exception as e:
        print(f"  [WARN] webdriver-manager: {e}")

    try:
        driver = webdriver.Chrome(options=opts)
        print("  ✔  ChromeDriver ready (system)")
        return driver
    except Exception as e:
        print(f"\n[ERROR] Chrome: {e}"); sys.exit(1)


def patch_driver(driver):
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',"
                       "{get:()=>undefined});"}
        )
    except Exception:
        pass


def warm_session(driver):
    print("  [1/3]  Loading NSE homepage …")
    driver.get(NSE_HOME)
    time.sleep(4)
    print(f"         Cookies: {[c['name'] for c in driver.get_cookies()]}")

    print("  [2/3]  Loading Stocks Traded page …")
    driver.get(NSE_PAGE)
    try:
        WebDriverWait(driver, PAGE_WAIT).until(
            EC.presence_of_element_located(
                (By.XPATH,
                 "//*[@id='cm_9'] | "
                 "//h2[contains(text(),'Stocks Traded')] | "
                 "//*[contains(text(),'Stocks Traded') and "
                 "    not(contains(@class,'nav'))]")
            )
        )
        print("         Page loaded ✔")
    except TimeoutException:
        print("         [WARN] timeout — continuing")

    print(f"         Settling {API_SETTLE}s …")
    time.sleep(API_SETTLE)
    print(f"         Cookies: {[c['name'] for c in driver.get_cookies()]}")


# ═══════════════════════════════════════════════════════════════
#  STEP 1B — XHR DATA FETCH
# ═══════════════════════════════════════════════════════════════

def _find_records_in_json(payload) -> list:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if any(k in payload[0] for k in ("symbol", "Symbol", "SYMBOL")):
            return payload
    if isinstance(payload, dict):
        for key in ("data", "stocksTradedData", "result", "rows",
                    "stockData", "DATA", "records", "stocks", "dataList"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
    return []


def fetch_via_xhr(driver) -> list:
    print("  [3/3]  Calling NSE API via sync XHR …")
    for url in NSE_APIS:
        print(f"         → {url}")
        try:
            res    = driver.execute_script(SYNC_XHR, url)
            status = res.get("status", -1)
            body   = res.get("body", "")
            print(f"           HTTP {status}  |  {len(body):,} chars")
            if status != 200 or not body:
                continue
            payload = json.loads(body)
            if isinstance(payload, dict):
                print(f"           JSON keys: {list(payload.keys())}")
            records = _find_records_in_json(payload)
            if records:
                print(f"  ✔  {len(records)} records via XHR")
                print(f"     Fields: {list(records[0].keys())[:10]}")
                return records
            print("           No stock list found in response")
        except json.JSONDecodeError as e:
            print(f"           JSON error: {e}")
        except Exception as e:
            print(f"           Error: {e}")
    return []


# ═══════════════════════════════════════════════════════════════
#  STEP 1C — CSV BUTTON FALLBACK
# ═══════════════════════════════════════════════════════════════

def _wait_for_csv(dl_dir: str) -> str:
    print(f"         Waiting {DOWNLOAD_WAIT}s for file", end="", flush=True)
    deadline = time.time() + DOWNLOAD_WAIT
    while time.time() < deadline:
        time.sleep(1)
        print(".", end="", flush=True)
        files = [f for f in
                 glob.glob(os.path.join(dl_dir, "*.csv")) +
                 glob.glob(os.path.join(dl_dir, "*.CSV"))
                 if not f.endswith(".crdownload")]
        if files:
            latest = max(files, key=os.path.getmtime)
            print(f"\n  ✔  {os.path.basename(latest)}")
            return latest
    print("\n  Timed out.")
    return ""


def fetch_via_csv_button(driver, dl_dir: str) -> str:
    print("  CSV button fallback …")
    xpaths = [
        "//a[contains(@onclick,'StocksTraded-download')]",
        "//a[contains(@onclick,'StocksTraded')]",
        ".//a[.//img[contains(@src,'xls') or contains(@src,'csv')]]",
        "//a[contains(@onclick,'download') and "
        "    not(contains(@onclick,'First')) and "
        "    not(contains(@onclick,'Prev')) and "
        "    not(contains(@onclick,'Next'))]",
    ]
    for xpath in xpaths:
        try:
            for el in driver.find_elements(By.XPATH, xpath):
                if el.is_displayed():
                    print(f"  Clicking: {el.get_attribute('outerHTML')[:100]}")
                    driver.execute_script("arguments[0].scrollIntoView(true);", el)
                    time.sleep(0.5)
                    driver.execute_script("arguments[0].click();", el)
                    path = _wait_for_csv(dl_dir)
                    if path:
                        return path
        except Exception:
            continue
    try:
        driver.execute_script("downloadCSV('StocksTraded-download');")
        return _wait_for_csv(dl_dir)
    except Exception as e:
        print(f"  JS call failed: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════
#  STEP 1D — NORMALISE TO STANDARD DATAFRAME
# ═══════════════════════════════════════════════════════════════

def safe_num(v) -> float:
    try:
        return float(str(v).replace(",", "").replace("–", "0")
                     .replace("−", "0").strip())
    except Exception:
        return 0.0


def normalise_json(records: list) -> pd.DataFrame:
    """NSE API: totalTradedValue is in raw ₹ → divide by 1e7 → Crores."""
    rows = []
    for d in records:
        sym = str(d.get("symbol", d.get("Symbol", ""))).strip()
        if not sym:
            continue
        tv_raw = safe_num(d.get("totalTradedValue", d.get("tradedValue", 0)))
        vol    = safe_num(d.get("totalTradedVolume", d.get("tradedQuantity", 0)))
        rows.append({
            "Symbol":           sym,
            "Company":          str(d.get("companyName", "")).strip(),
            "Series":           str(d.get("series", "EQ")).strip(),
            "LTP (₹)":          round(safe_num(d.get("lastPrice",
                                     d.get("closePrice", 0))), 2),
            "% Change":         round(safe_num(d.get("pChange", 0)), 2),
            "Mkt Cap (₹ Cr)":   round(safe_num(d.get("marketCap",
                                     d.get("market_cap", 0))), 2),
            "Volume (Lakhs)":   round(vol / 1e5, 2),
            "Value (₹ Crores)": round(tv_raw / 1e7, 2),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Value (₹ Crores)", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index += 1
    return df


def normalise_csv(path: str) -> pd.DataFrame:
    """
    NSE CSV confirmed headers:
      Symbol | Series | LTP | %chng | Mkt Cap (₹ Crores) | Volume (Lakhs) | Value (₹ Crores)
    Value (₹ Crores) is already in Crores — use directly.
    """
    try:
        df = pd.read_csv(path, thousands=",")
    except Exception as e:
        print(f"  CSV read error: {e}"); return pd.DataFrame()

    df.columns = df.columns.str.strip()
    print(f"  CSV columns: {list(df.columns)}")
    if not df.empty:
        print(f"  First row:   {df.iloc[0].to_dict()}")

    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl == "symbol":                                   col_map[col] = "Symbol"
        elif cl == "series":                                 col_map[col] = "Series"
        elif cl in ("ltp","last price","close","lastprice"): col_map[col] = "LTP (₹)"
        elif cl in ("%chng","%change","% change","pchange",
                    "% chng","per change","%chg"):           col_map[col] = "% Change"
        elif "mkt cap" in cl or "market cap" in cl:         col_map[col] = "Mkt Cap (₹ Cr)"
        elif "volume" in cl:                                 col_map[col] = "Volume (Lakhs)"
        elif "value" in cl:                                  col_map[col] = "Value (₹ Crores)"
        elif "company" in cl or cl == "name":                col_map[col] = "Company"

    df.rename(columns=col_map, inplace=True)
    print(f"  Mapped to:   {list(df.columns)}")

    for col in ["LTP (₹)", "% Change", "Mkt Cap (₹ Cr)",
                "Volume (Lakhs)", "Value (₹ Crores)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            ).fillna(0)

    for col, default in [("Company", ""), ("Series", "EQ"), ("Mkt Cap (₹ Cr)", 0.0)]:
        if col not in df.columns:
            df[col] = default

    if "Value (₹ Crores)" in df.columns:
        df.sort_values("Value (₹ Crores)", ascending=False, inplace=True)
        top = df["Value (₹ Crores)"].iloc[0]
        print(f"  Top Value: ₹{top:,.2f} Cr  "
              f"({'✔ correct' if top > 10 else '⚠ suspiciously low'})")

    df.reset_index(drop=True, inplace=True)
    df.index += 1
    print(f"  {len(df)} rows loaded")
    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — MASTER DOWNLOAD FUNCTION
# ═══════════════════════════════════════════════════════════════

def download_nse_data(headless: bool, from_csv: str) -> pd.DataFrame:
    if from_csv:
        print(f"\n  Loading manual CSV: {from_csv}")
        df = normalise_csv(from_csv)
        if df.empty:
            print("  [ERROR] CSV empty or unreadable."); sys.exit(1)
        return df

    if not SELENIUM_OK:
        print("[ERROR] selenium not installed."); sys.exit(1)

    dl_dir = tempfile.mkdtemp()
    driver = build_driver(headless, dl_dir)
    driver.set_page_load_timeout(60)
    patch_driver(driver)
    df = pd.DataFrame()

    try:
        warm_session(driver)
        records = fetch_via_xhr(driver)
        if records:
            df = normalise_json(records)
        if df.empty:
            print("\n  XHR returned no data — trying CSV button …")
            csv_path = fetch_via_csv_button(driver, dl_dir)
            if csv_path:
                df = normalise_csv(csv_path)
    except WebDriverException as e:
        print(f"\n[ERROR] WebDriver: {e}")
    finally:
        driver.quit()
        shutil.rmtree(dl_dir, ignore_errors=True)
        print("  Browser closed.")

    if df.empty:
        print("\n  ✗  Could not retrieve data from NSE.")
        print(f"  1. Open {NSE_PAGE}  →  click ↓ CSV")
        print("  2. python nse_pipeline.py --from-csv StocksTraded.csv")
        sys.exit(1)

    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — VALUE FILTER (₹ Crores + EQ series)
# ═══════════════════════════════════════════════════════════════

def filter_by_value(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only EQ-series stocks with Value > TRADED_VALUE_MIN_CR."""
    print(f"\n  Top 5 by Value before filter:")
    top5 = df[df["Value (₹ Crores)"] > 0].nlargest(5, "Value (₹ Crores)")
    for _, r in top5.iterrows():
        print(f"    {r['Symbol']:<12}  ₹{r['Value (₹ Crores)']:>10,.2f} Cr  "
              f"Series={r['Series']}")

    mask = (
        (df["Series"].str.strip().str.upper() == "EQ") &
        (df["Value (₹ Crores)"] > TRADED_VALUE_MIN_CR)
    )
    out = df[mask].copy()
    out.sort_values("Value (₹ Crores)", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — INDICATORS & EMA9 TRIPLE-TIMEFRAME FILTER
# ═══════════════════════════════════════════════════════════════

def ema_calc(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd_calc(close: pd.Series, fast=12, slow=26, signal=9):
    ml = ema_calc(close, fast) - ema_calc(close, slow)
    sl = ema_calc(ml, signal)
    return ml, sl, ml - sl


def fetch_ohlc(ticker: str, start: str, end: str,
               interval: str = "1d") -> pd.DataFrame:
    """Download OHLC from yfinance, flatten MultiIndex, strip timezone."""
    df = yf.download(ticker, start=start, end=end,
                     interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_htf_ema9(ticker: str,
                 daily_index: pd.DatetimeIndex) -> tuple:
    """
    Compute EMA9 on Weekly and Monthly closes, then forward-fill onto
    the daily bar index so every daily bar has the latest HTF EMA9 value.

    Returns
    -------
    (weekly_ema9_daily, monthly_ema9_daily) — both pd.Series on daily_index
    """
    end_dt    = (datetime.datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    w_start   = (datetime.datetime.today() - timedelta(days=WEEKLY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    m_start   = (datetime.datetime.today() - timedelta(days=MONTHLY_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    def _ffill(htf_series: pd.Series) -> pd.Series:
        if htf_series.empty:
            return pd.Series(np.nan, index=daily_index)
        merged = htf_series.reindex(
            htf_series.index.union(daily_index).sort_values()
        ).ffill()
        return merged.reindex(daily_index)

    # Weekly
    w_df    = fetch_ohlc(ticker, w_start, end_dt, "1wk")
    w_ema9  = ema_calc(w_df["Close"], EMA_PERIOD) if not w_df.empty \
              else pd.Series(dtype=float)

    # Monthly
    m_df    = fetch_ohlc(ticker, m_start, end_dt, "1mo")
    m_ema9  = ema_calc(m_df["Close"], EMA_PERIOD) if not m_df.empty \
              else pd.Series(dtype=float)

    return _ffill(w_ema9), _ffill(m_ema9)


def passes_ema9_filter(ohlc: pd.DataFrame,
                       w_ema9: pd.Series,
                       m_ema9: pd.Series) -> tuple[bool, dict]:
    """
    Check whether the LATEST daily close is above all three EMA9s.

    Returns
    -------
    (passes: bool, details: dict)
    details keys: close, d_ema9, w_ema9, m_ema9, above_d, above_w, above_m
    """
    daily_ema9 = ema_calc(ohlc["Close"], EMA_PERIOD)

    last_close = float(ohlc["Close"].iloc[-1])
    last_d     = float(daily_ema9.iloc[-1])

    # Forward-fill HTF values onto ohlc.index before reading last value
    w_aligned  = w_ema9.reindex(ohlc.index, method="ffill")
    m_aligned  = m_ema9.reindex(ohlc.index, method="ffill")
    last_w     = float(w_aligned.iloc[-1]) if not w_aligned.isna().all() else float("nan")
    last_m     = float(m_aligned.iloc[-1]) if not m_aligned.isna().all() else float("nan")

    above_d = last_close > last_d
    above_w = last_close > last_w  if not np.isnan(last_w) else False
    above_m = last_close > last_m  if not np.isnan(last_m) else False

    return (above_d and above_w and above_m), {
        "close":   last_close,
        "d_ema9":  last_d,
        "w_ema9":  last_w,
        "m_ema9":  last_m,
        "above_d": above_d,
        "above_w": above_w,
        "above_m": above_m,
    }


# ═══════════════════════════════════════════════════════════════
#  STEP 4 — CHART
# ═══════════════════════════════════════════════════════════════

def plot_chart(symbol: str, ohlc: pd.DataFrame,
               tv_cr: float, output_path: str,
               weekly_ema9: pd.Series,
               monthly_ema9: pd.Series):
    s  = STYLE
    n  = len(ohlc)
    xs = np.arange(n)

    daily_ema9        = ema_calc(ohlc["Close"], EMA_PERIOD)
    macd_l, sig, hist = macd_calc(ohlc["Close"])

    lookback           = min(RECENT_LOW_BARS, n)
    recent_window      = ohlc["Low"].iloc[-lookback:]
    recent_low_price   = recent_window.min()
    recent_low_bar_idx = n - lookback + int(recent_window.values.argmin())
    current_close      = ohlc["Close"].iloc[-1]
    pct_below          = (current_close - recent_low_price) / current_close * 100

    fig = plt.figure(figsize=(24, 10), facecolor=s["bg"])
    gs  = gridspec.GridSpec(2, 1, height_ratios=[7, 3],
                            hspace=0.04, top=0.93, bottom=0.08,
                            left=0.05, right=0.96)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    for ax in (ax1, ax2):
        ax.set_facecolor(s["panel_bg"])
        ax.tick_params(colors=s["subtext"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(s["border"])
        ax.grid(True, color=s["grid"], linewidth=0.4, alpha=0.6)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(ohlc.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        up = c >= o
        ax1.bar(i, h - l, bottom=l, width=CANDLE_WICK_WIDTH,
                color=s["wick_up" if up else "wick_down"], zorder=2)
        ax1.bar(i, abs(c - o), bottom=min(o, c), width=CANDLE_BODY_WIDTH,
                color=s["up_candle" if up else "down_candle"], zorder=3)

    # ── Daily EMA9 (orange solid) ─────────────────────────────────────────────
    ax1.plot(xs, daily_ema9.values,
             color=s["ema_color"], linewidth=1.6, linestyle="-",
             label=f"EMA {EMA_PERIOD} (Daily)", zorder=4)

    # ── Weekly EMA9 (purple dashed) ───────────────────────────────────────────
    w_aligned = weekly_ema9.reindex(ohlc.index, method="ffill")
    if not w_aligned.isna().all():
        ax1.plot(xs, w_aligned.values,
                 color=WEEKLY_EMA_COLOR, linewidth=1.5, linestyle="--",
                 label=f"EMA {EMA_PERIOD} (Weekly)", zorder=5, alpha=0.90)

    # ── Monthly EMA9 (cyan dotted) ────────────────────────────────────────────
    m_aligned = monthly_ema9.reindex(ohlc.index, method="ffill")
    if not m_aligned.isna().all():
        ax1.plot(xs, m_aligned.values,
                 color=MONTHLY_EMA_COLOR, linewidth=1.5, linestyle=":",
                 label=f"EMA {EMA_PERIOD} (Monthly)", zorder=5, alpha=0.90)

    # ── Recent-low dashed annotation ─────────────────────────────────────────
    ax1.hlines(recent_low_price, n - lookback, n - 0.5,
               colors=s["recent_low"], linewidths=1.2,
               linestyles="--", zorder=6)
    ax1.plot(recent_low_bar_idx, recent_low_price,
             marker="D", markersize=5, color=s["recent_low"],
             markeredgecolor=s["bg"], markeredgewidth=0.8, zorder=7)
    ax1.annotate(
        f"  {pct_below:.2f}%",
        xy=(n - 1, recent_low_price),
        xytext=(n - 1 + 0.8, recent_low_price),
        color=s["recent_low"], fontsize=7.5, fontweight="bold",
        va="center", ha="left", zorder=8, annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=s["panel_bg"],
                  edgecolor=s["recent_low"], alpha=0.85, linewidth=0.8),
    )

    ax1.set_xlim(-1, n + 1)
    pad = (ohlc["High"].max() - ohlc["Low"].min()) * 0.04
    ax1.set_ylim(ohlc["Low"].min() - pad, ohlc["High"].max() + pad)
    ax1.set_ylabel("Price (₹)", color=s["text"], fontsize=9)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = [
        mpatches.Patch(facecolor=s["up_candle"],   label="Bullish"),
        mpatches.Patch(facecolor=s["down_candle"], label="Bearish"),
        Line2D([0],[0], color=s["ema_color"],     linewidth=1.8, linestyle="-",
               label=f"EMA {EMA_PERIOD} (Daily)"),
        Line2D([0],[0], color=WEEKLY_EMA_COLOR,   linewidth=1.5, linestyle="--",
               label=f"EMA {EMA_PERIOD} (Weekly)"),
        Line2D([0],[0], color=MONTHLY_EMA_COLOR,  linewidth=1.5, linestyle=":",
               label=f"EMA {EMA_PERIOD} (Monthly)"),
        Line2D([0],[0], color=s["recent_low"],    linewidth=1.2, linestyle="--",
               label=f"{RECENT_LOW_BARS}-bar Low"),
    ]
    ax1.legend(handles=leg, loc="upper left", fontsize=7.5,
               framealpha=0.6, facecolor=s["bg"],
               edgecolor=s["border"], labelcolor=s["text"])

    # ── Right-axis price pills ────────────────────────────────────────────────
    lc = float(ohlc["Close"].iloc[-1])
    le = float(daily_ema9.iloc[-1])
    cc = s["up_candle"] if lc >= float(ohlc["Open"].iloc[-1]) else s["down_candle"]
    pills = [(lc, cc, f"₹{lc:,.2f}"),
             (le, s["ema_color"], f"D ₹{le:,.2f}")]
    if not w_aligned.isna().all():
        wv = float(w_aligned.iloc[-1])
        pills.append((wv, WEEKLY_EMA_COLOR, f"W ₹{wv:,.2f}"))
    if not m_aligned.isna().all():
        mv = float(m_aligned.iloc[-1])
        pills.append((mv, MONTHLY_EMA_COLOR, f"M ₹{mv:,.2f}"))

    for val, col, lbl in pills:
        ax1.annotate(lbl,
                     xy=(1, val), xycoords=("axes fraction", "data"),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=7.5, fontweight="bold", color=s["bg"],
                     ha="left", va="center", annotation_clip=False,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=col,
                               edgecolor="none", alpha=0.95))

    # ── MACD ──────────────────────────────────────────────────────────────────
    hcols = [s["hist_up"] if v >= 0 else s["hist_down"] for v in hist.values]
    ax2.bar(xs, hist.values, color=hcols, alpha=0.8, width=0.7, zorder=2,
            label="Histogram")
    ax2.plot(xs, macd_l.values, color=s["macd_line"],   linewidth=1.3,
             label="MACD",   zorder=3)
    ax2.plot(xs, sig.values,    color=s["signal_line"], linewidth=1.1,
             label="Signal", zorder=3)
    ax2.axhline(0, color=s["subtext"], linewidth=0.6, linestyle="--")
    ax2.set_ylabel("MACD", color=s["text"], fontsize=9)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.6,
               facecolor=s["bg"], edgecolor=s["border"],
               labelcolor=s["text"])

    # ── X-axis date labels ────────────────────────────────────────────────────
    step = max(n // 12, 1)
    ax2.set_xticks(xs[::step])
    ax2.set_xticklabels(
        [ohlc.index[i].strftime("%d %b '%y") for i in range(0, n, step)],
        rotation=30, ha="right", fontsize=7.5, color=s["subtext"])
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Title ─────────────────────────────────────────────────────────────────
    lc0  = float(ohlc["Close"].iloc[0])
    pct  = (lc - lc0) / lc0 * 100
    sign = "+" if pct >= 0 else ""
    ccol = s["up_candle"] if pct >= 0 else s["down_candle"]
    date = ohlc.index[-1].strftime("%d %b %Y")

    fig.text(0.05, 0.955, f"{symbol}  |  NSE  |  Daily",
             color=s["text"], fontsize=13, fontweight="bold")
    fig.text(0.05, 0.935,
             f"₹{lc:,.2f}   {sign}{pct:.2f}%  (1Y)"
             f"  │  Today's Value: ₹{tv_cr:,.1f} Cr",
             color=ccol, fontsize=10)
    fig.text(0.96, 0.955, f"Latest Close: {date}",
             color=s["text"], fontsize=9, ha="right", fontweight="bold")
    fig.text(0.96, 0.935,
             f"MACD ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})"
             f"  |  EMA {EMA_PERIOD} Daily / Weekly / Monthly  |  Bars: {n}",
             color=s["subtext"], fontsize=8, ha="right")

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=s["bg"], edgecolor="none")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  STEP 3+4 — SCAN + CHART GENERATOR
# ═══════════════════════════════════════════════════════════════

def scan_and_chart(value_filtered: pd.DataFrame):
    """
    For every stock in value_filtered:
      1. Download daily OHLC (1 year)
      2. Download weekly + monthly OHLC for HTF EMA9
      3. Check: Close > Daily EMA9  AND  Close > Weekly EMA9
                AND  Close > Monthly EMA9
      4. Chart only the stocks that pass all three conditions

    Returns
    -------
    charted  : list of symbols that passed and got charted
    rejected : list of symbols that failed the EMA9 filter
    errored  : list of symbols with download/other errors
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    end_dt    = datetime.datetime.today() + timedelta(days=1)
    start_dt  = end_dt - timedelta(days=PERIOD_DAYS + 1)
    end_str   = end_dt.strftime("%Y-%m-%d")
    start_str = start_dt.strftime("%Y-%m-%d")

    total    = len(value_filtered)
    charted, rejected, errored = [], [], []

    # Track rejection reasons for summary
    reject_log = []

    for idx, row in enumerate(value_filtered.itertuples(), 1):
        sym   = row.Symbol
        tv_cr = getattr(row, "Value (₹ Crores)", 0)
        ticker = sym if sym.endswith(EXCHANGE_SFX) else sym + EXCHANGE_SFX

        print(f"\n  [{idx:>4}/{total}]  {ticker:<22} "
              f"Value: ₹{tv_cr:>8,.1f} Cr")

        try:
            # ── Daily OHLC ────────────────────────────────────────────────────
            ohlc = fetch_ohlc(ticker, start_str, end_str, "1d")
            if ohlc.empty or len(ohlc) < MACD_SLOW + MACD_SIGNAL + 5:
                print(f"         ✗  Insufficient daily data ({len(ohlc)} bars)")
                errored.append(sym)
                continue
            ohlc = ohlc.tail(250)

            # ── Weekly + Monthly EMA9 ─────────────────────────────────────────
            w_ema9, m_ema9 = get_htf_ema9(ticker, ohlc.index)

            # ── EMA9 triple-timeframe filter ──────────────────────────────────
            passes, info = passes_ema9_filter(ohlc, w_ema9, m_ema9)

            d_mark = "✔" if info["above_d"] else "✗"
            w_mark = "✔" if info["above_w"] else "✗"
            m_mark = "✔" if info["above_m"] else "✗"

            print(f"         Close  ₹{info['close']:>9,.2f}"
                  f"  |  D-EMA9 ₹{info['d_ema9']:>9,.2f} {d_mark}"
                  f"  |  W-EMA9 ₹{info['w_ema9']:>9,.2f} {w_mark}"
                  f"  |  M-EMA9 ₹{info['m_ema9']:>9,.2f} {m_mark}")

            if not passes:
                failed_on = [tf for tf, ok in
                             [("Daily", info["above_d"]),
                              ("Weekly", info["above_w"]),
                              ("Monthly", info["above_m"])]
                             if not ok]
                reason = "Close below " + " & ".join(failed_on) + " EMA9"
                print(f"         ⛔  REJECTED — {reason}")
                rejected.append(sym)
                reject_log.append({"Symbol": sym, "Reason": reason,
                                   **{k: round(v, 2) for k, v in info.items()}})
                continue

            # ── All 3 conditions passed — save chart ──────────────────────────
            print(f"         ✅  PASS — saving chart …")
            out = os.path.join(OUTPUT_DIR, f"{sym}.png")
            plot_chart(sym, ohlc, tv_cr, out,
                       weekly_ema9=w_ema9,
                       monthly_ema9=m_ema9)
            print(f"         📊  {len(ohlc)} bars  →  {out}")
            charted.append(sym)

        except Exception:
            print(f"         ✗  Exception")
            traceback.print_exc()
            errored.append(sym)

    # Save rejection log
    if reject_log:
        pd.DataFrame(reject_log).to_csv("nse_ema9_rejected.csv", index=False)

    return charted, rejected, errored


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NSE Stocks Traded → Value Filter → EMA9 Triple Filter → Charts"
    )
    parser.add_argument("--visible", action="store_true",
                        help="Run Chrome with a visible window")
    parser.add_argument("--from-csv", metavar="FILE",
                        help="Skip browser — use a saved NSE CSV file")
    args   = parser.parse_args()
    headless = not args.visible

    run_time = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
    print(f"\n{'═'*70}")
    print(f"  NSE Pipeline  —  {run_time}")
    print(f"  Step 1 : Download NSE Stocks Traded")
    print(f"  Step 2 : Filter  Value > ₹{TRADED_VALUE_MIN_CR} Cr  (EQ series)")
    print(f"  Step 3 : EMA9 Filter — Close > Daily EMA9")
    print(f"                        AND Close > Weekly EMA9")
    print(f"                        AND Close > Monthly EMA9")
    print(f"  Step 4 : Chart qualifying stocks  →  {OUTPUT_DIR}/")
    print(f"{'═'*70}")

    # ── STEP 1: NSE download ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 1  —  NSE data download")
    print(f"{'─'*70}")
    all_df = download_nse_data(
        headless=headless,
        from_csv=getattr(args, "from_csv", None),
    )
    print(f"\n  Total records : {len(all_df)}")
    all_df.to_csv("nse_all_stocks.csv", index=False)
    print(f"  Saved         : nse_all_stocks.csv")

    # ── STEP 2: Value filter ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEP 2  —  Value filter: > ₹{TRADED_VALUE_MIN_CR} Cr  (EQ only)")
    print(f"{'─'*70}")
    value_filtered = filter_by_value(all_df)
    print(f"\n  Stocks after value filter : {len(value_filtered)}")

    if value_filtered.empty:
        print("\n  ⚠  No stocks passed the value filter.")
        print("     Check nse_all_stocks.csv → 'Value (₹ Crores)' column.")
        sys.exit(0)

    # Print top 25
    print(f"\n  {'#':>4}  {'Symbol':<12} {'Company':<26} "
          f"{'LTP':>8}  {'Value (Cr)':>11}  {'%Chg':>7}")
    print(f"  {'─'*75}")
    for i, row in value_filtered.head(25).iterrows():
        print(f"  {i+1:>4}  {row['Symbol']:<12} "
              f"{str(row.get('Company',''))[:24]:<26} "
              f"  ₹{row['LTP (₹)']:>7,.2f}"
              f"  ₹{row['Value (₹ Crores)']:>9,.1f} Cr"
              f"  {row['% Change']:>+7.2f}%")
    if len(value_filtered) > 25:
        print(f"  … {len(value_filtered) - 25} more")

    value_filtered.to_csv("nse_value_filtered.csv", index=False)
    print(f"\n  Saved : nse_value_filtered.csv  ({len(value_filtered)} stocks)")

    # ── STEPS 3+4: EMA9 filter + chart ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  STEPS 3+4  —  EMA9 triple-timeframe scan + charts")
    print(f"                Close > Daily EMA9  AND  Weekly EMA9  AND  Monthly EMA9")
    print(f"{'─'*70}")
    charted, rejected, errored = scan_and_chart(value_filtered)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  PIPELINE COMPLETE  —  {run_time}")
    print(f"  NSE records downloaded  : {len(all_df)}")
    print(f"  After value filter      : {len(value_filtered)}  "
          f"(Value > ₹{TRADED_VALUE_MIN_CR} Cr)")
    print(f"  ✅  Passed EMA9 filter  : {len(charted)}  →  {OUTPUT_DIR}/")
    print(f"  ⛔  Rejected by EMA9    : {len(rejected)}")
    print(f"  ✗   Errors/no data      : {len(errored)}")
    if charted:
        print(f"\n  Charted symbols:")
        for i in range(0, len(charted), 10):
            print(f"    {', '.join(charted[i:i+10])}")
    print(f"\n  Output files:")
    print(f"    nse_all_stocks.csv       — full NSE download")
    print(f"    nse_value_filtered.csv   — after value filter")
    print(f"    nse_ema9_rejected.csv    — rejection details")
    print(f"    {OUTPUT_DIR}/            — charts for passing stocks")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
