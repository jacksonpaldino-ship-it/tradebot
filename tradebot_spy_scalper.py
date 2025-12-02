"""
hybrid_tradebot_spy_signals.py

Hybrid trading bot:
- Primary: internal SPY scalper strategy (VWAP + EMA9 + volume + spread)
- Fallback: Google Sheet signals (CSV export) after a specified time if no trade executed
- Features: CSV trade log (trades.csv), stats (trade_stats.json), robust Alpaca calls with retries,
  in-process monitoring and exit logic (TP/SL), verbose prints for GitHub Actions logs.

SETUP for Google Sheets signal source:
1) Create a Google Sheet with a header row and columns (at minimum):
   symbol, action, qty, tp_dollars, sl_dollars, enabled, notes
   Example row:
   SPY,BUY,1,0.50,0.35,TRUE,test signal
2) Publish or "Share -> Get link" such that it can be exported as CSV. For simple usage:
   - Use the CSV export URL pattern:
     https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/export?format=csv&gid=<SHEET_GID>
   - Put that URL in SIGNAL_SHEET_CSV_URL below.
   - Make sure the sheet is accessible with the link (anyone with link) OR use other auth if you need privacy.
3) The bot reads the CSV and looks for the first row with enabled==TRUE (case-insensitive)
   and action==BUY (or SELL if you implement shorting). This is executed as the fallback trade.

ENV:
- ALPACA_API_KEY, ALPACA_SECRET_KEY must be in environment variables.
- The script uses Alpaca TradingClient (alpaca-py). Ensure dependencies installed:
  pip install alpaca-py yfinance pandas numpy pytz

NOTE: The script uses in-process monitoring after entry to handle TP/SL. The runner must remain alive
for MONITOR_TIMEOUT seconds at most.

TUNE the constants below to your risk tolerance and desired behavior.
"""

import os
import time
import json
import csv
from datetime import datetime, timezone, timedelta
import math

import pandas as pd
import numpy as np
import pytz
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -----------------------
# CONFIG
# -----------------------
# Primary strategy default (applies to SPY)
PRIMARY_SYMBOL = "SPY"

# Primary filters (kept relaxed for daily activity)
MIN_VOLUME = 200_000
MAX_SPREAD = 0.80
VWAP_ALLOWED_DISTANCE = 0.25
TRADE_QTY = 1

# Exit strategy (in USD)
TP_DOLLARS = 0.50     # take-profit above entry price
SL_DOLLARS = 0.35     # stop-loss below entry price

# Monitoring & timeouts
MONITOR_INTERVAL = 15      # seconds between price checks after entry
MONITOR_TIMEOUT = 60 * 60  # seconds to monitor before force-sell (1 hour)

# Files
TRADE_LOG_CSV = "trades.csv"
STATS_FILE = "trade_stats.json"

# Alpaca keys (from env)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # set to False for live

# SIGNAL (Google Sheet CSV) fallback
# Put your Google Sheet CSV export URL here. Example:
# "https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0"
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL", "").strip()

# Time (ET) to switch to signals fallback if no trade executed yet (local market time)
# Format: "HH:MM" in America/New_York timezone. Default 14:00 ET (2:00 PM ET).
SIGNAL_FALLBACK_TIME_ET = "14:00"

# If set True, fallback will only execute one signal per day
SIGNAL_ONLY_ONE_PER_DAY = True

# Whether to permit signals to override TP/SL if provided by sheet
ALLOW_SIGNAL_TP_SL = True

# -----------------------
# Utilities
# -----------------------
def now_utc():
    return datetime.now(timezone.utc)

def now_et():
    tz = pytz.timezone("America/New_York")
    return datetime.now(tz)

def now_str():
    return now_utc().astimezone().isoformat()

def safe_float(val):
    try:
        return float(val)
    except Exception:
        try:
            return float(val.iloc[0])
        except Exception:
            raise

def backoff_attempt(func, *args, max_attempts=5, base_delay=1.0, **kwargs):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                print(f"[backoff] Failed after {attempt} attempts: {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1)) + (0.1 * attempt)
            print(f"[backoff] Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

# -----------------------
# CSV & stats helpers
# -----------------------
def ensure_trade_log_exists():
    if not os.path.exists(TRADE_LOG_CSV):
        with open(TRADE_LOG_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "source",    # PRIMARY or SIGNAL
                "symbol",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "result",  # WIN / LOSS / TIMEOUT / ERROR
                "notes"
            ])

def append_trade_log(row: dict):
    ensure_trade_log_exists()
    with open(TRADE_LOG_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get("timestamp", ""),
            row.get("source", ""),
            row.get("symbol", ""),
            row.get("side", ""),
            row.get("qty", ""),
            f"{row.get('entry_price',''):.2f}" if row.get("entry_price") is not None else "",
            f"{row.get('exit_price',''):.2f}" if row.get("exit_price") is not None else "",
            f"{row.get('pnl',''):.2f}" if row.get("pnl") is not None else "",
            row.get("result", ""),
            row.get("notes", ""),
        ])

def load_stats():
    if not os.path.exists(STATS_FILE):
        return {"trades": 0, "wins": 0, "losses": 0, "last_signal_date": None}
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"trades": 0, "wins": 0, "losses": 0, "last_signal_date": None}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

# -----------------------
# Alpaca client init
# -----------------------
if not API_KEY or not SECRET_KEY:
    print("Missing Alpaca API keys in environment. Aborting.")
    exit(1)

client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

# -----------------------
# Primary strategy function (returns True if trade executed)
# -----------------------
def run_primary_strategy():
    symbol = PRIMARY_SYMBOL
    print(f"[{now_str()}] Running primary strategy for {symbol}")

    # Fetch 5m bars with yfinance
    df = yf.download(symbol, period="3d", interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        print("[primary] No data returned from yfinance. Skipping primary.")
        return False

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["EMA9"] = df["Close"].ewm(span=9).mean()

    row = df.iloc[-1]
    price = safe_float(row["Close"])
    vwap = safe_float(row["VWAP"])
    volume = safe_float(row["Volume"])
    candle_spread = safe_float(row["High"]) - safe_float(row["Low"])
    ema9 = safe_float(row["EMA9"])

    print(f"[{now_str()}][primary] Price: {price:.2f}, VWAP: {vwap:.2f}, Vol: {int(volume)}, Spread: {candle_spread:.2f}, EMA9: {ema9:.2f}")

    # Filters
    if volume < MIN_VOLUME:
        print(f"[primary] Volume {int(volume)} < {MIN_VOLUME}. Skipping.")
        return False
    if candle_spread > MAX_SPREAD:
        print(f"[primary] Spread {candle_spread:.2f} > {MAX_SPREAD}. Skipping.")
        return False
    if abs(price - vwap) > VWAP_ALLOWED_DISTANCE:
        print(f"[primary] Price {price:.2f} is {abs(price-vwap):.2f} from VWAP > {VWAP_ALLOWED_DISTANCE}. Skipping.")
        return False
    if price < ema9:
        print(f"[primary] Price {price:.2f} < EMA9 {ema9:.2f}. Skipping long trade.")
        return False

    # If reached here, place buy for PRIMARY_SYMBOL with configured TP/SL
    print(f"[{now_str()}][primary] Conditions met for {symbol}. Executing trade.")
    execute_trade(
        source="PRIMARY",
        symbol=symbol,
        side="BUY",
        qty=TRADE_QTY,
        tp_dollars=TP_DOLLARS,
        sl_dollars=SL_DOLLARS
    )
    return True

# -----------------------
# Signals fetcher (Google Sheet CSV)
# -----------------------
def fetch_signals_from_sheet(url):
    """
    Fetch CSV from URL and return a DataFrame.
    Expect columns: symbol, action, qty, tp_dollars, sl_dollars, enabled, notes
    """
    if not url:
        print("[signals] No SIGNAL_SHEET_CSV_URL configured.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(url)
        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[signals] Failed to fetch/parse CSV from sheet: {e}")
        return pd.DataFrame()

def choose_signal(df):
    """
    Choose the first enabled BUY signal row. Return dict or None.
    """
    if df.empty:
        return None
    # Standardize expected cols
    if "enabled" in df.columns:
        enabled_mask = df["enabled"].astype(str).str.strip().str.lower().isin(["true","1","yes","y"])
    else:
        enabled_mask = pd.Series([True]*len(df))
    df = df[enabled_mask]
    if df.empty:
        return None
    # prefer action == buy (case-insensitive)
    if "action" in df.columns:
        df_buy = df[df["action"].astype(str).str.strip().str.lower() == "buy"]
        if not df_buy.empty:
            row = df_buy.iloc[0]
        else:
            # take first enabled row
            row = df.iloc[0]
    else:
        row = df.iloc[0]

    # Parse fields safely
    sig = {}
    sig["symbol"] = str(row.get("symbol", "")).strip().upper()
    sig["action"] = str(row.get("action", "BUY")).strip().upper()
    try:
        sig["qty"] = int(row.get("qty",TRADE_QTY))
    except Exception:
        sig["qty"] = TRADE_QTY
    try:
        sig["tp_dollars"] = float(row.get("tp_dollars", TP_DOLLARS))
    except Exception:
        sig["tp_dollars"] = TP_DOLLARS
    try:
        sig["sl_dollars"] = float(row.get("sl_dollars", SL_DOLLARS))
    except Exception:
        sig["sl_dollars"] = SL_DOLLARS
    sig["notes"] = str(row.get("notes","")).strip()
    return sig

# -----------------------
# Execute trade (BUY only for now)
# -----------------------
def execute_trade(source, symbol, side, qty, tp_dollars, sl_dollars):
    """
    Places a market order (BUY). Monitors market price, then issues a MARKET SELL when TP/SL hit.
    Logs trade to CSV and updates stats.
    """
    side = side.upper()
    if side != "BUY":
        print(f"[{now_str()}] Only BUY side supported currently. Received side={side}. Skipping.")
        return

    print(f"[{now_str()}][execute] Placing MARKET BUY {qty} {symbol} (source={source})")
    buy_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )

    try:
        buy_order = backoff_attempt(client.submit_order, buy_request)
        print(f"[{now_str()}][execute] Buy submitted: {buy_order}")
    except Exception as e:
        print(f"[{now_str()}][execute] Buy order failed: {e}")
        append_trade_log({
            "timestamp": now_str(),
            "source": source,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": None,
            "exit_price": None,
            "pnl": None,
            "result": "ERROR",
            "notes": f"Buy failed: {e}"
        })
        return

    # Determine entry fill price
    entry_price = None
    order_id = getattr(buy_order, "id", None)
    if not order_id:
        try:
            entry_price = float(getattr(buy_order, "filled_avg_price", None) or getattr(buy_order, "filled_avg_price", None))
        except Exception:
            entry_price = None

    poll_attempts = 0
    max_poll = 10
    while (entry_price is None) and (poll_attempts < max_poll):
        time.sleep(1 + poll_attempts * 0.5)
        poll_attempts += 1
        try:
            if order_id:
                o = backoff_attempt(client.get_order, order_id)
                filled_avg = getattr(o, "filled_avg_price", None) or getattr(o, "filled_avg_price", None)
                if filled_avg:
                    entry_price = safe_float(filled_avg)
                    break
        except Exception as e:
            print(f"[order poll] attempt {poll_attempts} error: {e}")

    if entry_price is None:
        print("[warn] Couldn't get filled price from Alpaca; using recent market price as entry estimate.")
        try:
            recent = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not recent.empty:
                entry_price = safe_float(recent["Close"].iloc[-1])
            else:
                entry_price = None
        except Exception:
            entry_price = None

    if entry_price is None:
        print("[execute] Could not determine entry price; logging error and aborting sell monitoring.")
        append_trade_log({
            "timestamp": now_str(),
            "source": source,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": None,
            "exit_price": None,
            "pnl": None,
            "result": "ERROR",
            "notes": "No entry price found"
        })
        return

    entry_price = float(entry_price)
    print(f"[{now_str()}][execute] Entry price estimated: {entry_price:.4f}")

    # Monitor and exit
    tp_price = entry_price + tp_dollars
    sl_price = entry_price - sl_dollars
    print(f"[{now_str()}][execute] Monitoring for TP @ {tp_price:.4f} or SL @ {sl_price:.4f} (timeout {MONITOR_TIMEOUT}s)")

    start_monitor = time.time()
    exit_price = None
    result = None
    notes = ""

    while True:
        elapsed = time.time() - start_monitor
        if elapsed > MONITOR_TIMEOUT:
            notes = "Timeout reached"
            result = "TIMEOUT"
            break

        try:
            recent = yf.download(symbol, period="2d", interval="1m", progress=False, auto_adjust=True)
            if recent.empty:
                print("[monitor] no recent data; retrying")
                time.sleep(MONITOR_INTERVAL)
                continue
            last_px = safe_float(recent["Close"].iloc[-1])
        except Exception as e:
            print(f"[monitor] error fetching recent price: {e}")
            time.sleep(MONITOR_INTERVAL)
            continue

        print(f"[{now_str()}][monitor] Price: {last_px:.4f} (tp {tp_price:.4f}, sl {sl_price:.4f})")

        if last_px >= tp_price:
            result = "WIN"
            notes = "TP hit"
            exit_price = last_px
            break
        if last_px <= sl_price:
            result = "LOSS"
            notes = "SL hit"
            exit_price = last_px
            break

        time.sleep(MONITOR_INTERVAL)

    if exit_price is None:
        try:
            recent = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not recent.empty:
                exit_price = safe_float(recent["Close"].iloc[-1])
            else:
                exit_price = entry_price
        except Exception:
            exit_price = entry_price

    print(f"[{now_str()}][execute] Attempting SELL market {qty} {symbol} at market (exit estimate {exit_price:.4f}) for reason: {result} / {notes}")

    sell_request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    try:
        sell_order = backoff_attempt(client.submit_order, sell_request)
        print(f"[{now_str()}][execute] Sell submitted: {sell_order}")
    except Exception as e:
        print(f"[{now_str()}][execute] Sell order failed: {e}")
        notes += f" | Sell failed: {e}"
        result = result or "ERROR"

    pnl = (exit_price - entry_price) * qty
    pnl = round(pnl, 4)

    trade_row = {
        "timestamp": now_str(),
        "source": source,
        "symbol": symbol,
        "side": "LONG",
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "result": result or "UNKNOWN",
        "notes": notes
    }
    append_trade_log(trade_row)
    print(f"[{now_str()}][execute] Trade logged: PnL {pnl:.4f}, result: {result}, notes: {notes}")

    stats = load_stats()
    stats["trades"] = stats.get("trades", 0) + 1
    if result == "WIN":
        stats["wins"] = stats.get("wins", 0) + 1
    elif result == "LOSS":
        stats["losses"] = stats.get("losses", 0) + 1
    # track last signal date if this was a signal
    if source == "SIGNAL":
        stats["last_signal_date"] = now_et().strftime("%Y-%m-%d")
    save_stats(stats)
    print(f"[{now_str()}][execute] Updated stats: {stats}")

# -----------------------
# Fallback signal routine
# -----------------------
def run_signal_fallback_if_needed():
    """
    If no primary trade executed yet today and current ET >= SIGNAL_FALLBACK_TIME_ET,
    fetch signals from the sheet and execute the first enabled BUY signal.
    """
    print(f"[{now_str()}][signals] Checking fallback conditions")
    # Check time
    tz = pytz.timezone("America/New_York")
    current_et = datetime.now(tz)
    hhmm = current_et.strftime("%H:%M")
    if hhmm < SIGNAL_FALLBACK_TIME_ET:
        print(f"[signals] Current ET {hhmm} < fallback time {SIGNAL_FALLBACK_TIME_ET}. Not using signals yet.")
        return False

    # Check stats to avoid multiple signals per day if configured
    stats = load_stats()
    last_signal_date = stats.get("last_signal_date")
    today_str = current_et.strftime("%Y-%m-%d")
    if SIGNAL_ONLY_ONE_PER_DAY and last_signal_date == today_str:
        print(f"[signals] Already executed a signal today ({last_signal_date}). Skipping further signals.")
        return False

    df = fetch_signals_from_sheet(SIGNAL_SHEET_CSV_URL)
    if df.empty:
        print("[signals] No signals found in sheet.")
        return False

    sig = choose_signal(df)
    if not sig:
        print("[signals] No enabled signal rows found.")
        return False

    print(f"[{now_str()}][signals] Chosen signal: {sig}")
    # Execute the signal
    # Only BUY implemented
    if sig["action"].upper() != "BUY":
        print(f"[signals] Action {sig['action']} is not BUY. Skipping.")
        return False

    # Allow the signal to override TP/SL if enabled
    tp = sig["tp_dollars"] if ALLOW_SIGNAL_TP_SL else TP_DOLLARS
    sl = sig["sl_dollars"] if ALLOW_SIGNAL_TP_SL else SL_DOLLARS

    execute_trade(
        source="SIGNAL",
        symbol=sig["symbol"],
        side="BUY",
        qty=sig["qty"],
        tp_dollars=tp,
        sl_dollars=sl
    )
    return True

# -----------------------
# Main
# -----------------------
def main():
    print(f"[{now_str()}] Hybrid bot start")
    # Run primary strategy
    primary_executed = False
    try:
        primary_executed = run_primary_strategy()
    except Exception as e:
        print(f"[{now_str()}] Error running primary strategy: {e}")

    if primary_executed:
        print("[main] Primary executed a trade. Done for this run.")
        return

    # If no primary trade, check whether it's time to fallback to signals
    try:
        fallback_executed = run_signal_fallback_if_needed()
        if fallback_executed:
            print("[main] Signal fallback executed a trade. Done for this run.")
            return
        else:
            print("[main] No trade executed (primary skipped; signals not used or none).")
            return
    except Exception as e:
        print(f"[{now_str()}] Error during signal fallback: {e}")
        return

if __name__ == "__main__":
    main()
