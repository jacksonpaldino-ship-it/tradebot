"""
tradebot_spy_scalper.py

Features added:
- CSV trade logging (trades.csv) with entry/exit/PnL
- Win/loss tracking (stats.json)
- Robust Alpaca API calls with retries/backoff
- Automatic sell logic: monitor market price after buy and exit on TP or SL
- Clean yfinance usage (no FutureWarnings)
- Safe extraction of series values (no float(series) warnings)
- Verbose prints for GitHub Actions logs

Notes:
- This script places a MARKET buy, then monitors price and issues a MARKET sell when TP/SL hit.
- Monitoring is done in-process (polling market price every MONITOR_INTERVAL seconds), so the runner must stay alive while monitoring.
- Adjust TRADE_QTY, TP_DOLLARS, SL_DOLLARS, MONITOR_TIMEOUT as you like.
"""

import os
import time
import json
import csv
import math
from datetime import datetime, timezone, timedelta

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
SYMBOL = "SPY"

# Filters (kept relaxed to get ~1+ trade/day)
MIN_VOLUME = 200_000
MAX_SPREAD = 0.80
VWAP_ALLOWED_DISTANCE = 0.25
TRADE_QTY = 1

# Exit strategy (in USD)
TP_DOLLARS = 0.50     # take-profit amount above entry price
SL_DOLLARS = 0.35     # stop-loss amount below entry price

# Monitoring & timeouts
MONITOR_INTERVAL = 15      # seconds between price checks after entry
MONITOR_TIMEOUT = 60 * 60  # seconds to monitor before force-sell (1 hour)

# Files
TRADE_LOG_CSV = "trades.csv"
STATS_FILE = "trade_stats.json"

# Alpaca keys (expect to be in env)
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # set to False if using live

# -----------------------
# Utilities
# -----------------------
def now_str():
    return datetime.now(timezone.utc).astimezone().isoformat()

def safe_float(val):
    # accept numpy/pandas scalar or python scalar
    try:
        return float(val)
    except Exception:
        # fallback if it's a Series with one element
        try:
            return float(val.iloc[0])
        except Exception:
            raise

def backoff_attempt(func, *args, max_attempts=5, base_delay=1.0, **kwargs):
    """
    Call func(*args, **kwargs) with retry/backoff. Returns func result or raises last exception.
    """
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
        return {"trades": 0, "wins": 0, "losses": 0}
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"trades": 0, "wins": 0, "losses": 0}

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
# Download data
# -----------------------
print(f"[{now_str()}] Fetching 5m data for {SYMBOL}")
df = yf.download(
    SYMBOL,
    period="3d",
    interval="5m",
    progress=False,
    auto_adjust=True
)

if df.empty:
    print("No data returned from yfinance. Exiting.")
    exit(0)

# indicators
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
df["EMA9"] = df["Close"].ewm(span=9).mean()

row = df.iloc[-1]
price = safe_float(row["Close"])
vwap = safe_float(row["VWAP"])
volume = safe_float(row["Volume"])
candle_spread = safe_float(row["High"]) - safe_float(row["Low"])
ema9 = safe_float(row["EMA9"])

print(f"[{now_str()}] Price: {price:.2f}, VWAP: {vwap:.2f}, Vol: {int(volume)}, Spread: {candle_spread:.2f}, EMA9: {ema9:.2f}")

# -----------------------
# Filters
# -----------------------
if volume < MIN_VOLUME:
    print(f"Volume {int(volume)} < {MIN_VOLUME}. Skipping.")
    exit(0)

if candle_spread > MAX_SPREAD:
    print(f"Spread {candle_spread:.2f} > {MAX_SPREAD}. Skipping.")
    exit(0)

if abs(price - vwap) > VWAP_ALLOWED_DISTANCE:
    print(f"Price {price:.2f} is {abs(price-vwap):.2f} from VWAP > {VWAP_ALLOWED_DISTANCE}. Skipping.")
    exit(0)

if price < ema9:
    print(f"Price {price:.2f} < EMA9 {ema9:.2f}. Skipping long trade.")
    exit(0)

# -----------------------
# Place BUY order (market)
# -----------------------
print(f"[{now_str()}] Conditions met: placing MARKET BUY {TRADE_QTY} {SYMBOL}")

order_request = MarketOrderRequest(
    symbol=SYMBOL,
    qty=TRADE_QTY,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

try:
    buy_order = backoff_attempt(client.submit_order, order_request)
    print(f"[{now_str()}] Buy order submitted: {buy_order}")
except Exception as e:
    print(f"[{now_str()}] Buy order failed: {e}")
    # log failed attempt
    append_trade_log({
        "timestamp": now_str(),
        "symbol": SYMBOL,
        "side": "BUY",
        "qty": TRADE_QTY,
        "entry_price": None,
        "exit_price": None,
        "pnl": None,
        "result": "ERROR",
        "notes": f"Buy failed: {e}"
    })
    exit(1)

# Try to determine fill price. We'll poll order status a few times.
entry_price = None
order_id = getattr(buy_order, "id", None)
if not order_id:
    # Some Alpaca clients return object differently; attempt to extract filled_avg_price
    try:
        entry_price = float(buy_order.filled_avg_price)
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
            # check if filled or partially filled
            filled_avg = getattr(o, "filled_avg_price", None)
            if filled_avg:
                entry_price = safe_float(filled_avg)
                break
            # some clients expose 'filled_avg_price'
            filled_avg2 = getattr(o, "filled_avg_price", None)
            if filled_avg2:
                entry_price = safe_float(filled_avg2)
                break
    except Exception as e:
        print(f"[order poll] attempt {poll_attempts} error: {e}")

# Fallback: use recent market price if we couldn't get fill price
if entry_price is None:
    print("[warn] Couldn't get filled price from Alpaca; fetching recent market price as entry estimate.")
    try:
        recent = yf.download(SYMBOL, period="1d", interval="1m", progress=False, auto_adjust=True)
        if not recent.empty:
            entry_price = safe_float(recent["Close"].iloc[-1])
        else:
            entry_price = price  # last candle close from earlier
    except Exception:
        entry_price = price

print(f"[{now_str()}] Entry price estimated: {entry_price:.2f}")

# -----------------------
# Monitor for TP/SL or timeout, then SELL market
# -----------------------
tp_price = entry_price + TP_DOLLARS
sl_price = entry_price - SL_DOLLARS

print(f"[{now_str()}] Monitoring for TP @ {tp_price:.2f} or SL @ {sl_price:.2f} (timeout {MONITOR_TIMEOUT}s)")

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

    # fetch latest trade price (use yfinance 1m)
    try:
        recent = yf.download(SYMBOL, period="2d", interval="1m", progress=False, auto_adjust=True)
        if recent.empty:
            print("[monitor] no recent data; retrying")
            time.sleep(MONITOR_INTERVAL)
            continue
        last_px = safe_float(recent["Close"].iloc[-1])
    except Exception as e:
        print(f"[monitor] error fetching recent price: {e}")
        time.sleep(MONITOR_INTERVAL)
        continue

    # debug print
    print(f"[{now_str()}] Monitor price: {last_px:.2f} (tp {tp_price:.2f}, sl {sl_price:.2f})")

    # check TP/SL
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

# If exit_price not set but timed out, we will use market price at time of forced exit
if exit_price is None:
    # fetch last available price to use as exit
    try:
        recent = yf.download(SYMBOL, period="1d", interval="1m", progress=False, auto_adjust=True)
        if not recent.empty:
            exit_price = safe_float(recent["Close"].iloc[-1])
        else:
            exit_price = entry_price  # fallback
    except Exception:
        exit_price = entry_price

print(f"[{now_str()}] Attempting to SELL market {TRADE_QTY} {SYMBOL} at market (exit estimate {exit_price:.2f}) for reason: {result} / {notes}")

sell_request = MarketOrderRequest(
    symbol=SYMBOL,
    qty=TRADE_QTY,
    side=OrderSide.SELL,
    time_in_force=TimeInForce.DAY
)

try:
    sell_order = backoff_attempt(client.submit_order, sell_request)
    print(f"[{now_str()}] Sell order submitted: {sell_order}")
except Exception as e:
    print(f"[{now_str()}] Sell order failed: {e}")
    # we'll still log the trade using the estimated exit price
    notes += f" | Sell failed: {e}"
    result = result or "ERROR"

# Calculate PnL (approx)
pnl = (exit_price - entry_price) * TRADE_QTY
pnl = round(pnl, 4)

# Log trade to CSV
trade_row = {
    "timestamp": now_str(),
    "symbol": SYMBOL,
    "side": "LONG",
    "qty": TRADE_QTY,
    "entry_price": entry_price,
    "exit_price": exit_price,
    "pnl": pnl,
    "result": result or "UNKNOWN",
    "notes": notes
}
append_trade_log(trade_row)
print(f"[{now_str()}] Trade logged: PnL {pnl:.4f}, result: {result}, notes: {notes}")

# Update stats
stats = load_stats()
stats["trades"] = stats.get("trades", 0) + 1
if result == "WIN":
    stats["wins"] = stats.get("wins", 0) + 1
elif result == "LOSS":
    stats["losses"] = stats.get("losses", 0) + 1
save_stats(stats)
print(f"[{now_str()}] Updated stats: {stats}")

print(f"[{now_str()}] Run complete.")
