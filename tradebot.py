# tradebot.py
"""
Hybrid Trend + Mean-Reversion Tradebot
- Daily trend confirmation (SMA 20/50)
- Hourly entry signal (SMA 5/20)
- RSI filter for entry (buy dip when RSI < RSI_BUY_THRESH)
- ATR-based stop sizing
- Bracket orders (market entry + stop + take-profit)
- Dry-run mode available
"""

import os
import json
import math
import time
from datetime import datetime, date, time as dtime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi

# -----------------------
# USER CONFIG
# -----------------------
SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "PLTR", "CRSP"]  # your watchlist
DRY_RUN = False  # True = simulate; False = actually submit to Alpaca
PAPER = True    # keep True for paper trading; switch BASE_URL if going live

# Allocation / risk
EQUITY_RISK_PCT = 0.005      # risk per trade as fraction of equity (0.5%)
MAX_ALLOC_PER_TRADE_PCT = 0.20  # max allocation (dollar) per trade ~20% of equity
MAX_TOTAL_EXPOSURE_PCT = 0.80  # don't exceed 80% total exposure

# SMA windows
DAILY_SHORT = 20
DAILY_LONG = 50
HOUR_SHORT = 5
HOUR_LONG = 20

# Technical filters
RSI_PERIOD = 14
RSI_BUY_THRESH = 40    # allow buys when hourly RSI <= this (buy the dip)
RSI_SELL_THRESH = 60   # optional extra filter (not used for sells here)

# Stop/take sizes
MIN_STOP_PCT = 0.01      # minimum stop distance (1%)
MAX_STOP_PCT = 0.15      # max stop (15%)
TP_MULTIPLIER = 2.0      # take-profit = TP_MULTIPLIER * stop_distance

# Cooldown & safety
MAX_TRADES_PER_SYMBOL_PER_DAY = 1
TRADE_HISTORY_FILE = "trade_history.json"

# Market hours (Eastern time)
TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# Alpaca credentials from environment
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"

# -----------------------
# Alpaca init
# -----------------------
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version="v2")

# -----------------------
# Utility helpers
# -----------------------
def is_market_open():
    now = datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def load_trade_history():
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_trade_history(h):
    with open(TRADE_HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)

def trades_today_for(symbol):
    h = load_trade_history()
    today = date.today().isoformat()
    return h.get(today, {}).get(symbol, 0)

def record_trade_today(symbol):
    h = load_trade_history()
    today = date.today().isoformat()
    if today not in h:
        h[today] = {}
    h[today][symbol] = h[today].get(symbol, 0) + 1
    save_trade_history(h)

def get_account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Failed to fetch account equity:", e)
        return None

def total_open_exposure():
    try:
        positions = api.list_positions()
        return sum(float(p.market_value) for p in positions)
    except Exception:
        return 0.0

def get_position_qty(symbol):
    try:
        p = api.get_position(symbol)
        return int(float(p.qty))
    except Exception:
        return 0

# -----------------------
# Indicators: ATR and RSI
# -----------------------
def compute_atr(df, n=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------
# Data fetchers
# -----------------------
def fetch_daily(symbol, days="250d"):
    df = yf.download(symbol, period=days, interval="1d", progress=False)
    if df.empty:
        return df
    df["sma_short"] = df["Close"].rolling(DAILY_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(DAILY_LONG).mean()
    return df.dropna()

def fetch_hourly(symbol, days="90d"):
    df = yf.download(symbol, period=days, interval="1h", progress=False)
    if df.empty:
        return df
    # handle MultiIndex from yfinance occasionally
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    df["sma_short"] = df["Close"].rolling(HOUR_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(HOUR_LONG).mean()
    df["atr14"] = compute_atr(df, 14)
    df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
    return df.dropna()

# -----------------------
# Signal logic
# -----------------------
def daily_trend(symbol):
    df = fetch_daily(symbol)
    if df.empty or len(df) < DAILY_LONG + 2:
        return "HOLD"
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    # safe conversion to scalars
    try:
        latest_short = float(latest["sma_short"].iloc[0] if hasattr(latest["sma_short"], "iloc") else latest["sma_short"])
    except Exception:
        latest_short = float(latest["sma_short"])
    try:
        latest_long = float(latest["sma_long"].iloc[0] if hasattr(latest["sma_long"], "iloc") else latest["sma_long"])
    except Exception:
        latest_long = float(latest["sma_long"])
    try:
        prev_short = float(prev["sma_short"].iloc[0] if hasattr(prev["sma_short"], "iloc") else prev["sma_short"])
        prev_long = float(prev["sma_long"].iloc[0] if hasattr(prev["sma_long"], "iloc") else prev["sma_long"])
    except Exception:
        return "HOLD"

    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    if prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    return "HOLD"

def hourly_entry_signal(symbol):
    df = fetch_hourly(symbol)
    if df.empty or len(df) < HOUR_LONG + 2:
        return "HOLD", None
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    try:
        latest_short = float(latest["sma_short"].iloc[0] if hasattr(latest["sma_short"], "iloc") else latest["sma_short"])
        latest_long = float(latest["sma_long"].iloc[0] if hasattr(latest["sma_long"], "iloc") else latest["sma_long"])
        prev_short = float(prev["sma_short"].iloc[0] if hasattr(prev["sma_short"], "iloc") else prev["sma_short"])
        prev_long = float(prev["sma_long"].iloc[0] if hasattr(prev["sma_long"], "iloc") else prev["sma_long"])
    except Exception:
        return "HOLD", None

    rsi = float(latest["rsi"])
    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns else None
    # signal by SMA crossover
    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY", {"rsi": rsi, "atr": atr, "price": float(latest["Close"])}
    if prev_short >= prev_long and latest_short < latest_long:
        return "SELL", {"rsi": rsi, "atr": atr, "price": float(latest["Close"])}
    return "HOLD", {"rsi": rsi, "atr": atr, "price": float(latest["Close"])}

# -----------------------
# Position sizing & stop calc
# -----------------------
def compute_stop_and_qty(symbol, price, atr, equity):
    # use ATR if available; else use % fallback
    if atr and not np.isnan(atr) and atr > 0:
        stop_dist = max(atr, price * MIN_STOP_PCT)
    else:
        stop_dist = price * MIN_STOP_PCT
    # clamp
    stop_dist = max(stop_dist, price * MIN_STOP_PCT)
    stop_dist = min(stop_dist, price * MAX_STOP_PCT)

    # risk per share = stop_dist
    risk_per_share = stop_dist
    risk_budget = equity * EQUITY_RISK_PCT
    if risk_per_share <= 0:
        return None, 0
    raw_qty = math.floor(risk_budget / risk_per_share)
    # cap by allocation amount
    cap_qty = math.floor((equity * MAX_ALLOC_PER_TRADE_PCT) / price)
    qty = int(max(0, min(raw_qty, cap_qty)))
    stop_price = round(price - stop_dist, 2)
    tp_price = round(price + (stop_dist * TP_MULTIPLIER), 2)
    return (stop_price, tp_price), qty

# -----------------------
# Order submission
# -----------------------
def submit_bracket_buy(symbol, qty, stop_price, tp_price):
    if qty <= 0:
        print(f"[{symbol}] qty=0; skipping submit.")
        return None
    if DRY_RUN:
        print(f"[DRY RUN] Would submit bracket buy: {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id": "dryrun", "symbol": symbol, "qty": qty}
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={'limit_price': f"{tp_price:.2f}"},
            stop_loss={'stop_price': f"{stop_price:.2f}"}
        )
        print(f"[{symbol}] Submitted bracket order id={order.id}")
        return order
    except Exception as e:
        print(f"[{symbol}] Order submit error: {e}")
        return None

def submit_market_sell(symbol, qty):
    if qty <= 0:
        print(f"[{symbol}] no position to sell.")
        return None
    if DRY_RUN:
        print(f"[DRY RUN] Would submit market sell: {symbol} qty={qty}")
        return {"id": "dryrun_sell", "symbol": symbol, "qty": qty}
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
        print(f"[{symbol}] Submitted market sell id={order.id}")
        return order
    except Exception as e:
        print(f"[{symbol}] Sell submit error: {e}")
        return None

# -----------------------
# Main run
# -----------------------
def run_once():
    print(f"\n=== Tradebot run {datetime.now(TZ).isoformat()} ===")
    if not is_market_open():
        print("Market is closed — skipping.")
        return

    equity = get_account_equity()
    if equity is None:
        print("Cannot determine account equity, exiting.")
        return
    print(f"Account equity: ${equity:,.2f}")

    current_exposure = total_open_exposure()
    print(f"Current exposure: ${current_exposure:,.2f}")

    for symbol in SYMBOLS:
        try:
            print(f"\nChecking {symbol} ...")
            d_sig = daily_trend(symbol)
            h_sig, meta = hourly_entry_signal(symbol)
            print(f"  daily: {d_sig}, hourly: {h_sig}, rsi={meta.get('rsi') if meta else 'N/A'}")

            # require daily and hourly agree on direction for initial conservative approach
            if d_sig == "BUY" and h_sig == "BUY":
                # RSI filter: buy dips only when RSI <= threshold
                rsi = meta.get("rsi", None)
                atr = meta.get("atr", None)
                price = meta.get("price", None)
                if rsi is None or price is None:
                    print(f"  insufficient meta data; skip {symbol}")
                    continue
                if rsi > RSI_BUY_THRESH:
                    print(f"  RSI {rsi:.1f} > {RSI_BUY_THRESH}, not buying (wait dip).")
                    continue

                # cooldown
                if trades_today_for(symbol) >= MAX_TRADES_PER_SYMBOL_PER_DAY:
                    print(f"  reached max trades today for {symbol}.")
                    continue

                # skip if already long
                pos_qty = get_position_qty(symbol)
                if pos_qty > 0:
                    print(f"  already long qty={pos_qty}, skipping buy.")
                    continue

                (stop_price, tp_price), qty = compute_stop_and_qty(symbol, price, atr, equity)
                if qty <= 0:
                    print(f"  computed qty 0; skip {symbol}")
                    continue

                # exposure check
                projected = current_exposure + qty * price
                if projected > equity * MAX_TOTAL_EXPOSURE_PCT:
                    print(f"  would exceed total exposure ({projected:.2f} > {equity*MAX_TOTAL_EXPOSURE_PCT:.2f}), skip.")
                    continue

                # submit
                order = submit_bracket_buy(symbol, qty, stop_price, tp_price)
                if order:
                    record_trade_today(symbol)
                    current_exposure = projected

            elif d_sig == "SELL" and h_sig == "SELL":
                # close existing longs
                pos_qty = get_position_qty(symbol)
                if pos_qty > 0:
                    submit_market_sell(symbol, pos_qty)
                    record_trade_today(symbol)
                else:
                    print(f"  no long position to close for {symbol}")

            else:
                print("  no agreement on direction or HOLD -> no action")

            time.sleep(1.0)  # polite throttle

        except Exception as exc:
            print(f"  error for {symbol}: {exc}")

    print("\nRun complete.\n")

if __name__ == "__main__":
    run_once()
