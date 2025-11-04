"""
tradebot.py
Trend-following swing bot (daily + hourly SMA confirmation)
- Paper-trade first. Use env vars ALPACA_API_KEY and ALPACA_SECRET_KEY.
"""

import os
import json
import math
import time
from datetime import datetime, time as dtime, date
import pytz

import yfinance as yf
import pandas as pd
import alpaca_trade_api as tradeapi

# -------------------------
# CONFIG
# -------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # change to https://api.alpaca.markets for live

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "PLTR", "CRSP"]
# allocation & risk
ALLOC_PCT_PER_TRADE = 0.20     # allocate 20% of equity per new position (subject to MAX_TOTAL_EXPOSURE)
MAX_TOTAL_EXPOSURE = 0.80      # don't have more than 80% of equity invested at once
STOP_PCT = 0.06                # stop loss ~6% below entry (can be ATR-based later)
TP_MULTIPLIER = 2.0            # take-profit = TP_MULTIPLIER * stop distance (reward:risk)
MAX_TRADES_PER_SYMBOL_PER_DAY = 1  # allow 1 new entry per symbol per day
MIN_EQUITY = 1000              # don't trade if equity less than this

# SMA windows
DAILY_SHORT = 20
DAILY_LONG = 50
HOUR_SHORT = 5
HOUR_LONG = 20

# Market hours in Eastern
TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# History / cooldown file
TRADE_HISTORY_FILE = "trade_history.json"

# -------------------------
# INIT
# -------------------------
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

def is_market_open():
    now = datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def load_trade_history():
    if os.path.exists(TRADE_HISTORY_FILE):
        with open(TRADE_HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_trade_history(h):
    with open(TRADE_HISTORY_FILE, "w") as f:
        json.dump(h, f)

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

# -------------------------
# Data helpers
# -------------------------
def fetch_daily(symbol, period="200d"):
    # returns DataFrame of daily OHLCV with daily SMAs
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df.empty:
        return df
    df["sma_short"] = df["Close"].rolling(window=DAILY_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(window=DAILY_LONG).mean()
    return df.dropna()

def fetch_hourly(symbol, period="90d"):
    # fetch hourly bars (1h) for the last ~90 days (yfinance limits)
    df = yf.download(symbol, period=period, interval="1h", progress=False)
    if df.empty:
        return df
    # handle MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    df["sma_short"] = df["Close"].rolling(window=HOUR_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(window=HOUR_LONG).mean()
    return df.dropna()

# -------------------------
# Signal logic
# -------------------------
def daily_trend_confirms(symbol):
    df = fetch_daily(symbol)
    if df.empty or len(df) < DAILY_LONG + 2:
        return False
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    try:
        latest_short = float(latest["sma_short"])
        latest_long  = float(latest["sma_long"])
        prev_short   = float(prev["sma_short"])
        prev_long    = float(prev["sma_long"])
    except:
        return False

    # trend up: short > long and previous not above already
    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    if prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    return "HOLD"

def hourly_signal(symbol):
    df = fetch_hourly(symbol)
    if df.empty or len(df) < HOUR_LONG + 2:
        return "HOLD"
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    try:
        latest_short = float(latest["sma_short"])
        latest_long  = float(latest["sma_long"])
        prev_short   = float(prev["sma_short"])
        prev_long    = float(prev["sma_long"])
    except:
        return "HOLD"
    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    if prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    return "HOLD"

# -------------------------
# Position sizing
# -------------------------
def get_account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Failed to fetch account equity:", e)
        return None

def current_total_exposure():
    try:
        positions = api.list_positions()
        return sum(float(p.market_value) for p in positions)
    except:
        return 0.0

def compute_qty_for_allocation(symbol, price, equity):
    # Allocate ALLOC_PCT_PER_TRADE of equity to this symbol (dollar allocation)
    if equity is None or equity <= 0:
        return 0
    alloc_amount = equity * ALLOC_PCT_PER_TRADE
    raw_qty = math.floor(alloc_amount / price)
    return int(max(raw_qty, 0))

# -------------------------
# Orders
# -------------------------
def place_bracket_buy(symbol, qty, stop_price, take_profit_price):
    # uses Alpaca classic API bracket via submit_order
    try:
        if qty <= 0:
            print(f"[{symbol}] qty=0; skipping order.")
            return None
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={'limit_price': f"{take_profit_price:.2f}"},
            stop_loss={'stop_price': f"{stop_price:.2f}"}
        )
        print(f"[{symbol}] Bracket buy submitted (qty={qty}) id={order.id}")
        return order
    except Exception as e:
        print(f"[{symbol}] Failed to submit buy: {e}")
        return None

def place_market_sell(symbol, qty):
    try:
        if qty <= 0:
            print(f"[{symbol}] no position to sell.")
            return None
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day"
        )
        print(f"[{symbol}] Market sell submitted qty={qty}")
        return order
    except Exception as e:
        print(f"[{symbol}] Sell failed: {e}")
        return None

# -------------------------
# Main decision & execution
# -------------------------
def run_once():
    print(f"\n=== Tradebot run {datetime.now(TZ)} ===")
    if not is_market_open():
        print("Market closed. Exiting.")
        return

    equity = get_account_equity()
    if equity is None:
        print("Cannot fetch equity — aborting run.")
        return
    if equity < MIN_EQUITY:
        print("Equity too small to trade.")
        return

    total_exposure = current_total_exposure()
    print(f"Equity ${equity:,.2f}  Exposure ${total_exposure:,.2f}")

    for symbol in SYMBOLS:
        try:
            print(f"\nChecking {symbol} ...")
            dtrend = daily_trend_confirms(symbol)   # BUY / SELL / HOLD
            hsig = hourly_signal(symbol)            # BUY / SELL / HOLD
            print(f"  daily: {dtrend}, hourly: {hsig}")

            # only act if daily + hourly agree on a direction
            if dtrend == "BUY" and hsig == "BUY":
                # check cooldown
                if trades_today_for(symbol) >= MAX_TRADES_PER_SYMBOL_PER_DAY:
                    print(f"  {symbol} reached max trades today; skipping entry.")
                    continue

                # if already long, skip
                try:
                    pos = api.get_position(symbol)
                    pos_qty = int(float(pos.qty))
                except:
                    pos_qty = 0

                if pos_qty > 0:
                    print(f"  already have position qty={pos_qty}; skipping entry.")
                    continue

                # compute qty to allocate ALLOC_PCT_PER_TRADE of equity
                # get current market price
                bars = yf.download(symbol, period="5d", interval="1h", progress=False)
                if bars.empty:
                    print(f"  no price bars; skip {symbol}")
                    continue
                last_price = float(bars["Close"].iloc[-1])
                qty = compute_qty_for_allocation(symbol, last_price, equity)
                if qty <= 0:
                    print(f"  computed qty 0 (price {last_price:.2f}), skip")
                    continue

                # ensure not exceeding total exposure cap
                projected_exposure = total_exposure + (qty * last_price)
                if projected_exposure > equity * MAX_TOTAL_EXPOSURE:
                    print(f"  would exceed MAX_TOTAL_EXPOSURE; skip (projected {projected_exposure:.2f})")
                    continue

                # compute stop/take
                stop_price = round(last_price * (1 - STOP_PCT), 2)
                tp_price = round(last_price + (last_price - stop_price) * TP_MULTIPLIER, 2)

                # submit bracket buy
                order = place_bracket_buy(symbol, qty, stop_price, tp_price)
                if order:
                    record_trade_today(symbol)
                    total_exposure += qty * last_price

            elif dtrend == "SELL" and hsig == "SELL":
                # If both say SELL, close any existing long position
                try:
                    pos = api.get_position(symbol)
                    pos_qty = int(float(pos.qty))
                except:
                    pos_qty = 0
                if pos_qty > 0:
                    place_market_sell(symbol, pos_qty)
                    record_trade_today(symbol)
                else:
                    print(f"  no open position to close for {symbol}")

            else:
                print(f"  no agreement or HOLD for {symbol} -> no action")

            # polite pause to avoid rate limits
            time.sleep(1.0)

        except Exception as e:
            print(f"  error processing {symbol}: {e}")

    print("\nRun complete.\n")

if __name__ == "__main__":
    run_once()
