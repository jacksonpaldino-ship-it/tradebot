import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ========================================
# Alpaca Credentials
# ========================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ========================================
# Config
# ========================================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_SIZE = 1
STOP_LOSS_PCT = 0.005      # 0.5%
TAKE_PROFIT_PCT = 0.0075   # 0.75% profit target (slightly higher than SL)
VWAP_THRESHOLD = 0.01      # must be within ±1% of VWAP
VOL_WEIGHT = True          # boost symbols with stronger volume

# ========================================
# Fetch Data – FIXED
# ========================================
def fetch_data(symbol, limit=50):
    """Fetch minute bars for a symbol with correct symbol handling."""
    try:
        bars = api.get_bars(
            symbol,
            TimeFrame.Minute,
            limit=limit,
            adjustment='raw'
        ).df

        if bars.empty:
            raise ValueError("No bars returned")

        # Alpaca does NOT provide symbol column for single-symbol fetches
        if "symbol" not in bars.columns:
            bars["symbol"] = symbol

        return bars

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ========================================
# Score Logic (Your original BUY logic)
# ========================================
def score_symbol(df):
    last = df.iloc[-1]

    price = float(last["close"])
    vwap = float(last["vwap"]) if "vwap" in last else price
    vol = float(last["volume"]) if "volume" in last else 1

    high = float(last["high"])
    low = float(last["low"])
    spread = max(high - low, 1e-6)

    price_vwap_score = max(0, 1 - abs(price - vwap) / spread)
    vol_score = vol / df["volume"].max() if VOL_WEIGHT else 1

    total_score = 0.4 * price_vwap_score + 0.6 * vol_score
    return total_score, price, vwap

# ========================================
# Select Best Symbol to Buy
# ========================================
def select_candidate():
    candidates = []

    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is None:
            continue

        score, price, vwap = score_symbol(df)

        # must be near VWAP
        if abs(price - vwap) / vwap <= VWAP_THRESHOLD:
            candidates.append({"symbol": sym, "score": score, "price": price})

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

# ========================================
# Place Order
# ========================================
def place_order(symbol, qty, side="buy"):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        print(f"Submitted {side.upper()} {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Order failed {side.upper()} {symbol}: {e}")
        return None

# ========================================
# SELL Logic + Stop Loss Monitoring
# ========================================
def monitor_position(order):
    symbol = order.symbol
    print("Waiting for order fill...")

    # Wait for fill
    while True:
        try:
            o = api.get_order(order.id)
            if o.filled_avg_price is not None:
                entry = float(o.filled_avg_price)
                break
        except Exception:
            pass
        time.sleep(2)

    stop_loss = entry * (1 - STOP_LOSS_PCT)
    take_profit = entry * (1 + TAKE_PROFIT_PCT)

    print(f"Filled {symbol} at {entry}")
    print(f"SL: {stop_loss} | TP: {take_profit}")

    # monitor every 5 seconds
    while True:
        try:
            pos = api.get_position(symbol)
            current = float(pos.current_price)

            if current <= stop_loss:
                print(f"STOP LOSS triggered for {symbol} at {current}")
                place_order(symbol, int(pos.qty), side="sell")
                break

            if current >= take_profit:
                print(f"TAKE PROFIT hit for {symbol} at {current}")
                place_order(symbol, int(pos.qty), side="sell")
                break

        except Exception:
            # no position means exit
            break

        time.sleep(5)

# ========================================
# Main Entry
# ========================================
def main():
    now = dt.datetime.now(dt.timezone.utc)

    # US Market Hours: 13:30–20:00 UTC
    if now.hour < 13 or now.hour > 20:
        print("Market closed; exiting.")
        return

    candidate = select_candidate()
    if candidate is None:
        print("No valid candidate found; exiting.")
        return

    print(f"BUY candidate: {candidate['symbol']} @ {candidate['price']}")
    order = place_order(candidate["symbol"], TRADE_SIZE, side="buy")

    if order:
        monitor_position(order)


if __name__ == "__main__":
    main()
