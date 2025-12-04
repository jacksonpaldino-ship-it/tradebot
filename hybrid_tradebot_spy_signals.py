import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# === Alpaca API credentials ===
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# === Config ===
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_SIZE = 1  # shares per trade
STOP_LOSS_PCT = 0.5 / 100  # 0.5% stop-loss
TAKE_PROFIT_PCT = 0.5 / 100  # 0.5% profit target
VWAP_THRESHOLD = 0.01  # max distance from VWAP for buy
VOL_WEIGHT = True  # whether to weight score by volume

# === Helper Functions ===
def fetch_data(symbol, limit=50):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
        if bars.empty:
            raise ValueError("No bars data returned")
        bars = bars[bars['symbol'] == symbol]  # ensure correct symbol
        return bars
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def score_symbol(df):
    last = df.iloc[-1]
    price = float(last['close'])
    vwap = float(last['vwap']) if 'vwap' in last else price
    vol = float(last['volume']) if 'volume' in last else 1
    high, low = float(last['high']), float(last['low'])
    spread = high - low if high != low else 1e-6  # avoid divide by zero

    price_vwap_score = max(0, 1 - abs(price - vwap) / spread)
    vol_score = vol / df['volume'].max() if VOL_WEIGHT else 1
    total_score = 0.4*price_vwap_score + 0.6*vol_score
    return total_score, price, vwap, spread

def select_candidate():
    candidates = []
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is None:
            continue
        score, price, vwap, spread = score_symbol(df)
        if abs(price - vwap)/vwap <= VWAP_THRESHOLD:
            candidates.append({"symbol": sym, "score": score, "price": price})
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

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
    except APIError as e:
        print(f"Failed {side.upper()} {symbol}: {e}")
        return None

def monitor_position(order):
    symbol = order.symbol
    entry_price = float(order.filled_avg_price)
    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)

    print(f"Monitoring {symbol} entry {entry_price} TP {take_profit} SL {stop_loss}")

    while True:
        try:
            pos = api.get_position(symbol)
            current_price = float(pos.current_price)
            if current_price <= stop_loss or current_price >= take_profit:
                place_order(symbol, int(pos.qty), side="sell")
                print(f"Exited {symbol} at {current_price}")
                break
        except Exception:
            # position may not exist yet
            time.sleep(5)
        time.sleep(5)

# === Main Execution ===
def main():
    now = dt.datetime.now(dt.timezone.utc)
    if now.hour < 13 or now.hour > 20:  # US market hours ET: 9:30-16:00 -> UTC 13:30-20:00
        print("Market closed; exiting.")
        return

    candidate = select_candidate()
    if candidate is None:
        print("No valid candidate found; exiting.")
        return

    order = place_order(candidate["symbol"], TRADE_SIZE, side="buy")
    if order:
        monitor_position(order)

if __name__ == "__main__":
    main()
