import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame

# ==========================
# Alpaca API credentials
# ==========================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not all([API_KEY, API_SECRET, BASE_URL]):
    raise ValueError("API credentials missing. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ==========================
# Trading configuration
# ==========================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_QTY = 1  # modify for your account
MAX_PRICE_VWAP_DISTANCE = 0.5  # % of spread
MIN_VOL_THRESHOLD = 1e5  # minimum volume for trade
GUARANTEE_TRADE = True  # force at least one trade per day

# ==========================
# Fetch recent bars
# ==========================
def fetch_data(symbol, limit=50):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
        if bars.empty:
            print(f"No data for {symbol}")
            return None
        # Ensure column names are consistent
        bars = bars.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return bars
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ==========================
# Calculate VWAP
# ==========================
def calculate_vwap(df):
    return (df['close'] * df['volume']).sum() / df['volume'].sum()

# ==========================
# Score candidate
# ==========================
def score_candidate(df):
    last = df.iloc[-1]
    price = float(last["close"])
    vwap = calculate_vwap(df)
    volume = float(last["volume"])
    spread = float(last["high"]) - float(last["low"])
    spread_threshold = float(df["high"].max() - df["low"].min())
    
    # Adaptive scoring based on volatility and volume
    price_vwap_score = max(0, 1 - abs(price - vwap) / spread)  # closer to VWAP = better
    volume_score = min(1, volume / (df['volume'].rolling(20).mean().iloc[-1] + 1))
    spread_score = max(0, 1 - spread / (spread_threshold + 1e-5))
    
    # Total score weighted
    score = 0.5 * price_vwap_score + 0.3 * volume_score + 0.2 * spread_score
    return score, price, vwap

# ==========================
# Select top candidate
# ==========================
def select_candidate():
    candidates = []
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is not None and not df.empty:
            s, price, vwap = score_candidate(df)
            candidates.append({"symbol": sym, "score": s, "price": price, "vwap": vwap})
    
    if not candidates:
        return None
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    top = candidates[0]
    # Check if price too far from VWAP, otherwise fallback
    if abs(top["price"] - top["vwap"]) / (top["price"] + 1e-5) > MAX_PRICE_VWAP_DISTANCE:
        if GUARANTEE_TRADE:
            print(f"Price too far from VWAP for {top['symbol']}, but guaranteeing trade.")
            return top
        else:
            print(f"No suitable candidates; skipping trade.")
            return None
    return top

# ==========================
# Submit order
# ==========================
def submit_order(symbol, side="buy", qty=TRADE_QTY):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        print(f"Submitted {side.upper()} {qty} {symbol} at {datetime.now()}")
        return order
    except Exception as e:
        print(f"Failed to submit {side.upper()} {symbol}: {e}")
        return None

# ==========================
# Main execution
# ==========================
def main():
    print(f"Starting hybrid_tradebot at {datetime.now()}")
    candidate = select_candidate()
    if candidate:
        submit_order(candidate["symbol"], "buy", TRADE_QTY)
    elif GUARANTEE_TRADE:
        # Force trade on first symbol as fallback
        fallback = SYMBOLS[0]
        print(f"Forcing trade on {fallback}")
        submit_order(fallback, "buy", TRADE_QTY)
    else:
        print("No trade executed this run.")

if __name__ == "__main__":
    main()
