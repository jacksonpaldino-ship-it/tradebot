import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# ----------------------------
# CONFIGURATION
# ----------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # Change to live URL when ready
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_CANDLES = 50  # Lookback candles
GUARANTEE_TRADE = True
TP_PERCENT = 0.005  # 0.5%
SL_PERCENT = 0.002  # 0.2%

# ----------------------------
# CONNECT TO ALPACA
# ----------------------------
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ----------------------------
# DATA FUNCTIONS
# ----------------------------
def fetch_bars(symbol, limit=MAX_CANDLES):
    barset = api.get_bars(symbol, "5Min", limit=limit).df
    barset = barset[barset['symbol'] == symbol]
    barset = barset.sort_index()
    return barset

def get_last(barset):
    last = barset.iloc[-1]
    return float(last['close']), float(last['volume']), float(last['high']), float(last['low'])

# ----------------------------
# SIGNAL FUNCTIONS
# ----------------------------
def score_symbol(symbol):
    df = fetch_bars(symbol)
    if df.empty or len(df) < 2:
        return {"symbol": symbol, "score": 0}
    
    price, volume, high, low = get_last(df)
    
    # Volatility-based thresholds
    spread = high - low
    spread_threshold = df['high'].max() - df['low'].min()
    spread_score = max(0, 1 - (spread / (spread_threshold + 1e-6)))
    
    # Volume score (normalized)
    avg_vol = df['volume'].mean()
    vol_score = min(volume / (avg_vol + 1e-6), 1)
    
    # Price vs VWAP score
    vwap = np.average(df['close'], weights=df['volume'])
    price_score = max(0, 1 - abs(price - vwap)/vwap)
    
    total_score = 0.4*price_score + 0.3*vol_score + 0.3*spread_score
    
    return {"symbol": symbol, "score": total_score, "price": price, "vwap": vwap}

# ----------------------------
# TRADE EXECUTION
# ----------------------------
def execute_trade(candidate):
    symbol = candidate['symbol']
    price = candidate['price']
    
    qty = 1  # Fixed, change as needed
    
    tp = price * (1 + TP_PERCENT)
    sl = price * (1 - SL_PERCENT)
    
    try:
        print(f"Submitting BUY {qty} {symbol} at {price}")
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Trade executed: {symbol} entry {price} TP {tp} SL {sl}")
    except Exception as e:
        print(f"Failed to execute {symbol}: {e}")

# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    candidates = []
    for symbol in SYMBOLS:
        try:
            cand = score_symbol(symbol)
            candidates.append(cand)
        except Exception as e:
            print(f"Error scoring {symbol}: {e}")
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    top = candidates[0]
    print(f"Top candidate: {top}")
    
    # Guarantee at least one trade
    if GUARANTEE_TRADE or top['score'] > 0.5:
        execute_trade(top)
    else:
        print("No trade meets threshold today.")

if __name__ == "__main__":
    main()
