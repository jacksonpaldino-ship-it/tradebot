import os
import time
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame

# Alpaca credentials (from environment)
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

api = REST(API_KEY, API_SECRET, BASE_URL)

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_SIZE = 1  # number of shares per trade
GUARANTEE_TRADE = True
VOLATILITY_LOOKBACK = 20  # candles

TP_PERCENT = 0.5 / 100  # Take profit 0.5%
SL_PERCENT = 0.3 / 100  # Stop loss 0.3%

def fetch_data(symbol, limit=50):
    barset = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
    if barset.empty:
        return None
    df = barset[barset['symbol'] == symbol]
    return df

def calculate_score(df):
    last = df.iloc[-1]
    price = float(last["close"])
    vwap = float(df["close"].mean())  # simple VWAP proxy
    volume = float(last["volume"])
    spread = float(last["high"]) - float(last["low"])
    
    # Adaptive thresholds based on recent volatility
    recent_spreads = df["high"] - df["low"]
    spread_threshold = recent_spreads.mean() * 1.1  # 10% above avg
    vol_threshold = df["volume"].mean()
    
    score = 0
    # Price close to VWAP
    score += max(0, 1 - abs(price - vwap)/vwap)
    # Spread low
    score += max(0, 1 - spread/spread_threshold)
    # Volume high
    score += min(volume / vol_threshold, 1)
    
    return score, price, spread, volume, vwap

def select_candidate():
    candidates = []
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is None or df.empty:
            continue
        score, price, spread, volume, vwap = calculate_score(df)
        candidates.append({
            "symbol": sym,
            "score": score,
            "price": price,
            "spread": spread,
            "volume": volume,
            "vwap": vwap
        })
    if not candidates:
        return None
    # Rank by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

def submit_order(symbol, qty, side):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        return order
    except Exception as e:
        print(f"Order failed {symbol}: {e}")
        return None

def monitor_trade(symbol, entry_price):
    tp = entry_price * (1 + TP_PERCENT)
    sl = entry_price * (1 - SL_PERCENT)
    print(f"Monitoring {symbol} entry {entry_price:.2f} TP {tp:.2f} SL {sl:.2f}")
    while True:
        bar = fetch_data(symbol, limit=1)
        if bar is None or bar.empty:
            time.sleep(10)
            continue
        last_price = float(bar.iloc[-1]["close"])
        print(f"{symbol} price {last_price:.2f}")
        if last_price >= tp:
            print(f"{symbol} hit take profit at {last_price:.2f}")
            submit_order(symbol, TRADE_SIZE, "sell")
            break
        elif last_price <= sl:
            print(f"{symbol} hit stop loss at {last_price:.2f}")
            submit_order(symbol, TRADE_SIZE, "sell")
            break
        time.sleep(30)  # check every 30 seconds

def main():
    candidate = select_candidate()
    if candidate is None:
        print("No valid candidate. Exiting.")
        return
    
    print(f"Top candidate: {candidate['symbol']} score {candidate['score']:.3f}")
    
    # Check thresholds
    price_diff = abs(candidate["price"] - candidate["vwap"])/candidate["vwap"]
    if price_diff > 0.02:  # price too far from vwap
        print(f"{candidate['symbol']} price-vwap {price_diff:.3f} too far")
        if not GUARANTEE_TRADE:
            return
        print("Forcing top-ranked candidate due to guarantee flag.")
    
    order = submit_order(candidate["symbol"], TRADE_SIZE, "buy")
    if order is None:
        print(f"Buy order failed {candidate['symbol']}")
        return
    
    print(f"{candidate['symbol']} buy filled at {candidate['price']:.2f}")
    monitor_trade(candidate["symbol"], candidate['price'])

if __name__ == "__main__":
    main()
