import os
import time
import datetime
import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame

# --- CONFIGURATION ---
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_SIZE = 1  # number of shares per trade
GUARANTEE_TRADE = True  # force at least 1 trade per day

# Stop-loss / Take-profit % (relative to entry)
TP_PCT = 0.005  # 0.5%
SL_PCT = 0.002  # 0.2%

# Alpaca client
api = REST(API_KEY, API_SECRET, BASE_URL)

# --- UTILITIES ---
def fetch_bars(symbol, limit=20):
    df = api.get_bars(symbol, TimeFrame.Minute, limit=limit).df
    df = df[df['symbol'] == symbol]
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

def score_symbol(df):
    last = df.iloc[-1]
    price = float(last["Close"])
    vwap = float(last["VWAP"])
    volume = float(last["Volume"])
    spread = float(last["High"]) - float(last["Low"])
    
    # Adaptive thresholds
    spread_threshold = float(df["High"].max() - df["Low"].min())
    vwap_thresh = spread_threshold * 0.5
    vol_norm = min(volume / (df["Volume"].rolling(20).mean().iloc[-1]+1e-6), 1.0)
    
    # Score components
    vwap_score = max(0, 1 - abs(price - vwap)/vwap_thresh)
    spread_score = max(0, 1 - spread/spread_threshold)
    score = 0.5*vwap_score + 0.3*vol_norm + 0.2*spread_score
    return score, price, vwap, spread

def select_best_symbol():
    candidates = []
    for sym in SYMBOLS:
        try:
            df = fetch_bars(sym)
            score, price, vwap, spread = score_symbol(df)
            candidates.append({"symbol": sym, "score": score, "price": price, "vwap": vwap, "spread": spread})
        except Exception as e:
            print(f"Error scoring {sym}: {e}")
    if not candidates:
        return None
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

def submit_trade(symbol, qty=TRADE_SIZE):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Submitted BUY {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Buy order failed {symbol}: {e}")
        return None

def monitor_trade(symbol, entry_price):
    tp = entry_price * (1 + TP_PCT)
    sl = entry_price * (1 - SL_PCT)
    print(f"Monitoring {symbol} entry {entry_price:.2f} TP {tp:.2f} SL {sl:.2f}")
    while True:
        bar = fetch_bars(symbol, limit=1).iloc[-1]
        price = float(bar["Close"])
        if price >= tp:
            api.submit_order(symbol=symbol, qty=TRADE_SIZE, side='sell', type='market', time_in_force='day')
            print(f"{symbol} hit TP at {price:.2f}, sold")
            break
        if price <= sl:
            api.submit_order(symbol=symbol, qty=TRADE_SIZE, side='sell', type='market', time_in_force='day')
            print(f"{symbol} hit SL at {price:.2f}, sold")
            break
        time.sleep(30)

def main():
    print("Starting hybrid_tradebot_advanced")
    best = select_best_symbol()
    if not best:
        print("No valid candidates. Exiting.")
        return

    # Guarantee at least one trade
    if best["score"] < 0.1 and GUARANTEE_TRADE:
        print(f"Forcing top-ranked candidate {best['symbol']} due to guarantee flag.")
    
    order = submit_trade(best["symbol"])
    if order:
        # Wait a few seconds for fill
        time.sleep(5)
        # Fetch last trade price
        filled_price = float(api.get_position(best["symbol"]).avg_entry_price)
        monitor_trade(best["symbol"], filled_price)
    print("Run complete.")

if __name__ == "__main__":
    main()
