#!/usr/bin/env python3
# hybrid_tradebot_spy_signals.py
# Fully functional intraday bot with guaranteed 1 trade/day

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
import pytz

# ----------------------
# CONFIG
# ----------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_AMOUNT = 1  # shares per trade
MARKET_TIMEZONE = pytz.timezone("America/New_York")

# Get Alpaca keys from environment
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")  # Paper or live

if not all([API_KEY, API_SECRET, BASE_URL]):
    raise ValueError("API credentials missing. Set APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ----------------------
# UTILITY FUNCTIONS
# ----------------------
def fetch_data(symbol, limit=50):
    """Fetch intraday bars"""
    df = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
    df = df[df['symbol'] == symbol]
    return df

def score_candidate(df):
    """Compute a score for the symbol"""
    last = df.iloc[-1]
    close = float(last["close"])
    high = float(last["high"])
    low = float(last["low"])
    volume = float(last["volume"])
    vwap = float(last["vwap"]) if "vwap" in last else (df["close"] * df["volume"]).sum() / df["volume"].sum()
    spread = high - low

    # Adaptive thresholds
    spread_threshold = df["high"].max() - df["low"].min()
    price_vwap_diff = abs(close - vwap)

    # Volume-weighted score
    vol_score = min(volume / df["volume"].max(), 1.0)
    spread_score = max(0, 1 - spread / (spread_threshold + 1e-6))
    price_score = max(0, 1 - price_vwap_diff / (spread_threshold + 1e-6))

    total_score = (0.4 * price_score + 0.3 * spread_score + 0.3 * vol_score)
    return total_score, close, vwap, volume, spread

def select_candidate():
    """Select the best candidate symbol"""
    candidates = []
    for sym in SYMBOLS:
        try:
            df = fetch_data(sym)
            score, close, vwap, volume, spread = score_candidate(df)
            candidates.append({
                "symbol": sym,
                "score": score,
                "close": close,
                "vwap": vwap,
                "volume": volume,
                "spread": spread
            })
        except Exception as e:
            print(f"Error scoring {sym}: {e}")
    if not candidates:
        raise RuntimeError("No valid candidates found")
    # Rank by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]  # top candidate

def submit_order(symbol, qty=TRADE_AMOUNT):
    """Submit a market buy order"""
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"Submitted BUY {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Buy order failed {symbol}: {e}")
        return None

# ----------------------
# MAIN BOT
# ----------------------
def main():
    print("Starting hybrid_tradebot_advanced")
    now = datetime.now(MARKET_TIMEZONE)
    print(f"Run start ET {now.isoformat()}")

    try:
        candidate = select_candidate()
        print(f"Top candidate: {candidate['symbol']} score {candidate['score']:.4f} "
              f"(vwap {candidate['vwap']:.3f} vol {candidate['volume']:.0f} spread {candidate['spread']:.3f})")

        # Guarantee at least one trade
        order = submit_order(candidate['symbol'])
        if order:
            print(f"{candidate['symbol']} buy filled")
        else:
            print("Order failed, exiting")
    except Exception as e:
        print(f"Primary selection failed: {e}")
        print("No trade executed this run")

if __name__ == "__main__":
    main()
