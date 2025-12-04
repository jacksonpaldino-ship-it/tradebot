import pandas as pd
import numpy as np
import datetime as dt
import time
import alpaca_trade_api as tradeapi

# ---------------- CONFIG ----------------
API_KEY = "YOUR_ALPACA_KEY"
API_SECRET = "YOUR_ALPACA_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
VOL_WINDOW = 20  # lookback for volatility
GUARANTEE_TRADE = True
TRADE_SIZE = 1  # number of shares per trade
TP_MULT = 0.005  # take profit 0.5%
SL_MULT = 0.003  # stop loss 0.3%

# ---------------- ALPACA ----------------
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ---------------- UTILITIES ----------------
def get_barset(symbol, limit=50):
    try:
        df = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit).df
        df = df[df['symbol'] == symbol]
        return df
    except Exception as e:
        print(f"Failed to fetch bars for {symbol}: {e}")
        return pd.DataFrame()

def calc_volatility(df):
    df["spread"] = df["high"] - df["low"]
    vol = df["spread"].rolling(VOL_WINDOW).std().iloc[-1]
    return float(vol) if not np.isnan(vol) else 0.01

def score_symbol(df):
    last = df.iloc[-1, :]
    price = float(last["close"])
    vwap = float(last["vwap"])
    vol = float(last["volume"])
    spread = float(last["high"]) - float(last["low"])
    vol_score = vol / (vol + 1e-6)
    spread_score = 1 - spread / (spread + 1e-6)
    price_vwap_score = max(0, 1 - abs(price - vwap)/price)
    score = 0.4*price_vwap_score + 0.3*vol_score + 0.3*spread_score
    return score, price, vwap, spread, vol

def submit_order(symbol, qty, side="buy"):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
        print(f"Submitted {side.upper()} {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Buy order failed {symbol}: {e}")
        return None

def monitor_trade(symbol, entry_price):
    tp = entry_price*(1+TP_MULT)
    sl = entry_price*(1-SL_MULT)
    print(f"Monitoring {symbol} entry {entry_price:.2f} TP {tp:.2f} SL {sl:.2f}")
    for _ in range(30):  # 30 checks (adjust as needed)
        df = get_barset(symbol, limit=1)
        if df.empty:
            time.sleep(5)
            continue
        price = float(df.iloc[-1]["close"])
        if price >= tp:
            print(f"{symbol} hit TP {price:.2f}")
            submit_order(symbol, TRADE_SIZE, side="sell")
            break
        elif price <= sl:
            print(f"{symbol} hit SL {price:.2f}")
            submit_order(symbol, TRADE_SIZE, side="sell")
            break
        time.sleep(5)

# ---------------- MAIN ----------------
def main():
    candidates = []

    for sym in SYMBOLS:
        df = get_barset(sym)
        if df.empty:
            continue
        vol = calc_volatility(df)
        score, price, vwap, spread, volume = score_symbol(df)
        adaptive_spread = vol*2
        if spread <= adaptive_spread:
            candidates.append({"symbol": sym, "score": score, "price": price})

    if not candidates:
        print("No candidates found, checking guarantee...")
        if GUARANTEE_TRADE:
            # fallback to top-volume symbol
            volumes = {}
            for sym in SYMBOLS:
                df = get_barset(sym)
                if not df.empty:
                    volumes[sym] = float(df.iloc[-1]["volume"])
            if volumes:
                top_sym = max(volumes, key=volumes.get)
                df = get_barset(top_sym)
                price = float(df.iloc[-1]["close"])
                order = submit_order(top_sym, TRADE_SIZE)
                if order:
                    monitor_trade(top_sym, price)
        return

    # Rank candidates by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    order = submit_order(top["symbol"], TRADE_SIZE)
    if order:
        monitor_trade(top["symbol"], top["price"])

if __name__ == "__main__":
    print("Starting hybrid_tradebot_advanced")
    main()
