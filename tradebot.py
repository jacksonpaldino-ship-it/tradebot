import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
import pandas as pd
import time

# --- Alpaca keys from GitHub Secrets ---
import os
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# --- Settings ---
symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
COOLDOWN_FILE = "cooldown.csv"  # track daily trade counts
MAX_TRADES_PER_DAY = 2
SHORT_MA = 5
LONG_MA = 20

def get_bars(symbol):
    """Fetch 15m bars (max 60 days allowed)"""
    df = yf.download(symbol, period="60d", interval="15m", progress=False)
    df.dropna(inplace=True)
    df["sma_short"] = df["Close"].rolling(SHORT_MA).mean()
    df["sma_long"] = df["Close"].rolling(LONG_MA).mean()
    return df

def get_signal(df):
    """Return BUY / SELL / HOLD based on SMA crossover"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    latest_short = float(latest["sma_short"])
    latest_long = float(latest["sma_long"])
    prev_short = float(prev["sma_short"])
    prev_long = float(prev["sma_long"])

    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    elif prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    else:
        return "HOLD"

def is_market_open():
    """Check market hours (9:30–16:00 ET)"""
    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now <= close_time

def load_cooldown():
    """Track how many times each symbol traded today"""
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(COOLDOWN_FILE):
        df = pd.DataFrame(columns=["symbol", "date", "count"])
    else:
        df = pd.read_csv(COOLDOWN_FILE)
    df = df[df["date"] == today]
    return df

def save_cooldown(df):
    df.to_csv(COOLDOWN_FILE, index=False)

def increment_cooldown(df, symbol):
    today = datetime.now().strftime("%Y-%m-%d")
    if symbol in df["symbol"].values:
        df.loc[df["symbol"] == symbol, "count"] += 1
    else:
        df.loc[len(df)] = [symbol, today, 1]
    return df

def get_position(symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except:
        return 0.0

def execute_trade(symbol, signal):
    qty = 1  # adjust as needed
    if signal == "BUY":
        print(f"[{symbol}] 🟩 Buying {qty} share(s)")
        api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
    elif signal == "SELL":
        position = get_position(symbol)
        if position > 0:
            print(f"[{symbol}] 🟥 Selling {int(position)} share(s)")
            api.submit_order(symbol=symbol, qty=int(position), side="sell", type="market", time_in_force="day")

def main():
    print(f"\n=== Tradebot run {datetime.now()} ===")

    if not is_market_open():
        print("⏸️ Market closed — skipping trades.")
        return

    cooldown_df = load_cooldown()

    for symbol in symbols:
        print(f"\nChecking {symbol}...")
        try:
            df = get_bars(symbol)
            if df.empty or len(df) < LONG_MA:
                print(f"⚠️ Not enough data for {symbol}")
                continue

            signal = get_signal(df)
            print(f"[{symbol}] Signal = {signal}")

            # cooldown
            trades_today = cooldown_df.loc[cooldown_df["symbol"] == symbol, "count"].sum()
            if trades_today >= MAX_TRADES_PER_DAY:
                print(f"🕒 Skipping {symbol}: reached {MAX_TRADES_PER_DAY} trades today.")
                continue

            if signal in ["BUY", "SELL"]:
                execute_trade(symbol, signal)
                cooldown_df = increment_cooldown(cooldown_df, symbol)
                save_cooldown(cooldown_df)
            else:
                print(f"➖ No trade action for {symbol}")

            time.sleep(2)

        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")

    print("\n✅ Trade check complete. Exiting cleanly.")

if __name__ == "__main__":
    main()
