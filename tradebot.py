# tradebot.py
import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- API keys from GitHub secrets ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- Market hours ---
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# --- Stock symbols and cooldown tracking ---
symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
trade_log = {s: [] for s in symbols}  # store timestamps of trades for cooldown

# --- Helper functions ---
def get_latest_data(symbol):
    """Fetch last 60 days of 15-minute data."""
    df = yf.download(symbol, period="60d", interval="15m", progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = df[["Close"]].rename(columns={"Close": "close"})
    df["sma_short"] = df["close"].rolling(window=10).mean()
    df["sma_long"] = df["close"].rolling(window=30).mean()
    return df.dropna()

def get_signal(df):
    """Generate signal based on SMA crossovers."""
    latest_short = df["sma_short"].iloc[-1]
    latest_long = df["sma_long"].iloc[-1]
    prev_short = df["sma_short"].iloc[-2]
    prev_long = df["sma_long"].iloc[-2]

    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    elif prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    else:
        return "HOLD"

def trade(symbol, signal):
    """Submit trades based on signal, limited to 2 trades/day per symbol."""
    now = datetime.now()

    # Cooldown: max 2 trades per symbol per day
    trade_log[symbol] = [
        t for t in trade_log[symbol] if t.date() == now.date()
    ]
    if len(trade_log[symbol]) >= 2:
        print(f"🕒 Cooldown active for {symbol}, skipping trade.")
        return

    try:
        position = trading_client.get_open_position(symbol)
    except Exception:
        position = None

    if signal == "BUY" and not position:
        order = trading_client.submit_order(
            symbol=symbol,
            qty=2,
            side=OrderSide.BUY,
            type="market",
            time_in_force=TimeInForce.DAY,
        )
        trade_log[symbol].append(now)
        print(f"✅ Bought 2 shares of {symbol}")
    elif signal == "SELL" and position:
        order = trading_client.submit_order(
            symbol=symbol,
            qty=position.qty,
            side=OrderSide.SELL,
            type="market",
            time_in_force=TimeInForce.DAY,
        )
        trade_log[symbol].append(now)
        print(f"🟥 Sold {symbol}")
    else:
        print(f"➖ No trade action for {symbol}")

# --- Main Loop ---
def main():
    now = datetime.now().time()
    if not (MARKET_OPEN <= now <= MARKET_CLOSE):
        print(f"⏸ Market closed ({now}), skipping run.")
        return

    print(f"=== Tradebot run {datetime.now()} ===")

    for symbol in symbols:
        try:
            df = get_latest_data(symbol)
            signal = get_signal(df)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")
            trade(symbol, signal)
        except Exception as e:
            print(f"Failed bars for {symbol}: {e}")

    print("\n✅ Trade check complete. Exiting cleanly.")

if __name__ == "__main__":
    main()
