# tradebot.py
import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# === API Setup ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# === Config ===
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
COOLDOWN_HOURS = 3  # Don’t rebuy same symbol too soon
last_trade_times = {}

# === Data Fetcher ===
def get_latest_data(symbol):
    """Fetch 6 months of 15m price data from Yahoo Finance."""
    df = yf.download(symbol, period="6mo", interval="15m", progress=False)

    # Drop ticker level from MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)

    df = df[['Close']].rename(columns={'Close': 'close'})
    df['sma_short'] = df['close'].rolling(window=3).mean()
    df['sma_long'] = df['close'].rolling(window=7).mean()
    return df.dropna()

# === Signal Logic ===
def get_signal(df):
    """Generate trading signal based on SMA crossover."""
    if df.empty or len(df) < 7:
        return "HOLD"
    if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
        return "BUY"
    elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# === Trade Execution ===
def execute_trade(symbol, signal):
    """Send trade orders to Alpaca based on signal."""
    now = datetime.utcnow()
    position = None
    try:
        position = trading_client.get_open_position(symbol)
    except Exception:
        position = None

    # Cooldown check
    last_trade = last_trade_times.get(symbol)
    if last_trade and (now - last_trade).total_seconds() < COOLDOWN_HOURS * 3600:
        print(f"⏳ Skipping {symbol}: cooldown active.")
        return

    if signal == "BUY" and not position:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=2,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        last_trade_times[symbol] = now
        print(f"✅ Bought 2 shares of {symbol}")
    elif signal == "SELL" and position:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=position.qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        last_trade_times[symbol] = now
        print(f"🟥 Sold {symbol}")
    else:
        print(f"➖ No trade action for {symbol}")

# === Daily P&L ===
def calculate_daily_pnl():
    """Estimate paper P&L from Alpaca account."""
    try:
        account = trading_client.get_account()
        print(f"\nDaily P&L: ${float(account.equity) - float(account.last_equity):.2f}")
    except Exception as e:
        print(f"P&L check failed: {e}")

# === Main Run ===
if __name__ == "__main__":
    print(f"\n=== Tradebot run {datetime.now()} ===\n")

    for symbol in SYMBOLS:
        print(f"Checking {symbol}...")
        try:
            df = get_latest_data(symbol)
            signal = get_signal(df)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")
            execute_trade(symbol, signal)
        except Exception as e:
            print(f"Failed bars for {symbol}: {e}")

    calculate_daily_pnl()
    print("\n✅ Trade check complete. Exiting cleanly.\n")
