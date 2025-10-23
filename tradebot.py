import os
import yfinance as yf
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# === SETUP ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize Alpaca client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Stocks to trade
symbols = ["AAPL", "MSFT", "TSLA", "NVDA"]

# === FUNCTIONS ===
def get_latest_data(symbol):
    """Fetch last 5 days of price data."""
    try:
        df = yf.download(symbol, period="5d", interval="30m", progress=False)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_signal(df):
    """Simple momentum signal based on moving averages."""
    if df is None or df.empty:
        return "HOLD"
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    if df["SMA5"].iloc[-1] > df["SMA20"].iloc[-1]:
        return "BUY"
    elif df["SMA5"].iloc[-1] < df["SMA20"].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# === MAIN LOOP (runs once) ===
print(f"\n🚀 Trade bot started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for symbol in symbols:
    df = get_latest_data(symbol)
    signal = get_signal(df)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")

    # Check current position
    try:
        position = trading_client.get_open_position(symbol)
    except:
        position = None

    if signal == "BUY" and not position:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=2,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order)
        print(f"✅ Placed BUY order for {symbol}")
    elif signal == "SELL" and position:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=abs(int(float(position.qty))),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order)
        print(f"🟥 Placed SELL order for {symbol}")
    else:
        print(f"➖ No trade action for {symbol}")

print("\n✅ Trade check complete. Exiting cleanly.\n")

