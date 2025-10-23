import os
from datetime import datetime
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ==============================
# 🔐 API KEYS
# ==============================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ✅ Connect to Alpaca (paper trading)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ==============================
# ⚙️ Strategy parameters
# ==============================
symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
short_window = 5
long_window = 20

# ==============================
# 📈 Helper: Get signal
# ==============================
def get_signal(symbol):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        limit=long_window + 1
    )
    bars = data_client.get_stock_bars(request).df
    bars = bars[bars.index.get_level_values("symbol") == symbol]

    bars["SMA_Short"] = bars["close"].rolling(short_window).mean()
    bars["SMA_Long"] = bars["close"].rolling(long_window).mean()

    if bars["SMA_Short"].iloc[-2] < bars["SMA_Long"].iloc[-2] and bars["SMA_Short"].iloc[-1] > bars["SMA_Long"].iloc[-1]:
        return "BUY"
    elif bars["SMA_Short"].iloc[-2] > bars["SMA_Long"].iloc[-2] and bars["SMA_Short"].iloc[-1] < bars["SMA_Long"].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

# ==============================
# 💸 Helper: Place trade
# ==============================
def place_trade(symbol, side):
    try:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
        )
        print(f"✅ {side} order placed for {symbol}")
    except Exception as e:
        print(f"⚠️ Failed to place {side} order for {symbol}: {e}")

# ==============================
# 🚀 Main
# ==============================
def main():
    print(f"\n=== Running trade bot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    for symbol in symbols:
        signal = get_signal(symbol)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")

        if signal == "BUY":
            place_trade(symbol, "BUY")
        elif signal == "SELL":
            place_trade(symbol, "SELL")
        else:
            print(f"➖ No trade action for {symbol}")

    print("\n✅ Trade check complete. Exiting cleanly.\n")

if __name__ == "__main__":
    main()
