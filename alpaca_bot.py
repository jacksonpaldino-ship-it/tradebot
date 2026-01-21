import os
import asyncio
import logging
from datetime import datetime
import pandas as pd

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ------------------- CONFIGURATION -------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

# Symbols to track
SYMBOLS = ["AAPL", "TSLA", "MSFT"]

# Paper trading quantity
TRADE_QTY = 1

# Logging setup
logging.basicConfig(
    filename="alpaca_bot.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ------------------- VALIDATION -------------------
if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca API keys. Set them in your GitHub secrets.")

# ------------------- INITIALIZE CLIENTS -------------------
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
stream = StockDataStream(api_key=API_KEY, secret_key=SECRET_KEY)

# ------------------- HELPER FUNCTIONS -------------------
def log_trade(action, symbol, qty, price):
    logging.info(f"{action} | {symbol} | Qty: {qty} | Price: {price}")

async def handle_bars(bar):
    """
    Callback for streaming bars.
    Example logic: logs bar and can place a simple paper market order.
    """
    print(f"{bar.symbol} | Open: {bar.open} High: {bar.high} Low: {bar.low} Close: {bar.close}")
    logging.info(f"{bar.symbol} | Open: {bar.open} High: {bar.high} Low: {bar.low} Close: {bar.close}")

    # Example trade logic: buy if close > open
    try:
        if bar.close > bar.open:
            order_data = MarketOrderRequest(
                symbol=bar.symbol,
                qty=TRADE_QTY,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            order = trading_client.submit_order(order_data)
            log_trade("BUY", bar.symbol, TRADE_QTY, bar.close)
    except Exception as e:
        logging.error(f"Error submitting order for {bar.symbol}: {e}")

# ------------------- SUBSCRIBE TO SYMBOLS -------------------
for symbol in SYMBOLS:
    stream.subscribe_bars(handle_bars, symbol)

# ------------------- RUN STREAM -------------------
async def main():
    while True:
        try:
            await stream.run()
        except Exception as e:
            logging.error(f"Stream error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
