import os
import asyncio
import logging
from datetime import datetime
import pandas as pd
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Use GitHub secrets directly
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing Alpaca API keys in secrets!")

# Logging setup
logging.basicConfig(
    filename="alpaca_bot.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Trading client
trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True
)

# Stock stream
stream = StockDataStream(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY
)

# Strategy settings
SYMBOL = "AAPL"
POSITION_SIZE = 1
MAX_BARS = 10
bars_seen = 0
pnl = 0.0

async def handle_bars(bar):
    global bars_seen, pnl
    bars_seen += 1
    logger.info(f"New bar: {bar}")
    print(f"{datetime.now()} - New bar: {bar}")

    if bar.close > bar.open:
        order_data = MarketOrderRequest(
            symbol=SYMBOL,
            qty=POSITION_SIZE,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        logger.info(f"Submitted BUY order: {order}")
        print(f"BUY order submitted: {order}")
        pnl -= bar.close * POSITION_SIZE
    elif bar.close < bar.open:
        order_data = MarketOrderRequest(
            symbol=SYMBOL,
            qty=POSITION_SIZE,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_data)
        logger.info(f"Submitted SELL order: {order}")
        print(f"SELL order submitted: {order}")
        pnl += bar.close * POSITION_SIZE

    logger.info(f"Current P&L: {pnl}")
    print(f"Current P&L: {pnl}")

    if bars_seen >= MAX_BARS:
        await stream.stop_stream()

stream.subscribe_bars(handle_bars, SYMBOL)

async def main():
    logger.info("Starting Alpaca bot...")
    print("Starting Alpaca bot...")
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
