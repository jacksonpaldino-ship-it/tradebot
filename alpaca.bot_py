import os
import asyncio
import logging
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.live import StockDataStream
from dotenv import load_dotenv

# ------------------------------
# Load API Keys
# ------------------------------
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY_ID")
API_SECRET = os.getenv("ALPACA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # change to live if needed

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlpacaBot")

# ------------------------------
# Alpaca Client
# ------------------------------
client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET, paper=True)

# ------------------------------
# Configuration
# ------------------------------
SYMBOLS = ["AAPL", "TSLA"]  # Your watchlist
MAX_LOTS_PER_TRADE = 1  # 1 share per trade for simplicity, can adjust
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.05

positions = {}  # Keep track of open positions


# ------------------------------
# Trading functions
# ------------------------------
async def place_order(symbol, side, qty, order_type=OrderType.MARKET):
    try:
        order_data = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        order = client.submit_order(order_data)
        logger.info(f"Order submitted: {symbol} {side} {qty}")
        return order
    except Exception as e:
        logger.error(f"Failed to submit order for {symbol}: {e}")
        return None


async def handle_tick(data):
    symbol = data.symbol
    price = data.last
    logger.info(f"{symbol} tick: {price}")

    # Check if we have a position
    pos = positions.get(symbol)

    if not pos:
        # Example buy logic: just buy 1 lot
        await place_order(symbol, OrderSide.BUY, MAX_LOTS_PER_TRADE)
        positions[symbol] = {"buy_price": price, "qty": MAX_LOTS_PER_TRADE}
    else:
        buy_price = pos["buy_price"]
        # Stop loss
        if price <= buy_price * (1 - STOP_LOSS_PCT):
            await place_order(symbol, OrderSide.SELL, pos["qty"])
            logger.info(f"{symbol} STOP LOSS hit! Selling {pos['qty']}")
            positions.pop(symbol)
        # Take profit
        elif price >= buy_price * (1 + TAKE_PROFIT_PCT):
            await place_order(symbol, OrderSide.SELL, pos["qty"])
            logger.info(f"{symbol} TAKE PROFIT hit! Selling {pos['qty']}")
            positions.pop(symbol)


# ------------------------------
# Run Bot
# ------------------------------
async def main():
    # Subscribe to live trades
    for symbol in SYMBOLS:
        stream.subscribe_trades(handle_tick, symbol)
    await stream.run()


if __name__ == "__main__":
    asyncio.run(main())
