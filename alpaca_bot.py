import os
import asyncio
import logging
from datetime import datetime
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from dotenv import load_dotenv

# -----------------------
# Load environment secrets
# -----------------------
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not all([API_KEY, SECRET_KEY, BASE_URL]):
    raise ValueError("Missing Alpaca API keys or base URL!")

# -----------------------
# Configure logger
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("AlpacaPaperBot")

# -----------------------
# Configuration
# -----------------------
SYMBOLS = ["AAPL", "MSFT", "TSLA"]  # Example symbols, change as needed
MAX_POSITIONS = 3                   # Max simultaneous positions
CAPITAL_PER_TRADE = 1000           # USD per trade
STOP_LOSS_PCT = 0.98               # 2% stop loss
TAKE_PROFIT_PCT = 1.02             # 2% take profit

# -----------------------
# Alpaca Clients
# -----------------------
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
stream = StockDataStream(API_KEY, SECRET_KEY, base_url=BASE_URL)

# -----------------------
# Position tracking
# -----------------------
positions = {}  # symbol -> {"entry_price": float, "size": int}

# -----------------------
# Helper Functions
# -----------------------
async def place_order(symbol: str, qty: int, side: str):
    """Place a market order."""
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side),
            time_in_force=TimeInForce.DAY,
            order_type=OrderType.MARKET
        )
        order = trading_client.submit_order(order_request)
        logger.info(f"Order placed: {side} {qty} {symbol}")
        return order
    except Exception as e:
        logger.error(f"Failed to place order for {symbol}: {e}")
        return None

def calculate_qty(price: float) -> int:
    """Calculate shares to buy based on capital per trade."""
    qty = int(CAPITAL_PER_TRADE / price)
    return max(qty, 1)

# -----------------------
# Strategy Example
# -----------------------
async def simple_momentum_strategy(data):
    """Simple momentum strategy example: buy if price increased from last tick."""
    symbol = data.symbol
    price = float(data.price)

    # Skip if already in max positions
    if len(positions) >= MAX_POSITIONS:
        return

    # Buy if not in position
    if symbol not in positions:
        qty = calculate_qty(price)
        order = await place_order(symbol, qty, "BUY")
        if order:
            positions[symbol] = {"entry_price": price, "size": qty}
            logger.info(f"Entered position {symbol} at {price}")

    # Check stop loss / take profit
    if symbol in positions:
        entry_price = positions[symbol]["entry_price"]
        size = positions[symbol]["size"]

        # Stop loss
        if price <= entry_price * STOP_LOSS_PCT:
            await place_order(symbol, size, "SELL")
            logger.info(f"STOP LOSS triggered for {symbol} at {price}")
            del positions[symbol]

        # Take profit
        elif price >= entry_price * TAKE_PROFIT_PCT:
            await place_order(symbol, size, "SELL")
            logger.info(f"TAKE PROFIT triggered for {symbol} at {price}")
            del positions[symbol]

# -----------------------
# Streaming Callback
# -----------------------
async def on_trade(data):
    await simple_momentum_strategy(data)

# -----------------------
# Main
# -----------------------
async def main():
    # Subscribe to trades
    for symbol in SYMBOLS:
        stream.subscribe_trades(on_trade, symbol)
    logger.info(f"Subscribed to symbols: {SYMBOLS}")

    # Run the stream
    await stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
