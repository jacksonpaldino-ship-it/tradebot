import os
import asyncio
import logging
from datetime import datetime
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load secrets from environment variables (GitHub Secrets)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # default to paper

if not all([ALPACA_API_KEY, ALPACA_SECRET_KEY]):
    raise ValueError("Missing Alpaca API keys in environment variables")

# Initialize Alpaca Trading Client (Paper trading)
trading_client = TradingClient(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    paper=True  # explicitly paper trading
)

# Initialize Stock Data Stream
stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# Symbols to trade
SYMBOLS = ["AAPL", "TSLA", "MSFT"]  # Add your watchlist
POSITION_LIMIT = 100  # max shares per order


async def trade_on_tick(symbol: str, price: float):
    """Simple example: buy 100 shares if price drops more than 1% from previous close"""
    try:
        # Fetch last closing price
        barset = trading_client.get_bars(symbol, "1D", limit=2)
        prev_close = barset[-2].c if len(barset) > 1 else barset[-1].c

        change_pct = 100 * (price - prev_close) / prev_close
        logger.info(f"{symbol} price {price}, change {change_pct:.2f}%")

        # Example strategy: buy if price dropped > 1%
        if change_pct <= -1.0:
            logger.info(f"{symbol} condition met: buying {POSITION_LIMIT} shares")
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=POSITION_LIMIT,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            resp = trading_client.submit_order(order_request)
            logger.info(f"Order submitted: {resp}")
        else:
            logger.debug(f"{symbol} condition not met")
    except Exception as e:
        logger.error(f"Error in trade_on_tick for {symbol}: {e}")


# Callback for new trade updates
async def handle_trade_update(data):
    symbol = data.symbol
    price = data.price
    await trade_on_tick(symbol, price)


async def main():
    # Subscribe to trades
    for symbol in SYMBOLS:
        stream.subscribe_trades(handle_trade_update, symbol)

    logger.info("Starting Alpaca streaming...")
    await stream._run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
