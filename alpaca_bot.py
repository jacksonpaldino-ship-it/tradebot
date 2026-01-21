import os
import asyncio
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY_ID")
API_SECRET = os.getenv("ALPACA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca API keys")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AlpacaBot")

# -------------------------
# Client Initialization
# -------------------------
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET, base_url=BASE_URL)

# -------------------------
# Strategy Config
# -------------------------
SYMBOLS = ["AAPL", "TSLA", "MSFT"]
TRADE_SIZE = 1  # shares per trade
STOP_LOSS_PERCENT = 0.02  # 2%
TAKE_PROFIT_PERCENT = 0.04  # 4%
MAX_DAILY_LOSS = 200  # USD

POSITIONS = {}  # symbol -> {'qty': int, 'entry_price': float}
DAILY_PNL = 0

# -------------------------
# Utilities
# -------------------------
async def submit_order(symbol: str, qty: int, side: OrderSide):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        result = trading_client.submit_order(order)
        logger.info(f"Order submitted: {side.value} {qty} {symbol}")
        return result
    except Exception as e:
        logger.error(f"Order failed for {symbol}: {e}")

def calculate_stop_take(entry_price: float):
    stop = entry_price * (1 - STOP_LOSS_PERCENT)
    take = entry_price * (1 + TAKE_PROFIT_PERCENT)
    return stop, take

# -------------------------
# Trading Logic
# -------------------------
async def strategy(symbol: str, price: float):
    global DAILY_PNL

    # Skip if max daily loss reached
    if DAILY_PNL <= -MAX_DAILY_LOSS:
        logger.warning(f"Max daily loss reached. Skipping trades.")
        return

    pos = POSITIONS.get(symbol)
    if pos is None:
        # Example: simple breakout buy
        if price < 100:  # demo entry condition
            order = await submit_order(symbol, TRADE_SIZE, OrderSide.BUY)
            POSITIONS[symbol] = {'qty': TRADE_SIZE, 'entry_price': price}
            logger.info(f"Entered {symbol} at {price}")
    else:
        stop, take = calculate_stop_take(pos['entry_price'])
        if price <= stop or price >= take:
            side = OrderSide.SELL
            await submit_order(symbol, pos['qty'], side)
            pnl = (price - pos['entry_price']) * pos['qty']
            DAILY_PNL += pnl
            logger.info(f"Exited {symbol} at {price}, P&L: {pnl}")
            POSITIONS.pop(symbol)

# -------------------------
# Stream Handlers
# -------------------------
async def handle_quote(symbol, data):
    price = data.bid_price
    if price:
        await strategy(symbol, price)

async def handle_trade_update(data):
    logger.info(f"Trade update: {data}")

# -------------------------
# Main Async Runner
# -------------------------
async def main():
    for symbol in SYMBOLS:
        stream.subscribe_trades(handle_trade_update, symbol)
        stream.subscribe_quotes(lambda data, sym=symbol: asyncio.create_task(handle_quote(sym, data)), symbol)

    await stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
