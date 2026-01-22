import os
import asyncio
import logging
from datetime import datetime, time
import pandas as pd
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

# Load secrets (GitHub Actions secrets still work)
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing Alpaca API keys")

# Logging
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

# Market data stream
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

# HARD EOD cutoff (ET)
EOD_CUTOFF = time(15, 55)  # 3:55 PM ET

async def handle_bars(bar):
    global bars_seen, pnl

    # ----- HARD END-OF-DAY LIQUIDATION -----
    clock = trading_client.get_clock()
    now_et = clock.timestamp.time()

    if not clock.is_open or now_et >= EOD_CUTOFF:
        logger.info("EOD reached — closing all positions")
        print("EOD reached — closing all positions")

        try:
            trading_client.close_all_positions()
        except Exception as e:
            logger.error(f"EOD close error: {e}")

        await stream.stop_stream()
        return
    # --------------------------------------

    bars_seen += 1
    logger.info(f"New bar: {bar}")
    print(f"{datetime.now()} - New bar: {bar}")

    if bar.close > bar.open:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        )
        pnl -= bar.close * POSITION_SIZE
        logger.info(f"BUY order: {order}")

    elif bar.close < bar.open:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )
        pnl += bar.close * POSITION_SIZE
        logger.info(f"SELL order: {order}")

    print(f"Current P&L: {pnl}")

    if bars_seen >= MAX_BARS:
        await stream.stop_stream()

# Subscribe
stream.subscribe_bars(handle_bars, SYMBOL)

async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed — bot exiting")
        return

    logger.info("Starting Alpaca bot")
    print("Starting Alpaca bot")
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
