import os
import asyncio
import logging
from datetime import datetime, time
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

# Load API keys
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing Alpaca API keys")

# Clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ===== SETTINGS =====
SYMBOL = "AAPL"
POSITION_SIZE = 1
MAX_TRADES_PER_DAY = 3
MIN_BREAKOUT = 0.15  # minimum breakout distance (edge filter)
EOD_CUTOFF = time(15, 55)

# ===== STATE =====
opening_high = None
opening_low = None
trade_count = 0
realized_pnl = 0.0
position = 0
entry_price = 0.0
opening_range_complete = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# ----- OPENING RANGE TRACKING -----
async def handle_bars(bar):
    global opening_high, opening_low
    global trade_count, realized_pnl
    global position, entry_price
    global opening_range_complete

    clock = trading_client.get_clock()
    now_et = clock.timestamp.time()

    # ----- HARD EOD LIQUIDATION -----
    if not clock.is_open or now_et >= EOD_CUTOFF:
        if position != 0:
            trading_client.close_all_positions()
            logger.info("EOD liquidation executed")
        await stream.stop_stream()
        return

    # ----- BUILD FIRST 15 MIN RANGE -----
    if not opening_range_complete:
        if opening_high is None:
            opening_high = bar.high
            opening_low = bar.low
        else:
            opening_high = max(opening_high, bar.high)
            opening_low = min(opening_low, bar.low)

        if clock.timestamp.minute >= 45:  # 9:30–9:45 range
            opening_range_complete = True
            logger.info(f"Opening range set: H={opening_high}, L={opening_low}")
        return

    # ----- TRADE THROTTLE -----
    if trade_count >= MAX_TRADES_PER_DAY:
        return

    price = bar.close

    # ----- LONG BREAKOUT -----
    if price > opening_high + MIN_BREAKOUT and position == 0:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        )
        entry_price = price
        position = 1
        trade_count += 1
        logger.info("LONG breakout entered")

    # ----- SHORT BREAKOUT -----
    elif price < opening_low - MIN_BREAKOUT and position == 0:
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )
        entry_price = price
        position = -1
        trade_count += 1
        logger.info("SHORT breakout entered")

    # ----- EXIT LOGIC (simple mean reversion exit) -----
    elif position == 1 and price < opening_high:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )
        realized_pnl += price - entry_price
        position = 0
        logger.info(f"Exit long | Realized P&L: {realized_pnl}")

    elif position == -1 and price > opening_low:
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=POSITION_SIZE,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        )
        realized_pnl += entry_price - price
        position = 0
        logger.info(f"Exit short | Realized P&L: {realized_pnl}")

async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed")
        return

    print("Starting ORB bot")
    stream.subscribe_bars(handle_bars, SYMBOL)
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
