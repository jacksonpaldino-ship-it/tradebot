import asyncio
import datetime
import logging
import os
import sys
import json
import time
import random
from itertools import islice
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi

# ----------------------
# Broker Abstraction
# ----------------------
class BrokerInterface:
    """Abstraction layer to swap broker easily later."""
    def __init__(self, api_key, secret_key, base_url):
        self.api = tradeapi.REST(api_key, secret_key, base_url)
        self.account = self.api.get_account()

    # Market data
    async def get_last_price(self, symbol):
        barset = self.api.get_barset(symbol, 'minute', limit=1)
        return barset[symbol][0].c if barset[symbol] else None

    # Place order
    async def place_order(self, symbol, qty, side, type='market', time_in_force='day'):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            return order, True
        except Exception as e:
            return str(e), False

    # Get positions
    async def get_positions(self):
        return self.api.list_positions()

    # Cancel order
    async def cancel_order(self, order_id):
        return self.api.cancel_order(order_id)

# ----------------------
# Trading Bot
# ----------------------
class TradingBot:
    def __init__(self, broker: BrokerInterface, symbols, logger):
        self.broker = broker
        self.symbols = symbols
        self.logger = logger

        # Async locks
        self.locks = {s: asyncio.Lock() for s in symbols}

        # Data structures
        self.position_info = {}
        self.on_going_orders = {s: [] for s in symbols}
        self.on_going_orders_details = {}
        self.active_symbols = []
        self.max_price_seen = {s: 0 for s in symbols}
        self.past_prices_seen = {s: [] for s in symbols}
        self.average_price = {s: 0 for s in symbols}
        self.fund_available = 100_000  # adjustable

    # ----------------------
    # Core async loop
    # ----------------------
    async def run(self):
        self.logger.info("Bot started ...")
        await asyncio.gather(
            self.market_data_loop(),
            self.position_sizing_loop(),
            self.order_status_loop(),
            self.position_closure_loop()
        )

    # ----------------------
    # Market Data Processing
    # ----------------------
    async def market_data_loop(self):
        while True:
            for symbol in self.symbols:
                price = await self.broker.get_last_price(symbol)
                if price:
                    self.max_price_seen[symbol] = max(self.max_price_seen[symbol], price)
                    self.past_prices_seen[symbol].append(price)
                    if len(self.past_prices_seen[symbol]) > 20:
                        self.past_prices_seen[symbol].pop(0)
                    self.average_price[symbol] = sum(self.past_prices_seen[symbol]) / len(self.past_prices_seen[symbol])
            await asyncio.sleep(1)

    # ----------------------
    # Position Sizing
    # ----------------------
    async def position_sizing_loop(self):
        x = 3
        y = 5
        while len(self.active_symbols) < len(self.symbols):
            to_add = (s for s in self.symbols if s not in self.active_symbols)
            self.active_symbols.extend(islice(to_add, x))
            await asyncio.sleep(y)

    # ----------------------
    # Order Status Updater
    # ----------------------
    async def order_status_loop(self):
        while True:
            # Dummy: placeholder for real order updates
            await asyncio.sleep(5)

    # ----------------------
    # Position Closure
    # ----------------------
    async def position_closure_loop(self):
        while True:
            now_time = datetime.datetime.now(ZoneInfo("Asia/Taipei")).time()
            for symbol, info in self.position_info.items():
                if now_time >= datetime.time(13, 30):  # exit cutoff
                    async with self.locks[symbol]:
                        qty = info["size"]
                        if qty > 0:
                            order, success = await self.broker.place_order(symbol, qty, "sell")
                            if success:
                                self.logger.info(f"{symbol} exit order placed successfully")
                                self.position_info.pop(symbol)
            await asyncio.sleep(1)

# ----------------------
# Supervisor
# ----------------------
def main():
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL")

    logger = logging.getLogger("TradingBot")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    broker = BrokerInterface(api_key, secret_key, base_url)
    symbols = ["AAPL", "TSLA", "MSFT"]  # example

    bot = TradingBot(broker, symbols, logger)
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()
