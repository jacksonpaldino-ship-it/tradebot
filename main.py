import os
import asyncio
import logging
from datetime import time
from dotenv import load_dotenv

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

# ===== SETTINGS =====
SYMBOL = "AAPL"

RISK_PER_TRADE = 0.015      # 1.5% risk
RISK_REWARD = 2.5           # Bigger profit target
MAX_DAILY_LOSS = 0.03       # 3% max loss
ORB_MINUTES = 5
EOD_CUTOFF = time(15, 55)
# ====================

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

opening_high = None
opening_low = None
orb_complete = False
traded_today = False
start_equity = None

def calculate_position_size(entry, stop):
    account = trading_client.get_account()
    equity = float(account.equity)

    risk_amount = equity * RISK_PER_TRADE
    stop_distance = abs(entry - stop)

    qty = int(risk_amount / stop_distance)

    return max(qty, 1)

def flatten_all():
    trading_client.close_all_positions()

async def handle_bar(bar):
    global opening_high, opening_low, orb_complete
    global traded_today, start_equity

    clock = trading_client.get_clock()
    now = clock.timestamp.time()

    if not clock.is_open or now >= EOD_CUTOFF:
        flatten_all()
        await stream.stop_stream()
        return

    account = trading_client.get_account()

    if start_equity is None:
        start_equity = float(account.equity)

    if float(account.equity) <= start_equity * (1 - MAX_DAILY_LOSS):
        log.info("Daily loss limit hit.")
        flatten_all()
        await stream.stop_stream()
        return

    # ===== Build Opening Range =====
    market_open = time(9, 30)

    if not orb_complete:
        if now >= market_open and now < time(9, 30 + ORB_MINUTES):
            if opening_high is None:
                opening_high = bar.high
                opening_low = bar.low
            else:
                opening_high = max(opening_high, bar.high)
                opening_low = min(opening_low, bar.low)
            return
        else:
            orb_complete = True
            log.info(f"ORB High: {opening_high}, Low: {opening_low}")
            return

    if traded_today:
        return

    price = bar.close

    # ===== LONG BREAKOUT =====
    if price > opening_high:
        stop = opening_low
        qty = calculate_position_size(price, stop)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=price + (price - stop) * RISK_REWARD
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop
                )
            )
        )

        traded_today = True
        log.info(f"LONG {qty} shares")

    # ===== SHORT BREAKOUT =====
    elif price < opening_low:
        stop = opening_high
        qty = calculate_position_size(price, stop)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=price - (stop - price) * RISK_REWARD
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop
                )
            )
        )

        traded_today = True
        log.info(f"SHORT {qty} shares")

async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed.")
        return

    print("ORB Bot Running.")
    stream.subscribe_bars(handle_bar, SYMBOL)
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
