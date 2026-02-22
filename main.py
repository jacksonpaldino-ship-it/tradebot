import os
import asyncio
import logging
from datetime import time
from collections import deque
from dotenv import load_dotenv

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import StopLossRequest, TakeProfitRequest

# ========= SETTINGS =========
SYMBOL = "AAPL"

RISK_PER_TRADE = 0.015       # 1.5%
RISK_REWARD = 2.0
MAX_DAILY_LOSS = 0.02        # 2%
MAX_TRADES_PER_DAY = 3
EOD_CUTOFF = time(15, 55)

ORB_MINUTES = 15
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.8         # tighter stop for better sizing
# ============================

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# ========= STATE =========
bars = deque(maxlen=ATR_PERIOD)
opening_high = None
opening_low = None
range_complete = False
trades_today = 0
start_equity = None
# ==========================

def calculate_atr():
    if len(bars) < 2:
        return None
    trs = []
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)

def calculate_position_size(stop_distance):
    account = trading_client.get_account()
    equity = float(account.equity)

    risk_amount = equity * RISK_PER_TRADE
    qty = int(risk_amount / stop_distance)

    return max(qty, 10)

def flatten_all():
    trading_client.close_all_positions()

# ==========================

async def handle_bar(bar):
    global opening_high, opening_low
    global range_complete, trades_today
    global start_equity

    clock = trading_client.get_clock()
    now = clock.timestamp.time()

    # ----- EOD -----
    if not clock.is_open or now >= EOD_CUTOFF:
        flatten_all()
        await stream.stop_stream()
        return

    account = trading_client.get_account()

    if start_equity is None:
        start_equity = float(account.equity)

    if float(account.equity) <= start_equity * (1 - MAX_DAILY_LOSS):
        log.info("Daily loss limit reached")
        flatten_all()
        await stream.stop_stream()
        return

    bars.append(bar)

    # ----- BUILD ORB -----
    if not range_complete:
        if opening_high is None:
            opening_high = bar.high
            opening_low = bar.low
        else:
            opening_high = max(opening_high, bar.high)
            opening_low = min(opening_low, bar.low)

        if clock.timestamp.minute >= 45:
            range_complete = True
            log.info(f"Opening Range H={opening_high:.2f} L={opening_low:.2f}")
        return

    if trades_today >= MAX_TRADES_PER_DAY:
        return

    atr = calculate_atr()
    if atr is None:
        return

    stop_distance = atr * ATR_MULTIPLIER
    price = bar.close

    # ----- LONG BREAKOUT WITH MOMENTUM -----
    if price > opening_high and bar.close > bar.open:
        stop = price - stop_distance
        qty = calculate_position_size(stop_distance)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=price + stop_distance * RISK_REWARD
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop
                )
            )
        )

        trades_today += 1
        log.info(f"LONG {qty} shares | Risk ${stop_distance * qty:.2f}")

    # ----- SHORT BREAKDOWN WITH MOMENTUM -----
    elif price < opening_low and bar.close < bar.open:
        stop = price + stop_distance
        qty = calculate_position_size(stop_distance)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=price - stop_distance * RISK_REWARD
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop
                )
            )
        )

        trades_today += 1
        log.info(f"SHORT {qty} shares | Risk ${stop_distance * qty:.2f}")

# ==========================

async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed.")
        return

    print("1.5% Risk ORB Bot Running.")
    stream.subscribe_bars(handle_bar, SYMBOL)
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
