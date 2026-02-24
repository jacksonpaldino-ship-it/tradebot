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
SYMBOLS = ["AAPL", "NVDA", "TSLA", "AMD", "META"]

RISK_PER_TRADE = 0.015      # 1.5%
RISK_REWARD = 2.0
MAX_DAILY_LOSS = 0.03       # 3% account stop
MAX_TRADES_PER_DAY = 6
EOD_CUTOFF = time(15, 55)

ATR_PERIOD = 14
ATR_MULTIPLIER = 1.0
# ============================

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# ========= STATE =========
bars = {symbol: deque(maxlen=ATR_PERIOD) for symbol in SYMBOLS}
trades_today = 0
start_equity = None
# ==========================

def calculate_atr(symbol):
    data = bars[symbol]
    if len(data) < 2:
        return None
    trs = []
    for i in range(1, len(data)):
        high = data[i].high
        low = data[i].low
        prev_close = data[i-1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)

def calculate_position_size(stop_distance):
    account = trading_client.get_account()
    equity = float(account.equity)

    risk_amount = equity * RISK_PER_TRADE
    qty = int(risk_amount / stop_distance)

    return max(qty, 5)

def flatten_all():
    trading_client.close_all_positions()

# ==========================

async def handle_bar(bar):
    global trades_today, start_equity

    symbol = bar.symbol

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

    if trades_today >= MAX_TRADES_PER_DAY:
        return

    bars[symbol].append(bar)

    atr = calculate_atr(symbol)
    if atr is None:
        return

    stop_distance = atr * ATR_MULTIPLIER
    price = bar.close

    # Momentum breakout logic
    recent_high = max(b.high for b in bars[symbol])
    recent_low = min(b.low for b in bars[symbol])

    # ----- LONG MOMENTUM -----
    if price > recent_high and bar.close > bar.open:
        stop = price - stop_distance
        qty = calculate_position_size(stop_distance)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
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
        log.info(f"{symbol} LONG {qty}")

    # ----- SHORT MOMENTUM -----
    elif price < recent_low and bar.close < bar.open:
        stop = price + stop_distance
        qty = calculate_position_size(stop_distance)

        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
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
        log.info(f"{symbol} SHORT {qty}")

# ==========================

async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed.")
        return

    print("Multi-Symbol Momentum Bot Running.")
    for symbol in SYMBOLS:
        stream.subscribe_bars(handle_bar, symbol)

    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
