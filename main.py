import os
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

# ===== CONFIG =====
SYMBOL = "AAPL"
ORB_MINUTES = 5
RISK_PER_TRADE = 0.01          # 1% account risk
RR_RATIO = 2                   # 2R target
MAX_DAILY_LOSS = 0.02          # 2% daily stop
EOD_LIQUIDATION_TIME = time(15, 55)

# ===== SETUP =====
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(api_key, secret_key, paper=True)
data_stream = StockDataStream(api_key, secret_key)

ny_tz = ZoneInfo("America/New_York")

opening_high = None
opening_low = None
orb_complete = False
daily_loss_hit = False

# ===== HELPERS =====

def get_account_equity():
    account = trading_client.get_account()
    return float(account.equity)

def get_position():
    try:
        return trading_client.get_open_position(SYMBOL)
    except:
        return None

def calculate_position_size(entry, stop):
    equity = get_account_equity()
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = abs(entry - stop)

    if per_share_risk == 0:
        return 0

    qty = int(risk_amount / per_share_risk)
    return max(qty, 0)

def close_all_positions():
    trading_client.close_all_positions(cancel_orders=True)

# ===== MAIN BAR HANDLER =====

async def handle_bar(bar):
    global opening_high, opening_low, orb_complete, daily_loss_hit

    now = datetime.now(ny_tz).time()
    price = bar.close

    # EOD liquidation
    if now >= EOD_LIQUIDATION_TIME:
        close_all_positions()
        return

    # Build ORB
    if not orb_complete:
        if now >= time(9,30) and now < time(9,30 + ORB_MINUTES):
            if opening_high is None:
                opening_high = bar.high
                opening_low = bar.low
            else:
                opening_high = max(opening_high, bar.high)
                opening_low = min(opening_low, bar.low)
            return
        elif now >= time(9,30 + ORB_MINUTES):
            orb_complete = True

    # Guard
    if opening_high is None or opening_low is None:
        return

    # Skip if already in position
    if get_position() is not None:
        return

    # LONG breakout
    if price > opening_high:
        entry = price
        stop = opening_low
        target = entry + (entry - stop) * RR_RATIO

        qty = calculate_position_size(entry, stop)
        if qty <= 0:
            return

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)

    # SHORT breakout
    elif price < opening_low:
        entry = price
        stop = opening_high
        target = entry - (stop - entry) * RR_RATIO

        qty = calculate_position_size(entry, stop)
        if qty <= 0:
            return

        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)

# ===== RUN =====

data_stream.subscribe_bars(handle_bar, SYMBOL)

asyncio.run(data_stream._run_forever())
