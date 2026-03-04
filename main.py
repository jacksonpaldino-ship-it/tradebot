import os
import asyncio
from datetime import datetime, time
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.live import StockDataStream

# ===== CONFIG =====
SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]
ORB_BARS = 3
RISK_PER_TRADE = 0.015
RR_RATIO = 2
MAX_TRADES_PER_DAY = 4
DAILY_STOP_R = -3
DAILY_TARGET_R = 4
EOD_TIME = time(15, 55)

# ===== SETUP =====
api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(api_key, secret_key, paper=True)
stream = StockDataStream(api_key, secret_key)

ny = ZoneInfo("America/New_York")

# ===== STATE =====
state = {}
daily_r = 0
trades_today = 0
current_day = None

for symbol in SYMBOLS:
    state[symbol] = {
        "opening_high": None,
        "opening_low": None,
        "bar_count": 0,
        "orb_complete": False,
        "trades": 0
    }

# ===== HELPERS =====

def reset_day():
    global daily_r, trades_today
    daily_r = 0
    trades_today = 0
    for s in state:
        state[s]["opening_high"] = None
        state[s]["opening_low"] = None
        state[s]["bar_count"] = 0
        state[s]["orb_complete"] = False
        state[s]["trades"] = 0


def get_equity():
    return float(trading_client.get_account().equity)


def get_position(symbol):
    try:
        return trading_client.get_open_position(symbol)
    except:
        return None


def calculate_qty(entry, stop):
    equity = get_equity()
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = abs(entry - stop)

    if per_share_risk <= 0:
        return 0

    qty = int(risk_amount / per_share_risk)
    return max(qty, 0)


def close_all():
    trading_client.close_all_positions(cancel_orders=True)


# ===== MAIN HANDLER =====

async def handle_bar(bar):
    global daily_r, trades_today, current_day

    now = datetime.now(ny)
    today = now.date()
    symbol = bar.symbol
    price = bar.close

    if current_day != today:
        current_day = today
        reset_day()

    if now.time() >= EOD_TIME:
        close_all()
        return

    if daily_r <= DAILY_STOP_R or daily_r >= DAILY_TARGET_R:
        return

    if trades_today >= MAX_TRADES_PER_DAY:
        return

    if bar.timestamp.astimezone(ny).time() < time(9, 30):
        return

    s = state[symbol]

    # ===== BUILD ORB =====
    if not s["orb_complete"]:
        s["bar_count"] += 1

        if s["opening_high"] is None:
            s["opening_high"] = bar.high
            s["opening_low"] = bar.low
        else:
            s["opening_high"] = max(s["opening_high"], bar.high)
            s["opening_low"] = min(s["opening_low"], bar.low)

        if s["bar_count"] >= ORB_BARS:
            s["orb_complete"] = True
            print(f"{symbol} ORB built")

        return

    if s["opening_high"] is None or s["opening_low"] is None:
        return

    if get_position(symbol) is not None:
        return

    if s["trades"] >= 2:
        return

    # ===== LONG =====
    if price > s["opening_high"]:

        entry = price
        stop = s["opening_low"]
        target = entry + (entry - stop) * RR_RATIO

        qty = calculate_qty(entry, stop)
        if qty == 0:
            return

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss={"stop_price": stop},
            take_profit={"limit_price": target},
        )

        trading_client.submit_order(order)

        trades_today += 1
        s["trades"] += 1
        daily_r -= 1
        print(f"LONG {symbol} {qty}")

    # ===== SHORT =====
    elif price < s["opening_low"]:

        entry = price
        stop = s["opening_high"]
        target = entry - (stop - entry) * RR_RATIO

        qty = calculate_qty(entry, stop)
        if qty == 0:
            return

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss={"stop_price": stop},
            take_profit={"limit_price": target},
        )

        trading_client.submit_order(order)

        trades_today += 1
        s["trades"] += 1
        daily_r -= 1
        print(f"SHORT {symbol} {qty}")


# ===== SUBSCRIBE =====
for symbol in SYMBOLS:
    stream.subscribe_bars(handle_bar, symbol)

asyncio.run(stream._run_forever())
