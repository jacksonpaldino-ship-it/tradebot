import os
import asyncio
import logging
from datetime import datetime, time
from collections import deque

from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ================== CONFIG ==================
SYMBOL = "AAPL"
QTY = 1

ORB_MINUTES = 5
MIN_EDGE_CENTS = 0.20      # ---- EDGE FILTER ----
MAX_TRADES_PER_DAY = 20   # ---- TRADE THROTTLE ----
EOD_CUTOFF = time(15, 55) # ---- HARD FLATTEN ----

# ================== AUTH ==================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing Alpaca API keys")

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stream = StockDataStream(API_KEY, API_SECRET)

# ================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)
log = logging.getLogger()

# ================== STATE ==================
bars = deque(maxlen=ORB_MINUTES)
orb_high = None
orb_low = None

trades_today = 0
realized_pnl = 0.0
position = 0  # +1 long, -1 short, 0 flat

# ================== HELPERS ==================
def submit(side):
    global trades_today
    trading_client.submit_order(
        MarketOrderRequest(
            symbol=SYMBOL,
            qty=QTY,
            side=side,
            time_in_force=TimeInForce.DAY
        )
    )
    trades_today += 1

def flatten_all():
    global position
    trading_client.close_all_positions()
    position = 0

# ================== BAR HANDLER ==================
async def on_bar(bar):
    global orb_high, orb_low, position, realized_pnl

    # ----- HARD EOD LIQUIDATION -----
    clock = trading_client.get_clock()
    now_et = clock.timestamp.time()

    if not clock.is_open or now_et >= EOD_CUTOFF:
        log.info("EOD reached — flattening")
        flatten_all()
        await stream.stop_stream()
        return
    # --------------------------------

    bars.append(bar)

    # Build ORB
    if len(bars) < ORB_MINUTES:
        return

    if orb_high is None:
        orb_high = max(b.high for b in bars)
        orb_low = min(b.low for b in bars)
        log.info(f"ORB SET | High={orb_high:.2f} Low={orb_low:.2f}")
        return

    price = bar.close

    # ----- TRADE LIMIT -----
    if trades_today >= MAX_TRADES_PER_DAY:
        return

    # ----- LONG BREAKOUT -----
    if price > orb_high and position <= 0:
        edge = price - orb_high
        if edge >= MIN_EDGE_CENTS:
            if position < 0:
                realized_pnl += (entry_price - price) * QTY
            submit(OrderSide.BUY)
            position = 1
            entry_price = price
            log.info(f"LONG @ {price:.2f}")

    # ----- SHORT BREAKDOWN -----
    elif price < orb_low and position >= 0:
        edge = orb_low - price
        if edge >= MIN_EDGE_CENTS:
            if position > 0:
                realized_pnl += (price - entry_price) * QTY
            submit(OrderSide.SELL)
            position = -1
            entry_price = price
            log.info(f"SHORT @ {price:.2f}")

    log.info(
        f"Price={price:.2f} | Trades={trades_today} | Realized PnL={realized_pnl:.2f}"
    )

# ================== MAIN ==================
async def main():
    clock = trading_client.get_clock()
    if not clock.is_open:
        print("Market closed — exiting")
        return

    log.info("BOT STARTED")
    stream.subscribe_bars(on_bar, SYMBOL)
    await stream._run_forever()

if __name__ == "__main__":
    asyncio.run(main())
