import os
import logging
from datetime import datetime, time
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ====================== CONFIG ======================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

MAX_NOTIONAL_PER_TRADE = 2500      # <-- change to 300–500 for $2k account
MAX_TOTAL_EXPOSURE = 5000          # total intraday exposure cap
MAX_DAILY_LOSS_PCT = -0.01         # -1% kill switch
MIN_AVG_VOLUME = 1_000_000

ATR_PERIOD = 14
STOP_ATR_MULT = 1.2
TP_ATR_MULT = 1.5

EASTERN = pytz.timezone("America/New_York")

# ===================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

api = REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"],
)

# ====================== TIME LOGIC ======================

def now_et():
    return datetime.now(EASTERN)

def market_window():
    t = now_et().time()
    return time(9, 35) <= t <= time(15, 45)

def near_close():
    return now_et().time() >= time(15, 50)

# ====================== DATA ======================

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=60).df
    return df

def compute_atr(df):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_PERIOD).mean().iloc[-1]

# ====================== RISK ======================

def account_state():
    acct = api.get_account()
    return float(acct.equity), float(acct.last_equity)

def daily_pnl_pct():
    eq, last = account_state()
    return eq / last - 1

def current_exposure():
    exposure = 0
    for p in api.list_positions():
        exposure += abs(float(p.market_value))
    return exposure

# ====================== EXECUTION ======================

def flatten_all():
    for p in api.list_positions():
        side = "sell" if int(p.qty) > 0 else "buy"
        qty = abs(int(p.qty))
        logging.info(f"FORCE FLATTEN {p.symbol} qty={qty}")
        api.submit_order(
            symbol=p.symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

def place_trade(symbol, side, atr):
    price = api.get_latest_trade(symbol).price
    stop_dist = atr * STOP_ATR_MULT
    qty = int(MAX_NOTIONAL_PER_TRADE / price)

    if qty <= 0:
        return

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )

# ====================== STRATEGY ======================

def trade_signal(df):
    ema_fast = df["close"].ewm(span=9).mean().iloc[-1]
    ema_slow = df["close"].ewm(span=21).mean().iloc[-1]
    price = df["close"].iloc[-1]

    if ema_fast > ema_slow:
        return "buy"
    if ema_fast < ema_slow:
        return "sell"
    return None

# ====================== MAIN ======================

def run():
    logging.info("BOT START")

    if daily_pnl_pct() <= MAX_DAILY_LOSS_PCT:
        logging.warning("DAILY LOSS LIMIT HIT")
        return

    if near_close():
        logging.info("Market near close — flattening")
        flatten_all()
        return

    if not market_window():
        return

    exposure = current_exposure()
    if exposure >= MAX_TOTAL_EXPOSURE:
        logging.info("Exposure cap hit — no new trades")
        return

    for symbol in SYMBOLS:
        if any(p.symbol == symbol for p in api.list_positions()):
            continue

        bars = get_bars(symbol)
        if bars["volume"].mean() < MIN_AVG_VOLUME:
            continue

        atr = compute_atr(bars)
        if pd.isna(atr) or atr == 0:
            continue

        signal = trade_signal(bars)
        if signal:
            place_trade(symbol, signal, atr)

    logging.info("BOT END")

# ====================== RUN ======================

if __name__ == "__main__":
    run()
