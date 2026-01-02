import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

MAX_RISK_PER_TRADE = 0.01      # 1% equity per trade
MAX_DAILY_LOSS = -0.02         # -2% daily kill switch

COOLDOWN_MINUTES = 10
LOOP_MINUTES = 15
SLEEP_SECONDS = 60

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_ENTRY_MULT = 0.25
ATR_STOP_MULT = 0.6
ATR_TP_MULT = 1.0

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===== ENV VARS (MATCH YOUR SECRETS) =====

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL)

eastern = pytz.timezone("America/New_York")
cooldowns = {}

# ==========================================

def now_et():
    return datetime.now(eastern)

def market_open():
    return api.get_clock().is_open

def in_trade_window():
    t = now_et().time()
    return (
        datetime.strptime("09:35", "%H:%M").time() <= t <= datetime.strptime("11:30", "%H:%M").time()
        or
        datetime.strptime("13:30", "%H:%M").time() <= t <= datetime.strptime("15:50", "%H:%M").time()
    )

def get_equity():
    return float(api.get_account().equity)

def get_daily_pnl():
    acct = api.get_account()
    return float(acct.equity) / float(acct.last_equity) - 1

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def cooldown_active(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

# ========= DATA =========

def get_bars(symbol):
    bars = api.get_bars(symbol, "1Min", limit=120).df
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    return bars

def add_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

# ========= TRADING =========

def place_trade(symbol, side, price, atr, equity):
    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_dist = atr * ATR_STOP_MULT
    qty = int(risk_dollars / stop_dist)

    if qty <= 0:
        return

    if side == "buy":
        stop_price = round(price - stop_dist, 2)
        tp_price = round(price + atr * ATR_TP_MULT, 2)
    else:
        stop_price = round(price + stop_dist, 2)
        tp_price = round(price - atr * ATR_TP_MULT, 2)

    logging.info(
        f"{symbol} | {side.upper()} | qty={qty} entry={price:.2f} stop={stop_price} tp={tp_price}"
    )

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day",
        order_class="bracket",
        stop_loss={"stop_price": stop_price},
        take_profit={"limit_price": tp_price}
    )

    cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

# ========= MAIN LOOP =========

def run_cycle():
    if not market_open() or not in_trade_window():
        return

    equity = get_equity()
    daily_pnl = get_daily_pnl()

    logging.info(f"Equity: {equity:.2f} | Daily PnL: {daily_pnl:.2%}")

    if daily_pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT — STOPPING TRADES")
        return

    for symbol in SYMBOLS:
        try:
            if position_exists(symbol):
                continue

            if cooldown_active(symbol):
                continue

            df = add_indicators(get_bars(symbol))
            last = df.iloc[-1]
            prev = df.iloc[-2]

            atr = last["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            price = last["close"]
            momentum = abs(price - prev["close"]) > atr * ATR_ENTRY_MULT

            if not momentum:
                continue

            trend_up = last["ema_fast"] > last["ema_slow"]
            trend_down = last["ema_fast"] < last["ema_slow"]

            if trend_up and price > last["vwap"]:
                place_trade(symbol, "buy", price, atr, equity)

            elif trend_down and price < last["vwap"]:
                place_trade(symbol, "sell", price, atr, equity)

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

# ========= ENTRY =========

if __name__ == "__main__":
    logging.info("BOT START")
    start = time.time()

    while time.time() - start < LOOP_MINUTES * 60:
        run_cycle()
        time.sleep(SLEEP_SECONDS)

    logging.info("BOT END")
