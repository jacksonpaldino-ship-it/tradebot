import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

MAX_RISK_PER_TRADE = 0.01      # 1% equity
MAX_DAILY_LOSS = -0.02         # -2% kill switch
COOLDOWN_MINUTES = 10

LOOP_MINUTES = 15
SLEEP_SECONDS = 60

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_ENTRY_MULT = 0.25
ATR_STOP_MULT = 0.5
ATR_TP_MULT = 0.75

MIN_TICK = 0.01

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ✅ CORRECT ENV VARS
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
        datetime.strptime("09:35", "%H:%M").time() <= t <= datetime.strptime("11:00", "%H:%M").time()
        or
        datetime.strptime("13:30", "%H:%M").time() <= t <= datetime.strptime("15:45", "%H:%M").time()
    )

def get_equity():
    return float(api.get_account().equity)

def get_daily_pnl():
    acct = api.get_account()
    return float(acct.equity) / float(acct.last_equity) - 1

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=100).df
    return df[df["symbol"] == symbol]

def indicators(df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW).mean()
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def cooldown_active(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

def place_trade(symbol, side, price, atr, equity):
    if pd.isna(atr) or atr <= 0:
        return

    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_distance = atr * ATR_STOP_MULT
    qty = int(risk_dollars / stop_distance)

    if qty <= 0:
        return

    if side == "buy":
        stop = round(price - stop_distance, 2)
        tp = round(price + atr * ATR_TP_MULT, 2)
    else:
        stop = round(price + stop_distance, 2)
        tp = round(price - atr * ATR_TP_MULT - MIN_TICK, 2)

    logging.info(f"{symbol} | {side.upper()} | qty={qty} entry={price:.2f} stop={stop} tp={tp}")

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day",
        order_class="bracket",
        stop_loss={"stop_price": stop},
        take_profit={"limit_price": tp}
    )

    cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

# ==========================================

def run_cycle():
    if not market_open() or not in_trade_window():
        return

    equity = get_equity()
    daily_pnl = get_daily_pnl()

    logging.info(f"Equity: {equity:.2f} | Daily PnL: {daily_pnl:.2%}")

    if daily_pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT — HALTING TRADES")
        return

    for symbol in SYMBOLS:
        try:
            if position_exists(symbol) or cooldown_active(symbol):
                continue

            df = indicators(get_bars(symbol))
            if len(df) < ATR_PERIOD + 1:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]

            price = last["close"]
            atr = last["atr"]

            momentum = abs(price - prev["close"]) > atr * ATR_ENTRY_MULT

            if not momentum:
                continue

            if last["ema_fast"] > last["ema_slow"] and price > last["vwap"]:
                place_trade(symbol, "buy", price, atr, equity)

            elif last["ema_fast"] < last["ema_slow"] and price < last["vwap"]:
                place_trade(symbol, "sell", price, atr, equity)

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

# ==========================================

if __name__ == "__main__":
    logging.info("BOT START")
    start = time.time()

    while time.time() - start < LOOP_MINUTES * 60:
        run_cycle()
        time.sleep(SLEEP_SECONDS)

    logging.info("BOT END")
