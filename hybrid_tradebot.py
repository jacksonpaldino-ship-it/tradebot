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
MAX_DAILY_LOSS = -0.02         # -2% daily kill switch
MAX_BP_UTILIZATION = 0.20      # max 20% buying power per trade

COOLDOWN_MINUTES = 10
LOOP_MINUTES = 15
SLEEP_SECONDS = 60

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_ENTRY_MULT = 0.20
ATR_STOP_MULT = 0.6
ATR_TP_MULT = 1.2

MIN_TICK = 0.01

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

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

def get_account():
    return api.get_account()

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def cooldown_active(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=120).df.copy()
    df.index = pd.to_datetime(df.index)
    return df

def add_indicators(df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

# ==========================================

def place_trade(symbol, side, price, atr):
    acct = get_account()
    equity = float(acct.equity)
    buying_power = float(acct.buying_power)

    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_dist = max(atr * ATR_STOP_MULT, MIN_TICK)

    qty_risk = int(risk_dollars / stop_dist)
    qty_bp = int((buying_power * MAX_BP_UTILIZATION) / price)
    qty = min(qty_risk, qty_bp)

    if qty <= 0:
        return

    if side == "buy":
        stop = round(price - stop_dist, 2)
        tp = round(price + max(atr * ATR_TP_MULT, MIN_TICK), 2)
        tp = max(tp, price + MIN_TICK)
    else:
        stop = round(price + stop_dist, 2)
        tp = round(price - max(atr * ATR_TP_MULT, MIN_TICK), 2)
        tp = min(tp, price - MIN_TICK)

    logging.info(f"{symbol} | {side.upper()} | qty={qty} entry={price:.2f} stop={stop} tp={tp}")

    try:
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
    except Exception as e:
        logging.error(f"{symbol} ORDER FAILED — {e}")
    finally:
        cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

# ==========================================

def run_cycle():
    if not market_open() or not in_trade_window():
        return

    acct = get_account()
    equity = float(acct.equity)
    daily_pnl = float(acct.equity) / float(acct.last_equity) - 1

    logging.info(f"Equity: {equity:.2f} | Daily PnL: {daily_pnl:.2%}")

    if daily_pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT — STOPPING")
        return

    for symbol in SYMBOLS:
        try:
            if position_exists(symbol) or cooldown_active(symbol):
                continue

            df = add_indicators(get_bars(symbol))
            last, prev = df.iloc[-1], df.iloc[-2]

            atr = last["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            price = last["close"]
            momentum = abs(price - prev["close"]) > atr * ATR_ENTRY_MULT

            if not momentum:
                continue

            if last["ema_fast"] > last["ema_slow"] and price > last["vwap"]:
                place_trade(symbol, "buy", price, atr)

            elif last["ema_fast"] < last["ema_slow"] and price < last["vwap"]:
                place_trade(symbol, "sell", price, atr)

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
