import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

MAX_RISK_PER_TRADE = 0.005     # 0.5% equity
MAX_BP_UTIL = 0.25             # 25% buying power per trade
MAX_DAILY_LOSS = -0.02         # -2% kill switch
COOLDOWN_MINUTES = 15

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_STOP_MULT = 1.0
ATR_TP_MULT = 1.5

RUN_MINUTES = 10               # GitHub Action runtime
SLEEP_SECONDS = 60

# =========================================

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

# =========================================

def now_et():
    return datetime.now(eastern)

def market_open():
    return api.get_clock().is_open

def trade_window():
    t = now_et().time()
    return (
        (t >= datetime.strptime("09:35", "%H:%M").time() and
         t <= datetime.strptime("11:30", "%H:%M").time()) or
        (t >= datetime.strptime("13:30", "%H:%M").time() and
         t <= datetime.strptime("15:45", "%H:%M").time())
    )

def indicators(df):
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=100).df
    return df[df["symbol"] == symbol]

def in_position(symbol):
    try:
        api.get_position(symbol)
        return True
    except:
        return False

def cooldown_active(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

# =========================================

def place_trade(symbol, side, atr):
    account = api.get_account()

    equity = float(account.equity)
    buying_power = float(account.buying_power)
    price = float(api.get_last_trade(symbol).price)

    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_dist = atr * ATR_STOP_MULT

    qty_risk = int(risk_dollars / stop_dist)
    qty_bp = int((buying_power * MAX_BP_UTIL) / price)

    qty = min(qty_risk, qty_bp)

    if qty < 1:
        logging.info(f"{symbol} SKIP — qty too small")
        return

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )

    cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

# =========================================

def manage_exits():
    positions = api.list_positions()

    for p in positions:
        symbol = p.symbol
        qty = abs(int(float(p.qty)))
        entry = float(p.avg_entry_price)
        current = float(p.current_price)
        atr = float(indicators(get_bars(symbol)).iloc[-1]["atr"])

        if atr <= 0:
            continue

        stop = entry - atr * ATR_STOP_MULT if p.side == "long" else entry + atr * ATR_STOP_MULT
        tp = entry + atr * ATR_TP_MULT if p.side == "long" else entry - atr * ATR_TP_MULT

        exit_side = "sell" if p.side == "long" else "buy"

        if (p.side == "long" and (current <= stop or current >= tp)) or \
           (p.side == "short" and (current >= stop or current <= tp)):

            logging.info(f"{symbol} EXIT @ {current}")
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=exit_side,
                type="market",
                time_in_force="day"
            )

# =========================================

def run_cycle():
    if not market_open() or not trade_window():
        return

    account = api.get_account()
    daily_pnl = float(account.equity) / float(account.last_equity) - 1

    logging.info(f"Equity: {account.equity} | Daily PnL: {daily_pnl:.2%}")

    if daily_pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT — STOPPING")
        return

    manage_exits()

    for symbol in SYMBOLS:
        if in_position(symbol) or cooldown_active(symbol):
            continue

        df = indicators(get_bars(symbol))
        last = df.iloc[-1]
        prev = df.iloc[-2]

        if last["atr"] <= 0:
            continue

        momentum = abs(last["close"] - prev["close"]) > last["atr"] * 0.3

        if last["ema_fast"] > last["ema_slow"] and momentum:
            place_trade(symbol, "buy", last["atr"])

        elif last["ema_fast"] < last["ema_slow"] and momentum:
            place_trade(symbol, "sell", last["atr"])

# =========================================

if __name__ == "__main__":
    logging.info("BOT START")
    start = time.time()

    while time.time() - start < RUN_MINUTES * 60:
        run_cycle()
        time.sleep(SLEEP_SECONDS)

    logging.info("BOT END")
