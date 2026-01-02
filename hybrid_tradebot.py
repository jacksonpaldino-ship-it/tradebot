import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]
MAX_RISK_PER_TRADE = 0.01      # 1% of equity per trade
MAX_DAILY_LOSS = -0.02         # -2% daily kill switch
COOLDOWN_MINUTES = 10
LOOP_MINUTES = 15
SLEEP_SECONDS = 60

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_ENTRY_MULT = 0.25
ATR_STOP_MULT = 0.5
ATR_TP_MULT = 0.75

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Alpaca environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL)

eastern = pytz.timezone("America/New_York")
cooldowns = {}

# ==========================================

def market_open():
    clock = api.get_clock()
    return clock.is_open

def now_et():
    return datetime.now(eastern)

def in_trade_window():
    t = now_et().time()
    return (
        (t >= datetime.strptime("09:35", "%H:%M").time() and
         t <= datetime.strptime("11:00", "%H:%M").time()) or
        (t >= datetime.strptime("13:30", "%H:%M").time() and
         t <= datetime.strptime("15:45", "%H:%M").time())
    )

def get_equity():
    return float(api.get_account().equity)

def get_daily_pnl():
    acct = api.get_account()
    return float(acct.equity) / float(acct.last_equity) - 1

def get_bars(symbol):
    bars = api.get_bars(symbol, timeframe="1Min", limit=100).df
    if bars.empty:
        return None
    if 'symbol' in bars.columns:
        bars = bars[bars['symbol'] == symbol]
    bars = bars.reset_index()
    return bars if not bars.empty else None

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
    if symbol not in cooldowns:
        return False
    return now_et() < cooldowns[symbol]

def place_trade(symbol, side, price, atr, equity):
    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_distance = atr * ATR_STOP_MULT
    qty = max(int(risk_dollars / stop_distance), 1)

    if side == "buy":
        stop = round(price - stop_distance, 2)
        tp = round(price + atr * ATR_TP_MULT, 2)
        if tp <= price:
            tp = price + 0.01  # ensure valid bracket
    else:
        stop = round(price + stop_distance, 2)
        tp = round(price - atr * ATR_TP_MULT, 2)
        if tp >= price:
            tp = price - 0.01  # ensure valid bracket

    logging.info(f"{symbol} | {side.upper()} | qty={qty} entry={price} stop={stop} tp={tp}")

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
        cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)
    except Exception as e:
        logging.error(f"{symbol} ORDER FAILED — {e}")

# ==========================================

def run_cycle():
    if not market_open() or not in_trade_window():
        return

    equity = get_equity()
    daily_pnl = get_daily_pnl()
    logging.info(f"Equity: {equity:.2f} | Daily PnL: {daily_pnl:.2%}")

    if daily_pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT — STOPPING")
        return

    for symbol in SYMBOLS:
        try:
            if position_exists(symbol):
                continue
            if cooldown_active(symbol):
                continue

            df = get_bars(symbol)
            if df is None:
                logging.info(f"{symbol}: no data — skip")
                continue

            df = indicators(df)
            last = df.iloc[-1]
            price = last["close"]
            atr = last["atr"]

            if atr is None or atr == 0:
                continue

            trend_up = last["ema_fast"] > last["ema_slow"]
            trend_down = last["ema_fast"] < last["ema_slow"]
            momentum = abs(price - df.iloc[-2]["close"]) > atr * ATR_ENTRY_MULT

            if not momentum:
                continue

            if trend_up and price > last["vwap"]:
                place_trade(symbol, "buy", price, atr, equity)
            elif trend_down and price < last["vwap"]:
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
