import os
import time
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ===================== CONFIG =====================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

TIMEZONE = pytz.timezone("America/New_York")

MAX_RISK_PER_TRADE = 0.002        # 0.2% per trade (SAFE for $2k+)
MAX_TOTAL_EXPOSURE = 0.15         # 15% of account max exposure
MAX_POSITIONS = 3

ATR_PERIOD = 14
EMA_FAST = 9
EMA_SLOW = 21

MIN_ATR_DOLLARS = 0.15            # avoids dead symbols
COOLDOWN_MINUTES = 15

FLATTEN_TIME = "15:55"            # ALWAYS FLAT BEFORE CLOSE
START_TIME = "09:35"
END_TIME = "15:45"

BAR_LIMIT = 120

# ===================== LOGGING =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===================== API =====================

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca API credentials")

api = REST(API_KEY, API_SECRET, BASE_URL)

cooldowns = {}

# ===================== HELPERS =====================

def now_et():
    return datetime.now(TIMEZONE)

def market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def time_between(start, end):
    t = now_et().time()
    return start <= t <= end

def should_run():
    if not market_open():
        return False
    return time_between(
        datetime.strptime(START_TIME, "%H:%M").time(),
        datetime.strptime(END_TIME, "%H:%M").time()
    )

def nearing_close():
    return now_et().time() >= datetime.strptime(FLATTEN_TIME, "%H:%M").time()

def get_equity():
    return float(api.get_account().equity)

def get_positions():
    try:
        return api.list_positions()
    except:
        return []

def total_exposure():
    exposure = 0.0
    for p in get_positions():
        exposure += abs(float(p.market_value))
    return exposure

def get_bars(symbol, tf):
    bars = api.get_bars(symbol, tf, limit=BAR_LIMIT).df
    if bars.empty:
        return None
    return bars

def indicators(df):
    df = df.copy()
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

def on_cooldown(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

# ===================== ENTRY LOGIC (HF FIRST) =====================

def should_enter(df_1m, df_5m):
    last = df_1m.iloc[-1]
    prev = df_1m.iloc[-2]

    atr = last["atr"]
    if pd.isna(atr) or atr < MIN_ATR_DOLLARS:
        return None

    price_change = last["close"] - prev["close"]
    momentum = abs(price_change) > 0.15 * atr

    vwap_dist = last["close"] - last["vwap"]
    vwap_ok = abs(vwap_dist) > 0.1 * atr

    if not (momentum and vwap_ok):
        return None

    direction = "buy" if price_change > 0 else "sell"

    last_5m = df_5m.iloc[-1]
    trend_up = last_5m["ema_fast"] > last_5m["ema_slow"]
    trend_down = last_5m["ema_fast"] < last_5m["ema_slow"]

    if direction == "buy" and not trend_up:
        return None
    if direction == "sell" and not trend_down:
        return None

    return direction

# ===================== EXECUTION =====================

def place_trade(symbol, side, atr):
    equity = get_equity()
    risk_dollars = equity * MAX_RISK_PER_TRADE
    stop_distance = atr * 0.6

    qty = int(risk_dollars / stop_distance)
    if qty <= 0:
        return

    exposure_after = total_exposure() + (qty * atr)
    if exposure_after > equity * MAX_TOTAL_EXPOSURE:
        logging.info(f"{symbol} | exposure cap hit — skip")
        return

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)
    except Exception as e:
        logging.error(f"{symbol} ORDER FAILED — {e}")

# ===================== FLATTEN =====================

def flatten_all():
    for p in get_positions():
        qty = abs(int(float(p.qty)))
        if qty == 0:
            continue
        side = "sell" if p.side == "long" else "buy"
        logging.info(f"FORCE FLATTEN {p.symbol} qty={qty}")
        try:
            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
        except Exception as e:
            logging.error(f"FLATTEN FAILED {p.symbol} — {e}")

# ===================== MAIN =====================

def run():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed — exit")
        return

    if nearing_close():
        logging.info("Market near close — flattening")
        flatten_all()
        return

    if not should_run():
        logging.info("Outside trade window — exit")
        return

    positions = get_positions()
    if len(positions) >= MAX_POSITIONS:
        logging.info("Max positions reached — exit")
        return

    logging.info(f"Equity: {get_equity():.2f}")

    for symbol in SYMBOLS:
        if on_cooldown(symbol):
            continue
        if any(p.symbol == symbol for p in positions):
            continue

        df_1m = get_bars(symbol, "1Min")
        df_5m = get_bars(symbol, "5Min")

        if df_1m is None or df_5m is None:
            continue

        df_1m = indicators(df_1m)
        df_5m = indicators(df_5m)

        signal = should_enter(df_1m, df_5m)
        if signal:
            place_trade(symbol, signal, df_1m.iloc[-1]["atr"])

    logging.info("BOT END")

# ===================== ENTRY =====================

if __name__ == "__main__":
    run()
