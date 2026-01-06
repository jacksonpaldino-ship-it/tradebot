import os
import logging
from datetime import datetime, time as dtime
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

# Risk & sizing (SAFE for $100k, scales to $2k automatically)
RISK_PER_TRADE = 0.002        # 0.2% per trade
MAX_POSITION_PCT = 0.10       # max 10% equity per symbol
MAX_DAILY_LOSS = -0.01        # -1% daily kill switch

# Indicators (loosened so it actually trades)
EMA_FAST = 8
EMA_SLOW = 20
ATR_PERIOD = 14

# ATR based exits (valid for Alpaca constraints)
STOP_ATR = 0.6
TP_ATR = 1.0

# Trading windows (NY time)
MORNING_START = dtime(9, 35)
MORNING_END   = dtime(11, 30)
AFTERNOON_START = dtime(13, 30)
AFTERNOON_END   = dtime(15, 45)
FORCE_FLATTEN_TIME = dtime(15, 55)

# =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ENV VARS (correct names)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL)
eastern = pytz.timezone("America/New_York")

# =========================================

def now_et():
    return datetime.now(eastern)

def market_open():
    return api.get_clock().is_open

def in_trade_window():
    t = now_et().time()
    return (
        MORNING_START <= t <= MORNING_END or
        AFTERNOON_START <= t <= AFTERNOON_END
    )

def near_close():
    return now_et().time() >= FORCE_FLATTEN_TIME

def get_equity():
    return float(api.get_account().equity)

def daily_pnl():
    acct = api.get_account()
    return float(acct.equity) / float(acct.last_equity) - 1

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=120).df
    df = df.reset_index()
    return df

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

def get_position_qty(symbol):
    try:
        pos = api.get_position(symbol)
        return int(float(pos.qty))
    except:
        return 0

def flatten_all():
    positions = api.list_positions()
    for p in positions:
        qty = int(float(p.qty))
        if qty == 0:
            continue

        side = "sell" if qty > 0 else "buy"
        try:
            logging.info(f"FORCE CLOSE {p.symbol} qty={qty}")
            api.submit_order(
                symbol=p.symbol,
                qty=abs(qty),
                side=side,
                type="market",
                time_in_force="day"
            )
        except Exception as e:
            logging.warning(f"Flatten skip {p.symbol}: {e}")

def position_size(symbol, atr, price, equity):
    risk_dollars = equity * RISK_PER_TRADE
    raw_qty = int(risk_dollars / (atr * STOP_ATR))

    max_qty = int((equity * MAX_POSITION_PCT) / price)
    return max(0, min(raw_qty, max_qty))

def submit_bracket(symbol, side, qty, price, atr):
    if qty <= 0:
        return

    if side == "buy":
        stop = round(price - atr * STOP_ATR, 2)
        tp   = round(price + atr * TP_ATR, 2)
        if tp <= price + 0.01:
            tp = round(price + 0.02, 2)
    else:
        stop = round(price + atr * STOP_ATR, 2)
        tp   = round(price - atr * TP_ATR, 2)
        if tp >= price - 0.01:
            tp = round(price - 0.02, 2)

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

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

# =========================================

def run():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed")
        return

    if near_close():
        logging.info("Market near close — flattening")
        flatten_all()
        return

    equity = get_equity()
    pnl = daily_pnl()
    logging.info(f"Equity: {equity:.2f} | Daily PnL: {pnl:.2%}")

    if pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT — FLATTENING")
        flatten_all()
        return

    if not in_trade_window():
        logging.info("Outside trade window")
        return

    for symbol in SYMBOLS:
        try:
            if get_position_qty(symbol) != 0:
                continue

            df = indicators(get_bars(symbol))
            last = df.iloc[-1]
            prev = df.iloc[-2]

            atr = last["atr"]
            if pd.isna(atr) or atr == 0:
                continue

            price = float(api.get_latest_trade(symbol).price)

            # Loosened entry logic (momentum + trend)
            bullish = last["ema_fast"] > last["ema_slow"] and price > prev["close"]
            bearish = last["ema_fast"] < last["ema_slow"] and price < prev["close"]

            qty = position_size(symbol, atr, price, equity)

            if bullish:
                submit_bracket(symbol, "buy", qty, price, atr)
            elif bearish:
                submit_bracket(symbol, "sell", qty, price, atr)

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

    logging.info("BOT END")

# =========================================

if __name__ == "__main__":
    run()
