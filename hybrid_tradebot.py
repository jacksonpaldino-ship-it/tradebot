import os
import math
import logging
from datetime import datetime, time
import pytz
import pandas as pd
import numpy as np

from alpaca_trade_api import REST

# =========================
# CONFIG
# =========================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

TIMEZONE = pytz.timezone("America/New_York")

ACCOUNT_RISK_PCT = 0.0025
MAX_TRADES_PER_DAY = 3
MAX_SYMBOL_TRADES = 1
MAX_GROSS_EXPOSURE = 0.20

ATR_MULT_STOP = 1.0
ATR_MULT_TP = 1.5
ADX_MIN = 20

FORCE_FLATTEN_TIME = time(15, 55)

BAR_TIMEFRAME = "5Min"
LOOKBACK = 100

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# =========================
# API
# =========================
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# =========================
# UTILS
# =========================
def ny_now():
    return datetime.now(TIMEZONE)

def market_near_close():
    return ny_now().time() >= FORCE_FLATTEN_TIME

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# =========================
# INDICATORS
# =========================
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()

    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["adx"] = dx.rolling(14).mean()

    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    return df

# =========================
# DATA
# =========================
def get_bars(symbol):
    bars = api.get_bars(symbol, BAR_TIMEFRAME, limit=LOOKBACK).df
    if bars.empty:
        return None
    return indicators(bars)

# =========================
# RISK
# =========================
def calc_qty(price, atr, equity):
    risk_dollars = equity * ACCOUNT_RISK_PCT
    stop_dist = max(atr * ATR_MULT_STOP, price * 0.002)
    qty = math.floor(risk_dollars / stop_dist)
    return max(qty, 0)

# =========================
# POSITIONS
# =========================
def flatten_all():
    positions = api.list_positions()
    for p in positions:
        qty = abs(int(float(p.qty)))
        if qty == 0:
            continue
        side = "sell" if p.side == "long" else "buy"
        logging.info(f"FORCE CLOSE {p.symbol} qty={qty}")
        try:
            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
        except Exception as e:
            logging.error(f"FORCE CLOSE FAILED {p.symbol} — {e}")

# =========================
# PERFORMANCE LOGGING
# =========================
def log_performance():
    acct = api.get_account()
    equity = safe_float(acct.equity)
    last = safe_float(acct.last_equity)
    pnl = equity - last
    pct = (pnl / last * 100) if last else 0
    logging.info(f"Equity: {equity:.2f} | Daily PnL: {pct:.2f}%")

# =========================
# MAIN
# =========================
def run():
    logging.info("BOT START")

    if market_near_close():
        logging.info("Market near close — flattening")
        flatten_all()
        return

    acct = api.get_account()
    equity = safe_float(acct.equity)
    log_performance()

    positions = {p.symbol: p for p in api.list_positions()}
    trades_today = len(api.list_orders(status="all", after=ny_now().date().isoformat()))

    gross_exposure = sum(abs(float(p.market_value)) for p in positions.values())
    if gross_exposure / equity > MAX_GROSS_EXPOSURE:
        logging.info("Exposure cap hit — skip")
        return

    for symbol in SYMBOLS:
        if trades_today >= MAX_TRADES_PER_DAY:
            break
        if symbol in positions:
            continue

        df = get_bars(symbol)
        if df is None or len(df) < 30:
            continue

        last = df.iloc[-1]

        if last["adx"] < ADX_MIN:
            continue

        price = safe_float(last["close"])
        atr = safe_float(last["atr"])
        if atr <= 0:
            continue

        qty = calc_qty(price, atr, equity)
        if qty <= 0:
            continue

        side = None
        if last["ema9"] > last["ema21"] and price > last["vwap"]:
            side = "buy"
        elif last["ema9"] < last["ema21"] and price < last["vwap"]:
            side = "sell"

        if not side:
            continue

        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
            trades_today += 1
            logging.info(f"{symbol} | {side.upper()} | qty={qty}")
        except Exception as e:
            logging.error(f"{symbol} ORDER FAILED — {e}")

    logging.info("BOT END")

if __name__ == "__main__":
    run()
