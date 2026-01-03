import os
import logging
from datetime import datetime, time
import pytz
import math
import pandas as pd
from alpaca_trade_api import REST

# ===================== MODE =====================
# CHANGE THIS ONLY WHEN GOING LIVE
ACCOUNT_MODE = "LARGE"   # "LARGE" (~100k) or "SMALL" (~2k)

# ===================== SYMBOLS ==================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

# ===================== RISK CONFIG ==============
if ACCOUNT_MODE == "LARGE":
    RISK_PER_TRADE = 0.0015        # 0.15%
    MAX_DOLLARS_PER_TRADE = 15_000
    MAX_TOTAL_EXPOSURE = 0.30      # 30% of equity
else:  # SMALL ACCOUNT
    RISK_PER_TRADE = 0.003         # 0.30%
    MAX_DOLLARS_PER_TRADE = 600
    MAX_TOTAL_EXPOSURE = 0.40      # 40% of equity

ATR_STOP_MULT = 1.0
ATR_TP_MULT = 1.6
MIN_PRICE_INCREMENT = 0.01

MARKET_CLOSE_CUTOFF = time(15, 50)  # 3:50 PM ET
TIMEZONE = pytz.timezone("America/New_York")

# ===================== LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===================== ALPACA ===================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not all([API_KEY, API_SECRET, BASE_URL]):
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ===================== HELPERS ==================
def now_et():
    return datetime.now(TIMEZONE)

def market_closing():
    return now_et().time() >= MARKET_CLOSE_CUTOFF

def round_price(price, up=True):
    if up:
        return round(math.ceil(price / MIN_PRICE_INCREMENT) * MIN_PRICE_INCREMENT, 2)
    return round(math.floor(price / MIN_PRICE_INCREMENT) * MIN_PRICE_INCREMENT, 2)

def equity():
    return float(api.get_account().equity)

def open_positions():
    return api.list_positions()

def total_exposure():
    return sum(abs(float(p.market_value)) for p in open_positions())

def close_all_positions():
    for p in open_positions():
        side = "sell" if float(p.qty) > 0 else "buy"
        logging.info(f"FORCE CLOSE {p.symbol} qty={p.qty}")
        api.submit_order(
            symbol=p.symbol,
            qty=abs(int(float(p.qty))),
            side=side,
            type="market",
            time_in_force="day"
        )

def get_bars(symbol):
    df = api.get_bars(symbol, "5Min", limit=50).df
    return df if not df.empty else None

def atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ===================== CORE =====================
def run():
    logging.info("BOT START")

    if market_closing():
        logging.info("Market near close — flattening")
        close_all_positions()
        logging.info("BOT END")
        return

    acct_equity = equity()
    exposure_cap = acct_equity * MAX_TOTAL_EXPOSURE

    logging.info(f"Equity: {acct_equity:.2f}")

    held_symbols = {p.symbol for p in open_positions()}

    for symbol in SYMBOLS:
        if symbol in held_symbols:
            continue

        if total_exposure() >= exposure_cap:
            logging.info(f"{symbol} | exposure cap hit — skip")
            continue

        df = get_bars(symbol)
        if df is None or len(df) < 20:
            continue

        current_atr = atr(df)
        if current_atr is None or current_atr <= 0:
            continue

        last = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]

        # SIMPLE + STABLE EDGE
        if last <= prev:
            continue

        stop_dist = current_atr * ATR_STOP_MULT

        risk_dollars = acct_equity * RISK_PER_TRADE
        raw_qty = int(risk_dollars / stop_dist)

        dollar_cap_qty = int(MAX_DOLLARS_PER_TRADE / last)
        qty = min(raw_qty, dollar_cap_qty)

        if qty < 1:
            continue

        entry = round_price(last, up=True)
        stop = round_price(entry - stop_dist, up=False)
        tp = round_price(entry + current_atr * ATR_TP_MULT, up=True)

        if stop >= entry or tp <= entry:
            continue

        logging.info(f"{symbol} | BUY | qty={qty}")

        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day",
                order_class="bracket",
                stop_loss={"stop_price": stop},
                take_profit={"limit_price": tp}
            )
        except Exception as e:
            logging.error(f"{symbol} ORDER FAILED — {e}")

    logging.info("BOT END")

# ===================== ENTRY ====================
if __name__ == "__main__":
    run()
