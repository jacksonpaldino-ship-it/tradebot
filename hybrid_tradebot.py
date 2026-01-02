import os
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca_trade_api.rest import REST, APIError

# ================= CONFIG =================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

MAX_RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = -0.02
COOLDOWN_MINUTES = 10

EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14

ATR_STOP_MULT = 0.5
ATR_TP_MULT = 0.75
# =========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

api = REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"]
)

eastern = pytz.timezone("America/New_York")
cooldowns = {}

# ================= HELPERS =================
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
    return df if not df.empty else None

def indicators(df):
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

def position_exists(symbol):
    try:
        api.get_position(symbol)
        return True
    except APIError:
        return False

def cooldown_active(symbol):
    return symbol in cooldowns and now_et() < cooldowns[symbol]

# ================= ORDER LOGIC =================
def place_trade(symbol, side, atr, equity):
    try:
        risk = equity * MAX_RISK_PER_TRADE
        stop_dist = atr * ATR_STOP_MULT
        qty = max(int(risk / stop_dist), 1)

        logging.info(f"{symbol} | {side.upper()} | qty={qty}")

        # 1️⃣ MARKET ENTRY
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

        # 2️⃣ WAIT FOR FILL
        filled = None
        for _ in range(10):
            filled = api.get_order(order.id)
            if filled.filled_avg_price:
                break

        if not filled or not filled.filled_avg_price:
            raise Exception("Order not filled")

        base = float(filled.filled_avg_price)

        # 3️⃣ CALCULATE VALID STOPS
        if side == "buy":
            stop = round(base - stop_dist, 2)
            tp = round(base + atr * ATR_TP_MULT, 2)
        else:
            stop = round(base + stop_dist, 2)
            tp = round(base - atr * ATR_TP_MULT, 2)

        # 4️⃣ EXIT ORDERS (SEPARATE)
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell" if side == "buy" else "buy",
            type="stop",
            stop_price=stop,
            time_in_force="day"
        )

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell" if side == "buy" else "buy",
            type="limit",
            limit_price=tp,
            time_in_force="day"
        )

        cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)
        logging.info(f"{symbol} FILLED @ {base} | SL={stop} TP={tp}")

    except Exception as e:
        logging.error(f"{symbol} ORDER FAILED — {e}")

# ================= MAIN =================
def run():
    if not market_open() or not in_trade_window():
        return

    equity = get_equity()
    pnl = get_daily_pnl()
    logging.info(f"Equity: {equity:.2f} | Daily PnL: {pnl:.2%}")

    if pnl <= MAX_DAILY_LOSS:
        logging.warning("DAILY LOSS LIMIT HIT")
        return

    for symbol in SYMBOLS:
        if position_exists(symbol) or cooldown_active(symbol):
            continue

        df = get_bars(symbol)
        if df is None or len(df) < 30:
            continue

        df = indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]

        if pd.isna(last["atr"]):
            continue

        if last["ema_fast"] > last["ema_slow"] and last["close"] > last["vwap"]:
            place_trade(symbol, "buy", last["atr"], equity)

        elif last["ema_fast"] < last["ema_slow"] and last["close"] < last["vwap"]:
            place_trade(symbol, "sell", last["atr"], equity)

if __name__ == "__main__":
    logging.info("BOT START")
    run()
    logging.info("BOT END")
