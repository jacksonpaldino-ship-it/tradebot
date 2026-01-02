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

ATR_ENTRY_MULT = 0.25
ATR_STOP_MULT = 0.5
ATR_TP_MULT = 0.75
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= API =================
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
    return df.copy() if not df.empty else None

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

def place_trade(symbol, side, price, atr, equity):
    try:
        risk = equity * MAX_RISK_PER_TRADE
        stop_dist = atr * ATR_STOP_MULT
        qty = max(int(risk / stop_dist), 1)

        if side == "buy":
            stop = round(price - stop_dist, 2)
            tp = round(price + atr * ATR_TP_MULT, 2)
        else:
            stop = round(price + stop_dist, 2)
            tp = round(price - atr * ATR_TP_MULT, 2)

        if side == "buy":
            stop = min(stop, price - 0.01)
            tp = max(tp, price + 0.01)
        else:
            stop = max(stop, price + 0.01)
            tp = min(tp, price - 0.01)

        logging.info(f"{symbol} | {side.upper()} | qty={qty} entry={price} stop={stop} tp={tp}")

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            stop_loss={"stop_price": stop},
            take_profit={"limit_price": tp},
        )

        cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

    except APIError as e:
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
        try:
            if position_exists(symbol) or cooldown_active(symbol):
                continue

            df = get_bars(symbol)
            if df is None or len(df) < 30:
                continue

            df = indicators(df)
            last, prev = df.iloc[-1], df.iloc[-2]

            if pd.isna(last["atr"]) or last["atr"] <= 0:
                continue

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

if __name__ == "__main__":
    logging.info("BOT START")
    run()
    logging.info("BOT END")
