#!/usr/bin/env python3

import os
import math
from datetime import datetime, time, date
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================

SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY"
]

LOOKBACK_MIN = 3               # shorter momentum window
MIN_MOVE_PCT = 0.00035         # 0.035%
TP_PCT = 0.0015                # 0.15%
SL_PCT = 0.0012                # 0.12%

RISK_PER_TRADE = 0.005         # 0.5% equity risk
MAX_POSITION_PCT = 0.15        # max 15% equity per trade
MAX_TRADES_PER_RUN = 2         # increase frequency safely

DAILY_LOSS_LIMIT_PCT = 0.02    # 2% max daily loss

TZ = pytz.timezone("US/Eastern")

# Trade windows (ET)
TRADE_WINDOWS = [
    (time(9, 35), time(15, 45)),
]

# ================= ALPACA =================

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
)

# ================= UTILS =================

def log(msg):
    print(f"{datetime.now(TZ)} {msg}")

def market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def equity():
    return float(api.get_account().equity)

def buying_power():
    return float(api.get_account().buying_power)

def current_time_allowed():
    now = datetime.now(TZ).time()
    return any(start <= now <= end for start, end in TRADE_WINDOWS)

# ================= DAILY LOSS LOCK =================

def daily_pnl():
    today = datetime.now(TZ).date()
    try:
        activities = api.get_activities(activity_types="FILL")
    except Exception as e:
        log(f"Activity fetch error: {e}")
        return 0.0

    pnl = 0.0
    for a in activities:
        act_time = a.transaction_time
        if not isinstance(act_time, date):
            act_time = pd.to_datetime(act_time).date()

        if act_time != today:
            continue

        realized = getattr(a, "realized_pl", None)
        if realized is not None:
            pnl += float(realized)

    return pnl

# ================= DATA =================

def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    return df.tail(LOOKBACK_MIN + 2)

# ================= SIGNAL =================

def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK_MIN:
        return None

    recent = df.tail(LOOKBACK_MIN)

    start = recent["close"].iloc[0]
    end = recent["close"].iloc[-1]
    move = (end - start) / start

    avg_vol = recent["volume"].rolling(3).mean().iloc[-1]
    if avg_vol < 300:
        return None

    if abs(move) < MIN_MOVE_PCT:
        return None

    direction = "long" if move > 0 else "short"

    return {
        "symbol": symbol,
        "price": float(end),
        "score": abs(move),
        "side": direction
    }

# ================= ORDER =================

def calc_qty(price):
    eq = equity()
    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * SL_PCT
    qty_risk = math.floor(risk_dollars / per_share_risk)

    max_position_value = eq * MAX_POSITION_PCT
    qty_cap = math.floor(max_position_value / price)

    qty = min(qty_risk, qty_cap)
    return max(1, qty)

def round_price(p):
    return round(p, 2)

def submit_trade(sig):
    symbol = sig["symbol"]
    price = sig["price"]
    side = sig["side"]

    qty = calc_qty(price)
    if qty <= 0:
        return False

    if side == "long":
        tp = round_price(price * (1 + TP_PCT))
        sl = round_price(price * (1 - SL_PCT))
        order_side = "buy"
    else:
        tp = round_price(price * (1 - TP_PCT))
        sl = round_price(price * (1 + SL_PCT))
        order_side = "sell"

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=order_side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)}
        )
        log(f"{side.upper()} {symbol} qty={qty} tp={tp} sl={sl}")
        return True
    except Exception as e:
        log(f"ORDER ERROR {symbol}: {e}")
        return False

# ================= MAIN =================

def main():
    log("Cron run start")

    if not market_open():
        log("Market closed")
        return

    if not current_time_allowed():
        log("Outside trade window")
        return

    pnl = daily_pnl()
    if pnl < -equity() * DAILY_LOSS_LIMIT_PCT:
        log("Daily loss limit hit")
        return

    signals = []

    for sym in SYMBOLS:
        try:
            sig = get_signal(sym)
            if sig:
                signals.append(sig)
        except Exception as e:
            log(f"{sym} error {e}")

    if not signals:
        log("No signals")
        return

    signals.sort(key=lambda x: x["score"], reverse=True)

    trades = 0
    for sig in signals:
        if trades >= MAX_TRADES_PER_RUN:
            break
        if submit_trade(sig):
            trades += 1

if __name__ == "__main__":
    main()
