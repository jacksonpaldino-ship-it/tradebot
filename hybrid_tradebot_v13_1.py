#!/usr/bin/env python3

import os
import math
from datetime import datetime, date
import pytz

import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

LOOKBACK_MIN = 5
MIN_MOVE_PCT = 0.00035        # tuned
MIN_VOLUME = 500

TP_PCT = 0.0015               # 0.15%
SL_PCT = 0.0012               # 0.12%

RISK_PER_TRADE = 0.005        # 0.5% risk
MAX_POSITION_PCT = 0.15       # max 15% equity
MAX_DAILY_LOSS_PCT = 0.02     # 2% daily loss lock

TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ================= UTILS =================
def log(msg):
    print(f"{datetime.now(TZ)} {msg}")

def market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def within_trade_window():
    now = datetime.now(TZ).time()
    return (
        now >= datetime.strptime("09:35", "%H:%M").time() and
        now <= datetime.strptime("15:45", "%H:%M").time()
    )

def equity():
    return float(api.get_account().equity)

def buying_power():
    return float(api.get_account().buying_power)

# ================= DAILY LOSS LOCK =================
def daily_loss_exceeded():
    today = date.today().isoformat()
    try:
        activities = api.get_activities(
            activity_types="FILL",
            after=today
        )
    except Exception as e:
        log(f"Activity fetch error: {e}")
        return False

    realized_pnl = 0.0
    for act in activities:
        if act.side == "sell":
            realized_pnl += float(act.realized_pl)

    loss_limit = -equity() * MAX_DAILY_LOSS_PCT

    if realized_pnl <= loss_limit:
        log(f"DAILY LOSS LOCK HIT: PnL={realized_pnl:.2f}")
        return True

    return False

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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    return df.dropna().tail(LOOKBACK_MIN + 2)

# ================= SIGNAL =================
def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK_MIN:
        return None

    recent = df.tail(LOOKBACK_MIN)

    start = recent["close"].iloc[0]
    end = recent["close"].iloc[-1]
    move = (end - start) / start

    if move < MIN_MOVE_PCT:
        return None

    avg_vol = recent["volume"].mean()
    if avg_vol < MIN_VOLUME:
        return None

    green_candles = (recent["close"] > recent["open"]).sum()
    if green_candles < 3:
        return None

    return {
        "symbol": symbol,
        "price": float(end),
        "score": float(move)
    }

# ================= ORDER =================
def calc_qty(price):
    eq = equity()
    bp = buying_power()

    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * SL_PCT
    qty_risk = math.floor(risk_dollars / per_share_risk)

    max_position_value = eq * MAX_POSITION_PCT
    qty_cap = math.floor(max_position_value / price)

    qty = min(qty_risk, qty_cap)
    return max(1, qty)

def round_price(p):
    return round(p, 2)

def submit_trade(symbol, price):
    qty = calc_qty(price)
    if qty <= 0:
        return False

    tp = round_price(price * (1 + TP_PCT))
    sl = round_price(price * (1 - SL_PCT))

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)}
        )
        log(f"ENTER {symbol} qty={qty} tp={tp} sl={sl}")
        return True
    except Exception as e:
        log(f"ORDER ERROR {symbol}: {e}")
        return False

# ================= MAIN =================
def main():
    log("Run start")

    if not market_open():
        log("Market closed")
        return

    if not within_trade_window():
        log("Outside trade window")
        return

    if daily_loss_exceeded():
        log("Trading locked for the day")
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
        log("No entries")
        return

    signals.sort(key=lambda x: x["score"], reverse=True)

    for sig in signals:
        if submit_trade(sig["symbol"], sig["price"]):
            break

if __name__ == "__main__":
    main()
