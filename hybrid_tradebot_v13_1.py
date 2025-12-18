#!/usr/bin/env python3

import os
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

LOOKBACK = 3                 # EXTREMELY SHORT
RANGE_TRIGGER = 0.0006       # 0.06% 3-min range
BREAKOUT_TRIGGER = 0.0002    # 0.02% micro breakout

TP_PCT = 0.0015              # 0.15%
SL_PCT = 0.0012              # 0.12%
RISK_PER_TRADE = 0.02        # 2%

TZ = pytz.timezone("US/Eastern")

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
    return api.get_clock().is_open

def has_position():
    return len(api.list_positions()) > 0

def equity():
    return float(api.get_account().equity)

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    return df.dropna().tail(LOOKBACK + 1)

# ================= SIGNAL =================
def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK:
        return None

    recent = df.tail(LOOKBACK)

    high = recent["high"].max()
    low = recent["low"].min()
    last = recent["close"].iloc[-1]

    range_pct = (high - low) / last
    breakout_up = last > high * (1 - BREAKOUT_TRIGGER)

    # PURE ACTIVITY LOGIC
    if range_pct > RANGE_TRIGGER or breakout_up:
        score = range_pct
        return {
            "symbol": symbol,
            "price": float(last),
            "score": float(score),
        }

    return None

# ================= ORDER =================
def calc_qty(price):
    risk_dollars = equity() * RISK_PER_TRADE
    per_share = price * SL_PCT
    qty = int(risk_dollars / per_share)
    return max(1, qty)

def submit_trade(symbol, price):
    qty = calc_qty(price)

    tp = round(price * (1 + TP_PCT), 2)
    sl = round(price * (1 - SL_PCT), 2)

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)},
        )
        log(f"ENTER {symbol} qty={qty} tp={tp} sl={sl}")
    except Exception as e:
        log(f"ORDER ERROR {symbol}: {e}")

# ================= MAIN =================
def main():
    log("Run start")

    if not market_open():
        log("Market closed")
        return

    if has_position():
        log("Position open")
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

    # TAKE MOST ACTIVE SYMBOL
    signals.sort(key=lambda x: x["score"], reverse=True)
    best = signals[0]

    submit_trade(best["symbol"], best["price"])

if __name__ == "__main__":
    main()
