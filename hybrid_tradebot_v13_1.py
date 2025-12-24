#!/usr/bin/env python3

import os
import math
from datetime import datetime, timedelta
import pytz
import time

import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]  # ETFs to trade
LOOKBACK_MIN = 5                        # Aggressive lookback
MIN_MOVE_PCT = 0.0004                    # Minimum momentum for entry
RISK_PER_TRADE = 0.005                   # 0.5% equity per trade
TP_PCT = 0.0015                          # 0.15% profit target
SL_PCT = 0.0012                          # 0.12% stop loss
MIN_VOLUME = 500                          # Minimum last candle volume
MAX_POSITION_PCT = 0.15                  # 15% max equity per trade
MAX_DAILY_LOSS = 200                      # Max daily loss in dollars
MAX_TRADES_PER_DAY = 10                   # Max trades per day
SYMBOL_COOLDOWN_MIN = 10                  # Min minutes between trades on same symbol

TRADING_START = "09:30"
TRADING_END = "15:30"                     # Only trade within market hours

TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ================= STATE =================
last_trade_time = {}  # Tracks cooldown per symbol
daily_trades_count = 0
daily_loss_total = 0.0

# ================= UTILS =================
def log(msg):
    print(f"{datetime.now(TZ)} {msg}")

def market_open():
    try:
        clock = api.get_clock()
        if not clock.is_open:
            log(f"Market closed — next open {clock.next_open}")
        return clock.is_open
    except Exception as e:
        log(f"Clock error: {e}")
        return False

def equity():
    try:
        return float(api.get_account().equity)
    except:
        return 0.0

def buying_power():
    try:
        return float(api.get_account().buying_power)
    except:
        return 0.0

def daily_pnl():
    global daily_loss_total
    try:
        activities = api.get_activities(activity_types=["FILL"])
        pnl = 0.0
        today = datetime.now(TZ).date()
        for a in activities:
            act_time = a.transaction_time
            if hasattr(act_time, "date"):
                act_date = act_time.date()
            else:
                act_date = datetime.strptime(str(act_time), "%Y-%m-%dT%H:%M:%S.%fZ").date()
            if act_date == today:
                pnl += float(getattr(a, "amount", 0))
        daily_loss_total = -pnl if pnl < 0 else 0
        return daily_loss_total
    except Exception as e:
        log(f"Failed to calculate daily PnL: {e}")
        return daily_loss_total

def is_within_trading_hours():
    now = datetime.now(TZ).time()
    start = datetime.strptime(TRADING_START, "%H:%M").time()
    end = datetime.strptime(TRADING_END, "%H:%M").time()
    return start <= now <= end

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True
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
    return df.dropna().tail(LOOKBACK_MIN + 1)

# ================= SIGNAL =================
def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK_MIN:
        return None
    recent = df.tail(LOOKBACK_MIN)
    start = float(recent["close"].iloc[0])
    end = float(recent["close"].iloc[-1])
    move = (end - start) / start
    volume = int(recent["volume"].iloc[-1])
    if abs(move) < MIN_MOVE_PCT or volume < MIN_VOLUME:
        return None
    return {
        "symbol": symbol,
        "price": end,
        "score": abs(move),
        "direction": "long" if move > 0 else "short"
    }

# ================= ORDER =================
def calc_qty(price):
    eq = equity()
    bp = buying_power()
    if eq <= 0 or bp <= 0:
        return 0
    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * SL_PCT
    qty_risk = math.floor(risk_dollars / per_share_risk)
    max_position_value = eq * MAX_POSITION_PCT
    qty_cap = math.floor(max_position_value / price)
    qty = min(qty_risk, qty_cap)
    return max(1, qty)

def round_price(p):
    return round(p, 2)

def submit_trade(symbol, price, direction):
    global last_trade_time, daily_trades_count

    # Check cooldown
    now = datetime.now(TZ)
    if symbol in last_trade_time and (now - last_trade_time[symbol]).total_seconds() < SYMBOL_COOLDOWN_MIN * 60:
        log(f"Cooldown active for {symbol}")
        return False

    # Check daily trade limit
    if daily_trades_count >= MAX_TRADES_PER_DAY:
        log("Reached max daily trades")
        return False

    # Check daily loss lock
    if daily_pnl() <= -MAX_DAILY_LOSS:
        log(f"Daily loss limit reached: {daily_loss_total}")
        return False

    qty = calc_qty(price)
    if qty <= 0:
        log(f"Skip {symbol}: qty=0")
        return False

    tp_price = round_price(price * (1 + TP_PCT)) if direction == "long" else round_price(price * (1 - TP_PCT))
    sl_price = round_price(price * (1 - SL_PCT)) if direction == "long" else round_price(price * (1 + SL_PCT))

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy" if direction == "long" else "sell_short",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp_price)},
            stop_loss={"stop_price": str(sl_price)}
        )
        log(f"ENTER {symbol} qty={qty} dir={direction} tp={tp_price} sl={sl_price}")
        last_trade_time[symbol] = now
        daily_trades_count += 1
        return True
    except Exception as e:
        log(f"ORDER ERROR {symbol}: {e}")
        return False

# ================= MAIN =================
def main():
    log("Cron run start")
    if not market_open() or not is_within_trading_hours():
        log("Market closed or outside trading window")
        return

    signals = []
    for sym in SYMBOLS:
        try:
            sig = get_signal(sym)
            if sig:
                signals.append(sig)
        except Exception as e:
            log(f"{sym} error: {e}")

    if not signals:
        log("No signals")
        return

    # Sort by strongest momentum first
    signals.sort(key=lambda x: x["score"], reverse=True)

    for sig in signals:
        if submit_trade(sig["symbol"], sig["price"], sig["direction"]):
            break  # Only one trade per cron run

if __name__ == "__main__":
    main()
