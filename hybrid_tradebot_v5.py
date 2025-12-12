import os
import datetime as dt
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials.")

api = REST(API_KEY, API_SECRET, BASE_URL)

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 12        # hard ceiling (safety)
TARGET_TRADES_PER_DAY = 7      # typical daily volume
GUARANTEED_TRADES_PER_DAY = 1  # forced trades if zero so far today

LOOKBACK_MINUTES = 60
MIN_VOLUME = 50_000
RISK_PER_TRADE = 0.002  # 0.2% account value per trade
TRAILING_STOP = 0.003   # 0.3%

NY = pytz.timezone("America/New_York")

# ============================================================
# UTILITIES
# ============================================================

def now_et():
    return dt.datetime.now(NY)

def load_today_trade_count():
    """Count today's trades using Alpaca closed orders."""
    try:
        today = now_et().date()
        orders = api.list_orders(status="closed", limit=100)
        count = 0
        for o in orders:
            t = o.filled_at
            if t and t.astimezone(NY).date() == today:
                count += 1
        return count
    except Exception:
        return 0

def get_history(symbol):
    try:
        end = now_et()
        start = end - dt.timedelta(minutes=LOOKBACK_MINUTES + 5)
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1m",
            progress=False
        )
        if df.empty:
            return None
        df = df.tail(LOOKBACK_MINUTES)
        return df
    except Exception:
        return None

# ============================================================
# SCORING LOGIC
# ============================================================

def compute_score(df):
    """Lightweight signal: VWAP proximity + micro-trend strength."""
    if df is None or len(df) < 20:
        return None

    prices = df["Close"]
    volume = df["Volume"].iloc[-1]

    if volume < MIN_VOLUME:
        return None

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    vwap_val = vwap.iloc[-1]

    price = prices.iloc[-1]
    vwap_dist = abs(price - vwap_val) / vwap_val

    # Micro trend
    slope = (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]

    # Score: 1 = strong alignment, 0 = trash
    score = max(0, 1 - vwap_dist * 10) + max(0, slope * 8)
    return float(score)

def pick_trade_candidate():
    scored = []
    for sym in SYMBOLS:
        df = get_history(sym)
        score = compute_score(df)
        if score is None:
            continue
        scored.append((sym, score))

    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0]  # (symbol, score)

# ============================================================
# TRADING
# ============================================================

def get_account_value():
    try:
        acc = api.get_account()
        return float(acc.equity)
    except:
        return 50_000  # fallback

def calc_position_size(price):
    equity = get_account_value()
    dollar_risk = equity * RISK_PER_TRADE
    qty = max(1, int(dollar_risk / (price * TRAILING_STOP)))
    return qty

def enter_trade(symbol):
    # price via Alpaca
    quote = api.get_latest_trade(symbol)
    price = float(quote.price)

    qty = calc_position_size(price)

    try:
        o = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"ENTER BUY {symbol} x{qty} @ {price}")
        return symbol, price, qty
    except APIError as e:
        print(f"Failed BUY {symbol}: {e}")
        return None

def trail_stop_monitor(symbol, entry, qty):
    """Immediate, fast trailing stop."""
    try:
        latest = api.get_latest_trade(symbol)
        price = float(latest.price)
    except:
        return False

    if price <= entry * (1 - TRAILING_STOP):
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day"
            )
            print(f"STOP HIT {symbol} @ {price}")
            return True
        except:
            return False

    return False

# ============================================================
# MAIN LOOP (SINGLE RUN PER GITHUB ACTION EXECUTION)
# ============================================================

def main():
    now = now_et()
    print(now.isoformat(), "Run start ET")

    # If market closed → exit
    if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
        print("Market closed.")
        return

    today_trades = load_today_trade_count()

    # If already hit ceiling → exit
    if today_trades >= MAX_TRADES_PER_DAY:
        print("Max trades for day reached.")
        return

    symbol_score = pick_trade_candidate()

    # If no strong candidate:
    if symbol_score is None:
        if today_trades < GUARANTEED_TRADES_PER_DAY:
            print("No candidate – performing guaranteed daily trade.")
            sym = SYMBOLS[np.random.randint(len(SYMBOLS))]
            enter_trade(sym)
        else:
            print("No candidates. No forced trade needed.")
        return

    symbol, score = symbol_score

    # If scoring too weak, but forced trade quota not met
    if score < 0.2:
        if today_trades < GUARANTEED_TRADES_PER_DAY:
            print("Weak signals – performing guaranteed daily trade.")
            sym = SYMBOLS[np.random.randint(len(SYMBOLS))]
            enter_trade(sym)
        else:
            print("Weak signals, skipping.")
        return

    # Otherwise execute the real trade
    result = enter_trade(symbol)
    if result:
        sym, price, qty = result
        trail_stop_monitor(sym, price, qty)

    print("Run complete.")


if __name__ == "__main__":
    main()
