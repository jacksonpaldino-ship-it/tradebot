import os
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca_trade_api import REST, TimeFrame

# ================== CONFIG ==================
NY_TZ = ZoneInfo("America/New_York")

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "TSLA",
    "QQQ", "SPY"
]

# Loosened strategy so it actually trades
BREAKOUT_LOOKBACK_DAYS = 5
SMA_EXIT_DAYS = 10
MAX_HOLD_DAYS = 7

RISK_PER_TRADE = 0.01
STOP_PCT = 0.04
TAKE_PROFIT_PCT = 0.08
MAX_POSITIONS = 5
MAX_NOTIONAL_PER_TRADE = 0.25

RUN_AFTER_CLOSE_ONLY = True
MIN_PRICE = 5.0

# ================== ENV / API ==================
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ================== HELPERS ==================
def ny_now():
    return datetime.now(NY_TZ)

def today_str():
    return ny_now().strftime("%Y-%m-%d")

def get_clock():
    return api.get_clock()

def get_equity():
    acct = api.get_account()
    return float(acct.equity)

def list_positions_map():
    pos = {}
    for p in api.list_positions():
        pos[p.symbol] = p
    return pos

def get_daily_bars(symbol: str, limit: int):
    return list(api.get_bars(symbol, TimeFrame.Day, limit=limit))

def sma(values):
    return sum(values) / len(values) if values else None

def calc_qty(equity: float, price: float):
    if price <= 0:
        return 0

    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = price * STOP_PCT
    if per_share_risk <= 0:
        return 0

    qty_risk = math.floor(risk_amount / per_share_risk)
    max_notional = equity * MAX_NOTIONAL_PER_TRADE
    qty_cap = math.floor(max_notional / price)

    qty = min(qty_risk, qty_cap)
    return max(int(qty), 0)

def get_last_entry_date(symbol: str, days_back: int = 30):
    after = (ny_now() - timedelta(days=days_back)).date().isoformat()
    acts = api.get_activities(activity_types="FILL", direction="desc", after=after)
    for a in acts:
        try:
            if getattr(a, "symbol", None) != symbol:
                continue
            if getattr(a, "side", "").lower() == "buy":
                ts = getattr(a, "transaction_time", None)
                if ts:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY_TZ).date()
        except Exception:
            continue
    return None

def close_position(symbol: str):
    try:
        api.close_position(symbol)
        print(f"[EXIT] Closed {symbol}")
    except Exception as e:
        print(f"[EXIT] Failed closing {symbol}: {e}")

def submit_bracket_buy(symbol: str, qty: int, ref_price: float):
    tp = round(ref_price * (1.0 + TAKE_PROFIT_PCT), 2)
    sl = round(ref_price * (1.0 - STOP_PCT), 2)

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
        print(f"[ENTRY] BUY {symbol} qty={qty} ref={ref_price:.2f} TP={tp:.2f} SL={sl:.2f}")
    except Exception as e:
        print(f"[ENTRY] Failed BUY {symbol}: {e}")

# ================== STRATEGY ==================
def should_enter_long(symbol: str, bars):
    need = BREAKOUT_LOOKBACK_DAYS + 2
    if len(bars) < need:
        return (False, None)

    last = bars[-1]
    close = float(last.c)
    if close < MIN_PRICE:
        return (False, None)

    prev_window = bars[-(BREAKOUT_LOOKBACK_DAYS + 1):-1]
    highest_prev_high = max(float(b.h) for b in prev_window)

    # Easier trigger: close above previous N-day high
    if close > highest_prev_high:
        return (True, close)

    return (False, None)

def should_exit(symbol: str, bars, entry_date):
    if len(bars) < SMA_EXIT_DAYS + 1:
        return False

    closes = [float(b.c) for b in bars]
    last_close = closes[-1]
    ma = sma(closes[-SMA_EXIT_DAYS:])

    if ma is not None and last_close < ma:
        return True

    if entry_date is not None:
        days_held = (ny_now().date() - entry_date).days
        if days_held >= MAX_HOLD_DAYS:
            return True

    return False

# ================== MAIN ==================
def main():
    print(f"\n=== Swing Bot Run ({today_str()}) ===")

    clock = get_clock()
    if RUN_AFTER_CLOSE_ONLY and clock.is_open:
        print("Market is open. Exiting because RUN_AFTER_CLOSE_ONLY=True.")
        return

    equity = get_equity()
    print(f"Equity: ${equity:,.2f}")

    positions = list_positions_map()

    # 1) exits first
    for sym, pos in list(positions.items()):
        if sym not in SYMBOLS:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(SMA_EXIT_DAYS, BREAKOUT_LOOKBACK_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        entry_date = get_last_entry_date(sym, days_back=45)
        if should_exit(sym, bars, entry_date):
            close_position(sym)

    positions = list_positions_map()
    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        print("No entry slots available.")
        return

    candidates = []
    for sym in SYMBOLS:
        if sym in positions:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(SMA_EXIT_DAYS, BREAKOUT_LOOKBACK_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        ok, ref_price = should_enter_long(sym, bars)
        if not ok:
            continue

        prev_window = bars[-(BREAKOUT_LOOKBACK_DAYS + 1):-1]
        breakout_level = max(float(b.h) for b in prev_window)
        strength = ref_price - breakout_level
        candidates.append((strength, sym, ref_price))

    candidates.sort(reverse=True, key=lambda x: x[0])

    if not candidates:
        print("No entry signals today.")
        return

    for strength, sym, ref_price in candidates[:slots]:
        qty = calc_qty(equity, ref_price)
        if qty <= 0:
            print(f"[SKIP] {sym} qty=0")
            continue

        submit_bracket_buy(sym, qty, ref_price)

    print("Done.")

if __name__ == "__main__":
    main()
