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

# VERY LOOSE SETTINGS TO FORCE MORE TRADES
SMA_ENTRY_DAYS = 3
SMA_EXIT_DAYS = 3
MAX_HOLD_DAYS = 5

RISK_PER_TRADE = 0.01
STOP_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
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
    """
    Very loose entry:
    - enough bars
    - price above yesterday close
    - price above 3-day SMA
    """
    need = max(SMA_ENTRY_DAYS, 3) + 1
    if len(bars) < need:
        return (False, None, None)

    closes = [float(b.c) for b in bars]
    close_today = closes[-1]
    close_yesterday = closes[-2]
    entry_sma = sma(closes[-SMA_ENTRY_DAYS:])

    if close_today < MIN_PRICE:
        return (False, None, None)

    if entry_sma is None:
        return (False, None, None)

    # strength score = daily percent move
    strength = (close_today / close_yesterday) - 1.0

    if close_today > close_yesterday and close_today > entry_sma:
        return (True, close_today, strength)

    return (False, None, None)

def should_exit(symbol: str, bars, entry_date):
    if len(bars) < SMA_EXIT_DAYS + 1:
        return False

    closes = [float(b.c) for b in bars]
    close_today = closes[-1]
    exit_sma = sma(closes[-SMA_EXIT_DAYS:])

    if exit_sma is not None and close_today < exit_sma:
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

    # 1) Exits first
    for sym, pos in list(positions.items()):
        if sym not in SYMBOLS:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(SMA_ENTRY_DAYS, SMA_EXIT_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        entry_date = get_last_entry_date(sym, days_back=30)
        if should_exit(sym, bars, entry_date):
            close_position(sym)

    # refresh after exits
    positions = list_positions_map()
    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        print("No entry slots available.")
        return

    # 2) Entries
    candidates = []
    for sym in SYMBOLS:
        if sym in positions:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(SMA_ENTRY_DAYS, SMA_EXIT_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        ok, ref_price, strength = should_enter_long(sym, bars)
        if not ok:
            continue

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

        print(f"[SIGNAL] {sym} strength={strength:.4%}")
        submit_bracket_buy(sym, qty, ref_price)

    print("Done.")

if __name__ == "__main__":
    main()
