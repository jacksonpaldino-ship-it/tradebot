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

MAX_POSITIONS = 3
MAX_HOLD_DAYS = 5

RISK_PER_TRADE = 0.01
STOP_PCT = 0.03
TAKE_PROFIT_PCT = 0.06
MAX_NOTIONAL_PER_TRADE = 0.30

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
    return float(api.get_account().equity)

def list_positions_map():
    out = {}
    for p in api.list_positions():
        out[p.symbol] = p
    return out

def get_daily_bars(symbol: str, limit: int):
    return list(api.get_bars(symbol, TimeFrame.Day, limit=limit))

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
def score_symbol(symbol: str, bars):
    """
    Validation mode:
    score = today's % return from yesterday close
    """
    if len(bars) < 2:
        return None

    close_today = float(bars[-1].c)
    close_yesterday = float(bars[-2].c)

    if close_today < MIN_PRICE or close_yesterday <= 0:
        return None

    return (close_today / close_yesterday) - 1.0, close_today

def should_exit(symbol: str, entry_date, top_symbols):
    """
    Exit if:
    - held too long
    - symbol is no longer in today's top ranked list
    """
    if entry_date is not None:
        days_held = (ny_now().date() - entry_date).days
        if days_held >= MAX_HOLD_DAYS:
            return True

    if symbol not in top_symbols:
        return True

    return False

# ================== MAIN ==================
def main():
    print(f"\n=== Validation Swing Bot Run ({today_str()}) ===")

    clock = get_clock()
    if RUN_AFTER_CLOSE_ONLY and clock.is_open:
        print("Market is open. Exiting because RUN_AFTER_CLOSE_ONLY=True.")
        return

    equity = get_equity()
    print(f"Equity: ${equity:,.2f}")

    positions = list_positions_map()

    # Rank all symbols by 1-day return
    ranked = []
    for sym in SYMBOLS:
        try:
            bars = get_daily_bars(sym, limit=3)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        scored = score_symbol(sym, bars)
        if scored is None:
            continue

        strength, ref_price = scored
        ranked.append((strength, sym, ref_price))

    ranked.sort(reverse=True, key=lambda x: x[0])

    if not ranked:
        print("No ranked symbols. Data issue or market timing issue.")
        return

    target_symbols = [sym for _, sym, _ in ranked[:MAX_POSITIONS]]
    print(f"Top symbols today: {target_symbols}")

    # Exit anything no longer in top ranks or held too long
    for sym in list(positions.keys()):
        if sym not in SYMBOLS:
            continue

        entry_date = get_last_entry_date(sym, days_back=30)
        if should_exit(sym, entry_date, target_symbols):
            close_position(sym)

    # Refresh after exits
    positions = list_positions_map()
    current_symbols = set(positions.keys())

    # Fill available slots with top-ranked names
    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        print("No entry slots available.")
        return

    for strength, sym, ref_price in ranked:
        if slots <= 0:
            break
        if sym in current_symbols:
            continue

        qty = calc_qty(equity, ref_price)
        if qty <= 0:
            print(f"[SKIP] {sym} qty=0")
            continue

        print(f"[SIGNAL] {sym} strength={strength:.4%}")
        submit_bracket_buy(sym, qty, ref_price)
        slots -= 1

    print("Done.")

if __name__ == "__main__":
    main()
