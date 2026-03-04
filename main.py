import os
import math
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

from alpaca_trade_api import REST, TimeFrame


# ================== CONFIG ==================
NY_TZ = ZoneInfo("America/New_York")

# Universe (add/remove freely)
SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "TSLA",
    "QQQ", "SPY"
]

# Strategy
BREAKOUT_LOOKBACK_DAYS = 20     # "Donchian" breakout lookback
SMA_EXIT_DAYS = 20              # exit if close < SMA20
MAX_HOLD_DAYS = 10              # hard time stop
MARKET_FILTER_SYMBOL = "SPY"    # only trade long if SPY > SMA200
MARKET_FILTER_SMA_DAYS = 200

# Risk / sizing
RISK_PER_TRADE = 0.01           # 1% equity risk per trade
STOP_PCT = 0.05                 # 5% stop
TAKE_PROFIT_PCT = 0.10          # 10% take profit
MAX_POSITIONS = 5               # max concurrent positions
MAX_NOTIONAL_PER_TRADE = 0.25   # cap each trade to 25% of equity

# Operational
RUN_AFTER_CLOSE_ONLY = True     # prevents intraday PDT-like behavior
MIN_PRICE = 5.0                 # skip penny-ish
MIN_AVG_VOLUME = 1_000_000      # liquidity filter (daily avg volume)


# ================== ENV / API ==================
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment (GitHub secrets).")

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
        try:
            pos[p.symbol] = p
        except Exception:
            continue
    return pos

def get_daily_bars(symbol: str, limit: int):
    """
    Returns list of bars (oldest -> newest). Uses alpaca_trade_api TimeFrame.Day.
    """
    bars = api.get_bars(symbol, TimeFrame.Day, limit=limit)
    # alpaca_trade_api returns a BarSet-like that can be iterated
    return list(bars)

def sma(values):
    return sum(values) / len(values) if values else None

def calc_qty(equity: float, price: float):
    """
    Risk-based sizing:
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = price * STOP_PCT
    qty = floor(risk_amount / per_share_risk)
    also cap notional to MAX_NOTIONAL_PER_TRADE * equity
    """
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

def market_regime_ok():
    """
    Long-only filter:
    SPY close must be above SMA200.
    If data insufficient, return True (don’t brick the bot).
    """
    need = MARKET_FILTER_SMA_DAYS + 5
    bars = get_daily_bars(MARKET_FILTER_SYMBOL, need)
    if len(bars) < MARKET_FILTER_SMA_DAYS + 1:
        return True

    closes = [float(b.c) for b in bars]
    last_close = closes[-1]
    ma = sma(closes[-MARKET_FILTER_SMA_DAYS:])
    if ma is None:
        return True
    return last_close > ma

def avg_volume_ok(bars, lookback=20):
    if len(bars) < lookback + 1:
        return False
    vols = [float(b.v) for b in bars[-lookback:]]
    return (sum(vols) / len(vols)) >= MIN_AVG_VOLUME

def get_last_entry_date(symbol: str, days_back: int = 30):
    """
    Best-effort: look back through FILL activities to find the most recent BUY fill for this symbol.
    (Used for MAX_HOLD_DAYS.)
    """
    after = (ny_now() - timedelta(days=days_back)).date().isoformat()
    acts = api.get_activities(activity_types="FILL", direction="desc", after=after)
    for a in acts:
        try:
            if getattr(a, "symbol", None) != symbol:
                continue
            side = getattr(a, "side", "").lower()
            if side == "buy":
                # transaction_time is ISO-like string
                ts = getattr(a, "transaction_time", None)
                if ts:
                    # Example: "2026-02-23T20:55:04.123Z"
                    # Parse as UTC then convert to NY.
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(NY_TZ)
                    return dt.date()
        except Exception:
            continue
    return None

def close_position(symbol: str):
    try:
        api.close_position(symbol)
        print(f"[EXIT] Closed position: {symbol}")
    except Exception as e:
        print(f"[EXIT] Failed closing {symbol}: {e}")

def submit_bracket_buy(symbol: str, qty: int, ref_price: float):
    """
    Bracket order: market entry + TP limit + SL stop.
    Prices computed from ref_price (latest close).
    """
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


# ================== STRATEGY LOGIC ==================
def should_enter_long(symbol: str, bars):
    """
    Signal:
    - Must have enough history
    - Price >= MIN_PRICE
    - Avg volume filter
    - Today's close > highest high of previous BREAKOUT_LOOKBACK_DAYS
    """
    need = BREAKOUT_LOOKBACK_DAYS + 2
    if len(bars) < need:
        return (False, None)

    last = bars[-1]
    close = float(last.c)
    if close < MIN_PRICE:
        return (False, None)

    if not avg_volume_ok(bars, lookback=20):
        return (False, None)

    prev_window = bars[-(BREAKOUT_LOOKBACK_DAYS + 1):-1]  # exclude last bar
    highest_prev_high = max(float(b.h) for b in prev_window)

    if close > highest_prev_high:
        return (True, close)

    return (False, None)

def should_exit(symbol: str, bars, entry_date):
    """
    Exit if:
    - time stop hit (>= MAX_HOLD_DAYS)
    - trend break: close < SMA_EXIT_DAYS SMA
    """
    if len(bars) < max(SMA_EXIT_DAYS, 5) + 1:
        return False

    closes = [float(b.c) for b in bars]
    last_close = closes[-1]
    ma = sma(closes[-SMA_EXIT_DAYS:])
    if ma is None:
        return False

    # Trend break exit
    if last_close < ma:
        return True

    # Time stop exit
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
        # This avoids intraday behavior and reduces PDT/same-day churn.
        print("Market is open. Exiting because RUN_AFTER_CLOSE_ONLY=True.")
        return

    equity = get_equity()
    print(f"Equity: ${equity:,.2f}")

    # Market regime filter (long-only)
    if not market_regime_ok():
        print("Market filter: SPY not above SMA200 -> no new entries today.")
        allow_new_entries = False
    else:
        allow_new_entries = True

    positions = list_positions_map()

    # 1) Manage exits first
    for sym, pos in list(positions.items()):
        if sym not in SYMBOLS:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(MARKET_FILTER_SMA_DAYS, SMA_EXIT_DAYS, BREAKOUT_LOOKBACK_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        entry_date = get_last_entry_date(sym, days_back=45)
        if should_exit(sym, bars, entry_date):
            close_position(sym)

    # Refresh positions after exits
    positions = list_positions_map()

    # 2) New entries (up to MAX_POSITIONS)
    slots = MAX_POSITIONS - len(positions)
    if slots <= 0:
        print(f"No entry slots (MAX_POSITIONS={MAX_POSITIONS}).")
        return

    if not allow_new_entries:
        return

    # Scan candidates and rank by "breakout strength" (close - breakout level)
    candidates = []
    for sym in SYMBOLS:
        if sym in positions:
            continue

        try:
            bars = get_daily_bars(sym, limit=max(MARKET_FILTER_SMA_DAYS, SMA_EXIT_DAYS, BREAKOUT_LOOKBACK_DAYS) + 5)
        except Exception as e:
            print(f"[DATA] Failed bars for {sym}: {e}")
            continue

        ok, ref_price = should_enter_long(sym, bars)
        if not ok or ref_price is None:
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
            print(f"[SKIP] {sym} qty computed 0 (equity too small or stop too wide).")
            continue

        submit_bracket_buy(sym, qty, ref_price)

    print("Done.")


if __name__ == "__main__":
    main()
