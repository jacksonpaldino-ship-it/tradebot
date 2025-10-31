# tradebot.py
# Aggressive/managed position-sizing overlay for your SMA strategy
# -> Paper-test thoroughly. No guarantees. Use at your own risk.

import os
import time
import math
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, time as dtime
import alpaca_trade_api as tradeapi

# ---------------------------
# CONFIG / RISK PARAMETERS
# ---------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]

# Position sizing / risk controls (tweak these carefully)
TARGET_PCT_PER_SYMBOL = 0.12      # aim to allocate ~12% of equity per symbol (not exact)
MAX_TOTAL_EXPOSURE = 0.60         # do not exceed 60% of equity invested
RISK_PCT_PER_TRADE = 0.005        # risk 0.5% of equity per trade (if stop is hit)
MAX_TRADES_PER_SYMBOL_PER_DAY = 2
REWARD_TO_RISK = 2.0              # take-profit multiple of stop distance (2:1)
SHORT_MA = 5
LONG_MA = 20

# Market hours (ET)
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
TZ = pytz.timezone("US/Eastern")

# Alpaca client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Local cooldown tracking
trade_counts = {}  # { "YYYY-MM-DD": {symbol: count, ...} }

# ---------------------------
# Utilities
# ---------------------------
def is_market_open_now():
    now = datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def get_account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Failed to fetch account equity:", e)
        return None

def get_open_positions_value():
    """Return total market value of open positions (float)."""
    try:
        positions = api.list_positions()
        total = sum(float(p.market_value) for p in positions)
        return total
    except Exception:
        return 0.0

def get_position_qty(symbol):
    try:
        p = api.get_position(symbol)
        return int(float(p.qty))
    except Exception:
        return 0

# ---------------------------
# Market data & indicators
# ---------------------------
def fetch_15m(symbol, period="60d"):
    """Return dataframe with 15m bars and ATR (14) computed."""
    df = yf.download(symbol, period=period, interval="15m", progress=False)
    if df.empty:
        return df
    df = df.dropna()
    # ensure single index
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    # indicators
    df["sma_short"] = df["Close"].rolling(window=SHORT_MA).mean()
    df["sma_long"] = df["Close"].rolling(window=LONG_MA).mean()
    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df.dropna()

# ---------------------------
# Signal function
# ---------------------------
def get_signal_from_df(df):
    # return "BUY","SELL","HOLD"
    if df is None or df.empty or len(df) < LONG_MA + 2:
        return "HOLD"
    latest = df.tail(1).iloc[0]
    prev = df.tail(2).iloc[0]
    try:
        latest_short = float(latest["sma_short"])
        latest_long = float(latest["sma_long"])
        prev_short = float(prev["sma_short"])
        prev_long = float(prev["sma_long"])
    except Exception:
        return "HOLD"

    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    if prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    return "HOLD"

# ---------------------------
# Position sizing
# ---------------------------
def compute_order_qty(symbol, price, stop_distance, equity):
    """
    Compute integer qty such that:
      qty * stop_distance ≈ equity * RISK_PCT_PER_TRADE
    where stop_distance is in price units (e.g. $).
    Make sure resulting exposure <= TARGET_PCT_PER_SYMBOL * equity and overall exposure limits.
    """
    if stop_distance <= 0 or equity is None or equity <= 0:
        return 0
    risk_per_trade = equity * RISK_PCT_PER_TRADE
    raw_qty = math.floor(risk_per_trade / stop_distance)
    if raw_qty < 1:
        return 0
    # cap by target exposure per symbol
    cap_qty = math.floor((equity * TARGET_PCT_PER_SYMBOL) / price)
    qty = min(raw_qty, cap_qty)
    return int(max(qty, 0))

# ---------------------------
# Order placement (bracket)
# ---------------------------
def submit_bracket_order(symbol, side, qty, entry_price, stop_price, take_profit_price):
    """
    Use Alpaca classic API bracket order: market entry with stop & take profit
    note: returns order response or raises
    """
    if qty <= 0:
        raise ValueError("qty must be > 0")
    try:
        # alpaca_trade_api submit_order supports order_class parameter for bracket
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={'limit_price': str(round(take_profit_price, 2))},
            stop_loss={'stop_price': str(round(stop_price, 2)), 'limit_price': str(round(stop_price * 0.995, 2))}
        )
        return order
    except Exception as e:
        raise

# ---------------------------
# Trade execution wrapper
# ---------------------------
def attempt_trade(symbol, df, equity):
    # Get latest price
    last = df.tail(1).iloc[0]
    price = float(last["Close"])
    atr = float(df["atr14"].iloc[-1]) if "atr14" in df.columns else None
    # fallback stop distance as a percent if ATR missing
    if atr and atr > 0:
        stop_dist = max(atr, price * 0.005)   # min 0.5% of price
    else:
        stop_dist = price * 0.01  # fallback 1%
    stop_price = round(price - stop_dist if True else price - stop_dist, 2)  # for BUY, stop below
    tp_price = round(price + (stop_dist * REWARD_TO_RISK), 2)

    # compute qty using stop distance (we risk RISK_PCT_PER_TRADE of equity)
    qty = compute_order_qty(symbol, price, stop_dist, equity)

    # safety checks
    if qty <= 0:
        print(f"[{symbol}] computed qty=0 (stop_dist={stop_dist:.2f}), skipping")
        return None

    # check total exposure after this potential buy
    current_open_value = get_open_positions_value()
    if (current_open_value + qty * price) > equity * MAX_TOTAL_EXPOSURE:
        print(f"[{symbol}] would exceed MAX_TOTAL_EXPOSURE; skipping (open {current_open_value:.2f}, trying {qty*price:.2f})")
        return None

    # submit bracket buy order
    try:
        print(f"[{symbol}] Submitting BUY bracket: qty={qty}, entry≈{price:.2f}, stop={stop_price}, tp={tp_price}")
        order = submit_bracket_order(symbol, "buy", qty, price, stop_price, tp_price)
        print(f"[{symbol}] Order submitted id={order.id}")
        return order
    except Exception as e:
        print(f"[{symbol}] Order failed: {e}")
        return None

def attempt_sell_existing(symbol):
    # Sell entire position with a market order (use when SELL signal)
    pos_qty = get_position_qty(symbol)
    if pos_qty <= 0:
        print(f"[{symbol}] No position to sell.")
        return None
    try:
        order = api.submit_order(symbol=symbol, qty=pos_qty, side="sell", type="market", time_in_force="day")
        print(f"[{symbol}] Market sell submitted qty={pos_qty}")
        return order
    except Exception as e:
        print(f"[{symbol}] Sell failed: {e}")
        return None

# ---------------------------
# Daily tracking helpers
# ---------------------------
def todays_key():
    return datetime.now(TZ).strftime("%Y-%m-%d")

def trades_today_count(symbol):
    day = todays_key()
    if day not in trade_counts:
        return 0
    return trade_counts[day].get(symbol, 0)

def record_trade_today(symbol):
    day = todays_key()
    if day not in trade_counts:
        trade_counts[day] = {}
    trade_counts[day][symbol] = trade_counts[day].get(symbol, 0) + 1

# ---------------------------
# Main run
# ---------------------------
def main():
    print(f"\n=== Tradebot run {datetime.now(TZ)} ===")
    if not is_market_open_now():
        print("Market closed — skipping.")
        return

    equity = get_account_equity()
    if equity is None:
        print("Unable to fetch account equity; aborting run.")
        return
    print(f"Account equity: ${equity:,.2f}")

    for symbol in SYMBOLS:
        print(f"\nChecking {symbol}...")
        try:
            df = fetch_15m(symbol)
            if df is None or df.empty or len(df) < (LONG_MA + 5):
                print(f"[{symbol}] insufficient data, skipping")
                continue

            signal = get_signal_from_df(df)
            print(f"[{symbol}] Signal = {signal}")

            # cooldown: max trades per symbol per day
            if trades_today_count(symbol) >= MAX_TRADES_PER_SYMBOL_PER_DAY:
                print(f"[{symbol}] max trades today reached, skipping")
                continue

            if signal == "BUY":
                # only buy if not already long
                pos_qty = get_position_qty(symbol)
                if pos_qty > 0:
                    print(f"[{symbol}] already long qty={pos_qty}, skipping buy")
                    continue
                order = attempt_trade(symbol, df, equity)
                if order:
                    record_trade_today(symbol)

            elif signal == "SELL":
                # close existing long positions
                pos_qty = get_position_qty(symbol)
                if pos_qty > 0:
                    order = attempt_sell_existing(symbol)
                    if order:
                        record_trade_today(symbol)
                else:
                    print(f"[{symbol}] no open position to close")

            else:
                print(f"[{symbol}] HOLD")

            time.sleep(1)  # polite throttle

        except Exception as e:
            print(f"[{symbol}] Error: {e}")

    print("\n✅ Run complete.\n")

if __name__ == "__main__":
    main()
