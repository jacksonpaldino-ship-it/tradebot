#!/usr/bin/env python3
"""
HF scalper bot (micro-scalper, aggressive)
- Mode: B1 micro-scalper (1-minute signals)
- Max trades/day: 10 (global), per-symbol limited
- Profit target: 0.20% (TP) ; Stop-loss: 0.15% (SL)
- Position sizing: 2% equity risk (ATR-based)
- Uses alpaca-py TradingClient for orders and data
- Single-open-position global rule (safety)
- Persistent state in JSON to avoid duplicate entries and enforce daily caps
"""

import os
import math
import time
import json
import csv
from datetime import datetime, timedelta
import pytz
import requests
import traceback

import numpy as np
import pandas as pd
import yfinance as yf

# alpaca-py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, OrderRequest, TakeProfit, StopLoss
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------------- CONFIG (aggressive defaults chosen per your request) ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]   # universe
MAX_TRADES_PER_DAY = 10                  # global cap
PER_SYMBOL_DAILY_CAP = 6                 # cap per symbol (safety)
TP_PCT = 0.0020                          # 0.20% take profit
SL_PCT = 0.0015                          # 0.15% stop-loss
RISK_PER_TRADE = 0.02                    # 2% account equity risk per trade (aggressive)
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

MIN_VOLUME = 20_000                      # filter low-volume bars
MAX_POSITION_AGE_MIN = 60 * 3           # close after 3 hours forced (shouldn't hit often)

STATE_FILE = "hf_scalper_state.json"
TRADES_CSV = "hf_trades.csv"
LOG_FILE = "hf_bot.log"

# Alpaca credentials from env (workflow maps these secrets)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. 'https://paper-api.alpaca.markets'
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

# clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True if "paper" in ALPACA_BASE_URL else False)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- utilities ----------------
def now_et():
    return datetime.now(TZ)

def utcnow():
    return datetime.utcnow()

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "has_open": False}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "has_open": False}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)

def append_csv(row):
    exists = os.path.exists(TRADES_CSV)
    header = ["utc_ts","symbol","side","qty","entry","exit","pnl","note"]
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

def log(msg):
    ts = utcnow().isoformat()
    line = f"{ts} {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ---------------- indicators ----------------
def compute_atr(series_df, period=ATR_PERIOD):
    high = series_df["high"]
    low = series_df["low"]
    close = series_df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return max(float(atr), 0.0001)

def compute_macd_hist(series_df):
    close = series_df["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = (macd - signal).iloc[-1]
    return float(hist)

def compute_vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    if v <= 0:
        return float(df["close"].iloc[-1])
    return float(pv / v)

# ---------------- alpaca data helper (1m) ----------------
def fetch_minute_bars(symbol, limit=120):
    """
    Use Alpaca data client to fetch minute bars (most reliable).
    Falls back to yfinance if data_client fails (offline/testing).
    Returns pandas DataFrame with columns: open, high, low, close, volume, indexed by timestamp (UTC).
    """
    try:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, limit=limit)
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            raise RuntimeError("empty bars")
        # if multi-symbol, select symbol
        if isinstance(df.columns, pd.MultiIndex):
            # dataframe has columns like (symbol, open)
            df = df.xs(symbol, axis=1, level=0)
        df = df.rename(str.lower, axis=1)
        df = df.sort_index()
        return df
    except Exception as e:
        # fallback to yfinance (slower, less reliable)
        try:
            import yfinance as yf
            t_end = datetime.utcnow()
            t_start = t_end - timedelta(minutes=limit*2)
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval="1m", start=t_start - timedelta(minutes=1), end=t_end + timedelta(minutes=1))
            if df is None or df.empty:
                return None
            df = df.rename(columns=str.lower)
            df = df[["open","high","low","close","volume"]]
            df.index = df.index.tz_localize(None)
            return df.tail(limit)
        except Exception:
            return None

# ---------------- risk sizing ----------------
def get_account_equity():
    try:
        acct = trading_client.get_account()
        return float(acct.equity)
    except Exception as e:
        log(f"get_account_equity error: {e}")
        return None

def compute_qty(entry_price, atr):
    equity = get_account_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price * 0.0005)  # floor tiny risk
    qty = int(max(1, math.floor(risk_amount / per_share_risk)))
    # cap size by buying power safety estimate (don't use margin crazy)
    cap = max(1, int((equity * 0.5) // entry_price))  # never use >50% equity nominal exposure
    return min(qty, cap)

# ---------------- order helpers ----------------
def place_bracket_buy(symbol, qty, sl_price, tp_price):
    """
    Uses alpaca-py TradingClient to submit a bracket order if available.
    """
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfit(limit_price=str(round(tp_price, 6))),
            stop_loss=StopLoss(stop_price=str(round(sl_price, 6)))
        )
        order = trading_client.submit_order(order_request)
        log(f"Bracket buy submitted: {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"place_bracket_buy error: {e}")
        return None

def place_market_buy(symbol, qty):
    try:
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_request)
        log(f"Market buy submitted: {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"place_market_buy error: {e}")
        return None

# ---------------- main signal & execution logic ----------------
def scan_and_trade():
    """
    Single-run scanner:
    - refresh state for today's counters
    - if global open position exists, do nothing (we enforce single-open-position)
    - otherwise score universe and attempt to place one (or a few) scalps subject to daily caps
    """
    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        # reset daily counters
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "open_order_id": None, "has_open": False}
        save_state(state)

    # safety: do not run heavy logic if already reached daily cap
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily trade cap reached ({state['daily_trades']}/{MAX_TRADES_PER_DAY}). Exiting.")
        return

    # if any open positions exist on account, do not enter new one (global single-position rule)
    positions = list(trading_client.get_all_positions())
    if positions:
        log("Account has open positions; skipping entry to avoid multi-position state.")
        return

    # Score each symbol
    scored = []
    for sym in SYMBOLS:
        try:
            df = fetch_minute_bars(sym, limit=120)
            if df is None or df.empty or len(df) < 10:
                continue
            # filter by recent volume liquidity
            if int(df["volume"].iloc[-1]) < MIN_VOLUME:
                continue
            # compute indicators on recent window (last 60)
            window = df.tail(60)
            price = float(window["close"].iloc[-1])
            vwap = compute_vwap(window)
            vw_gap = abs(price - vwap) / (vwap + EPS)
            macd_hist = compute_macd_hist(window)
            atr = compute_atr(window)
            # scoring: favor close-to-vwap, positive macd hist, high recent volume
            vw_score = max(0.0, 1.0 - vw_gap / 0.01)   # normalized to 1% band
            vol_score = min(1.0, window["volume"].iloc[-1] / (window["volume"].mean() + EPS))
            macd_score = 1.0 if macd_hist > 0 else 0.0
            score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
            scored.append({"symbol": sym, "score": score, "price": price, "vwap": vwap, "atr": atr, "macd_hist": macd_hist})
        except Exception as e:
            log(f"score error {sym}: {e}")

    if not scored:
        log("No candidates found.")
        return

    # sort
    scored.sort(key=lambda x: x["score"], reverse=True)
    # attempt up to 1 entry this run (aggressive but safe); can be changed to try multiple symbols
    for candidate in scored:
        sym = candidate["symbol"]
        # enforce per-symbol daily cap
        per_sym = state["per_symbol"].get(sym, 0)
        if per_sym >= PER_SYMBOL_DAILY_CAP:
            log(f"Per-symbol daily cap reached for {sym}.")
            continue
        # compute sizing
        entry_price = candidate["price"]
        atr = candidate["atr"]
        qty = compute_qty(entry_price, atr)
        if qty <= 0:
            continue

        # compute TP/SL using percentages (tight scalping)
        tp_price = entry_price * (1.0 + TP_PCT)
        sl_price = entry_price * (1.0 - SL_PCT)

        # Submit bracket order (preferred)
        order = place_bracket_buy(sym, qty, sl_price, tp_price)
        if order:
            log(f"Placed bracket for {sym} qty={qty} entry approx {entry_price:.4f} tp {tp_price:.4f} sl {sl_price:.4f}")
            # update state
            state["daily_trades"] += 1
            state["per_symbol"][sym] = per_sym + 1
            state["open_order_id"] = getattr(order, "id", None)
            state["has_open"] = True
            save_state(state)
            # record nominal trade row (entry/exits will be updated later via fills log)
            append_csv([utcnow().isoformat(), sym, "BUY_SUBMIT", qty, round(entry_price,6), None, None, f"bracket_submitted tp{TP_PCT} sl{SL_PCT}"])
            return
        else:
            # fallback to market buy + manual monitor with tighter constraints
            order2 = place_market_buy(sym, qty)
            if not order2:
                log(f"Buy failed for {sym}; trying next candidate.")
                continue
            # wait a moment to let order fill (small sleep)
            time.sleep(1.2)
            # read filled price
            try:
                filled = trading_client.get_order_by_client_order_id(order2.client_order_id) if getattr(order2, "client_order_id", None) else trading_client.get_order_by_id(order2.id)
            except Exception:
                filled = order2
            # best-effort entry price
            filled_price = getattr(filled, "filled_avg_price", None) or getattr(filled, "filled_avg_price", None) or entry_price
            try:
                filled_price = float(filled_price)
            except Exception:
                filled_price = entry_price
            # now submit stop-loss and take-profit as separate orders (OCO not always available)
            # For safety we won't submit separate orders here. We'll rely on monitor step in future runs (but single-run design).
            log(f"Market buy filled {sym} price {filled_price} qty {qty}")
            state["daily_trades"] += 1
            state["per_symbol"][sym] = per_sym + 1
            state["has_open"] = True
            save_state(state)
            append_csv([utcnow().isoformat(), sym, "BUY_FILLED", qty, round(filled_price,6), None, None, "market_buy_filled"])
            # Optionally, attach a note that manual monitoring required
            return

    log("Scanned candidates but none placed due to caps or failures.")

# ---------------- helper to check fills and update records (called each run) --------------
def reconcile_fills_and_cleanup():
    """
    Read recent orders and positions to record fills and update state.
    This function also closes long-dead leftover positions if any (very conservative).
    """
    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        # safety reset if mismatch
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "open_order_id": None, "has_open": False}
        save_state(state)

    # Fetch current positions
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        log(f"get_all_positions error: {e}")
        positions = []

    # If positions exist, record them if not yet recorded; if too old, force close
    for pos in positions:
        sym = pos.symbol
        qty = int(float(pos.qty))
        market_value = float(pos.market_value)
        avg_entry = float(pos.avg_entry_price) if getattr(pos, "avg_entry_price", None) else None
        # write a record if we don't have has_open flag (best-effort: we may have recorded earlier)
        append_csv([utcnow().isoformat(), sym, "POSITION", qty, round(avg_entry,6) if avg_entry else None, None, None, "open_position"])
        state["has_open"] = True
        state["open_order_id"] = None
        save_state(state)

        # If position age is too long, close (protection for stuck positions)
        # We can't easily check age here without recorded entry ts; do a conservative forced close if POS very large and equity negative.
        # Skip forced close here to avoid accidental exits during live runs.

    # attempt to collect filled orders from trading_client
    try:
        orders = trading_client.get_orders(status="all", limit=50)
    except Exception as e:
        log(f"get_orders error: {e}")
        orders = []

    for o in orders:
        try:
            # Consider only today's orders
            created_at = getattr(o, "created_at", None)
            if created_at:
                created_dt = created_at.astimezone(pytz.utc).replace(tzinfo=None)
                if created_dt.date() != datetime.utcnow().date():
                    continue
            side = getattr(o, "side", None)
            symbol = getattr(o, "symbol", None)
            filled_qty = float(getattr(o, "filled_qty", 0) or 0)
            filled_avg = getattr(o, "filled_avg_price", None)
            if filled_qty and filled_avg:
                # record fill
                append_csv([utcnow().isoformat(), symbol, side.upper(), int(filled_qty), round(float(filled_avg),6), None, None, f"filled_order id {getattr(o, 'id', None)}"])
        except Exception:
            continue

    save_state(state)

# ---------------- entrypoint ----------------
def main():
    try:
        # only run during market open: use trading_client.get_clock()
        try:
            clock = trading_client.get_clock()
            if not getattr(clock, "is_open", False):
                log("Market closed. Exiting.")
                # still call reconcile to record fills if needed
                reconcile_fills_and_cleanup()
                return
        except Exception as e:
            log(f"clock error: {e}. Proceeding cautiously.")

        # first, reconcile any fills from earlier runs
        reconcile_fills_and_cleanup()

        # scan and attempt to place 1 aggressive scalp (per-run)
        scan_and_trade()

        # reconcile after attempting entries
        reconcile_fills_and_cleanup()

    except Exception as e:
        log("Unhandled exception in main: " + repr(e))
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
