#!/usr/bin/env python3
"""
hybrid_tradebot_spy_signals.py

Production-ready hybrid intraday scalper:
- multi-symbol scoring (VWAP/volume/spread + adaptive thresholds)
- guaranteed 1 entry per market day (only during market open)
- market buy + TP (limit) + SL (stop) exit logic, with monitoring to cancel losers
- trade logging (trades.csv) and stats (trade_stats.json)
- Google Sheets fallback (if SIGNAL_SHEET_CSV_URL provided)
- robust Alpaca error handling and retries
"""

import os
import time
import csv
import json
import math
import requests
from datetime import datetime, timedelta
import pytz

import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# -------------------- CONFIG --------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]        # primary universe
MIN_VOLUME = 30_000                           # baseline minimum volume
VWAP_MAX_PCT = 0.02                           # baseline VWAP deviation allowed (2%)
MAX_SPREAD_DOLLARS = 2.0                      # baseline spread limit
GUARANTEE_TRADE = True                        # force top candidate if none pass (only when market open)
MONITOR_INTERVAL = 6                           # seconds between checks after entry
MONITOR_TIMEOUT = 60 * 30                      # seconds to wait for TP/SL (30 minutes)
RISK_PER_TRADE = 0.002                         # percent of account equity risk per trade (0.2%)
TP_MULT = 1.5                                  # risk * TP_MULT = take-profit distance (e.g. 1.5R)
SL_MULT = 1.0                                  # risk * SL_MULT = stop-loss distance (1R)
MAX_RETRY = 5
EPS = 1e-6

# Secrets/environment (you said you use these secret names)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. https://paper-api.alpaca.markets

SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL")  # optional fallback

TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
TRADED_DAY_FILE = "traded_day.json"  # persist the date we traded

TZ = pytz.timezone("US/Eastern")

if not (API_KEY and API_SECRET and BASE_URL):
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL.")

# Alpaca client
api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# -------------------- HELPERS --------------------
def now_et():
    return datetime.now(TZ)

def read_traded_day():
    if os.path.exists(TRADED_DAY_FILE):
        try:
            with open(TRADED_DAY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def write_traded_day(d):
    with open(TRADED_DAY_FILE, "w") as f:
        json.dump(d, f)

def append_trade_csv(trade):
    header = ["timestamp", "symbol", "side", "qty", "entry", "exit", "pnl", "result", "notes"]
    exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            trade.get("timestamp"),
            trade.get("symbol"),
            trade.get("side"),
            trade.get("qty"),
            trade.get("entry"),
            trade.get("exit"),
            trade.get("pnl"),
            trade.get("result"),
            trade.get("notes", "")
        ])

def update_stats(trade):
    stats = {}
    if os.path.exists(STATS_JSON):
        try:
            with open(STATS_JSON, "r") as f:
                stats = json.load(f)
        except Exception:
            stats = {}
    s = stats.get(trade["symbol"], {"wins":0,"losses":0,"trades":0, "pnl":0.0})
    s["trades"] += 1
    pnl = trade.get("pnl", 0.0) or 0.0
    s["pnl"] += pnl
    if trade.get("result") == "WIN":
        s["wins"] += 1
    elif trade.get("result") == "LOSS":
        s["losses"] += 1
    stats[trade["symbol"]] = s
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

def safe_get_bars(symbol, limit=100, timeframe=TimeFrame.Minute):
    """Fetch bars and return DataFrame or None. Handles Alpaca empty responses."""
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit, adjustment='raw').df
        if bars is None or bars.empty:
            return None
        # Keep only symbol rows if multi-symbol returned
        if 'symbol' in bars.columns:
            bars = bars[bars['symbol'] == symbol]
        if bars.empty:
            return None
        # Make sure columns names: close/high/low/volume exist
        return bars
    except Exception as e:
        print(f"safe_get_bars error {symbol}: {e}")
        return None

def fetch_sheet_signals():
    if not SIGNAL_SHEET_CSV_URL:
        return []
    try:
        resp = requests.get(SIGNAL_SHEET_CSV_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            df = df[df["symbol"] != ""]
        else:
            return []
        return df.to_dict("records")
    except Exception as e:
        print(f"fetch_sheet_signals error: {e}")
        return []

# -------------------- RISK / SIZE --------------------
def account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print(f"account_equity error: {e}")
        return None

def compute_qty(symbol, entry_price):
    """Simple sizing: risk per trade default (RISK_PER_TRADE of equity) divided by dollar risk per share."""
    equity = account_equity()
    if equity is None:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    # estimate dollar risk per share = entry_price * SL_MULT * some small factor (we'll set minimal $0.5)
    dollar_risk_share = max(0.5, entry_price * 0.01 * SL_MULT)
    qty = int(max(1, math.floor(risk_amount / dollar_risk_share)))
    return qty

# -------------------- SCORING --------------------
def compute_vwap(df):
    # df expected to have 'close' and 'volume' columns
    vol = (df['close'] * df['volume']).sum()
    vols = df['volume'].sum()
    if vols <= 0:
        return float(df['close'].iloc[-1])
    return float(vol / vols)

def score_symbol(symbol):
    df = safe_get_bars(symbol, limit=120)
    if df is None or df.empty:
        return None
    # compute VWAP and metrics on recent window
    window = min(len(df), 60)
    dfw = df.tail(window)
    vwap = compute_vwap(dfw)
    price = float(dfw['close'].iloc[-1])
    volume = float(dfw['volume'].iloc[-1])
    high = float(dfw['high'].iloc[-1])
    low = float(dfw['low'].iloc[-1])
    spread = max((high - low), EPS)
    # adaptive thresholds
    spread_avg = float((dfw['high'] - dfw['low']).mean())
    vol_avg = float(dfw['volume'].mean()) if dfw['volume'].mean() > 0 else 1.0
    # components
    vw_gap = abs(price - vwap) / (vwap + EPS)            # relative gap
    vw_score = max(0.0, 1.0 - vw_gap / (VWAP_MAX_PCT + EPS))  # normalized
    vol_score = min(1.0, volume / (vol_avg + EPS))
    spread_score = max(0.0, 1.0 - (spread / (spread_avg + EPS)))
    # final score weights
    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * spread_score
    return {
        "symbol": symbol,
        "score": float(score),
        "price": price,
        "vwap": vwap,
        "volume": volume,
        "spread": spread,
        "spread_avg": spread_avg,
        "vol_avg": vol_avg
    }

# -------------------- ORDER HELPERS --------------------
def submit_market_order(symbol, qty, side):
    """Submit market order with backoff; do not retry client 4xx errors."""
    attempt = 0
    while attempt < MAX_RETRY:
        attempt += 1
        try:
            order = api.submit_order(symbol=symbol, qty=qty, side=side,
                                      type='market', time_in_force='day')
            return order
        except APIError as e:
            msg = str(e)
            # client error check (400-range)
            if "400" in msg or "Bad Request" in msg or "symbol is required" in msg:
                print(f"submit_market_order client error: {msg} (not retrying)")
                return None
            print(f"submit_market_order attempt {attempt} failed: {e}, retrying...")
            time.sleep(1.5 ** attempt)
        except Exception as e:
            print(f"submit_market_order unknown error {e}, attempt {attempt}")
            time.sleep(1.5 ** attempt)
    print("submit_market_order failed after retries")
    return None

def submit_tp_sl_orders(symbol, qty, entry_price, sl_price, tp_price):
    """Place TP (limit) and SL (stop) orders and return their IDs. Use monitor to cancel when one fills."""
    tp_id = None
    sl_id = None
    try:
        # Place take-profit limit sell
        tp = api.submit_order(symbol=symbol, qty=qty, side='sell', type='limit',
                              time_in_force='gtc', limit_price=str(round(tp_price, 6)))
        tp_id = tp.id
    except Exception as e:
        print(f"TP order error: {e}")
        tp_id = None

    try:
        # Place stop-loss sell (stop_market)
        sl = api.submit_order(symbol=symbol, qty=qty, side='sell', type='stop',
                              time_in_force='gtc', stop_price=str(round(sl_price, 6)))
        sl_id = sl.id
    except Exception as e:
        print(f"SL order error: {e}")
        sl_id = None

    return tp_id, sl_id

def cancel_order(order_id):
    try:
        api.cancel_order(order_id)
    except Exception as e:
        print(f"cancel_order error {order_id}: {e}")

def get_order_status(order_id):
    try:
        return api.get_order(order_id).status
    except Exception:
        return None

# -------------------- MONITORING & EXECUTION --------------------
def monitor_and_manage(symbol, qty, entry_price, sl_price, tp_price, timeout=MONITOR_TIMEOUT):
    print(f"Monitoring {symbol}: entry {entry_price:.4f}, TP {tp_price:.4f}, SL {sl_price:.4f}")
    tp_id, sl_id = submit_tp_sl_orders(symbol, qty, entry_price, sl_price, tp_price)
    start = time.time()
    filled = False
    result = None
    exit_price = None

    while time.time() - start < timeout:
        # check orders
        if tp_id:
            tp_status = get_order_status(tp_id)
            if tp_status in ("filled", "partial", "done"):
                # filled (partial/done treat as filled)
                filled = True
                result = "WIN"
                # determine fill price (we'll read fill price from orders)
                try:
                    o = api.get_order(tp_id)
                    exit_price = float(o.filled_avg_price or tp_price)
                except Exception:
                    exit_price = tp_price
                # cancel stop
                if sl_id:
                    cancel_order(sl_id)
                break
        if sl_id:
            sl_status = get_order_status(sl_id)
            if sl_status in ("filled", "partial", "done"):
                filled = True
                result = "LOSS"
                try:
                    o = api.get_order(sl_id)
                    exit_price = float(o.filled_avg_price or sl_price)
                except Exception:
                    exit_price = sl_price
                if tp_id:
                    cancel_order(tp_id)
                break
        # also check if position closed for other reasons
        time.sleep(MONITOR_INTERVAL)

    if not filled:
        # timeout: attempt to cancel orders and market exit
        print("Monitor timeout, attempting to exit market and cancel OCO orders")
        # cancel outstanding
        if tp_id:
            cancel_order(tp_id)
        if sl_id:
            cancel_order(sl_id)
        # try a market sell
        try:
            sell = submit_market_order(symbol, qty, "sell")
            if sell:
                # we don't have filled price reliably; use last trade price
                bars = safe_get_bars(symbol, limit=1)
                exit_price = float(bars['close'].iloc[-1]) if bars is not None else None
                result = "TIMEOUT"
        except Exception as e:
            print(f"Timeout exit market sell failed: {e}")
            result = "UNKNOWN"

    # compute pnl
    pnl = None
    if exit_price is not None:
        pnl = round((exit_price - entry_price) * qty, 6)
    trade_rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "side": "LONG",
        "qty": qty,
        "entry": round(entry_price, 6),
        "exit": round(exit_price, 6) if exit_price else None,
        "pnl": pnl,
        "result": result,
        "notes": "TP/SL monitored"
    }
    append_trade_csv(trade_rec)
    update_stats(trade_rec)
    return trade_rec

# -------------------- FALLBACK SHEET --------------------
def fallback_from_sheet():
    signals = fetch_sheet_signals()
    for s in signals:
        symbol = s.get("symbol")
        action = (s.get("action") or "BUY").upper()
        qty = int(s.get("qty", 1))
        if not symbol:
            continue
        if action == "BUY":
            # simple market buy with default TP/SL using current price
            bars = safe_get_bars(symbol, limit=5)
            if bars is None:
                continue
            price = float(bars['close'].iloc[-1])
            qty_calc = qty
            order = submit_market_order(symbol, qty_calc, "buy")
            if order:
                # compute sl/tp using percent approach
                sl_price = price * (1 - 0.005)
                tp_price = price * (1 + 0.01)
                return monitor_and_manage(symbol, qty_calc, price, sl_price, tp_price)
    return None

# -------------------- MAIN STRATEGY --------------------
def run_once():
    # Check market open
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            print(f"Market closed (timestamp {clock.timestamp}). Exiting without trading.")
            return None
    except Exception as e:
        print(f"Failed to get clock: {e}. Aborting run.")
        return None

    # Ensure not already traded today
    traded = read_traded_day()
    today = now_et().strftime("%Y-%m-%d")
    if traded.get("date") == today:
        print("Already executed today's guaranteed trade. Exiting.")
        return None

    # Score symbols
    scored = []
    for s in SYMBOLS:
        try:
            data = score_symbol(s)
            if data:
                scored.append(data)
        except Exception as e:
            print(f"score_symbol error {s}: {e}")

    if not scored:
        print("No scored symbols (no data). Try fallback sheet.")
        fb = fallback_from_sheet()
        if fb:
            traded["date"] = today
            write_traded_day(traded)
            return fb
        return None

    # rank
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[0]
    print(f"Top candidate: {top['symbol']} score {top['score']:.4f} price {top['price']:.4f} vwap {top['vwap']:.4f}")

    # Filter checks (adaptive)
    # allow if within adaptive vw_pct threshold based on spread_avg
    vw_pct = abs(top['price'] - top['vwap']) / (top['vwap'] + EPS)
    adaptive_vwap_pct = max(0.01, (top['spread_avg'] / (top['vwap'] + EPS)) * 1.5)  # scale
    if vw_pct > adaptive_vwap_pct:
        print(f"{top['symbol']} price-vwap {vw_pct:.4f} > adaptive {adaptive_vwap_pct:.4f} => skipping primary")
        # attempt fallback
        fb = fallback_from_sheet()
        if fb:
            traded["date"] = today
            write_traded_day(traded)
            return fb
        # if guarantee enabled, force top after logging
        if not GUARANTEE_TRADE:
            return None
        print("Guarantee flag ON — forcing top candidate despite filter")

    # compute qty using sizing
    qty = compute_qty(top['symbol'], top['price'])
    if qty <= 0:
        qty = 1

    # Submit market buy
    order = submit_market_order(top['symbol'], qty, "buy")
    if not order:
        print("Buy order failed. Exiting.")
        return None

    # Determine entry price (best effort)
    entry_price = None
    try:
        # attempt to find fill price from order
        o = api.get_order(order.id)
        entry_price = float(o.filled_avg_price) if getattr(o, 'filled_avg_price', None) else None
    except Exception:
        entry_price = None

    if entry_price is None:
        bars = safe_get_bars(top['symbol'], limit=1)
        if bars is not None:
            entry_price = float(bars['close'].iloc[-1])
    if entry_price is None:
        print("Could not determine entry price, aborting management.")
        return None

    # Determine stop and take prices based on volatility and risk
    # Estimate per-share dollar risk = entry_price * (spread_avg/vwap) approx or small floor
    est_risk_per_share = max(0.1, abs(top['price'] - top['vwap']), top['spread'] * 0.5)
    sl_price = entry_price - (est_risk_per_share * SL_MULT)
    tp_price = entry_price + (est_risk_per_share * TP_MULT)

    # Monitor TP/SL and manage
    trade_rec = monitor_and_manage(top['symbol'], qty, entry_price, sl_price, tp_price)

    # persist traded day so guarantee doesn't run again
    traded["date"] = today
    write_traded_day(traded)
    return trade_rec

# -------------------- ENTRYPOINT --------------------
if __name__ == "__main__":
    print(f"Run start ET {now_et().isoformat()}")
    try:
        rec = run_once()
        if rec:
            print("Trade recorded:", rec)
        else:
            print("No trade executed this run.")
    except Exception as e:
        print("Unhandled error in run:", e)
