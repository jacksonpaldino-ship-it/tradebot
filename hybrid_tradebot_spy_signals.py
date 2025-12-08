#!/usr/bin/env python3
"""
Hybrid tradebot — full rewrite.

Features:
- Uses your VWAP/volume/spread buy logic + MACD crossover confirmation
- ATR(14) volatility-based conservative position sizing (0.5% equity risk)
- Trailing stop based on ATR (ratchets upward)
- Attempts bracket orders (TP/SL), falls back to market + monitor
- Guarantees max 1 entry per ET day; forces a trade at FORCE_TRADE_HOUR:FORCE_TRADE_MIN if none executed
- Robust Alpaca API error handling and retries
- Logs trades to trades.csv and stats to trade_stats.json
- Uses secrets: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
"""

import os
import time
import math
import csv
import json
from datetime import datetime, timedelta, time as dtime
import pytz
import requests

import numpy as np
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]   # primary universe; add extras if desired
TRADE_LOG_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
TRADED_DAY_FILE = "traded_day.json"

# Sizing / risk
RISK_PER_TRADE = 0.005   # 0.5% of equity per trade (conservative)
ATR_PERIOD = 14
MIN_SL_PCT = 0.002       # 0.2% minimum per-share SL

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Trailing stop
ATR_TRAIL_MULT = 1.0

# Filters / scoring
VWAP_MAX_PCT = 0.02      # used to normalize score (2% baseline)
MIN_VOLUME = 30000

# Guarantee / scheduling
GUARANTEE_TRADE = True
FORCE_TRADE_HOUR = 15    # ET hour to force trade if none (15 = 3pm ET)
FORCE_TRADE_MIN = 30     # ET minute to force trade (3:30pm ET)
RUN_INTERVAL_SECONDS = 10 * 60  # not used inside script; GH runs every 10m

# Monitoring
MONITOR_INTERVAL = 5     # seconds
MONITOR_TIMEOUT = 60 * 45  # seconds (45 min)

# Alpaca secrets (must match your repo secrets)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. "https://paper-api.alpaca.markets"

if not (API_KEY and API_SECRET and BASE_URL):
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

# Alpaca client
api = REST(API_KEY, API_SECRET, BASE_URL)

# Timezone
TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Utilities ----------------
def now_et():
    return datetime.now(TZ)

def now_utc():
    return datetime.utcnow()

def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)

def append_trade_csv(row):
    header = ["timestamp","symbol","side","qty","entry","exit","pnl","result","notes"]
    exists = os.path.exists(TRADE_LOG_CSV)
    with open(TRADE_LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

def update_stats(tr):
    stats = load_json(STATS_JSON)
    s = stats.get(tr["symbol"], {"wins":0,"losses":0,"trades":0,"pnl":0.0})
    s["trades"] += 1
    pnl = tr.get("pnl") or 0.0
    s["pnl"] = round(s.get("pnl",0.0) + pnl,6)
    if tr.get("result") == "WIN":
        s["wins"] += 1
    elif tr.get("result") == "LOSS":
        s["losses"] += 1
    stats[tr["symbol"]] = s
    save_json(STATS_JSON, stats)

# ---------------- Data helpers ----------------
def safe_get_bars(symbol, limit=200):
    """
    Fetch recent minute bars from Alpaca. Returns DataFrame or None.
    Normalizes columns to lower-case: open, high, low, close, volume
    """
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
        if bars is None or bars.empty:
            return None
        # If bars contains multi-index columns (symbol, open, ...), flatten
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(1)
        bars = bars.rename(str.lower, axis="columns")
        # sometimes symbol column missing for single-symbol fetch
        if "symbol" not in bars.columns:
            bars["symbol"] = symbol
        bars = bars.sort_index()
        return bars
    except Exception as e:
        print(f"safe_get_bars error {symbol}: {e}")
        return None

# ---------------- Indicators ----------------
def compute_atr(df, period=ATR_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(max(atr, 0.0001))

def compute_vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    if v <= 0:
        return float(df["close"].iloc[-1])
    return float(pv / v)

def compute_macd_hist(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return float(hist.iloc[-1]), float(macd.iloc[-1]), float(sig.iloc[-1])

# ---------------- Scoring & entry logic ----------------
def score_symbol(symbol):
    df = safe_get_bars(symbol, limit=200)
    if df is None or df.empty:
        return None
    window = min(120, len(df))
    dfw = df.iloc[-window:].copy()
    price = float(dfw["close"].iloc[-1])
    vwap = compute_vwap(dfw)
    volume = int(dfw["volume"].iloc[-1])
    high = float(dfw["high"].iloc[-1])
    low = float(dfw["low"].iloc[-1])
    spread = max(high - low, EPS)
    spread_avg = float((dfw["high"] - dfw["low"]).mean()) if len(dfw) > 1 else spread
    vol_avg = float(dfw["volume"].mean()) if len(dfw) > 1 else volume

    vw_gap = abs(price - vwap) / (vwap + EPS)
    vw_score = max(0.0, 1.0 - vw_gap / (VWAP_MAX_PCT + EPS))
    vol_score = min(1.0, volume / (vol_avg + EPS))
    spread_score = max(0.0, 1.0 - (spread / (spread_avg + EPS)))
    base_score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * spread_score

    macd_hist, macd_val, macd_sig = compute_macd_hist(dfw)
    macd_ok = macd_hist > 0

    if volume < MIN_VOLUME:
        return None

    atr = compute_atr(dfw)

    return {
        "symbol": symbol,
        "score": float(base_score),
        "price": price,
        "vwap": vwap,
        "volume": volume,
        "spread": spread,
        "spread_avg": spread_avg,
        "vol_avg": vol_avg,
        "macd_hist": macd_hist,
        "macd_ok": macd_ok,
        "atr": atr,
        "df": dfw
    }

# ---------------- Position sizing ----------------
def get_account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print("get_account_equity error:", e)
        return None

def compute_qty(entry_price, atr):
    equity = get_account_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price * MIN_SL_PCT)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    return qty

# ---------------- Order helpers ----------------
def submit_market_order(symbol, qty, side="buy", max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            order = api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="day")
            print(f"Submitted {side.upper()} {qty} {symbol}")
            return order
        except Exception as e:
            attempt += 1
            print(f"submit_market_order attempt {attempt} error: {e}")
            time.sleep(1.0 * attempt)
    return None

def submit_bracket_buy(symbol, qty, sl_price, tp_price):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(round(tp_price,6))},
            stop_loss={"stop_price": str(round(sl_price,6))}
        )
        print(f"Submitted bracket buy {qty} {symbol}")
        return order
    except Exception as e:
        print(f"submit_bracket_buy failed: {e}")
        return None

# ---------------- Monitoring & exits ----------------
def monitor_position(symbol, entry_price, qty, atr, initial_sl):
    """
    Monitor a live position: trailing ATR stop (ratchets up), VWAP breakdown exit, or timeout exit.
    """
    trail_stop = initial_sl
    start = time.time()
    print(f"Monitoring {symbol}: entry={entry_price:.6f} qty={qty} init_sl={initial_sl:.6f}")
    while True:
        # Timeout guard
        if time.time() - start > MONITOR_TIMEOUT:
            print("Monitor timeout, forcing exit")
            sell = submit_market_order(symbol, qty, "sell")
            exit_price = None
            if sell:
                try:
                    o = api.get_order(sell.id)
                    exit_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else None
                except Exception:
                    exit_price = None
            if exit_price is None:
                bars = safe_get_bars(symbol, limit=1)
                exit_price = float(bars["close"].iloc[-1]) if bars is not None else None
            pnl = (exit_price - entry_price) * qty if exit_price is not None else None
            record_trade(symbol, qty, entry_price, exit_price, pnl, "TIMEOUT")
            return

        bars = safe_get_bars(symbol, limit=5)
        if bars is None:
            time.sleep(MONITOR_INTERVAL)
            continue
        last_price = float(bars["close"].iloc[-1])
        recent_atr = compute_atr(bars) or atr
        candidate_trail = last_price - recent_atr * ATR_TRAIL_MULT
        if candidate_trail > trail_stop:
            trail_stop = candidate_trail

        # trailing stop hit
        if last_price <= trail_stop:
            print(f"Trailing stop hit: {symbol} @ {last_price:.6f} (trail {trail_stop:.6f})")
            sell = submit_market_order(symbol, qty, "sell")
            exit_price = None
            if sell:
                try:
                    o = api.get_order(sell.id)
                    exit_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else None
                except Exception:
                    exit_price = None
            if exit_price is None:
                exit_price = last_price
            pnl = (exit_price - entry_price) * qty
            record_trade(symbol, qty, entry_price, exit_price, pnl, "TRAIL_SL")
            return

        # VWAP breakdown exit
        vwap_recent = compute_vwap(bars.tail(min(len(bars), 60)))
        if last_price < vwap_recent:
            print(f"VWAP breakdown exit: {symbol} price {last_price:.6f} vwap {vwap_recent:.6f}")
            sell = submit_market_order(symbol, qty, "sell")
            exit_price = None
            if sell:
                try:
                    o = api.get_order(sell.id)
                    exit_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else None
                except Exception:
                    exit_price = None
            if exit_price is None:
                exit_price = last_price
            pnl = (exit_price - entry_price) * qty
            record_trade(symbol, qty, entry_price, exit_price, pnl, "VWAP_EXIT")
            return

        time.sleep(MONITOR_INTERVAL)

# ---------------- Recording ----------------
def record_trade(symbol, qty, entry, exit_price, pnl, result, notes=""):
    ts = datetime.utcnow().isoformat()
    row = [ts, symbol, "LONG", qty, round(entry,6) if entry is not None else None,
           round(exit_price,6) if exit_price is not None else None,
           round(pnl,6) if pnl is not None else None, result, notes]
    append_trade_csv(row)
    tr = {"timestamp": ts, "symbol": symbol, "qty": qty, "entry": entry, "exit": exit_price, "pnl": pnl, "result": result, "notes": notes}
    update_stats(tr)
    print("Recorded trade:", tr)

# ---------------- Fallback Google Sheet (optional) ----------------
def fetch_sheet_signals():
    url = os.getenv("SIGNAL_SHEET_CSV_URL")
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "symbol" not in df.columns:
            return []
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        return df.to_dict("records")
    except Exception as e:
        print("fetch_sheet_signals error:", e)
        return []

# ---------------- Main run (guarantee 1 trade/day) ----------------
def run_once():
    # 1) check market is open
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            print(f"Market closed; exiting. ({clock.timestamp})")
            return None
    except Exception as e:
        print("Clock error:", e)
        return None

    traded = load_json(TRADED_DAY_FILE)
    today = now_et().strftime("%Y-%m-%d")
    if traded.get("date") == today:
        print("Already traded today. Exiting.")
        return None

    # 2) score all symbols
    candidates = []
    for s in SYMBOLS:
        try:
            info = score_symbol(s)
            if not info:
                continue
            # require MACD histogram > 0 (momentum)
            if not info["macd_ok"]:
                continue
            candidates.append(info)
        except Exception as e:
            print(f"score error {s}: {e}")

    # 3) fallback to sheet signals if none
    if not candidates:
        print("No primary candidates found; trying sheet fallback.")
        sheet = fetch_sheet_signals()
        for row in sheet:
            sym = row.get("symbol")
            if not sym:
                continue
            bars = safe_get_bars(sym, limit=20)
            if not bars:
                continue
            price = float(bars["close"].iloc[-1])
            atr = compute_atr(bars)
            qty = 1
            order = submit_market_order(sym, qty, "buy")
            if order:
                entry_price = None
                try:
                    o = api.get_order(order.id)
                    entry_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else price
                except Exception:
                    entry_price = price
                sl = entry_price - max(atr, entry_price * MIN_SL_PCT)
                monitor_position(sym, entry_price, qty, atr, sl)
                traded["date"] = today
                save_json(TRADED_DAY_FILE, traded)
                return True
        # maybe force trade later
    # If we have candidates, rank and choose best
    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[0]
        print(f"Top candidate: {top['symbol']} score {top['score']:.4f} price {top['price']:.4f} macd_hist {top['macd_hist']:.6f}")

        # adaptive acceptance
        vw_gap_pct = abs(top["price"] - top["vwap"]) / (top["vwap"] + EPS)
        adaptive_limit = max(0.005, (top["spread_avg"] / (top["vwap"] + EPS)) * 1.2)
        if vw_gap_pct > adaptive_limit:
            print(f"{top['symbol']} price-vwap {vw_gap_pct:.4f} > adaptive {adaptive_limit:.4f} -> skipping")
            if not GUARANTEE_TRADE:
                return None
            print("Guarantee ON: forcing top candidate")

        # sizing & order
        atr = top["atr"]
        sl_price = top["price"] - max(atr, top["price"] * MIN_SL_PCT)
        qty = compute_qty(top["price"], atr)
        if qty < 1:
            qty = 1

        tp_price = top["price"] + (top["price"] - sl_price) * 1.5
        bracket = submit_bracket_buy(top["symbol"], qty, sl_price, tp_price)
        entry_price = None
        if bracket:
            # read filled price if available
            try:
                o = api.get_order(bracket.id)
                entry_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else None
            except Exception:
                entry_price = None
        else:
            order = submit_market_order(top["symbol"], qty, "buy")
            if not order:
                print("Buy failed; aborting run.")
                return None
            try:
                o = api.get_order(order.id)
                entry_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else None
            except Exception:
                entry_price = None
            if entry_price is None:
                bars = safe_get_bars(top["symbol"], limit=1)
                entry_price = float(bars["close"].iloc[-1]) if bars is not None else top["price"]

        # monitor
        monitor_position(top["symbol"], entry_price, qty, atr, sl_price)
        traded["date"] = today
        save_json(TRADED_DAY_FILE, traded)
        return True

    # 4) If reached FORCE_TRADE time and guarantee ON, force top-ranked symbol (even if not matching all filters)
    now = now_et()
    if GUARANTEE_TRADE and (now.hour > FORCE_TRADE_HOUR or (now.hour == FORCE_TRADE_HOUR and now.minute >= FORCE_TRADE_MIN)):
        print("Force-trade window reached; selecting highest-liquidity symbol to force trade.")
        # simple selection: symbol with highest recent volume
        best = None
        best_vol = 0
        for s in SYMBOLS:
            df = safe_get_bars(s, limit=20)
            if df is None:
                continue
            vol = int(df["volume"].iloc[-1])
            if vol > best_vol:
                best_vol = vol
                best = s
        if best:
            bars = safe_get_bars(best, limit=20)
            price = float(bars["close"].iloc[-1])
            atr = compute_atr(bars)
            qty = compute_qty(price, atr)
            if qty < 1:
                qty = 1
            sl_price = price - max(atr, price * MIN_SL_PCT)
            order = submit_market_order(best, qty, "buy")
            entry_price = None
            if order:
                try:
                    o = api.get_order(order.id)
                    entry_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else price
                except Exception:
                    entry_price = price
                monitor_position(best, entry_price, qty, atr, sl_price)
                traded["date"] = today
                save_json(TRADED_DAY_FILE, traded)
                return True

    print("No trade executed this run.")
    return None

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    print("Run start ET", now_et().isoformat())
    try:
        r = run_once()
        if r:
            print("Trade executed / recorded.")
        else:
            print("No trade executed this run.")
    except Exception as e:
        print("Unhandled error:", e)
