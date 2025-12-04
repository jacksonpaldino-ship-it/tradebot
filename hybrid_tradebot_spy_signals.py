#!/usr/bin/env python3
"""
hybrid_tradebot_spy_signals.py

- Keeps your buy logic (VWAP/volume/spread scoring + forced top candidate)
- Adds sell logic: bracket TP/SL (preferred) + VWAP-break exit + monitored fallback
- Sizing computed from account equity & risk per trade
- One entry per symbol per calendar day (persisted)
- Only runs during market hours (uses Alpaca clock)
- Logs trades and stats
"""

import os
import time
import math
import csv
import json
from datetime import datetime, timedelta
import pytz

import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame, APIError

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]      # your universe
TRADE_QTY_MIN = 1
RISK_PER_TRADE = 0.002                     # 0.2% of equity per trade for sizing
DEFAULT_SL_PCT = 0.003                     # 0.3% stop loss (fallback)
DEFAULT_TP_MULT = 1.6                      # take-profit = SL * TP_MULT (approx 1.6R)
GUARANTEE_TRADE = True                     # force top candidate if nothing passes filters
VWAP_MAX_PCT = 0.02                        # baseline allowed deviation for scoring
MONITOR_INTERVAL = 6                       # seconds between checks when monitoring
MONITOR_TIMEOUT = 60 * 30                  # 30 minutes max monitor after entry
TRADED_DAY_FILE = "traded_today.json"
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- SECRETS (your current names) ----------------
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. https://paper-api.alpaca.markets

if not (API_KEY and API_SECRET and BASE_URL):
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

# Alpaca client
api = REST(API_KEY, API_SECRET, BASE_URL)

# ---------------- persistence helpers ----------------
def read_traded_today():
    if os.path.exists(TRADED_DAY_FILE):
        try:
            with open(TRADED_DAY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def write_traded_today(d):
    with open(TRADED_DAY_FILE, "w") as f:
        json.dump(d, f)

def append_trade(trade):
    header = ["timestamp", "symbol", "side", "qty", "entry", "exit", "pnl", "result", "notes"]
    exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            trade.get("timestamp"), trade.get("symbol"), trade.get("side"),
            trade.get("qty"), trade.get("entry"), trade.get("exit"),
            trade.get("pnl"), trade.get("result"), trade.get("notes", "")
        ])
    # update stats
    stats = {}
    if os.path.exists(STATS_JSON):
        try:
            with open(STATS_JSON, "r") as f:
                stats = json.load(f)
        except Exception:
            stats = {}
    s = stats.get(trade["symbol"], {"wins":0,"losses":0,"trades":0,"pnl":0.0})
    s["trades"] += 1
    pnl = trade.get("pnl") or 0.0
    s["pnl"] = round(s.get("pnl",0.0) + pnl, 6)
    if trade.get("result") == "WIN":
        s["wins"] += 1
    elif trade.get("result") == "LOSS":
        s["losses"] += 1
    stats[trade["symbol"]] = s
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

# ---------------- data helpers ----------------
def now_et():
    return datetime.now(TZ)

def safe_get_bars(symbol, limit=120):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
        if bars is None or bars.empty:
            return None
        if 'symbol' in bars.columns:
            bars = bars[bars['symbol'] == symbol]
            if bars.empty:
                return None
        # ensure columns include close, high, low, volume
        return bars
    except Exception as e:
        print(f"safe_get_bars error {symbol}: {e}")
        return None

def compute_vwap(df):
    # robust VWAP over df
    try:
        v = (df['close'] * df['volume']).sum()
        vv = df['volume'].sum()
        if vv <= 0:
            return float(df['close'].iloc[-1])
        return float(v / vv)
    except Exception:
        return float(df['close'].iloc[-1])

# ---------------- scoring (your buy logic) ----------------
def score_symbol(symbol):
    df = safe_get_bars(symbol, limit=120)
    if df is None or df.empty:
        return None
    window = min(len(df), 60)
    dfw = df.tail(window)
    vwap = compute_vwap(dfw)
    price = float(dfw['close'].iloc[-1])
    vol = float(dfw['volume'].iloc[-1])
    high = float(dfw['high'].iloc[-1])
    low = float(dfw['low'].iloc[-1])
    spread = max(high - low, EPS)
    spread_avg = float((dfw['high'] - dfw['low']).mean() if len(dfw) > 1 else spread)
    vol_avg = float(dfw['volume'].mean() if len(dfw) > 1 else vol)
    vw_gap = abs(price - vwap) / (vwap + EPS)
    vw_score = max(0.0, 1.0 - vw_gap / (VWAP_MAX_PCT + EPS))
    vol_score = min(1.0, vol / (vol_avg + EPS))
    spread_score = max(0.0, 1.0 - (spread / (spread_avg + EPS)))
    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * spread_score
    return {"symbol": symbol, "score": float(score), "price": price, "vwap": vwap,
            "volume": vol, "spread": spread, "spread_avg": spread_avg, "vol_avg": vol_avg}

# ---------------- sizing ----------------
def account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        print("account_equity error:", e)
        return None

def compute_qty(entry_price, sl_price):
    equity = account_equity()
    if equity is None:
        return TRADE_QTY_MIN
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(0.01, abs(entry_price - sl_price))
    qty = int(max(TRADE_QTY_MIN, math.floor(risk_amount / (per_share_risk + EPS))))
    return qty

# ---------------- order helpers ----------------
def submit_market_buy_with_bracket(symbol, qty, sl_price, tp_price):
    """
    Try bracket order (Alpaca) with take_profit and stop_loss.
    Returns order object or None.
    """
    try:
        # note: alpaca_trade_api expects dicts for stop and take
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(round(tp_price, 6))},
            stop_loss={"stop_price": str(round(sl_price, 6))}
        )
        return order
    except APIError as e:
        print(f"Bracket order APIError: {e}")
        # fall through to manual if bracket unsupported
        return None
    except Exception as e:
        print(f"Bracket order failed: {e}")
        return None

def submit_market_buy(symbol, qty):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
        return order
    except Exception as e:
        print("submit_market_buy error:", e)
        return None

def submit_market_sell(symbol, qty):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
        return order
    except Exception as e:
        print("submit_market_sell error:", e)
        return None

# ---------------- monitoring & exit rules ----------------
def get_order_filled_price(order):
    try:
        # order.filled_avg_price may be None initially; fetch fresh
        o = api.get_order(order.id)
        if getattr(o, "filled_avg_price", None):
            return float(o.filled_avg_price)
        # fallback: check latest bar
        bars = safe_get_bars(o.symbol, limit=1)
        if bars is not None:
            return float(bars['close'].iloc[-1])
    except Exception:
        pass
    return None

def monitor_position_simple(symbol, entry_price, qty, sl_price, tp_price, bracket_used):
    """
    If bracket order used, Alpaca will manage TP/SL server-side; we monitor to log fills.
    If bracket not used, we monitor price and execute market sells on conditions:
      - price <= sl_price (stop)
      - price >= tp_price (take profit)
      - price < VWAP (VWAP-break exit)
      - timeout -> market exit
    """
    start = time.time()
    while time.time() - start < MONITOR_TIMEOUT:
        bars = safe_get_bars(symbol, limit=5)
        if bars is None:
            time.sleep(MONITOR_INTERVAL)
            continue
        last_price = float(bars['close'].iloc[-1])
        vwap = compute_vwap(bars.tail(min(len(bars), 60)))
        # check VWAP breakdown exit
        if last_price < vwap:
            print(f"{symbol} VWAP breakdown exit at {last_price:.6f} (vwap {vwap:.6f})")
            sell = submit_market_sell(symbol, qty)
            exit_price = None
            if sell:
                # try to get fill price
                try:
                    o = api.get_order(sell.id)
                    exit_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
                except Exception:
                    exit_price = last_price
            else:
                exit_price = last_price
            pnl = (exit_price - entry_price) * qty if exit_price is not None else None
            rec = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "sell",
                   "qty": qty, "entry": entry_price, "exit": exit_price, "pnl": pnl, "result": "VWAP_EXIT", "notes": ""}
            append_trade(rec)
            return rec

        # check TP/SL if bracket not used
        if not bracket_used:
            if last_price >= tp_price:
                print(f"{symbol} hit TP at {last_price:.6f}")
                sell = submit_market_sell(symbol, qty)
                exit_price = None
                if sell:
                    try:
                        o = api.get_order(sell.id)
                        exit_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else last_price
                    except Exception:
                        exit_price = last_price
                pnl = (exit_price - entry_price) * qty if exit_price is not None else None
                rec = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "sell",
                       "qty": qty, "entry": entry_price, "exit": exit_price, "pnl": pnl, "result": "WIN", "notes": "TP"}
                append_trade(rec)
                return rec
            if last_price <= sl_price:
                print(f"{symbol} hit SL at {last_price:.6f}")
                sell = submit_market_sell(symbol, qty)
                exit_price = None
                if sell:
                    try:
                        o = api.get_order(sell.id)
                        exit_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else last_price
                    except Exception:
                        exit_price = last_price
                pnl = (exit_price - entry_price) * qty if exit_price is not None else None
                rec = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "sell",
                       "qty": qty, "entry": entry_price, "exit": exit_price, "pnl": pnl, "result": "LOSS", "notes": "SL"}
                append_trade(rec)
                return rec
        # else if bracket used, check if position still open (Alpaca will have closed automatically)
        else:
            # check positions
            try:
                positions = api.list_positions()
                symbols = [p.symbol for p in positions]
                if symbol not in symbols:
                    # position closed; find last trade fill price from orders or last trade
                    bars = safe_get_bars(symbol, limit=1)
                    exit_price = float(bars['close'].iloc[-1]) if bars is not None else None
                    # attempt to infer result by comparing exit to entry
                    pnl = (exit_price - entry_price) * qty if exit_price is not None else None
                    result = "WIN" if exit_price and exit_price > entry_price else "LOSS"
                    rec = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "sell",
                           "qty": qty, "entry": entry_price, "exit": exit_price, "pnl": pnl, "result": result, "notes": "bracket_filled"}
                    append_trade(rec)
                    return rec
            except Exception:
                pass

        time.sleep(MONITOR_INTERVAL)

    # timeout: try to exit market
    print("Monitor timeout; attempting market exit")
    sell = submit_market_sell(symbol, qty)
    exit_price = None
    if sell:
        try:
            o = api.get_order(sell.id)
            exit_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
        except Exception:
            pass
    bars = safe_get_bars(symbol, limit=1)
    if exit_price is None and bars is not None:
        exit_price = float(bars['close'].iloc[-1])
    pnl = (exit_price - entry_price) * qty if exit_price is not None else None
    rec = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "side": "sell",
           "qty": qty, "entry": entry_price, "exit": exit_price, "pnl": pnl, "result": "TIMEOUT", "notes": ""}
    append_trade(rec)
    return rec

# ---------------- fallback google sheet ----------------
def fetch_sheet_signals():
    url = os.getenv("SIGNAL_SHEET_CSV_URL")
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        if "symbol" not in df.columns:
            return []
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        return df.to_dict("records")
    except Exception as e:
        print("fetch_sheet_signals error:", e)
        return []

# ---------------- main run ----------------
def run_once():
    # only trade during market open
    try:
        clock = api.get_clock()
        if not clock.is_open:
            print("Market closed; exiting.")
            return None
    except Exception as e:
        print("Failed to get clock:", e)
        return None

    traded = read_traded_today()
    today = now_et().strftime("%Y-%m-%d")
    if traded.get("date") == today:
        print("Guaranteed trade already executed today; exiting.")
        return None

    # score symbols (your buy logic)
    scored = []
    for s in SYMBOLS:
        try:
            data = score_symbol(s)
            if data:
                scored.append(data)
        except Exception as e:
            print("score_symbol error", s, e)
    if not scored:
        print("No symbols scored; trying sheet fallback")
        sheet_res = fetch_sheet_signals()
        for row in sheet_res:
            sym = row.get("symbol")
            if not sym:
                continue
            bars = safe_get_bars(sym, limit=5)
            if not bars:
                continue
            price = float(bars['close'].iloc[-1])
            qty = int(row.get("qty", 1))
            order = submit_market_buy(sym, qty)
            if order:
                traded["date"] = today
                write_traded_today(traded)
                # compute basic sl/tp
                sl = price * (1 - DEFAULT_SL_PCT)
                tp = price + (price - sl) * DEFAULT_TP_MULT
                return monitor_position_simple(sym, price, qty, sl, tp, bracket_used=False)
        return None

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[0]
    print(f"Top candidate: {top['symbol']} score {top['score']:.4f} price {top['price']:.4f} vwap {top['vwap']:.4f}")

    # adaptive VWAP filter
    vw_gap_pct = abs(top['price'] - top['vwap']) / (top['vwap'] + EPS)
    adaptive_vwap_pct = max(0.005, (top['spread_avg'] / (top['vwap'] + EPS)) * 1.2)
    if vw_gap_pct > adaptive_vwap_pct and not GUARANTEE_TRADE:
        print(f"{top['symbol']} price too far from VWAP ({vw_gap_pct:.4f} > {adaptive_vwap_pct:.4f}), skipping")
        return None

    # calculate provisional SL (based on estimated volatility)
    est_risk_per_share = max(0.01, abs(top['price'] - top['vwap']), top['spread'] * 0.6)
    sl_price = top['price'] - est_risk_per_share
    tp_price = top['price'] + est_risk_per_share * DEFAULT_TP_MULT
    qty = compute_qty(top['price'], sl_price)
    if qty < TRADE_QTY_MIN:
        qty = TRADE_QTY_MIN

    # try bracket order first
    bracket_order = submit_market_buy_with_bracket(top['symbol'], qty, sl_price, tp_price)
    entry_price = None
    bracket_used = False
    if bracket_order:
        bracket_used = True
        # get filled price (best-effort)
        try:
            filled = get_order = api.get_order(bracket_order.id)
            entry_price = float(get_order.filled_avg_price) if getattr(get_order, "filled_avg_price", None) else None
        except Exception:
            entry_price = None
    else:
        # fallback to market buy then manual TP/SL
        order = submit_market_buy(top['symbol'], qty)
        if not order:
            print("Market buy failed")
            return None
        try:
            o = api.get_order(order.id)
            entry_price = float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) else None
        except Exception:
            entry_price = None
        if entry_price is None:
            bars = safe_get_bars(top['symbol'], limit=1)
            entry_price = float(bars['close'].iloc[-1]) if bars is not None else top['price']

    # monitor & manage
    rec = monitor_position_simple(top['symbol'], entry_price, qty, sl_price, tp_price, bracket_used)
    traded["date"] = today
    write_traded_today(traded)
    return rec

# ---------------- entrypoint ----------------
if __name__ == "__main__":
    print("Run start ET", now_et().isoformat())
    try:
        res = run_once()
        if res:
            print("Trade result:", res)
        else:
            print("No trade executed this run.")
    except Exception as exc:
        print("Unhandled exception:", exc)
