# hybrid_tradebot_advanced_alpacapy.py
"""
Advanced hybrid trading bot using alpaca-py (TradingClient).
- Adaptive ATR/VWAP-based filters
- Volume-weighted scoring and multi-symbol ranking
- Google Sheets fallback (CSV export URL via SIGNAL_SHEET_CSV_URL)
- Optional guarantee: force top-ranked trade once per run/day
- In-process monitor for TP/SL (uses yfinance 1m)
- Robust conversions and tolerant CSV parsing
- Logs trades to trades.csv and stats to trade_stats.json
- Uses environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY, SIGNAL_SHEET_CSV_URL
"""

import os
import time
import json
import math
import logging
from io import StringIO
from datetime import datetime
import pytz

import pandas as pd
import numpy as np
import yfinance as yf
import requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -----------------------------
# CONFIG
# -----------------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
CANDLE_INTERVAL = "5m"
LOOKBACK_DAYS = 3

# default exit targets (percent)
DEFAULT_TP_PCT = 0.005   # 0.5%
DEFAULT_SL_PCT = 0.003   # 0.3%

# adaptive multipliers
VWAP_VOLATILITY_MULTIPLIER = 0.7
MAX_SPREAD_MULTIPLIER = 1.5
MIN_VOLUME_MULTIPLIER = 0.5

# scoring weights
SCORE_WEIGHT_VWAP = 0.45
SCORE_WEIGHT_VOLUME = 0.30
SCORE_WEIGHT_SPREAD = 0.25

# fallback sheet (env)
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL", "").strip()

# guarantee trade if nothing else (per run or per-day logic can be added)
GUARANTEE_ONE_TRADE_PER_DAY = True
FALLBACK_TIME_ET = 14  # hour in ET when sheet fallback is allowed

# Alpaca (env)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # set to False for live

# files & monitoring
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
LOG_FILE = "bot.log"
MONITOR_INTERVAL = 15  # seconds for 1m polling
MONITOR_TIMEOUT = 60 * 60  # 1 hour

# -----------------------------
# VALIDATION
# -----------------------------
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required.")

# -----------------------------
# logging
# -----------------------------
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting hybrid_tradebot_advanced_alpacapy")

# -----------------------------
# Alpaca client
# -----------------------------
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# -----------------------------
# Utilities
# -----------------------------
def now_et():
    return datetime.now(pytz.timezone("America/New_York"))

def safe_float(x):
    """Convert pandas scalar/Series/numpy scalar/python scalar to float safely."""
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        try:
            return float(getattr(x, "item", lambda: x)())
        except Exception:
            raise ValueError(f"Could not convert value to float: {x!r}")

def backoff_attempt(func, *args, max_attempts=5, base_delay=1.0, **kwargs):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                logging.error(f"Backoff failed after {attempt} attempts: {e}")
                raise
            delay = base_delay * (2 ** (attempt - 1)) + 0.1 * attempt
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s")
            time.sleep(delay)

# -----------------------------
# Logs & stats helpers
# -----------------------------
def ensure_trade_log():
    if not os.path.exists(TRADES_CSV):
        df = pd.DataFrame(columns=[
            "timestamp", "source", "symbol", "side", "qty",
            "entry_price", "exit_price", "pnl", "result", "notes",
            "tp_price", "sl_price"
        ])
        df.to_csv(TRADES_CSV, index=False)

def append_trade_log(trade: dict):
    ensure_trade_log()
    pd.DataFrame([trade]).to_csv(TRADES_CSV, mode="a", header=False, index=False)
    # update stats
    stats = {}
    if os.path.exists(STATS_JSON):
        try:
            with open(STATS_JSON, "r") as f:
                stats = json.load(f)
        except Exception:
            stats = {}
    sym = trade["symbol"]
    if sym not in stats:
        stats[sym] = {"trades": 0, "wins": 0, "losses": 0}
    stats[sym]["trades"] += 1
    if trade.get("result") == "WIN":
        stats[sym]["wins"] += 1
    elif trade.get("result") == "LOSS":
        stats[sym]["losses"] += 1
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

# -----------------------------
# Market data helpers
# -----------------------------
def get_recent_bars(symbol, period_days=LOOKBACK_DAYS, interval=CANDLE_INTERVAL):
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        logging.error(f"yfinance error fetching {symbol}: {e}")
        return pd.DataFrame()

def compute_atr(df, n=14):
    if df.empty or len(df) < 2:
        return None
    tr = (df["High"] - df["Low"]).abs()
    atr = tr.tail(n).mean()
    return safe_float(atr)

def compute_vwap(df):
    if df.empty:
        return None
    return (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

def compute_recent_volume_median(df):
    if df.empty:
        return None
    return safe_float(df["Volume"].tail(20).median())

# -----------------------------
# Candidate scoring
# -----------------------------
def score_candidate(symbol):
    df = get_recent_bars(symbol)
    if df.empty:
        logging.info(f"No data for {symbol}")
        return None
    vwap_series = compute_vwap(df)
    if vwap_series is None:
        return None
    df = df.copy()
    df["VWAP"] = vwap_series
    last = df.iloc[-1]
    try:
        price = safe_float(last["Close"])
        vwap = safe_float(last["VWAP"])
        volume = safe_float(last["Volume"])
        spread = safe_float(last["High"]) - safe_float(last["Low"])
    except Exception as e:
        logging.error(f"safe_float error for {symbol}: {e}")
        return None

    atr = compute_atr(df, n=14) or max(0.01, abs(price) * 0.0005)
    atr = max(0.0001, atr)

    allowed_vwap_dist = VWAP_VOLATILITY_MULTIPLIER * atr
    allowed_spread = MAX_SPREAD_MULTIPLIER * atr
    median_vol = compute_recent_volume_median(df) or 1000.0
    min_volume = MIN_VOLUME_MULTIPLIER * median_vol

    # normalized metrics
    vwap_dist = abs(price - vwap)
    vwap_score = max(0.0, 1.0 - min(vwap_dist / (allowed_vwap_dist * 2 + 1e-9), 1.0))

    volume_ratio = volume / (median_vol + 1e-9)
    if volume_ratio <= 0.5:
        volume_score = 0.0
    else:
        volume_score = min(1.0, (volume_ratio - 0.5) / 1.5)

    spread_score = max(0.0, 1.0 - min(spread / (allowed_spread * 2 + 1e-9), 1.0))

    combined_score = (SCORE_WEIGHT_VWAP * vwap_score +
                      SCORE_WEIGHT_VOLUME * volume_score +
                      SCORE_WEIGHT_SPREAD * spread_score)

    logging.info(f"{symbol} score {combined_score:.4f} (vwap {vwap_score:.3f} vol {volume_score:.3f} spread {spread_score:.3f})")
    return {
        "symbol": symbol,
        "price": price,
        "vwap": vwap,
        "volume": volume,
        "median_volume": median_vol,
        "spread": spread,
        "atr": atr,
        "allowed_vwap_dist": allowed_vwap_dist,
        "allowed_spread": allowed_spread,
        "min_volume": min_volume,
        "vwap_score": vwap_score,
        "volume_score": volume_score,
        "spread_score": spread_score,
        "combined_score": combined_score
    }

# -----------------------------
# Trading helpers (alpaca-py)
# -----------------------------
def submit_market_order(symbol, qty, side_str):
    side = str(side_str).strip().upper()
    if side == "BUY":
        side_enum = OrderSide.BUY
    elif side == "SELL":
        side_enum = OrderSide.SELL
    else:
        logging.error(f"Invalid side: {side_str}")
        return None
    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=TimeInForce.DAY)
    try:
        order = backoff_attempt(client.submit_order, req)
        logging.info(f"Submitted {side} {qty} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Submit market order failed {symbol} {side}: {e}")
        return None

def get_filled_price_from_order_obj(order_obj):
    # alpaca-py order object fields vary; attempt common attributes
    for attr in ("filled_avg_price", "filled_avg", "filled_avgprice", "filled_price"):
        try:
            val = getattr(order_obj, attr, None)
            if val is not None:
                return safe_float(val)
        except Exception:
            continue
    # fallback: poll order by id
    order_id = getattr(order_obj, "id", None) or getattr(order_obj, "order_id", None)
    if order_id:
        try:
            o = backoff_attempt(client.get_order, order_id)
            val = getattr(o, "filled_avg_price", None) or getattr(o, "filled_avg", None)
            if val is not None:
                return safe_float(val)
        except Exception:
            pass
    return None

def monitor_and_exit(symbol, qty, entry_price, tp_price, sl_price):
    start = time.time()
    exit_price = None
    result = None
    notes = ""
    logging.info(f"Monitoring {symbol} entry {entry_price:.4f} TP {tp_price:.4f} SL {sl_price:.4f}")
    last_px = entry_price
    while True:
        if time.time() - start > MONITOR_TIMEOUT:
            notes = "timeout"
            result = "TIMEOUT"
            break
        try:
            recent = yf.download(symbol, period="2d", interval="1m", progress=False, auto_adjust=True)
            if recent.empty:
                time.sleep(MONITOR_INTERVAL)
                continue
            last_px = safe_float(recent["Close"].iloc[-1])
        except Exception as e:
            logging.warning(f"Monitor fetch error {e}")
            time.sleep(MONITOR_INTERVAL)
            continue

        logging.info(f"Monitor {symbol} price {last_px:.4f}")
        if last_px >= tp_price:
            exit_price = last_px
            result = "WIN"
            notes = "tp_hit"
            break
        if last_px <= sl_price:
            exit_price = last_px
            result = "LOSS"
            notes = "sl_hit"
            break

        time.sleep(MONITOR_INTERVAL)

    # execute market sell
    sell_order = submit_market_order(symbol, qty, "SELL")
    if sell_order:
        filled = get_filled_price_from_order_obj(sell_order)
        if filled:
            exit_price = filled

    if exit_price is None:
        exit_price = last_px or entry_price

    pnl = (exit_price - entry_price) * qty
    trade = {
        "timestamp": datetime.now(pytz.utc).isoformat(),
        "source": "PRIMARY",
        "symbol": symbol,
        "side": "LONG",
        "qty": qty,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": round(pnl, 4),
        "result": result,
        "notes": notes,
        "tp_price": tp_price,
        "sl_price": sl_price
    }
    append_trade_log(trade)
    logging.info(f"Trade closed {trade}")
    return trade

# -----------------------------
# Sheet helpers (robust)
# -----------------------------
def fetch_signals_from_sheet(url):
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch sheet: {e}")
        return []

# -----------------------------
# Candidate selection & execution
# -----------------------------
def choose_best_candidates(symbols):
    metrics = []
    for s in symbols:
        try:
            m = score_candidate(s)
            if m:
                metrics.append(m)
        except Exception as e:
            logging.error(f"Error scoring {s}: {e}")
    metrics.sort(key=lambda x: x["combined_score"], reverse=True)
    return metrics

def execute_candidate(candidate, force=False):
    symbol = candidate["symbol"]
    price = candidate["price"]
    qty = 1
    tp_price = price * (1 + DEFAULT_TP_PCT)
    sl_price = price * (1 - DEFAULT_SL_PCT)

    if not force:
        if candidate["volume"] < candidate["min_volume"]:
            logging.info(f"{symbol} volume {candidate['volume']} < min {candidate['min_volume']:.1f}")
            return None
        if candidate["spread"] > candidate["allowed_spread"]:
            logging.info(f"{symbol} spread {candidate['spread']:.4f} > allowed {candidate['allowed_spread']:.4f}")
            return None
        if abs(price - candidate["vwap"]) > (candidate["allowed_vwap_dist"] * 2):
            logging.info(f"{symbol} price-vwap {abs(price-candidate['vwap']):.4f} too far")
            return None

    buy_order = submit_market_order(symbol, qty, "BUY")
    if not buy_order:
        logging.error(f"Buy order failed for {symbol}")
        return None

    entry_price = get_filled_price_from_order_obj(buy_order) or price
    logging.info(f"{symbol} buy filled at {entry_price:.4f}")
    return monitor_and_exit(symbol, qty, entry_price, tp_price, sl_price)

# -----------------------------
# Orchestrator
# -----------------------------
def main():
    logging.info(f"Run start ET {now_et().isoformat()}")
    candidates = choose_best_candidates(SYMBOLS)
    if candidates:
        top = candidates[0]
        logging.info(f"Top candidate: {top['symbol']} score {top['combined_score']:.4f}")
        trade = execute_candidate(top, force=False)
        if trade:
            logging.info("Primary trade executed.")
            return
        # try next two
        for cand in candidates[1:3]:
            trade = execute_candidate(cand, force=False)
            if trade:
                logging.info("Primary trade executed on lower-ranked candidate.")
                return

    # fallback sheet signals if after time threshold
    if now_et().hour >= FALLBACK_TIME_ET:
        logging.info("Checking fallback sheet signals")
        signals = fetch_signals_from_sheet(SIGNAL_SHEET_CSV_URL)
        for sig in signals:
            try:
                symbol = str(sig.get("symbol","")).upper()
                action = str(sig.get("action","BUY")).upper()
                qty = int(sig.get("qty", 1))
                # map action to alpaca-py side
                if action not in ("BUY","SELL"):
                    logging.info(f"Unsupported action in sheet: {action}")
                    continue
                order = submit_market_order(symbol, qty, action)
                if not order:
                    continue
                entry = get_filled_price_from_order_obj(order) or 0.0
                tp_price = entry * (1 + DEFAULT_TP_PCT)
                sl_price = entry * (1 - DEFAULT_SL_PCT)
                tp_val = sig.get("take_profit", "")
                sl_val = sig.get("stop_loss", "")
                try:
                    if tp_val not in [None, ""]:
                        tp_num = float(tp_val)
                        if tp_num <= 1:
                            tp_price = entry * (1 + tp_num)
                        else:
                            tp_price = tp_num
                except Exception:
                    pass
                try:
                    if sl_val not in [None, ""]:
                        sl_num = float(sl_val)
                        if sl_num <= 1:
                            sl_price = entry * (1 - sl_num)
                        else:
                            sl_price = sl_num
                except Exception:
                    pass
                monitor_and_exit(symbol, qty, entry, tp_price, sl_price)
                logging.info(f"Executed sheet signal for {symbol}")
                return
            except Exception as e:
                logging.error(f"Error executing sheet signal: {e}")

    # guarantee mode: force top candidate
    if GUARANTEE_ONE_TRADE_PER_DAY and candidates:
        logging.info("Forcing top-ranked candidate due to guarantee flag.")
        if execute_candidate(candidates[0], force=True):
            logging.info("Forced trade executed.")
            return

    logging.info("Run complete. No trade executed this run.")

if __name__ == "__main__":
    main()
