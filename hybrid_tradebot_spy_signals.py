# hybrid_tradebot_advanced.py
"""
Advanced hybrid trading bot — corrected and hardened.

Features:
- Multi-symbol adaptive filtering (ATR-based)
- Volume-weighted probability scoring and ranking
- Optional guaranteed trade per day (force top candidate)
- Google Sheet fallback (CSV export URL)
- Robust Alpaca order submission with backoff
- In-process monitoring for TP/SL and market exits
- CSV trade logging and per-symbol stats (trade_stats.json)
- Uses .iloc[...] and robust conversions to avoid pandas FutureWarning / Series truth errors
"""

import os
import time
import math
import json
import logging
from datetime import datetime
from io import StringIO

import pytz
import requests
import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -------------------------
# CONFIG
# -------------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
CANDLE_INTERVAL = "5m"
LOOKBACK_DAYS = 3

DEFAULT_TP_PCT = 0.005     # 0.5%
DEFAULT_SL_PCT = 0.003     # 0.3%

# Adaptive multipliers
VWAP_VOLATILITY_MULTIPLIER = 0.7
MAX_SPREAD_MULTIPLIER = 1.5
MIN_VOLUME_MULTIPLIER = 0.5

# Scoring weights
SCORE_WEIGHT_VWAP = 0.45
SCORE_WEIGHT_VOLUME = 0.30
SCORE_WEIGHT_SPREAD = 0.25

# Fallback sheet (CSV export URL)
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL", "").strip()

# Guarantee one trade per day (force top candidate)
GUARANTEE_ONE_TRADE_PER_DAY = True

# Fallback time (ET hour)
FALLBACK_TIME_ET = 14

# Alpaca config (env)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # set False for live

# Files & monitoring
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
LOG_FILE = "bot.log"
MONITOR_INTERVAL = 15
MONITOR_TIMEOUT = 60 * 60  # 1 hour

# -------------------------
# VALIDATION
# -------------------------
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required.")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting hybrid_tradebot_advanced (corrected)")

# -------------------------
# Alpaca client
# -------------------------
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# -------------------------
# Utilities
# -------------------------
def now_et():
    return datetime.now(pytz.timezone("America/New_York"))

def safe_float(x):
    """Robust conversion to float for pandas scalars/Series/numpy types/py scalars."""
    try:
        # pandas Series or Index — return first element
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        # numpy scalar or python scalar
        return float(x)
    except Exception:
        # as last resort, try item()
        try:
            return float(getattr(x, "item", lambda: x)())
        except Exception:
            raise ValueError(f"Could not convert {type(x)} value to float: {x}")

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

# -------------------------
# Logging / stats helpers
# -------------------------
def ensure_logs():
    if not os.path.exists(TRADES_CSV):
        df = pd.DataFrame(columns=[
            "timestamp", "source", "symbol", "side", "qty",
            "entry_price", "exit_price", "pnl", "result", "notes",
            "tp_price", "sl_price"
        ])
        df.to_csv(TRADES_CSV, index=False)

def append_trade_log(trade: dict):
    ensure_logs()
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

# -------------------------
# Market data & metrics
# -------------------------
def get_recent_bars(symbol, period_days=LOOKBACK_DAYS, interval=CANDLE_INTERVAL):
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        logging.error(f"yfinance fetch error {symbol}: {e}")
        return pd.DataFrame()

def compute_atr(df, n=14):
    if df.empty or len(df) < 2:
        return None
    high = df["High"]
    low = df["Low"]
    tr = (high - low).abs()
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

# -------------------------
# Scoring
# -------------------------
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

# -------------------------
# Trading helpers
# -------------------------
def submit_market_order(symbol, qty, side):
    try:
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY)
        order = backoff_attempt(client.submit_order, req)
        logging.info(f"Submitted {side} {qty} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Submit market order failed {symbol}: {e}")
        return None

def get_filled_price_from_order(order):
    # Try multiple attributes depending on client shape
    for attr in ("filled_avg_price", "filled_avg", "filled_avgprice", "filled_price"):
        try:
            val = getattr(order, attr, None)
            if val is not None:
                return safe_float(val)
        except Exception:
            continue
    # Some objects store fills in a dict-like field
    try:
        if hasattr(order, "fills") and order.fills:
            first = order.fills[0]
            if isinstance(first, dict) and "price" in first:
                return safe_float(first["price"])
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

    # Execute sell market
    sell_order = submit_market_order(symbol, qty, "SELL")
    if sell_order:
        fp = get_filled_price_from_order(sell_order)
        if fp:
            exit_price = fp

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

# -------------------------
# Sheet helpers
# -------------------------
def fetch_signals_from_sheet(url):
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch sheet: {e}")
        return []

# -------------------------
# Candidate selection & execution
# -------------------------
def choose_best_candidates(symbols):
    metrics = []
    for s in symbols:
        try:
            m = score_candidate(s)
            if m:
                metrics.append(m)
        except Exception as e:
            logging.error(f"Error scoring {s}: {e}")
    if not metrics:
        return []
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
        logging.error(f"Buy order failed {symbol}")
        return None

    entry_price = get_filled_price_from_order(buy_order) or price
    logging.info(f"{symbol} buy filled at {entry_price:.4f}")
    return monitor_and_exit(symbol, qty, entry_price, tp_price, sl_price)

# -------------------------
# Orchestrator
# -------------------------
def main():
    logging.info(f"Run start ET {now_et().isoformat()}")
    candidates = choose_best_candidates(SYMBOLS)
    if candidates:
        top = candidates[0]
        logging.info(f"Top candidate: {top['symbol']} score {top['combined_score']:.4f} (vwap {top['vwap_score']:.3f} vol {top['volume_score']:.3f} spread {top['spread_score']:.3f})")
        trade = execute_candidate(top, force=False)
        if trade:
            logging.info("Primary trade executed.")
            return
        # try next two
        for cand in candidates[1:3]:
            trade = execute_candidate(cand, force=False)
            if trade:
                logging.info("Trade executed on lower-ranked candidate.")
                return

    # fallback sheet if after time threshold
    if now_et().hour >= FALLBACK_TIME_ET:
        logging.info("Checking fallback sheet signals")
        signals = fetch_signals_from_sheet(SIGNAL_SHEET_CSV_URL)
        for sig in signals:
            try:
                symbol = str(sig.get("symbol","")).upper()
                action = str(sig.get("action","BUY")).upper()
                qty = int(sig.get("qty", 1))
                order = submit_market_order(symbol, qty, action)
                if not order:
                    continue
                entry = get_filled_price_from_order(order) or 0.0
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

    # guarantee trade if enabled
    if GUARANTEE_ONE_TRADE_PER_DAY and candidates:
        logging.info("No primary/sheet trade executed; forcing top-ranked candidate due to guarantee flag.")
        if execute_candidate(candidates[0], force=True):
            logging.info("Forced trade executed.")
            return

    logging.info("Run complete. No trade executed this run.")

if __name__ == "__main__":
    main()
