# hybrid_tradebot_advanced.py
"""
Advanced hybrid trading bot
- Multi-symbol primary strategy with adaptive thresholds (ATR/volatility-based)
- Volume-weighted probability scoring to rank symbols
- Google Sheet fallback (optional) after fallback time
- Guaranteed trade option (force top-ranked candidate if nothing passes)
- Stop-loss / take-profit per trade (defaults or from sheet)
- Robust Alpaca order submission with backoff/retry
- CSV logging (trades.csv) and per-symbol stats (trade_stats.json)
- Avoids pandas FutureWarnings by using .iloc[0] where appropriate

Config section below - tune to your account and risk profile.
"""

import os
import time
import math
import json
import logging
from datetime import datetime
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
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]  # symbols to evaluate
LOOKBACK_MINUTES = 60 * 6  # lookback in minutes for volatility/volume (e.g., 6 hours)
CANDLE_INTERVAL = "5m"     # interval for data used in signals

# Base (fallback) percent targets - used to compute TP/SL from entry price
DEFAULT_TP_PCT = 0.005     # 0.5% take profit
DEFAULT_SL_PCT = 0.003     # 0.3% stop loss

# Volatility multiplier controls aggressiveness of thresholds
VWAP_VOLATILITY_MULTIPLIER = 0.7  # multiplies ATR to allow VWAP distance
MAX_SPREAD_MULTIPLIER = 1.5       # allows up to this * ATR for candle spread
MIN_VOLUME_MULTIPLIER = 0.5       # require recent volume >= multiplier * median(volume)

# Scoring weights (how much each factor contributes to final score; sum not required)
SCORE_WEIGHT_VWAP = 0.45
SCORE_WEIGHT_VOLUME = 0.30
SCORE_WEIGHT_SPREAD = 0.25

# Fallback to sheet config
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL", "").strip()

# Guarantee at least one trade per day if True (will force top-ranked candidate)
GUARANTEE_ONE_TRADE_PER_DAY = True

# Fallback time (ET hour) to use sheet if primary did not trade
FALLBACK_TIME_ET = 14  # 2pm ET

# Alpaca config (from env)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER = True  # set False for live

# Files
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
LOG_FILE = "bot.log"

# Monitoring/exit parameters (in seconds)
MONITOR_INTERVAL = 15
MONITOR_TIMEOUT = 60 * 60  # 1 hour max

# -------------------------
# SAFETY / VALIDATION
# -------------------------
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required.")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())  # also print to console
logging.info("Starting hybrid_tradebot_advanced")

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
    try:
        return float(x)
    except Exception:
        try:
            return float(x.iloc[0])
        except Exception:
            raise

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
# Logging + Stats helpers
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
    df = pd.DataFrame([trade])
    df.to_csv(TRADES_CSV, mode="a", header=False, index=False)
    # update stats
    stats = {}
    if os.path.exists(STATS_JSON):
        with open(STATS_JSON, "r") as f:
            try:
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
# Market metrics & scoring
# -------------------------
def get_recent_bars(symbol, period_days=3, interval=CANDLE_INTERVAL):
    """Return dataframe from yfinance for recent bars (auto_adjust True)."""
    try:
        df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False, auto_adjust=True)
        return df
    except Exception as e:
        logging.error(f"yfinance fetch error {symbol}: {e}")
        return pd.DataFrame()

def compute_atr(df, n=14):
    """Compute ATR-like measure using high-low range average (simple) over n bars."""
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
    return df["Volume"].tail(20).median()

def score_candidate(symbol):
    """
    Returns a dict with metrics and a combined score for ranking.
    Higher score = better candidate.
    """
    df = get_recent_bars(symbol, period_days=3, interval=CANDLE_INTERVAL)
    if df.empty:
        logging.info(f"No data for {symbol}")
        return None

    vwap_series = compute_vwap(df)
    if vwap_series is None:
        return None
    df["VWAP"] = vwap_series
    last = df.iloc[-1]
    price = safe_float(last["Close"])
    vwap = safe_float(last["VWAP"])
    volume = safe_float(last["Volume"])
    spread = safe_float(last["High"]) - safe_float(last["Low"])

    # ATR (volatility) baseline
    atr = compute_atr(df, n=14) or max(0.01, abs(price) * 0.0005)
    if atr <= 0:
        atr = max(0.01, abs(price) * 0.0005)

    # adaptive thresholds derived from recent volatility
    allowed_vwap_dist = VWAP_VOLATILITY_MULTIPLIER * atr
    allowed_spread = MAX_SPREAD_MULTIPLIER * atr
    min_volume = MIN_VOLUME_MULTIPLIER * compute_recent_volume_median(df) if compute_recent_volume_median(df) is not None else MIN_VOLUME_MULTIPLIER * 1000

    # compute normalized metrics for scoring (0..1 higher is better)
    # VWAP closeness: closer -> higher score (1 when price==vwap)
    vwap_dist = abs(price - vwap)
    vwap_score = max(0.0, 1.0 - min(vwap_dist / (allowed_vwap_dist * 2 + 1e-9), 1.0))

    # Volume score: higher than median -> higher score; saturate at 2x median
    median_vol = compute_recent_volume_median(df) or 1.0
    volume_ratio = volume / (median_vol + 1e-9)
    volume_score = min(1.0, (volume_ratio - 0.5) / 1.5) if volume_ratio >= 0.5 else 0.0

    # Spread score: smaller spread -> higher score
    spread_score = max(0.0, 1.0 - min(spread / (allowed_spread * 2 + 1e-9), 1.0))

    # combine with weights
    combined_score = (SCORE_WEIGHT_VWAP * vwap_score +
                      SCORE_WEIGHT_VOLUME * volume_score +
                      SCORE_WEIGHT_SPREAD * spread_score)

    metrics = {
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
        "combined_score": combined_score,
    }
    return metrics

# -------------------------
# Trading helpers (entry + monitoring exit)
# -------------------------
def submit_market_order(symbol, qty, side):
    """Submit market order with backoff and return order object or None"""
    try:
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide(side), time_in_force=TimeInForce.DAY)
        order = backoff_attempt(client.submit_order, req)
        logging.info(f"Submitted {side} {qty} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Submit market order failed {symbol} {side}: {e}")
        return None

def get_filled_price_from_order(order):
    """Try to extract filled price from order object, robust to client variants"""
    try:
        price = getattr(order, "filled_avg_price", None) or getattr(order, "filled_avg", None)
        if price is not None:
            return safe_float(price)
    except Exception:
        pass
    return None

def monitor_and_exit(symbol, qty, entry_price, tp_price, sl_price):
    """Poll market price until TP or SL hit or timeout, then execute market sell"""
    start = time.time()
    exit_price = None
    result = None
    notes = ""
    logging.info(f"Monitoring {symbol} entry {entry_price:.4f} TP {tp_price:.4f} SL {sl_price:.4f}")
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

    # perform sell market
    sell_order = submit_market_order(symbol, qty, "SELL")
    if sell_order:
        filled = get_filled_price_from_order(sell_order)
        if filled:
            exit_price = filled
    if exit_price is None:
        # fallback to last_px or entry_price
        exit_price = exit_price or entry_price

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
# Signal sheet helpers
# -------------------------
def fetch_signals_from_sheet(url):
    if not url:
        return []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        # require enabled column true if present
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch sheet: {e}")
        return []

# -------------------------
# Main logic: evaluate, rank, choose candidate
# -------------------------
def choose_best_candidate(symbols):
    candidates = []
    for s in symbols:
        try:
            m = score_candidate(s)
            if m:
                candidates.append(m)
        except Exception as e:
            logging.error(f"Error scoring {s}: {e}")
    if not candidates:
        return None
    # sort by combined_score descending
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    return candidates

def execute_candidate(candidate, force=False):
    """
    candidate: metrics dict from score_candidate
    force: if True, we'll execute even if some criteria fail (for guarantee)
    """
    symbol = candidate["symbol"]
    price = candidate["price"]
    qty = 1  # TODO: position sizing later
    # derive TP/SL from price and candidate ATR (conservative)
    tp_price = price * (1 + DEFAULT_TP_PCT)
    sl_price = price * (1 - DEFAULT_SL_PCT)

    # If not forced, enforce adaptive minimum conditions
    if not force:
        if candidate["volume"] < candidate["min_volume"]:
            logging.info(f"{symbol} volume {candidate['volume']} < min {candidate['min_volume']} -> skip")
            return None
        if candidate["spread"] > candidate["allowed_spread"]:
            logging.info(f"{symbol} spread {candidate['spread']:.4f} > allowed {candidate['allowed_spread']:.4f} -> skip")
            return None
        # price distance from VWAP should be within allowed_vwap_dist
        if abs(price - candidate["vwap"]) > candidate["allowed_vwap_dist"] * 2:
            logging.info(f"{symbol} price-vwap {abs(price-candidate['vwap']):.4f} too far -> skip")
            return None

    # Submit market buy
    order = submit_market_order(symbol, qty, "BUY")
    if not order:
        logging.error(f"Buy order failed for {symbol}")
        return None
    # attempt to get filled price
    entry_price = get_filled_price_from_order(order) or price
    logging.info(f"{symbol} buy filled at {entry_price:.4f}")
    # monitor and exit
    trade = monitor_and_exit(symbol, qty, entry_price, tp_price, sl_price)
    return trade

# -------------------------
# Orchestrator
# -------------------------
def main():
    logging.info(f"Run start ET {now_et().isoformat()}")
    candidates = choose_best_candidate(SYMBOLS)
    if candidates:
        top = candidates[0]
        logging.info(f"Top candidate: {top['symbol']} score {top['combined_score']:.4f} vwap_score {top['vwap_score']:.3f} vol_score {top['volume_score']:.3f} spread_score {top['spread_score']:.3f}")
        # Try execute top candidate normally
        trade = execute_candidate(top, force=False)
        if trade:
            logging.info("Primary trade executed successfully.")
            return
        # If not executed, try next candidates
        for cand in candidates[1:3]:  # attempt next two candidates
            trade = execute_candidate(cand, force=False)
            if trade:
                logging.info("Primary trade executed on a lower-ranked candidate.")
                return

    # If we reach here, no candidate executed
    # If time passed fallback threshold, try sheet signals
    current_et_hour = now_et().hour
    if current_et_hour >= FALLBACK_TIME_ET:
        logging.info("Checking fallback sheet signals")
        signals = fetch_signals_from_sheet(SIGNAL_SHEET_CSV_URL)
        for sig in signals:
            try:
                symbol = str(sig.get("symbol","")).upper()
                action = str(sig.get("action","BUY")).upper()
                qty = int(sig.get("qty", 1))
                # allow optional TP/SL in sheet (absolute price or percent)
                tp_val = sig.get("take_profit", "")
                sl_val = sig.get("stop_loss", "")
                # if values look like percent (e.g., 0.5%), handle else ignore
                order = submit_market_order(symbol, qty, action)
                if not order:
                    continue
                entry = get_filled_price_from_order(order) or 0.0
                # interpret tp/sl: if provided as numeric > 1 assume absolute price; if between 0 and 1 assume pct
                tp_price = entry * (1 + DEFAULT_TP_PCT)
                sl_price = entry * (1 - DEFAULT_SL_PCT)
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
                # monitor & exit
                monitor_and_exit(symbol, qty, entry, tp_price, sl_price)
                logging.info(f"Executed sheet signal for {symbol}")
                return
            except Exception as e:
                logging.error(f"Error executing sheet signal: {e}")

    # If still no trade and guarantee is enabled, force top-ranked candidate
    if GUARANTEE_ONE_TRADE_PER_DAY and candidates:
        logging.info("No primary/sheet trade executed; forcing top-ranked candidate due to guarantee flag.")
        forced = execute_candidate(candidates[0], force=True)
        if forced:
            logging.info("Forced trade executed.")
            return

    logging.info("Run complete. No trade executed this run.")

if __name__ == "__main__":
    main()
