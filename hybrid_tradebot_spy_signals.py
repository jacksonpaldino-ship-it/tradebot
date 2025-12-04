#!/usr/bin/env python3
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz

API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

FORCE_DAILY_TRADE = True     # <--- guarantees at least 1 trade/day
SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_EXPORT_LINK_HERE"

def safe_float(x):
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def fetch_data(symbol):
    try:
        barset = api.get_bars(
            symbol,
            "1Min",
            limit=60
        )

        if len(barset) == 0:
            return None

        df = pd.DataFrame([{
            "time": b.t,
            "open": b.o,
            "high": b.h,
            "low": b.l,
            "close": b.c,
            "volume": b.v
        } for b in barset])

        # VWAP
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        return df
    except Exception as e:
        print(f"Fetch error {symbol}: {e}")
        return None


def score_symbol(symbol, df):
    last = df.iloc[-1]
    price = safe_float(last["close"])
    vwap = safe_float(last["vwap"])

    # Volatility filter (adaptive)
    vol = df["close"].pct_change().rolling(20).std().iloc[-1]
    if np.isnan(vol) or vol == 0:
        vol = 0.002  # fallback

    # tighter filters during quiet periods, looser during high volatility
    max_spread_allowed = max(0.15, 6 * vol)  
    max_dist_allowed = max(0.10, 4 * vol)

    spread = safe_float(last["high"]) - safe_float(last["low"])
    dist = abs(price - vwap)

    # Hard rejects
    if spread > max_spread_allowed:
        return -1, f"spread {spread:.3f} > allowed {max_spread_allowed:.3f}"
    if dist > max_dist_allowed * price:
        return -1, f"price-vwap {dist:.3f} too far (limit {max_dist_allowed*price:.3f})"

    # Scoring
    score = 0
    score += (1 - dist / (max_dist_allowed * price)) * 0.6
    score += (1 - spread / max_spread_allowed) * 0.2

    # Volume-weighted signal
    vnorm = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
    vnorm = min(vnorm, 3)
    score += (vnorm / 3) * 0.2

    return score, "ok"


def read_sheet_signals():
    try:
        csv = requests.get(SHEET_URL).text
        df = pd.read_csv(pd.compat.StringIO(csv))
        return df
    except Exception as e:
        print(f"Sheet read failed: {e}")
        return None


def place_market_buy(symbol):
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side="buy",  # <-- FIXED (alpaca requires lowercase)
            type="market",
            time_in_force="day"
        )
        print(f"✔ BUY sent for {symbol}")
        return True
    except Exception as e:
        print(f"Order failed {symbol}: {e}")
        return False


def run():
    print("Starting hybrid_tradebot_advanced")
    print("Run start ET", datetime.now(pytz.timezone("US/Eastern")))

    results = []

    # --- score each symbol ---
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is None:
            continue

        score, reason = score_symbol(sym, df)
        results.append((sym, score, reason))

    if len(results) == 0:
        print("No data available.")
        return

    # rank
    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0]
    print(f"Top candidate: {best[0]} score {best[1]:.4f} ({best[2]})")

    if best[1] > 0:
        print("Conditions met → placing trade")
        place_market_buy(best[0])
        return

    print("Primary filters rejected all symbols.")

    # --- fallback sheet ---
    sheet = read_sheet_signals()
    if sheet is not None and "Buy" in sheet.columns:
        row = sheet[sheet["Buy"] == 1].head(1)
        if len(row) > 0:
            sym = row.iloc[0]["Symbol"]
            print(f"Sheet triggered: BUY {sym}")
            place_market_buy(sym)
            return

    # --- FORCE TRADE (guaranteed daily) ---
    if FORCE_DAILY_TRADE:
        print("No trade executed → forcing trade on top-ranked symbol.")
        place_market_buy(best[0])
        return

    print("Run complete. No trade executed.")


if __name__ == "__main__":
    run()
