import os
import pandas as pd
from datetime import datetime, timezone
from alpaca_trade_api.rest import REST, TimeFrame

# -----------------------
# CONFIGURATION
# -----------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TRADE_QUANTITY = 1  # number of shares/contracts to buy
GUARANTEE_TRADE = True

# Read secrets from GitHub Actions environment
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise ValueError("API credentials missing. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(API_KEY, API_SECRET, BASE_URL)

# -----------------------
# DATA FETCHING
# -----------------------
def fetch_data(symbol, limit=50):
    """Fetch minute bars and return DataFrame"""
    try:
        barset = api.get_bars(symbol, TimeFrame.Minute, limit=limit, adjustment='raw').df
        barset = barset[barset['symbol'] == symbol]
        return barset
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# -----------------------
# SCORING FUNCTION
# -----------------------
def score_candidate(df):
    """Compute a score based on price-vwap distance, spread, and volume"""
    last = df.iloc[-1]
    last_price = float(last["close"])
    vwap = float(last["vwap"]) if "vwap" in last else last_price
    spread = float(last["high"]) - float(last["low"])
    spread_threshold = float(df["high"].max() - df["low"].min())
    volume_factor = float(last["volume"]) / (df["volume"].mean() + 1)
    
    # Adaptive thresholds based on spread
    spread_score = max(0, 1 - (spread / (spread_threshold + 1e-6)))
    price_vwap_score = max(0, 1 - abs(last_price - vwap)/ (vwap + 1e-6))
    score = 0.4 * spread_score + 0.4 * price_vwap_score + 0.2 * volume_factor
    return score

# -----------------------
# CANDIDATE SELECTION
# -----------------------
def select_candidate():
    candidates = []
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if df is not None and not df.empty:
            try:
                score = score_candidate(df)
                candidates.append({"symbol": sym, "score": score, "last_price": float(df.iloc[-1]["close"])})
            except Exception as e:
                print(f"Error scoring {sym}: {e}")
    
    if not candidates:
        return None
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    print(f"Top candidate: {top['symbol']} score {top['score']:.4f}")
    return top

# -----------------------
# TRADE EXECUTION
# -----------------------
def place_trade(candidate):
    try:
        symbol = candidate["symbol"]
        qty = TRADE_QUANTITY
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        print(f"Submitted BUY {qty} {symbol}")
        return order
    except Exception as e:
        print(f"Buy order failed {candidate['symbol']}: {e}")
        return None

# -----------------------
# MAIN LOOP
# -----------------------
def main():
    print(f"Starting hybrid_tradebot at {datetime.now(timezone.utc).astimezone().isoformat()}")
    
    candidate = select_candidate()
    order = None

    if candidate:
        order = place_trade(candidate)
    
    # Guarantee a trade if nothing executed
    if GUARANTEE_TRADE and order is None and candidate:
        print(f"Forcing top candidate trade: {candidate['symbol']}")
        order = place_trade(candidate)

    if order:
        print("Trade executed, monitoring disabled for simplicity in this version.")
    else:
        print("No trade executed this run.")

if __name__ == "__main__":
    main()
