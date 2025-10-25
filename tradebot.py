import os
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, date, time as dtime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Alpaca credentials
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Stock list
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]

# === Market Hours Check ===
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
now = datetime.now().time()
if not (MARKET_OPEN <= now <= MARKET_CLOSE):
    print("⏸️ Market is closed — skipping trading until next session.")
    exit()

print(f"=== Tradebot run {datetime.now()} ===")

# === Cooldown Tracking File ===
COOLDOWN_FILE = "trade_history.json"
MAX_TRADES_PER_DAY = 2

def load_trade_history():
    if os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE, "r") as f:
            return json.load(f)
    return {}

def save_trade_history(history):
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(history, f)

def record_trade(symbol):
    today_str = str(date.today())
    history = load_trade_history()
    if today_str not in history:
        history[today_str] = {}
    history[today_str][symbol] = history[today_str].get(symbol, 0) + 1
    save_trade_history(history)

def trades_today(symbol):
    today_str = str(date.today())
    history = load_trade_history()
    return history.get(today_str, {}).get(symbol, 0)

def get_signal(df):
    """Generate trading signal based on SMA crossover."""
    if len(df) < 20:
        return "HOLD"

    latest_short = df['sma_short'].iloc[-1]
    latest_long = df['sma_long'].iloc[-1]
    prev_short = df['sma_short'].iloc[-2]
    prev_long = df['sma_long'].iloc[-2]

    # Detect crossover: short crosses above long = BUY, crosses below = SELL
    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    elif prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    else:
        return "HOLD"

def place_order(symbol, side):
    try:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        record_trade(symbol)
        print(f"✅ Placed {side} order for {symbol}")
    except Exception as e:
        print(f"❌ Order failed for {symbol}: {e}")

def main():
    for symbol in SYMBOLS:
        print(f"\nChecking {symbol}...")
        try:
            df = yf.download(symbol, period="60d", interval="15m", progress=False)
            df.dropna(inplace=True)

            if df.empty:
                print(f"⚠️ No data for {symbol}")
                continue

            signal = get_signal(df)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")

            if trades_today(symbol) >= MAX_TRADES_PER_DAY:
                print(f"🕓 Cooldown active — already traded {symbol} {MAX_TRADES_PER_DAY}x today.")
                continue

            positions = trading_client.get_all_positions()
            held_symbols = [p.symbol for p in positions]

            if signal == "BUY" and symbol not in held_symbols:
                place_order(symbol, "BUY")
            elif signal == "SELL" and symbol in held_symbols:
                place_order(symbol, "SELL")
            else:
                print(f"➖ No trade action for {symbol}")

        except Exception as e:
            print(f"Failed bars for {symbol}: {e}")

    print("\n✅ Trade check complete. Exiting cleanly.\n")

if __name__ == "__main__":
    main()

