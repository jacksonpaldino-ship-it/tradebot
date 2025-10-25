import os
import yfinance as yf
import pandas as pd
from datetime import datetime, time as dtime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Alpaca credentials from GitHub secrets
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Stocks to trade
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]

# === Market Hours Check ===
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
now = datetime.now().time()

if not (MARKET_OPEN <= now <= MARKET_CLOSE):
    print("⏸️ Market is closed — skipping trading until next session.")
    exit()

print(f"=== Tradebot run {datetime.now()} ===")

def get_signal(df):
    short_ma = df["Close"].rolling(window=5).mean()
    long_ma = df["Close"].rolling(window=20).mean()

    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        return "BUY"
    elif short_ma.iloc[-1] < long_ma.iloc[-1]:
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
        print(f"✅ Placed {side} order for {symbol}")
    except Exception as e:
        print(f"❌ Order failed for {symbol}: {e}")

def main():
    total_pnl = 0
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

    # Calculate daily P&L
    try:
        today = datetime.now().date()
        closed_orders = trading_client.get_orders(status="closed")
        pnl_today = sum(
            float(o.filled_avg_price or 0) * (1 if o.side == "sell" else -1)
            for o in closed_orders
            if o.submitted_at and o.submitted_at.date() == today
        )
        print(f"\nDaily P&L: ${pnl_today:.2f}")
    except Exception as e:
        print(f"Error calculating P&L: {e}")

    print("\n✅ Trade check complete. Exiting cleanly.\n")

if __name__ == "__main__":
    main()
