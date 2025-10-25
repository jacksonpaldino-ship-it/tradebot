# tradebot.py
import os
import requests
import yfinance as yf
from datetime import datetime, time as dtime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# === CONFIG ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
EMAIL_TO = "jackson.paldino@icloud.com"
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]

# === SETUP ===
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- Helper: send email (simple, uses MailThis.to free relay) ---
def send_email(subject, message):
    try:
        requests.post(
            "https://mailthis.to/jacksonpaldino",
            data={
                "email": EMAIL_TO,
                "subject": subject,
                "message": message,
            },
            timeout=10
        )
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print(f"Email error: {e}")

# --- Strategy functions ---
def get_latest_data(symbol):
    df = yf.download(symbol, period="60d", interval="15m", progress=False)
    df["sma_short"] = df["Close"].rolling(window=3).mean()
    df["sma_long"] = df["Close"].rolling(window=7).mean()
    return df.dropna()

def get_signal(df):
    if df["sma_short"].iloc[-1] > df["sma_long"].iloc[-1]:
        return "BUY"
    elif df["sma_short"].iloc[-1] < df["sma_long"].iloc[-1]:
        return "SELL"
    return "HOLD"

def trade(symbol):
    df = get_latest_data(symbol)
    signal = get_signal(df)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: {signal}")
    position = None
    try:
        position = trading_client.get_open_position(symbol)
    except:
        pass

    if signal == "BUY" and not position:
        trading_client.submit_order(
            MarketOrderRequest(symbol=symbol, qty=2, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        )
        msg = f"✅ Bought 2 shares of {symbol} at {datetime.now().strftime('%H:%M:%S')}"
        print(msg)
        send_email(f"Trade Alert: BUY {symbol}", msg)
    elif signal == "SELL" and position:
        trading_client.submit_order(
            MarketOrderRequest(symbol=symbol, qty=position.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        )
        msg = f"🟥 Sold {symbol} at {datetime.now().strftime('%H:%M:%S')}"
        print(msg)
        send_email(f"Trade Alert: SELL {symbol}", msg)
    else:
        print(f"➖ No trade action for {symbol}")

# --- Main run ---
def main():
    now = datetime.now().time()
    if not (MARKET_OPEN <= now <= MARKET_CLOSE):
        print("Market closed, skipping run.")
        return

    print(f"\n=== Tradebot run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    for symbol in SYMBOLS:
        try:
            trade(symbol)
        except Exception as e:
            print(f"Error trading {symbol}: {e}")

    print("\n✅ Trade check complete. Exiting cleanly.\n")

if __name__ == "__main__":
    main()
