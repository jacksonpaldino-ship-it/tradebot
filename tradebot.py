# tradebot.py
import os
import time
import smtplib
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dtime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- API keys from GitHub secrets ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
EMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# --- Market hours ---
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

# --- Stock list and cooldown tracking ---
symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
trade_log = {s: [] for s in symbols}
daily_trades = []

# --- Email helper ---
def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"📧 Email sent: {subject}")

# --- Data and trading helpers ---
def get_latest_data(symbol):
    """Fetch last 60 days of 15-minute data."""
    df = yf.download(symbol, period="60d", interval="15m", progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df = df[["Close"]].rename(columns={"Close": "close"})
    df["sma_short"] = df["close"].rolling(window=10).mean()
    df["sma_long"] = df["close"].rolling(window=30).mean()
    return df.dropna()

def get_signal(df):
    """Generate BUY/SELL/HOLD signal based on SMA crossovers."""
    latest_short = df["sma_short"].iloc[-1]
    latest_long = df["sma_long"].iloc[-1]
    prev_short = df["sma_short"].iloc[-2]
    prev_long = df["sma_long"].iloc[-2]

    if prev_short <= prev_long and latest_short > latest_long:
        return "BUY"
    elif prev_short >= prev_long and latest_short < latest_long:
        return "SELL"
    else:
        return "HOLD"

def trade(symbol, signal):
    """Execute trades with cooldown and record actions."""
    now = datetime.now()
    trade_log[symbol] = [t for t in trade_log[symbol] if t.date() == now.date()]

    if len(trade_log[symbol]) >= 2:
        return f"🕒 Cooldown active for {symbol}, skipping."

    try:
        position = trading_client.get_open_position(symbol)
    except Exception:
        position = None

    if signal == "BUY" and not position:
        trading_client.submit_order(
            symbol=symbol,
            qty=2,
            side=OrderSide.BUY,
            type="market",
            time_in_force=TimeInForce.DAY,
        )
        trade_log[symbol].append(now)
        daily_trades.append(f"✅ Bought 2 shares of {symbol}")
        return f"✅ Bought 2 shares of {symbol}"

    elif signal == "SELL" and position:
        trading_client.submit_order(
            symbol=symbol,
            qty=position.qty,
            side=OrderSide.SELL,
            type="market",
            time_in_force=TimeInForce.DAY,
        )
        trade_log[symbol].append(now)
        daily_trades.append(f"🟥 Sold {symbol}")
        return f"🟥 Sold {symbol}"

    else:
        return f"➖ No trade action for {symbol}"

# --- Daily P&L summary ---
def get_daily_summary():
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        return f"Equity: ${equity:,.2f}\nCash: ${cash:,.2f}"
    except Exception as e:
        return f"Error fetching account summary: {e}"

# --- Main function ---
def main():
    now = datetime.now().time()
    if not (MARKET_OPEN <= now <= MARKET_CLOSE):
        print(f"⏸ Market closed ({now}), skipping run.")
        return

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = [f"=== Tradebot run {run_time} ==="]

    for symbol in symbols:
        try:
            df = get_latest_data(symbol)
            signal = get_signal(df)
            action = trade(symbol, signal)
            log.append(f"{symbol}: {signal} → {action}")
        except Exception as e:
            log.append(f"{symbol}: Error → {e}")

    log.append("\nAccount Summary:\n" + get_daily_summary())
    message = "\n".join(log)
    print(message)

    send_email(f"Tradebot Run Report - {run_time}", message)

    # Send daily summary at ~4:05 PM ET
    if datetime.now().hour == 16 and datetime.now().minute >= 5:
        summary = "=== End-of-Day Summary ===\n" + "\n".join(daily_trades) + "\n\n" + get_daily_summary()
        send_email("Tradebot Daily Summary", summary)
        daily_trades.clear()

    print("✅ Run complete.\n")

if __name__ == "__main__":
    main()
