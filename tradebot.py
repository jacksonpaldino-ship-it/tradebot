# ============================
#  DAILY MOMENTUM VWAP TRADE BOT
#  Uses alpaca-py (NEW SDK)
#  Strategy:
#    1. At market open → scan momentum
#    2. Buy strongest ticker whose price > VWAP
#    3. Sell near close OR momentum reversal
# ============================

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import time

# =======================================
#   CONFIG
# =======================================

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(API_KEY, SECRET_KEY, paper=True)

WATCHLIST = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMD", "CRSP", "PLTR"]

MAX_POSITION = 1      # only 1 ticker per day
RISK_PER_TRADE = 0.25 # 25% of account max
CLOSE_TIME = "15:55"  # sell before close


# =======================================
#  Helpers
# =======================================

def now_est():
    return datetime.now(pytz.timezone("America/New_York"))

def get_vwap(df):
    pv = df["Close"] * df["Volume"]
    return pv.cumsum() / df["Volume"].cumsum()

def get_momentum_score(df):
    change_10 = df["Close"].pct_change(10).iloc[-1]
    change_3  = df["Close"].pct_change(3).iloc[-1]
    score = (change_10 * 0.7) + (change_3 * 0.3)
    return score


# =======================================
#  PICK THE BEST STOCK OF THE DAY
# =======================================

def pick_best_stock():
    results = []

    for ticker in WATCHLIST:
        df = yf.download(ticker, period="15d", interval="5m")
        if len(df) < 20:
            continue

        df["VWAP"] = get_vwap(df)
        price = df["Close"].iloc[-1]
        vwap  = df["VWAP"].iloc[-1]

        if price < vwap:
            continue  # only trade strong stocks

        score = get_momentum_score(df)
        results.append((ticker, score))

    if not results:
        return None

    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0]


# =======================================
#  TRADING FUNCTIONS
# =======================================

def buy_ticker(ticker):
    account = client.get_account()
    buying_power = float(account.buying_power)
    spend = buying_power * RISK_PER_TRADE

    price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    qty = int(spend // price)
    if qty < 1:
        print(f"Not enough to buy {ticker}")
        return

    order = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )
    client.submit_order(order)
    print(f"🟢 Bought {qty} shares of {ticker}")


def close_all_positions():
    positions = client.get_all_positions()
    for p in positions:
        order = MarketOrderRequest(
            symbol=p.symbol,
            qty=p.qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        client.submit_order(order)
        print(f"🔴 Sold {p.qty} shares of {p.symbol}")


# =======================================
#  MAIN LOGIC
# =======================================

def main():
    print("\n=== NEW MOMENTUM VWAP BOT RUN ===")

    est = now_est()
    print(f"Time: {est}")

    positions = client.get_all_positions()
    has_position = len(positions) > 0
    
    # 1. If no position AND before sell time → buy something
    if not has_position:
        if est.strftime("%H:%M") < CLOSE_TIME:
            print("📈 No open positions — scanning for best stock to BUY...")

            ticker = pick_best_stock()

            if ticker:
                print(f"🔥 Best pick today: {ticker}")
                buy_ticker(ticker)
            else:
                print("⚠ No strong stock today")
        else:
            print("⏳ Too close to market close — will not open new positions")

    # 2. If we DO have a position and it's close to market close → sell everything
    else:
        print("📦 Currently holding a position")

        if est.strftime("%H:%M") >= CLOSE_TIME:
            print("🕒 Time to SELL before close...")
            close_all_positions()
        else:
            print("📘 Holding until near close")

    print("Done.\n")

