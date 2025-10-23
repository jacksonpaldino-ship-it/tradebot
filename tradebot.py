# tradebot.py
# SMA crossover bot — 15m bars, cooldown, daily P&L email summary

import os
import math
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -----------------------
# CONFIG
# -----------------------
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
SHORT_WINDOW = 5
LONG_WINDOW = 20
POSITION_SIZE_PCT = 0.02
MAX_POSITIONS = 6
STOP_LOSS_PCT = 0.06
TAKE_PROFIT_PCT = 0.08
COOLDOWN_RUNS = 4   # don’t buy the same stock again until 4 runs (~1 hour if cron=15m)

EMAIL_FROM = os.getenv("BOT_EMAIL_FROM")
EMAIL_PASS = os.getenv("BOT_EMAIL_PASS")
EMAIL_TO   = os.getenv("BOT_EMAIL_TO")

# -----------------------
# API Clients
# -----------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY env vars")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

COOLDOWN_FILE = "cooldown.json"
PANDL_FILE = "pnl_log.json"

# -----------------------
# Helpers
# -----------------------
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def fetch_bars(symbol):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        limit=LONG_WINDOW * 4,  # ~5 hours of data
    )
    bars = data_client.get_stock_bars(req).df
    if ("symbol" in bars.index.names) and (symbol in bars.index.get_level_values("symbol")):
        b = bars[bars.index.get_level_values("symbol") == symbol].copy()
    else:
        b = bars.copy()
    b = b.resample("15min").last().dropna()
    return b

def calc_signal(bars):
    if bars is None or bars.empty or len(bars) < LONG_WINDOW + 1:
        return "HOLD"
    if "close" not in bars.columns and "Close" in bars.columns:
        bars = bars.rename(columns={"Close": "close"})
    bars["sma_short"] = bars["close"].rolling(SHORT_WINDOW).mean()
    bars["sma_long"] = bars["close"].rolling(LONG_WINDOW).mean()
    prev_short = bars["sma_short"].iloc[-2]
    prev_long  = bars["sma_long"].iloc[-2]
    cur_short  = bars["sma_short"].iloc[-1]
    cur_long   = bars["sma_long"].iloc[-1]
    if prev_short <= prev_long and cur_short > cur_long:
        return "BUY"
    elif prev_short >= prev_long and cur_short < cur_long:
        return "SELL"
    else:
        return "HOLD"

def place_market_order(symbol, qty, side):
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        order = trading_client.submit_order(req)
        return order
    except Exception as e:
        print(f"⚠️ Error placing {side} for {symbol}: {e}")
        return None

def get_account_cash():
    acc = trading_client.get_account()
    return float(acc.cash)

def get_positions_map():
    pos = {}
    for p in trading_client.get_all_positions():
        pos[p.symbol] = p
    return pos

# -----------------------
# Core Trading Logic
# -----------------------
def main():
    print(f"\n=== Tradebot run {datetime.now()} ===")
    cash = get_account_cash()
    positions = get_positions_map()
    cooldowns = load_json(COOLDOWN_FILE)

    trades_made = []

    for symbol in SYMBOLS:
        print(f"\nChecking {symbol}...")
        try:
            bars = fetch_bars(symbol)
        except Exception as e:
            print(f"Failed bars for {symbol}: {e}")
            continue
        signal = calc_signal(bars)
        print(f"Signal: {signal}")

        position = positions.get(symbol)
        qty_held = int(float(position.qty)) if position else 0
        entry_price = float(position.avg_entry_price) if position else None
        last_price = float(bars["close"].iloc[-1])

        # handle cooldowns
        cooldowns.setdefault(symbol, 0)
        if cooldowns[symbol] > 0:
            cooldowns[symbol] -= 1

        if signal == "SELL" and position:
            print(f"Selling {qty_held} of {symbol}")
            order = place_market_order(symbol, qty_held, "SELL")
            if order:
                trades_made.append(f"Sold {qty_held} {symbol} @ {last_price:.2f}")
                cooldowns[symbol] = COOLDOWN_RUNS

        elif signal == "BUY" and not position and cooldowns[symbol] == 0:
            if len(positions) >= MAX_POSITIONS:
                print("Max positions reached.")
                continue
            allocation = cash * POSITION_SIZE_PCT
            qty = max(1, int(allocation / last_price))
            print(f"Buying {qty} of {symbol}")
            order = place_market_order(symbol, qty, "BUY")
            if order:
                trades_made.append(f"Bought {qty} {symbol} @ {last_price:.2f}")
                cooldowns[symbol] = COOLDOWN_RUNS
                positions = get_positions_map()
                cash -= qty * last_price

        elif position:
            pnl_pct = (last_price - entry_price) / entry_price
            if pnl_pct <= -STOP_LOSS_PCT:
                print(f"Stop loss triggered for {symbol}")
                order = place_market_order(symbol, qty_held, "SELL")
                if order:
                    trades_made.append(f"Stop loss sell {symbol} ({pnl_pct:.2%})")
                    cooldowns[symbol] = COOLDOWN_RUNS
            elif pnl_pct >= TAKE_PROFIT_PCT:
                print(f"Take profit triggered for {symbol}")
                order = place_market_order(symbol, qty_held, "SELL")
                if order:
                    trades_made.append(f"Take profit sell {symbol} ({pnl_pct:.2%})")
                    cooldowns[symbol] = COOLDOWN_RUNS
            else:
                print(f"Holding {symbol} ({pnl_pct:.2%})")

    save_json(COOLDOWN_FILE, cooldowns)
    summarize_pnl(trades_made)

# -----------------------
# P&L + Email
# -----------------------
def summarize_pnl(trades_made):
    account = trading_client.get_account()
    equity = float(account.equity)
    last_log = load_json(PANDL_FILE)
    last_equity = last_log.get("equity", equity)
    pnl_today = equity - last_equity
    msg = f"Daily P&L: ${pnl_today:.2f}\n\nRecent trades:\n" + ("\n".join(trades_made) if trades_made else "No trades today.")
    print(msg)

    save_json(PANDL_FILE, {"equity": equity, "updated": datetime.now().isoformat()})
    if EMAIL_FROM and EMAIL_TO and EMAIL_PASS:
        send_email("Tradebot Report", msg)

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
        print("✅ Email report sent")
    except Exception as e:
        print(f"⚠️ Email send failed: {e}")

if __name__ == "__main__":
    main()
