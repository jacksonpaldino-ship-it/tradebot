import time
from datetime import datetime, time as dtime
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

API_KEY = "PKLQFRGIDZI7L2MBEZ3TGBOZEY"
SECRET_KEY = "3hW9TartmCaLuYHRUPumUN5Qd3R822Xoda8tc5FPbQmM"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_latest_data(symbol):
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    df = df[['Close']].rename(columns={'Close': 'close'})
    df['sma_short'] = df['close'].rolling(window=3).mean()
    df['sma_long'] = df['close'].rolling(window=7).mean()
    return df

def get_signal(df):
    if len(df) < 7:
        return "HOLD"
    if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
        return "BUY"
    elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
        return "SELL"
    else:
        return "HOLD"

def trade(symbol):
    df = get_latest_data(symbol)
    signal = get_signal(df)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol} → {signal}")

    position = None
    try:
        position = trading_client.get_open_position(symbol)
    except:
        position = None

    if signal == "BUY" and not position:
        order = trading_client.submit_order(
            symbol=symbol, qty=2, side=OrderSide.BUY, type="market", time_in_force=TimeInForce.GTC
        )
        print(f"✅ Bought 2 shares of {symbol}")
    elif signal == "SELL" and position:
        order = trading_client.submit_order(
            symbol=symbol, qty=position.qty, side=OrderSide.SELL, type="market", time_in_force=TimeInForce.GTC
        )
        print(f"🟥 Sold {symbol}")
    else:
        print("No trade action taken.")

symbols = ["AAPL", "TSLA", "MSFT"]

MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)

while True:
    now = datetime.now().time()
    if MARKET_OPEN <= now <= MARKET_CLOSE:
        for symbol in symbols:
            trade(symbol)
        time.sleep(300)
    else:
        print(f"Market closed. Sleeping at {datetime.now():%H:%M:%S}")
        time.sleep(1800)
