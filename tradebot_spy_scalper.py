import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import OrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ==========================
# CONFIG
# ==========================
API_KEY = "YOUR_KEY"
API_SECRET = "YOUR_SECRET"
PAPER = True

SYMBOL = "SPY"
RISK_PER_TRADE = 0.02         # 2% account equity per trade
STOP_LOSS_PCT = 0.002         # 0.2%
TAKE_PROFIT_PCT = 0.0035      # 0.35%
MIN_SPREAD = 0.02             # Avoid bad fills
MIN_VOLUME = 500000           # Avoid dead candles

# ==========================
# Alpaca Client
# ==========================
client = TradingClient(API_KEY, API_SECRET, paper=PAPER)

# ==========================
# VWAP FUNCTION
# ==========================
def compute_vwap(df):
    pv = (df["Close"] * df["Volume"]).cumsum()
    v = df["Volume"].cumsum()
    return pv / v

# ==========================
# MAIN STRATEGY EXECUTION
# ==========================
def run_bot():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est)

    # Avoid running outside market hours
    if now.hour < 9 or (now.hour == 9 and now.minute < 35) or now.hour > 15:
        print("Market closed or pre-OR. Exiting.")
        return

    # ========= Download 5-min SPY data =========
    df = yf.download(SYMBOL, period="3d", interval="5m", progress=False)

    if df is None or len(df) < 30:
        print("Not enough data.")
        return

    df["VWAP"] = compute_vwap(df)

    # latest candle
    row = df.iloc[-1]
    price = float(row["Close"])
    vwap = float(row["VWAP"])
    volume = float(row["Volume"])

    # Check volume
    if volume < MIN_VOLUME:
        print("Volume too low, skipping.")
        return

    # Check spread
    candle_spread = float(row["High"]) - float(row["Low"])
    if candle_spread > MIN_SPREAD:
        print("Spread too big, skipping.")
        return

    # ========= Determine trend =========
    vwap_prev = float(df["VWAP"].iloc[-3])
    vwap_slope_up = vwap > vwap_prev
    vwap_slope_down = vwap < vwap_prev

    # ========= Check existing positions =========
    positions = client.get_all_positions()
    already_in_position = any(pos.symbol == SYMBOL for pos in positions)

    if already_in_position:
        print("Already in position → no new trade.")
        return

    # ========= Compute position size =========
    account = client.get_account()
    equity = float(account.equity)
    risk_amount = equity * RISK_PER_TRADE
    qty = int(risk_amount / price)

    if qty < 1:
        print("Qty < 1, skipping.")
        return

    # ========= Determine signal =========
    signal = None

    # LONG: price > VWAP and VWAP sloping up
    if price > vwap and vwap_slope_up:
        signal = "LONG"

    # SHORT: price < VWAP and VWAP sloping down
    elif price < vwap and vwap_slope_down:
        signal = "SHORT"

    # No trade
    if signal is None:
        print("No signal → exiting.")
        return

    # ========= Compute TP / SL =========
    if signal == "LONG":
        sl = price * (1 - STOP_LOSS_PCT)
        tp = price * (1 + TAKE_PROFIT_PCT)
        side = OrderSide.BUY

    else:
        sl = price * (1 + STOP_LOSS_PCT)
        tp = price * (1 - TAKE_PROFIT_PCT)
        side = OrderSide.SELL

    # ========= Place bracket order =========
    try:
        order = OrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=side,
            type="market",
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": round(tp, 2)},
            stop_loss={"stop_price": round(sl, 2)},
        )

        client.submit_order(order)
        print(f"Placed {signal} order: qty={qty}, TP={tp:.2f}, SL={sl:.2f}")

    except Exception as e:
        print("Order error:", e)


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    run_bot()
    print("Run complete.")
