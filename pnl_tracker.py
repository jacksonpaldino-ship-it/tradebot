import os
from alpaca_trade_api import REST
from datetime import datetime
import pytz

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
TZ = pytz.timezone("America/New_York")

def run():
    acct = api.get_account()
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    equity = float(acct.equity)
    last_equity = float(acct.last_equity)
    daily_pnl = equity - last_equity

    print("\n=== Alpaca P&L Summary ===")
    print(f"Date: {today}")
    print(f"Equity: ${equity:,.2f}")
    print(f"Daily P&L: ${daily_pnl:,.2f}")

    positions = api.list_positions()
    if positions:
        print("\nOpen Positions:")
        for p in positions:
            print(
                f"{p.symbol} | Qty: {p.qty} | "
                f"P/L: ${float(p.unrealized_pl):,.2f}"
            )
    else:
        print("\nNo open positions")

if __name__ == "__main__":
    run()
