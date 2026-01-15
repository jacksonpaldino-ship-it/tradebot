from alpaca_trade_api import REST
from datetime import datetime
import pytz

api = REST(api_version="v2")
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
