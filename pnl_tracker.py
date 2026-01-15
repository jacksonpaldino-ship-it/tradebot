import os
import pytz
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

# ================= CONFIG =================

BASE_URL = "https://paper-api.alpaca.markets"
TZ = pytz.timezone("America/New_York")

# ============== API INIT ==================

api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    BASE_URL,
    api_version="v2"
)

# ============== HELPERS ===================

def get_account():
    return api.get_account()

def get_closed_trades(start, end):
    return api.list_orders(
        status="closed",
        after=start.isoformat(),
        until=end.isoformat(),
        limit=500,
        direction="desc"
    )

def calculate_pnl(start, end):
    orders = get_closed_trades(start, end)
    pnl = 0.0

    for o in orders:
        if o.filled_avg_price and o.qty:
            qty = float(o.qty)
            price = float(o.filled_avg_price)
            side = o.side
            value = qty * price
            pnl += value if side == "sell" else -value

    return pnl

# ============== MAIN ======================

def run():
    now = datetime.now(TZ)

    start_day = now.replace(hour=9, minute=30, second=0, microsecond=0)
    start_week = start_day - timedelta(days=start_day.weekday())
    start_month = start_day.replace(day=1)

    daily_pnl = calculate_pnl(start_day, now)
    weekly_pnl = calculate_pnl(start_week, now)
    monthly_pnl = calculate_pnl(start_month, now)

    acct = get_account()

    print("\n=== Running P&L Tracker ===\n")
    print(f"📊 Alpaca Account Summary ({now.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    print("────────────────────────────────────────")
    print(f"💰 Equity:          ${float(acct.equity):,.2f}")
    print(f"🏦 Cash:            ${float(acct.cash):,.2f}")
    print(f"⚡ Buying Power:    ${float(acct.buying_power):,.2f}")
    print("────────────────────────────────────────")
    print(f"📆 Daily P&L:       ${daily_pnl:,.2f}")
    print(f"📅 Weekly P&L:      ${weekly_pnl:,.2f}")
    print(f"🗓 Monthly P&L:     ${monthly_pnl:,.2f}")
    print("────────────────────────────────────────\n")

if __name__ == "__main__":
    run()
