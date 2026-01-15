import os
from datetime import datetime, timedelta
import pytz
from alpaca_trade_api import REST

# ================== ENV ==================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
TZ = pytz.timezone("America/New_York")

# ================== HELPERS ==================
def get_equity_at(date_str):
    """
    Returns equity at the END of the given date.
    """
    hist = api.get_portfolio_history(
        date_start=date_str,
        date_end=date_str,
        timeframe="1D"
    )
    if not hist.equity:
        return None
    return hist.equity[-1]

# ================== MAIN ==================
def run():
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")

    acct = api.get_account()
    current_equity = float(acct.equity)

    # ---------- DAILY ----------
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    equity_yesterday = get_equity_at(yesterday)
    daily_pnl = (
        current_equity - equity_yesterday
        if equity_yesterday is not None
        else 0.0
    )

    # ---------- WEEKLY ----------
    week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
    equity_week_start = get_equity_at(week_start)
    weekly_pnl = (
        current_equity - equity_week_start
        if equity_week_start is not None
        else 0.0
    )

    # ---------- MONTHLY ----------
    month_start = now.replace(day=1).strftime("%Y-%m-%d")
    equity_month_start = get_equity_at(month_start)
    monthly_pnl = (
        current_equity - equity_month_start
        if equity_month_start is not None
        else 0.0
    )

    # ---------- OUTPUT ----------
    print("\n=== Alpaca P&L Summary ===")
    print(f"Date: {today}")
    print("────────────────────────────────────")
    print(f"Equity:        ${current_equity:,.2f}")
    print(f"Daily P&L:     ${daily_pnl:,.2f}")
    print(f"Weekly P&L:    ${weekly_pnl:,.2f}")
    print(f"Monthly P&L:   ${monthly_pnl:,.2f}")
    print("────────────────────────────────────")

    positions = api.list_positions()
    if positions:
        print("Open Positions:")
        for p in positions:
            print(
                f"{p.symbol} | Qty: {p.qty} | "
                f"P/L: ${float(p.unrealized_pl):,.2f}"
            )
    else:
        print("No open positions")

if __name__ == "__main__":
    run()
