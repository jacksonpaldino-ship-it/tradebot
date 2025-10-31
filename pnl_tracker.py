import os
import csv
import datetime
import alpaca_trade_api as tradeapi
from colorama import Fore, Style, init

init(autoreset=True)

# 🔑 Alpaca credentials (use your existing env vars)
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # or live endpoint

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

def get_account_summary():
    """Pull and display account-level P&L summary"""
    account = api.get_account()

    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    portfolio_value = float(account.portfolio_value)
    pnl_today = float(account.equity) - float(account.last_equity)

    pnl_color = Fore.GREEN if pnl_today >= 0 else Fore.RED
    sign = "+" if pnl_today >= 0 else "-"

    print(f"\n📊 {Style.BRIGHT}Alpaca Account Summary ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"────────────────────────────────────────────────────────")
    print(f"💰 Equity:          ${equity:,.2f}")
    print(f"🏦 Cash:            ${cash:,.2f}")
    print(f"⚡ Buying Power:    ${buying_power:,.2f}")
    print(f"📈 Portfolio Value: ${portfolio_value:,.2f}")
    print(f"📆 Daily P&L:       {pnl_color}{sign}${abs(pnl_today):,.2f}{Style.RESET_ALL}")
    print(f"────────────────────────────────────────────────────────")

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
        "portfolio_value": portfolio_value,
        "daily_pnl": pnl_today
    }

def get_positions_summary():
    """Pull open positions and compute unrealized gains/losses"""
    positions = api.list_positions()

    if not positions:
        print("No open positions.\n")
        return []

    print(f"{Style.BRIGHT}Current Open Positions:{Style.RESET_ALL}")
    print("────────────────────────────────────────────────────────")
    data = []
    for pos in positions:
        symbol = pos.symbol
        qty = int(float(pos.qty))
        market_value = float(pos.market_value)
        unrealized_pl = float(pos.unrealized_pl)
        unrealized_plpc = float(pos.unrealized_plpc) * 100

        color = Fore.GREEN if unrealized_pl >= 0 else Fore.RED
        sign = "+" if unrealized_pl >= 0 else "-"

        print(f"{symbol:<6} | Qty: {qty:<3} | Value: ${market_value:>8,.2f} | P/L: {color}{sign}${abs(unrealized_pl):,.2f} ({unrealized_plpc:+.2f}%){Style.RESET_ALL}")
        data.append({
            "symbol": symbol,
            "qty": qty,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc
        })

    print("────────────────────────────────────────────────────────\n")
    return data

def log_to_csv(account_data):
    """Log account summary to CSV for daily tracking"""
    file_path = "pnl_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "equity", "cash", "buying_power", "portfolio_value", "daily_pnl"])
        writer.writerow([
            account_data["timestamp"],
            account_data["equity"],
            account_data["cash"],
            account_data["buying_power"],
            account_data["portfolio_value"],
            account_data["daily_pnl"]
        ])

def main():
    print("=== Running P&L Tracker ===")
    account_data = get_account_summary()
    get_positions_summary()
    log_to_csv(account_data)
    print("✅ Logged to pnl_log.csv\n")

if __name__ == "__main__":
    main()
