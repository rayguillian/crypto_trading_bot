from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

client = Client(api_key, api_secret)

try:
    # Test API connection
    status = client.get_system_status()
    print("System status:", status)
    
    # Get account information
    account = client.get_account()
    print("\nAccount status:", account['accountType'])
    
    # Get current BTC price
    btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
    print("\nBTC Price:", btc_price['price'])

except Exception as e:
    print("Error:", str(e))
