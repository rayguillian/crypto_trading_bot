# Cryptocurrency Trading Bot

An advanced automated cryptocurrency trading bot with AI integration, backtesting visualization, and robust risk management.

## Features

- Modular strategy framework
- Interactive backtesting visualization dashboard
- Real-time trading execution
- Strategy optimization
- Performance analytics

## Project Structure

```
crypto_trading_bot/
├── app.py                 # Flask application
├── frontend/             # Frontend React application
│   ├── components/       # React components
│   ├── pages/            # Next.js pages
│   └── package.json      # Frontend dependencies
├── core/                 # Core trading functionality
│   ├── backtest.py       # Backtesting engine
│   └── strategy.py       # Strategy base class
├── strategies/           # Trading strategies
├── models/               # Data models
└── config/              # Configuration files
```

## Prerequisites

- Python 3.8+
- Node.js 14+
- Binance account with API access

## Installation

1. Clone and set up Python environment:
```bash
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
npm run build
cd ..
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Binance API credentials
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the dashboard:
- Open http://localhost:5000
- Navigate to Backtest page
- Select strategy and timeframe

3. Run backtests:
```bash
python run_backtest.py --strategy=MyStrategy
```

## Development

1. Run frontend in development mode:
```bash
cd frontend
npm run dev
```

2. Run backend:
```bash
python app.py
```

## API Endpoints

- GET `/api/strategies`: List available strategies
- GET `/api/backtest-results`: Get backtest results
  - Query params: `strategy`, `timeframe`

## Creating Strategies

1. Create new strategy file in `strategies/`:
```python
from core.strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Strategy logic
        pass
```

2. Run backtest:
```bash
python run_backtest.py --strategy=MyStrategy
```

3. View results in dashboard

## Testing

```bash
pytest tests/
```

## Contributing

1. Fork repository
2. Create feature branch
3. Submit pull request

## License

MIT License