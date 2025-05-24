# Algorithmic Trading Bot

A comprehensive, professional-grade algorithmic trading bot with advanced backtesting capabilities, multiple trading strategies, risk management, and real-time execution.

## 🚀 Features

### Trading Strategies
- **Mean Reversion**: Bollinger Bands and RSI-based strategy
- **Momentum**: MACD and trend-following strategy  
- **Breakout**: Consolidation pattern detection and breakout trading
- **ML Ensemble**: Random Forest and Gradient Boosting machine learning models
- **Pairs Trading**: Statistical arbitrage strategy (optional)

### Risk Management
- **Position Sizing**: Kelly Criterion, volatility-based, ATR-based, and dynamic sizing
- **Risk Metrics**: VaR, CVaR, maximum drawdown, correlation analysis
- **Real-time Monitoring**: Continuous risk assessment with alerts
- **Stop Loss**: Automatic stop-loss and take-profit orders

### Backtesting & Analysis
- **Comprehensive Backtesting**: Historical performance simulation
- **Performance Metrics**: Sharpe ratio, alpha, beta, information ratio
- **Visualization**: Interactive charts and performance reports
- **Walk-forward Analysis**: Out-of-sample testing capabilities

### Execution & Monitoring
- **Live Trading**: Real-time order execution via Alpaca API
- **Paper Trading**: Risk-free testing environment
- **Multi-timeframe**: Support for various timeframes and instruments
- **Real-time Data**: Live market data integration

### Notifications & Reporting
- **Email Alerts**: Trade notifications, risk alerts, and performance reports
- **Performance Reports**: Daily, weekly, and monthly automated reports
- **Dashboard**: Web-based monitoring dashboard
- **Logging**: Comprehensive logging system with rotation

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
All dependencies are listed in `requirements.txt`:

```bash
# Core trading and data
alpaca-trade-api>=3.0.0
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
requests>=2.28.0

# Technical analysis
ta-lib>=0.4.0
talib>=0.4.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Database and caching
sqlalchemy>=2.0.0
redis>=4.6.0

# Utilities and notifications
python-dotenv>=1.0.0
jinja2>=3.1.0
schedule>=1.2.0
pyyaml>=6.0
```

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd algotrade
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install TA-Lib (Technical Analysis Library)

#### On macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

#### On Windows:
```bash
# Download and install from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXXm-win_amd64.whl
```

### 5. Configure Environment
```bash
cp .env.template .env
# Edit .env with your API keys and configuration
```

## ⚙️ Configuration

### 1. Environment Variables
Edit the `.env` file with your configuration:

```bash
# Trading API (Required)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Email Notifications (Optional)
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=recipient@email.com

# Trading Settings
TRADING_MODE=paper  # Start with paper trading
AUTO_TRADING_ENABLED=true
MAX_PORTFOLIO_RISK=0.02
```

### 2. Get API Keys

#### Alpaca Markets (Free Paper Trading)
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Generate API keys in your dashboard
3. Use paper trading URL for testing

#### Alpha Vantage (Optional, for additional data)
1. Sign up at [Alpha Vantage](https://www.alphavantage.co/)
2. Get free API key (500 requests/day)

### 3. Email Setup (Gmail Example)
1. Enable 2-factor authentication on Gmail
2. Generate an App Password
3. Use the App Password in EMAIL_PASSWORD

## 🚀 Usage

### Quick Start
```bash
# Run the trading bot
python main.py

# Run backtesting only
python main.py --backtest-only

# Run with specific strategies
python main.py --strategies mean_reversion,momentum
```

### Command Line Options
```bash
python main.py [OPTIONS]

Options:
  --backtest-only     Run backtesting without live trading
  --paper-trading     Force paper trading mode
  --strategies TEXT   Comma-separated list of strategies to use
  --config PATH       Path to custom configuration file
  --log-level TEXT    Set logging level (DEBUG, INFO, WARNING, ERROR)
  --help             Show help message
```

### Backtesting
```python
from backtesting.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy

# Initialize backtesting
engine = BacktestEngine(initial_capital=100000)
strategy = MeanReversionStrategy()

# Run backtest
results = engine.backtest(
    strategy=strategy,
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# View results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Live Trading
```python
from execution.execution_engine import ExecutionEngine
from strategies.strategy_manager import StrategyManager

# Initialize execution engine
execution_engine = ExecutionEngine(config)

# Initialize strategy manager
strategy_manager = StrategyManager(config)

# Start live trading
execution_engine.start_trading(strategy_manager)
```

## 📊 Monitoring & Visualization

### Performance Dashboard
The bot generates interactive HTML reports in the `reports/` directory:
- Portfolio performance charts
- Risk metrics dashboard
- Trade history and analysis
- Strategy performance comparison

### Email Reports
Configure email settings to receive:
- Daily performance summaries
- Trade execution alerts
- Risk management warnings
- Weekly/monthly detailed reports

### Logging
Comprehensive logging system with separate loggers for:
- Trading operations (`logs/trading.log`)
- Strategy signals (`logs/strategy.log`)
- Risk management (`logs/risk.log`)
- Performance metrics (`logs/performance.log`)

## 🔒 Risk Management

### Built-in Safety Features
- **Maximum Portfolio Risk**: Limits total portfolio exposure
- **Position Sizing**: Intelligent position sizing based on volatility
- **Stop Losses**: Automatic stop-loss orders
- **Drawdown Protection**: Pauses trading if losses exceed limits
- **Real-time Monitoring**: Continuous risk assessment

### Risk Metrics
- Value at Risk (VaR) at 95% and 99% confidence levels
- Conditional Value at Risk (CVaR)
- Maximum drawdown tracking
- Correlation risk analysis
- Stress testing scenarios

## 📈 Strategies

### 1. Mean Reversion Strategy
```python
# Configuration in config/config.py
MEAN_REVERSION_CONFIG = {
    'bollinger_period': 20,
    'bollinger_std': 2,
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70
}
```

### 2. Momentum Strategy
```python
# MACD-based momentum strategy
MOMENTUM_CONFIG = {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'trend_period': 50
}
```

### 3. ML Ensemble Strategy
```python
# Machine learning configuration
ML_CONFIG = {
    'models': ['random_forest', 'gradient_boosting'],
    'features': ['technical', 'price_action'],
    'retrain_frequency': 'weekly',
    'lookback_window': 60
}
```

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_strategies.py

# Run with coverage
python -m pytest tests/ --cov=.
```

### Backtesting Validation
```bash
# Run comprehensive backtesting
python scripts/validate_strategies.py

# Test specific date ranges
python scripts/backtest_period.py --start 2020-01-01 --end 2023-12-31
```

## 📁 Project Structure

```
algotrade/
├── config/
│   └── config.py              # Configuration settings
├── data/
│   ├── data_manager.py        # Data fetching and management
│   └── trading_data.db        # SQLite database (created automatically)
├── strategies/
│   ├── base_strategy.py       # Base strategy class
│   ├── mean_reversion.py      # Mean reversion strategy
│   ├── momentum.py            # Momentum strategy
│   ├── breakout.py            # Breakout strategy
│   ├── ml_ensemble.py         # ML ensemble strategy
│   ├── pairs_trading.py       # Pairs trading strategy
│   └── strategy_manager.py    # Multi-strategy coordination
├── backtesting/
│   └── backtest_engine.py     # Backtesting framework
├── execution/
│   └── execution_engine.py    # Live trading execution
├── risk_management/
│   ├── risk_calculator.py     # Risk metrics calculation
│   ├── position_sizer.py      # Position sizing algorithms
│   └── risk_monitor.py        # Real-time risk monitoring
├── visualization/
│   ├── performance_visualizer.py  # Performance charts
│   ├── trading_charts.py      # Technical analysis charts
│   └── risk_visualizer.py     # Risk analysis visualizations
├── utils/
│   ├── helpers.py             # Utility functions
│   ├── decorators.py          # Function decorators
│   ├── logging_config.py      # Logging configuration
│   └── notifications.py       # Email notification system
├── logs/                      # Log files (created automatically)
├── reports/                   # Generated reports (created automatically)
├── models/                    # ML models (created automatically)
├── requirements.txt           # Python dependencies
├── .env.template             # Environment configuration template
├── main.py                   # Main trading bot entry point
└── README.md                 # This file
```

## 🚨 Important Warnings

### ⚠️ FINANCIAL RISK DISCLAIMER
- **TRADING INVOLVES SUBSTANTIAL RISK OF LOSS**
- Past performance does not guarantee future results
- Only trade with money you can afford to lose
- Always start with paper trading before going live
- Understand all strategies before deploying them
- Monitor your bot regularly, especially initially

### 🔒 Security Best Practices
- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Use strong, unique passwords
- Enable two-factor authentication where possible
- Monitor API usage and set up alerts

### 📊 Trading Best Practices
- Start with small position sizes
- Diversify across multiple strategies
- Set strict risk limits
- Monitor performance regularly
- Keep detailed records
- Stay informed about market conditions
- Have a plan for handling losses

## 🐛 Troubleshooting

### Common Issues

#### 1. TA-Lib Installation Error
```bash
# On macOS with M1/M2 chip
arch -x86_64 brew install ta-lib
arch -x86_64 pip install TA-Lib
```

#### 2. API Connection Issues
- Verify API keys are correct
- Check network connectivity
- Ensure API rate limits aren't exceeded
- Confirm market hours for live data

#### 3. Data Issues
- Check data provider status
- Verify symbol formats
- Ensure sufficient historical data
- Check for market holidays/closures

#### 4. Email Notifications Not Working
- Verify SMTP settings
- Use App Passwords for Gmail
- Check firewall settings
- Verify recipient email addresses

### Getting Help
1. Check the logs in the `logs/` directory
2. Review error messages carefully
3. Verify configuration settings
4. Test with paper trading first
5. Start with single strategies before using multiple

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
isort .
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation for changes

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for trading API
- [TA-Lib](https://ta-lib.org/) for technical analysis
- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [Plotly](https://plotly.com/) for interactive visualizations
- [scikit-learn](https://scikit-learn.org/) for machine learning

## 📞 Support

For questions, issues, or contributions:
1. Check existing issues and documentation
2. Create detailed bug reports with logs
3. Include system information and configuration
4. Test with minimal examples when possible

---

**Remember: Trading is risky. Never risk money you cannot afford to lose. This software is for educational and research purposes. Use at your own risk.**