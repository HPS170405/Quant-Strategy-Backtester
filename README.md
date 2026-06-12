# 📈 Quantitative Trading Strategy Backtester

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quant-strategy-backtester-hps.streamlit.app/)

A professional-grade, high-performance quantitative trading backtesting framework and interactive Streamlit dashboard. Build, test, and analyze momentum, moving-average crossover, and mean-reversion strategies on historical market data with transaction costs, slippage, position sizing, and risk controls.

---

## ✨ Features

- **Backtesting Engine**: Event-driven execution logic simulating commissions, slippage, and position tracking.
- **Three Core Strategies**:
  - **Moving Average Crossover**: Crossover signals using SMA or EMA with custom fast/slow lookback periods.
  - **Momentum (ROC / RSI)**: Breaks down price trends using Rate of Change or Relative Strength Index values.
  - **Mean Reversion (Bollinger Bands)**: Trades asset deviations from the rolling mean with standardized standard deviation envelopes.
- **Risk Management & Position Sizing**:
  - Stop Loss (SL) and Take Profit (TP) trigger simulation.
  - Sizing models including Fixed Capital, Fixed Units, and Percent of Equity.
- **Stunning UI Dashboard**:
  - Interactive Plotly equity curves vs. Buy & Hold benchmark.
  - Drawdown area plots and return distribution histograms.
  - Buy/Sell trigger markers on price charts.
  - Searchable trade ledger log and multi-strategy side-by-side comparison.
- **Hybrid Data Source**: Pulls historical daily price data via Yahoo Finance or falls back to an offline-capable deterministic Geometric Brownian Motion synthetic data generator.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Data Engineering**: Pandas, NumPy
- **Visualizations**: Plotly, Streamlit
- **API Fetching**: yfinance

---

## 🚀 Getting Started

### 1. Installation

Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

To launch the Streamlit dashboard app locally:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### 3. Run Unit Tests

Execute the unit test suite covering loaders, engine calculations, strategies, and performance metrics:

```bash
python -m unittest tests/test_backtester.py
```

---

## 📊 Performance Metrics Computed

- **Total Return % & CAGR %**: Compound annualized growth rate.
- **Sharpe & Sortino Ratios**: Risk-adjusted returns focusing on total variance and downside volatility respectively.
- **Drawdowns**: Maximum Drawdown % and Maximum Drawdown Duration (days).
- **Trade Statistics**: Win Rate %, Profit Factor, average trade returns, and average hold duration.
