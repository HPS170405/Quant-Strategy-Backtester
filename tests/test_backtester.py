import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import generate_synthetic_data, load_data
from src.strategies import MACrossoverStrategy, MomentumStrategy, MeanReversionStrategy
from src.engine import Backtester
from src.metrics import calculate_metrics

class TestQuantBacktester(unittest.TestCase):
    def setUp(self):
        # Generate clean synthetic data for SPY
        self.start_date = "2023-01-01"
        self.end_date = "2023-06-30"
        self.df = generate_synthetic_data("SPY", self.start_date, self.end_date)
        
    def test_data_loader(self):
        self.assertIsNotNone(self.df)
        self.assertFalse(self.df.empty)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, self.df.columns)
        self.assertTrue(isinstance(self.df.index, pd.DatetimeIndex))

    def test_crossover_strategy(self):
        strat = MACrossoverStrategy(fast_window=10, slow_window=30, allow_short=True)
        signals = strat.generate_signals(self.df)
        self.assertEqual(len(signals), len(self.df))
        # Warmup period should be 0 (slow_window - 1 days)
        self.assertEqual(signals.iloc[0], 0)
        self.assertTrue(all(sig in [-1, 0, 1] for sig in signals.unique()))

    def test_momentum_strategy(self):
        strat = MomentumStrategy(lookback_period=10, indicator="ROC")
        signals = strat.generate_signals(self.df)
        self.assertEqual(len(signals), len(self.df))
        self.assertTrue(all(sig in [0, 1] for sig in signals.unique()))

    def test_mean_reversion_strategy(self):
        strat = MeanReversionStrategy(window=10, num_std=1.5)
        signals = strat.generate_signals(self.df)
        self.assertEqual(len(signals), len(self.df))
        self.assertTrue(all(sig in [0, 1] for sig in signals.unique()))

    def test_backtest_engine(self):
        strat = MACrossoverStrategy(fast_window=10, slow_window=20, allow_short=False)
        signals = strat.generate_signals(self.df)
        
        backtester = Backtester(
            df=self.df,
            signals=signals,
            initial_capital=100000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            sizing_type="percent_equity",
            sizing_value=0.5, # 50% equity sizing
            stop_loss_pct=0.02, # 2% SL
            take_profit_pct=0.05 # 5% TP
        )
        
        equity_curve, trades_df = backtester.run()
        
        self.assertEqual(len(equity_curve), len(self.df))
        self.assertIn('Equity', equity_curve.columns)
        self.assertIn('Cash', equity_curve.columns)
        self.assertIn('Benchmark', equity_curve.columns)
        
        # Calculate metrics
        metrics = calculate_metrics(equity_curve['Equity'], trades_df)
        self.assertIn('total_return', metrics)
        self.assertIn('cagr', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)

if __name__ == "__main__":
    unittest.main()
