import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    Core Backtesting Engine.
    Simulates daily trading of an asset based on strategy signals, including:
    - Capital & portfolio tracking
    - Slippage & commissions
    - Position sizing models
    - Risk controls (Stop-Loss and Take-Profit)
    - Detailed trade ledger and daily equity tracking
    """
    def __init__(self, df: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000.0,
                 commission_pct: float = 0.001, slippage_pct: float = 0.0005,
                 sizing_type: str = "percent_equity", sizing_value: float = 1.0,
                 stop_loss_pct: float = 0.0, take_profit_pct: float = 0.0):
        """
        Parameters:
        - df: DataFrame with OHLCV daily data.
        - signals: Series of target signals (-1, 0, 1), indexed by Date.
        - initial_capital: Starting cash.
        - commission_pct: Commission per trade as a fraction of trade value (e.g. 0.001 = 0.1%).
        - slippage_pct: Slippage per trade as a fraction of asset price (e.g. 0.0005 = 0.05%).
        - sizing_type: 'percent_equity', 'fixed_capital', or 'fixed_units'.
        - sizing_value: Value corresponding to sizing_type (e.g. 0.1 = 10% equity, 10000 = $10k per trade, 100 = 100 shares).
        - stop_loss_pct: Stop-loss trigger threshold (e.g. 0.02 = 2%). 0.0 disables.
        - take_profit_pct: Take-profit trigger threshold (e.g. 0.05 = 5%). 0.0 disables.
        """
        # Align index
        self.df = df.copy()
        self.signals = signals.reindex(df.index).fillna(0).astype(int)
        
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        self.sizing_type = sizing_type
        self.sizing_value = sizing_value
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Output placeholders
        self.equity_curve = None
        self.trades = []

    def run(self):
        """
        Runs the backtest simulation.
        """
        logger.info("Initializing backtest run...")
        
        # State variables
        cash = self.initial_capital
        position_size = 0.0  # units of the asset
        position_direction = 0  # 1: Long, -1: Short, 0: Flat
        entry_price = 0.0
        entry_date = None
        
        # Lists for tracking daily stats
        daily_equity = []
        daily_cash = []
        daily_holdings = []
        
        # We loop through index using values for performance
        dates = self.df.index
        opens = self.df['Open'].values
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        
        # Signals shifted by 1 day to execute on NEXT day open (prevents lookahead bias)
        # Shift the signals so signals[t] determines trade execution on t+1 open
        # We can also do target execution at t open based on t signal if t signal is computed at t-1 close.
        # Shift signals by 1 to make it execute on next bar open:
        shifted_signals = self.signals.shift(1).fillna(0).astype(int).values
        
        for i in range(len(self.df)):
            date = dates[i]
            open_p = opens[i]
            high_p = highs[i]
            low_p = lows[i]
            close_p = closes[i]
            
            # Target signal for this bar (computed based on previous closes)
            target_signal = shifted_signals[i]
            
            # --- 1. Risk Controls / Intra-bar Exits (Stop Loss & Take Profit) ---
            exit_triggered = False
            exit_reason = "SIGNAL"
            exit_price = 0.0
            
            if position_direction != 0:
                # Long position risk checks
                if position_direction == 1:
                    stop_p = entry_price * (1.0 - self.stop_loss_pct) if self.stop_loss_pct > 0 else -1.0
                    tp_p = entry_price * (1.0 + self.take_profit_pct) if self.take_profit_pct > 0 else float('inf')
                    
                    # Check if stop-loss was hit
                    # We check if Low is below or equal to Stop price
                    # And Take Profit: if High is above or equal to TP price
                    if self.stop_loss_pct > 0 and low_p <= stop_p:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                        # Exit at stop price, or Open if Open gapped below stop price
                        exit_price = min(open_p, stop_p)
                    elif self.take_profit_pct > 0 and high_p >= tp_p:
                        exit_triggered = True
                        exit_reason = "TAKE_PROFIT"
                        # Exit at TP price, or Open if Open gapped above TP price
                        exit_price = max(open_p, tp_p)
                
                # Short position risk checks
                elif position_direction == -1:
                    stop_p = entry_price * (1.0 + self.stop_loss_pct) if self.stop_loss_pct > 0 else float('inf')
                    tp_p = entry_price * (1.0 - self.take_profit_pct) if self.take_profit_pct > 0 else -1.0
                    
                    if self.stop_loss_pct > 0 and high_p >= stop_p:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                        exit_price = max(open_p, stop_p)
                    elif self.take_profit_pct > 0 and low_p <= tp_p:
                        exit_triggered = True
                        exit_reason = "TAKE_PROFIT"
                        exit_price = min(open_p, tp_p)
            
            if exit_triggered:
                # Handle execution of SL/TP
                # Apply slippage & commission
                if position_direction == 1:
                    # Selling to exit Long
                    executed_exit = exit_price * (1.0 - self.slippage_pct)
                    trade_value = position_size * executed_exit
                    commission = trade_value * self.commission_pct
                    pnl = trade_value - (position_size * entry_price) - commission
                    pnl_pct = (executed_exit / entry_price - 1.0) - self.commission_pct - self.slippage_pct
                    
                    cash += (trade_value - commission)
                else: # position_direction == -1
                    # Buying to cover Short
                    executed_exit = exit_price * (1.0 + self.slippage_pct)
                    trade_value = position_size * executed_exit
                    commission = trade_value * self.commission_pct
                    pnl = (position_size * entry_price) - trade_value - commission
                    pnl_pct = (entry_price / executed_exit - 1.0) - self.commission_pct - self.slippage_pct
                    
                    # When shorting, cash was increased at entry by entry_value
                    # So now we subtract the cover value + commission
                    cash -= (trade_value + commission)
                
                self.trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': executed_exit,
                    'direction': 'LONG' if position_direction == 1 else 'SHORT',
                    'size': position_size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100.0,
                    'exit_reason': exit_reason
                })
                
                position_size = 0.0
                position_direction = 0
                entry_price = 0.0
                entry_date = None
            
            # --- 2. Signal-Based Trade Execution ---
            # If no SL/TP exit happened, check for signal-based entry/exit
            if not exit_triggered and target_signal != position_direction:
                # First, close existing position if any
                if position_direction != 0:
                    # Closing trade at the Open price of today
                    if position_direction == 1:
                        # Sell Long
                        executed_exit = open_p * (1.0 - self.slippage_pct)
                        trade_value = position_size * executed_exit
                        commission = trade_value * self.commission_pct
                        pnl = trade_value - (position_size * entry_price) - commission
                        pnl_pct = (executed_exit / entry_price - 1.0) - self.commission_pct - self.slippage_pct
                        
                        cash += (trade_value - commission)
                    else: # position_direction == -1
                        # Cover Short
                        executed_exit = open_p * (1.0 + self.slippage_pct)
                        trade_value = position_size * executed_exit
                        commission = trade_value * self.commission_pct
                        pnl = (position_size * entry_price) - trade_value - commission
                        pnl_pct = (entry_price / executed_exit - 1.0) - self.commission_pct - self.slippage_pct
                        
                        cash -= (trade_value + commission)
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': executed_exit,
                        'direction': 'LONG' if position_direction == 1 else 'SHORT',
                        'size': position_size,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100.0,
                        'exit_reason': 'SIGNAL'
                    })
                    
                    position_size = 0.0
                    position_direction = 0
                    entry_price = 0.0
                    entry_date = None
                
                # Open new position if target signal is not flat (0)
                if target_signal != 0:
                    # Sizing logic based on current equity
                    current_equity = cash
                    
                    if self.sizing_type == "percent_equity":
                        trade_capital = current_equity * self.sizing_value
                    elif self.sizing_type == "fixed_capital":
                        trade_capital = min(self.sizing_value, current_equity)
                    elif self.sizing_type == "fixed_units":
                        trade_capital = self.sizing_value * open_p
                    else:
                        trade_capital = current_equity
                    
                    # Ensure trade capital doesn't exceed cash
                    trade_capital = min(trade_capital, cash)
                    
                    if trade_capital > 0:
                        position_direction = target_signal
                        entry_date = date
                        
                        if position_direction == 1:
                            # Buy Long
                            entry_price = open_p * (1.0 + self.slippage_pct)
                            # Size = trade_capital / entry_price
                            position_size = trade_capital / entry_price
                            commission = (position_size * entry_price) * self.commission_pct
                            cash -= (position_size * entry_price + commission)
                        else: # position_direction == -1
                            # Sell Short
                            entry_price = open_p * (1.0 - self.slippage_pct)
                            position_size = trade_capital / entry_price
                            commission = (position_size * entry_price) * self.commission_pct
                            # Cash increases on short sale, but we subtract commission
                            cash += (position_size * entry_price - commission)
            
            # --- 3. Daily Accounting & Valuation ---
            # Valuation is done at the close price of the day
            if position_direction == 1:
                # Long: position value goes up/down with close price
                holdings_val = position_size * close_p
                equity = cash + holdings_val
            elif position_direction == -1:
                # Short: we sold at entry_price, cash went up. We must cover at close_p.
                # Profit = (entry - close) * size
                # Value of short liability is size * close_p.
                # Net equity = Cash - LiabilityValue (since cash includes short proceeds)
                holdings_val = -position_size * close_p
                equity = cash + (2.0 * position_size * entry_price) - (position_size * close_p)
                # Wait, let's verify cash accounting:
                # Initial cash was C0. 
                # At short entry: cash = C0 + (size * entry_price) - commission.
                # At day close, equity = current cash - (size * close_p)
                # Since current cash is C0 + size * entry_price - commission,
                # equity = (C0 + size * entry_price - commission) - (size * close_p)
                # This equals: cash_level - size * close_p. Let's make sure it matches.
                # Yes, Cash is: cash. Liability is: position_size * close_p.
                # Net equity is cash - liability. Let's write it that way, it's simpler!
                equity = cash - (position_size * close_p)
                # Wait! Let's double check.
                # Let's say cash = 100k. We short 1 share at $100.
                # Cash becomes 100k + 100 = 100,100 (ignoring fees).
                # If Close is $100, equity = 100,100 - 100 = 100k. Correct!
                # If Close is $90 (we are making money), equity = 100,100 - 90 = 100,010. Correct!
                # If Close is $110 (we are losing money), equity = 100,100 - 110 = 99,990. Correct!
                # Yes, equity = cash - (position_size * close_p) for short! That is mathematically correct.
            else:
                holdings_val = 0.0
                equity = cash
                
            daily_equity.append(equity)
            daily_cash.append(cash)
            daily_holdings.append(holdings_val)
            
        # Build results DataFrame
        self.equity_curve = pd.DataFrame({
            'Equity': daily_equity,
            'Cash': daily_cash,
            'Holdings': daily_holdings
        }, index=self.df.index)
        
        # Benchmark comparison: Buy and Hold Strategy
        # Buy at Open of day 1, hold to Close of last day
        benchmark_shares = self.initial_capital / (opens[0] * (1.0 + self.slippage_pct))
        benchmark_commission = (benchmark_shares * opens[0] * (1.0 + self.slippage_pct)) * self.commission_pct
        benchmark_cash = self.initial_capital - (benchmark_shares * opens[0] * (1.0 + self.slippage_pct)) - benchmark_commission
        
        self.equity_curve['Benchmark'] = benchmark_cash + (benchmark_shares * self.df['Close'])
        
        # Format trades list to DataFrame
        self.trades_df = pd.DataFrame(self.trades)
        if not self.trades_df.empty:
            self.trades_df['duration'] = (self.trades_df['exit_date'] - self.trades_df['entry_date']).dt.days
        else:
            self.trades_df = pd.DataFrame(columns=['entry_date', 'entry_price', 'exit_date', 'exit_price', 
                                                    'direction', 'size', 'pnl', 'pnl_pct', 'exit_reason', 'duration'])
        
        logger.info(f"Backtest run complete. Total Trades: {len(self.trades_df)}")
        return self.equity_curve, self.trades_df
