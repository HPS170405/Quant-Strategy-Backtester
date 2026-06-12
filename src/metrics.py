import pandas as pd
import numpy as np

def calculate_metrics(equity_series: pd.Series, trades_df: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates detailed performance metrics from the equity curve and trade log.
    
    Parameters:
    - equity_series: Series of daily equity values, indexed by Date.
    - trades_df: DataFrame containing the trade log.
    - risk_free_rate: Annual risk-free rate (e.g. 0.02 for 2%).
    
    Returns:
    - dict of metrics.
    """
    metrics = {}
    
    if equity_series.empty:
        return metrics

    # 1. Basic returns
    start_equity = equity_series.iloc[0]
    end_equity = equity_series.iloc[-1]
    total_return = (end_equity / start_equity) - 1.0
    metrics['total_return'] = total_return * 100.0
    
    # Calculate years elapsed
    days_elapsed = (equity_series.index[-1] - equity_series.index[0]).days
    years = max(days_elapsed / 365.25, 0.0027)  # prevent division by zero or negative
    
    # CAGR
    if end_equity > 0:
        cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0
    metrics['cagr'] = cagr * 100.0
    
    # 2. Daily returns stats
    daily_returns = equity_series.pct_change().dropna()
    
    # Annualized Sharpe and Sortino
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - daily_rf
    
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        mean_excess = excess_returns.mean()
        std_returns = daily_returns.std()
        
        # Sharpe
        sharpe = (mean_excess / std_returns) * np.sqrt(252)
        
        # Sortino (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            downside_std = downside_returns.std()
            sortino = (mean_excess / downside_std) * np.sqrt(252)
        else:
            sortino = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0
        
    metrics['sharpe_ratio'] = sharpe
    metrics['sortino_ratio'] = sortino
    metrics['volatility'] = daily_returns.std() * np.sqrt(252) * 100.0
    
    # 3. Drawdown Calculations
    cum_max = equity_series.cummax()
    drawdowns = (equity_series / cum_max) - 1.0
    max_drawdown = drawdowns.min()
    metrics['max_drawdown'] = max_drawdown * 100.0
    
    # Max Drawdown Duration
    is_in_dd = drawdowns < 0
    dd_duration = 0
    current_dd_duration = 0
    
    # Vectorized or simple search for max drawdown duration in calendar days
    # Let's do a simple calculation of peak to recovery
    peak_date = equity_series.index[0]
    max_dd_days = 0
    
    # Find max peak to recovery time
    current_peak_val = start_equity
    current_peak_date = equity_series.index[0]
    
    for idx, date in enumerate(equity_series.index):
        val = equity_series.iloc[idx]
        if val >= current_peak_val:
            current_peak_val = val
            current_peak_date = date
        else:
            duration = (date - current_peak_date).days
            if duration > max_dd_days:
                max_dd_days = duration
                
    metrics['max_drawdown_duration_days'] = max_dd_days
    
    # 4. Trade-level statistics
    total_trades = len(trades_df)
    metrics['total_trades'] = total_trades
    
    if total_trades > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades
        metrics['win_rate'] = win_rate * 100.0
        
        total_gains = winning_trades['pnl'].sum()
        total_losses = abs(losing_trades['pnl'].sum())
        
        if total_losses > 0:
            profit_factor = total_gains / total_losses
        else:
            profit_factor = float('inf') if total_gains > 0 else 1.0
            
        metrics['profit_factor'] = profit_factor
        metrics['avg_trade_pnl_pct'] = trades_df['pnl_pct'].mean()
        metrics['avg_trade_duration_days'] = trades_df['duration'].mean()
        
        # Max win / Max loss
        metrics['max_win_pct'] = trades_df['pnl_pct'].max()
        metrics['max_loss_pct'] = trades_df['pnl_pct'].min()
    else:
        metrics['win_rate'] = 0.0
        metrics['profit_factor'] = 0.0
        metrics['avg_trade_pnl_pct'] = 0.0
        metrics['avg_trade_duration_days'] = 0.0
        metrics['max_win_pct'] = 0.0
        metrics['max_loss_pct'] = 0.0
        
    return metrics
