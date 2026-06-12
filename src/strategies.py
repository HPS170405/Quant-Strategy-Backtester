import pandas as pd
import numpy as np

class Strategy:
    """
    Base Strategy Class. All quantitative strategies should inherit from this.
    """
    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Processes the input dataframe and returns a Series of trading signals:
         1: Go Long
        -1: Go Short (if allowed, else Flat/Cash)
         0: Flat / Cash
        """
        raise NotImplementedError("Strategies must implement generate_signals method.")


class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.
    Generates long/short signals based on the crossover of a fast and a slow moving average.
    """
    def __init__(self, fast_window: int = 50, slow_window: int = 200, ma_type: str = "SMA", allow_short: bool = False):
        super().__init__(f"{ma_type} Crossover ({fast_window}/{slow_window})")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.ma_type = ma_type.upper()
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        signals = pd.Series(index=df.index, data=0, dtype=int)
        
        # Calculate moving averages
        if self.ma_type == "EMA":
            fast_ma = close.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow_window, adjust=False).mean()
        else: # Default SMA
            fast_ma = close.rolling(window=self.fast_window).mean()
            slow_ma = close.rolling(window=self.slow_window).mean()

        # Signal generation
        # Buy/Long (1) when Fast MA > Slow MA
        # Sell/Short (-1) when Fast MA < Slow MA
        # Default flat (0) where we lack enough data
        
        # Determine raw positioning
        position = np.where(fast_ma > slow_ma, 1, -1 if self.allow_short else 0)
        
        # Fill first slow_window - 1 elements with 0 because MA is not calculated yet
        position[:self.slow_window - 1] = 0
        
        signals.iloc[:] = position
        return signals


class MomentumStrategy(Strategy):
    """
    Momentum Strategy based on Rate of Change (ROC) or Relative Strength Index (RSI).
    """
    def __init__(self, lookback_period: int = 14, indicator: str = "RSI", 
                 rsi_upper: float = 70.0, rsi_lower: float = 30.0, allow_short: bool = False):
        super().__init__(f"Momentum ({indicator} - {lookback_period})")
        self.lookback_period = lookback_period
        self.indicator = indicator.upper()
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.allow_short = allow_short

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / np.where(loss == 0, 1e-10, loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        signals = pd.Series(index=df.index, data=0, dtype=int)
        
        if self.indicator == "RSI":
            rsi = self._calculate_rsi(close, self.lookback_period)
            
            # Simple RSI trend follower:
            # Go Long (1) when RSI > 50 (bullish momentum)
            # Go Short/Flat when RSI < 50 (bearish momentum)
            # Alternatively, can implement swing breakout momentum: 
            # Go Long if RSI crosses above 50, etc. Let's do a reliable momentum trend indicator:
            # Long when RSI > 50, Short/Flat when RSI <= 50.
            position = np.where(rsi > 50, 1, -1 if self.allow_short else 0)
            
            # Mask the warm-up period
            position[:self.lookback_period] = 0
            signals.iloc[:] = position
            
        else: # Default ROC (Rate of Change)
            # ROC = ((Close_t - Close_t-n) / Close_t-n) * 100
            roc = close.pct_change(periods=self.lookback_period) * 100
            
            # Go Long if momentum is positive, Short/Flat if negative
            position = np.where(roc > 0, 1, -1 if self.allow_short else 0)
            position[:self.lookback_period] = 0
            signals.iloc[:] = position
            
        return signals


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy based on Bollinger Bands.
    """
    def __init__(self, window: int = 20, num_std: float = 2.0, allow_short: bool = False):
        super().__init__(f"Mean Reversion (BB {window}/{num_std})")
        self.window = window
        self.num_std = num_std
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        signals = pd.Series(index=df.index, data=0, dtype=int)
        
        # Calculate Bollinger Bands
        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()
        
        upper_band = rolling_mean + (self.num_std * rolling_std)
        lower_band = rolling_mean - (self.num_std * rolling_std)
        
        # Let's iterate or vectorize the state-machine for Bollinger Bands
        # State machine logic:
        # If price falls below lower band: Buy/Long (1)
        # If price rises above upper band: Short (-1) or Flat (0)
        # If price crosses the middle band (mean): Close existing positions (0)
        # We'll use a loop to accurately simulate this state-machine signal
        
        current_state = 0 # 0: flat, 1: long, -1: short
        state_log = []
        
        for i in range(len(df)):
            if i < self.window - 1:
                state_log.append(0)
                continue
                
            price = close.iloc[i]
            upper = upper_band.iloc[i]
            lower = lower_band.iloc[i]
            mean = rolling_mean.iloc[i]
            
            if current_state == 0:
                # Flat -> Entry rules
                if price <= lower:
                    current_state = 1
                elif price >= upper:
                    current_state = -1 if self.allow_short else 0
            elif current_state == 1:
                # Long -> Exit rules (take profit / mean reversion to mid band)
                if price >= mean:
                    # Reverted to mean, close position
                    current_state = 0
            elif current_state == -1:
                # Short -> Exit rules (reverted to mid band)
                if price <= mean:
                    # Reverted to mean, close position
                    current_state = 0
                    
            state_log.append(current_state)
            
        signals.iloc[:] = state_log
        return signals
