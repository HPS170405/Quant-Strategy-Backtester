import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates highly realistic daily OHLCV synthetic market data using Geometric Brownian Motion (GBM).
    Adjusts drift, volatility, and baseline price based on the asset ticker signature.
    """
    logger.info(f"Generating synthetic data for ticker: {ticker}")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create business day index
    date_range = pd.date_range(start=start, end=end, freq='B')
    n_days = len(date_range)
    if n_days == 0:
        # Fallback to daily calendar days if no business days
        date_range = pd.date_range(start=start, end=end, freq='D')
        n_days = len(date_range)
        
    if n_days == 0:
        raise ValueError("Start date must be before end date.")

    # Ticker specific parameters (drift annualized, volatility annualized, start price)
    ticker_profiles = {
        "SPY": {"drift": 0.08, "vol": 0.16, "base_price": 400.0, "volume_base": 80_000_000},
        "QQQ": {"drift": 0.12, "vol": 0.22, "base_price": 350.0, "volume_base": 50_000_000},
        "AAPL": {"drift": 0.15, "vol": 0.25, "base_price": 180.0, "volume_base": 60_000_000},
        "MSFT": {"drift": 0.14, "vol": 0.23, "base_price": 350.0, "volume_base": 25_000_000},
        "TSLA": {"drift": 0.25, "vol": 0.45, "base_price": 200.0, "volume_base": 90_000_000},
        "BTC-USD": {"drift": 0.40, "vol": 0.60, "base_price": 40000.0, "volume_base": 25_000_000_000},
        "ETH-USD": {"drift": 0.35, "vol": 0.70, "base_price": 2500.0, "volume_base": 15_000_000_000},
        "GLD": {"drift": 0.05, "vol": 0.12, "base_price": 190.0, "volume_base": 8_000_000},
    }
    
    # Default parameters for unknown tickers
    profile = ticker_profiles.get(ticker.upper(), {"drift": 0.10, "vol": 0.28, "base_price": 100.0, "volume_base": 5_000_000})
    
    mu = profile["drift"]
    sigma = profile["vol"]
    s0 = profile["base_price"]
    vol_base = profile["volume_base"]
    
    # Geometric Brownian Motion simulation
    dt = 1.0 / 252.0  # Daily time increment
    
    # Daily returns
    # St = St-1 * exp((mu - 0.5 * sigma^2)*dt + sigma*sqrt(dt)*Z)
    np.random.seed(hash(ticker) % 2**32) # Seed based on ticker to get reproducible synthetic data for same ticker
    random_shocks = np.random.normal(0, 1, n_days)
    
    price_path = np.zeros(n_days)
    price_path[0] = s0
    
    for t in range(1, n_days):
        drift_term = (mu - 0.5 * sigma**2) * dt
        shock_term = sigma * np.sqrt(dt) * random_shocks[t]
        price_path[t] = price_path[t-1] * np.exp(drift_term + shock_term)
        
    # Generate OHLC
    close_prices = price_path
    open_prices = np.zeros(n_days)
    high_prices = np.zeros(n_days)
    low_prices = np.zeros(n_days)
    volumes = np.zeros(n_days)
    
    # First day open
    open_prices[0] = s0 * (1 + np.random.normal(0, 0.005))
    
    # Daily price variations
    for t in range(n_days):
        if t > 0:
            # Open is close of yesterday + some gap
            gap_std = sigma * np.sqrt(dt) * 0.15 # gap is a fraction of daily volatility
            open_prices[t] = close_prices[t-1] * np.exp(np.random.normal(0, gap_std))
        
        # High and Low are functions of Open and Close
        max_oc = max(open_prices[t], close_prices[t])
        min_oc = min(open_prices[t], close_prices[t])
        
        # Intraday ranges
        intraday_vol = sigma * np.sqrt(dt) * 0.8
        high_prices[t] = max_oc * (1.0 + abs(np.random.normal(0, intraday_vol)))
        low_prices[t] = min_oc * (1.0 - abs(np.random.normal(0, intraday_vol)))
        
        # Volume: lognormal, scales higher on volatile days
        pct_change = abs((close_prices[t] - open_prices[t]) / open_prices[t])
        vol_multiplier = 1.0 + (pct_change / (sigma * np.sqrt(dt))) * 0.5
        volumes[t] = int(vol_base * np.random.lognormal(0, 0.3) * vol_multiplier)
        
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=date_range)
    
    # Set index name
    df.index.name = 'Date'
    return df

def load_data(ticker: str, start_date: str, end_date: str, force_synthetic: bool = False) -> pd.DataFrame:
    """
    Loads historical daily price data. First tries yfinance. 
    If that fails, falls back to generating highly realistic synthetic data.
    """
    if force_synthetic:
        return generate_synthetic_data(ticker, start_date, end_date)
        
    try:
        logger.info(f"Downloading historical data for {ticker} from {start_date} to {end_date} via yfinance...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty or len(df) < 5:
            logger.warning(f"No data retrieved or dataset too small for {ticker} from yfinance. Falling back to synthetic.")
            return generate_synthetic_data(ticker, start_date, end_date)
            
        # Clean multi-index columns if yfinance returns them (newer yfinance versions can sometimes do this)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure required columns are present
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                # Try finding case-insensitive match or raise
                matches = [c for c in df.columns if c.lower() == col.lower()]
                if matches:
                    df = df.rename(columns={matches[0]: col})
                else:
                    raise KeyError(f"Required column '{col}' missing from downloaded data.")
                    
        # Sort index and drop NaNs
        df = df[required_cols].sort_index().dropna()
        # Ensure indices are DatetimeIndex
        df.index = pd.to_datetime(df.index)
        
        logger.info(f"Successfully loaded {len(df)} rows for {ticker} from yfinance.")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}. Falling back to synthetic data.")
        return generate_synthetic_data(ticker, start_date, end_date)

if __name__ == "__main__":
    # Test data loading
    df = load_data("AAPL", "2023-01-01", "2023-12-31")
    print(df.head())
