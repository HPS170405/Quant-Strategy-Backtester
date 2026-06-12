import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Append current dir to system path to import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data
from src.strategies import MACrossoverStrategy, MomentumStrategy, MeanReversionStrategy
from src.engine import Backtester
from src.metrics import calculate_metrics

# Page configuration
st.set_page_config(
    page_title="Quant Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance styling and give it a premium dark-mode feeling
st.markdown("""
<style>
    .reportview-container {
        background: #0A0F1D;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #00E5FF;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #121A2E;
        border-radius: 4px;
        color: #94A3B8;
        padding-left: 20px;
        padding-right: 20px;
        border: 1px solid #1E293B;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E5FF !important;
        color: #0A0F1D !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("📈 Quantitative Strategy Backtesting Dashboard")
st.markdown("Evaluate, optimize, and compare algorithmic trading strategies with historical market data.")

# ----------------- SIDEBAR -----------------
st.sidebar.header("🛠️ Configuration Panel")

# Section 1: Asset Selection
st.sidebar.subheader("1. Asset & Period")
asset_options = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "BTC-USD", "ETH-USD", "GLD", "Custom Ticker"]
selected_asset_choice = st.sidebar.selectbox("Select Asset Ticker", asset_options, index=0)

if selected_asset_choice == "Custom Ticker":
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. NVDA, AMZN)", value="NVDA").upper()
else:
    ticker = selected_asset_choice

# Date Selection
default_start = datetime.today() - timedelta(days=365 * 3) # 3 years ago
default_end = datetime.today()
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

# Data Source
data_mode = st.sidebar.radio("Data Extraction Mode", ["Yahoo Finance (Online)", "Synthetic Data Fallback (Offline)"], index=0)
force_synth = (data_mode == "Synthetic Data Fallback (Offline)")

# Section 2: Strategy Definition
st.sidebar.subheader("2. Trading Strategy")
strat_type = st.sidebar.selectbox(
    "Choose Strategy",
    ["Moving Average Crossover", "Momentum (ROC/RSI)", "Mean Reversion (Bollinger Bands)"]
)

allow_short = st.sidebar.checkbox("Allow Short Selling", value=False, help="Enable short positioning when sell signals trigger.")

# Display Strategy Specific Options
if strat_type == "Moving Average Crossover":
    ma_type = st.sidebar.selectbox("Moving Average Type", ["SMA", "EMA"])
    fast_window = st.sidebar.slider("Fast Window (Days)", min_value=5, max_value=100, value=50)
    slow_window = st.sidebar.slider("Slow Window (Days)", min_value=20, max_value=300, value=200)
    
    if fast_window >= slow_window:
        st.sidebar.error("Fast window must be smaller than slow window!")
        
    strategy = MACrossoverStrategy(fast_window=fast_window, slow_window=slow_window, ma_type=ma_type, allow_short=allow_short)

elif strat_type == "Momentum (ROC/RSI)":
    momentum_indicator = st.sidebar.selectbox("Momentum Indicator", ["RSI", "ROC"])
    lookback_period = st.sidebar.slider("Lookback Period (Days)", min_value=3, max_value=100, value=14)
    
    strategy = MomentumStrategy(lookback_period=lookback_period, indicator=momentum_indicator, allow_short=allow_short)

else:  # Mean Reversion (Bollinger Bands)
    bb_window = st.sidebar.slider("Bollinger Bands Window (Days)", min_value=5, max_value=100, value=20)
    bb_std = st.sidebar.slider("Standard Deviation Multiplier", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    
    strategy = MeanReversionStrategy(window=bb_window, num_std=bb_std, allow_short=allow_short)

# Section 3: Portfolio & Risk Parameters
st.sidebar.subheader("3. Capital & Risk Controls")
initial_capital = st.sidebar.number_input("Starting Balance ($)", min_value=1000.0, value=100000.0, step=1000.0)

sizing_type_choice = st.sidebar.selectbox("Position Sizing Model", ["Percent of Equity", "Fixed Capital", "Fixed Units"])
sizing_map = {"Percent of Equity": "percent_equity", "Fixed Capital": "fixed_capital", "Fixed Units": "fixed_units"}
sizing_type = sizing_map[sizing_type_choice]

if sizing_type == "percent_equity":
    sizing_value = st.sidebar.slider("Percent of Equity per Trade", min_value=5, max_value=100, value=100) / 100.0
elif sizing_type == "fixed_capital":
    sizing_value = st.sidebar.number_input("Fixed Dollar Amount per Trade ($)", min_value=100.0, value=10000.0, step=500.0)
else:
    sizing_value = st.sidebar.number_input("Fixed Units per Trade", min_value=1.0, value=100.0, step=1.0)

commission_pct = st.sidebar.slider("Commission Rate (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100.0
slippage_pct = st.sidebar.slider("Slippage Rate (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01) / 100.0

# Stop loss / Take profit sliders
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.5) / 100.0
take_profit_pct = st.sidebar.slider("Take Profit (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5) / 100.0


# ----------------- DATA LOADING & ENGINE EXECUTION -----------------
@st.cache_data(show_spinner="Downloading market data...")
def get_market_data(ticker, start, end, force_synth):
    return load_data(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), force_synthetic=force_synth)

try:
    df = get_market_data(ticker, start_date, end_date, force_synth)
except Exception as e:
    st.error(f"Failed to load data for {ticker}: {e}")
    st.stop()

# Generate Signals
signals = strategy.generate_signals(df)

# Run Backtest
backtester = Backtester(
    df=df,
    signals=signals,
    initial_capital=initial_capital,
    commission_pct=commission_pct,
    slippage_pct=slippage_pct,
    sizing_type=sizing_type,
    sizing_value=sizing_value,
    stop_loss_pct=stop_loss_pct,
    take_profit_pct=take_profit_pct
)

with st.spinner("Simulating portfolio backtest..."):
    equity_curve, trades_df = backtester.run()

# Calculate Metrics
metrics = calculate_metrics(equity_curve['Equity'], trades_df)
benchmark_metrics = calculate_metrics(equity_curve['Benchmark'], pd.DataFrame())


# ----------------- MAIN PANEL LAYOUT -----------------

# Tabs setup
tab_dash, tab_signals, tab_trades, tab_compare, tab_data = st.tabs([
    "📊 Performance Dashboard",
    "📈 Trading Signals Explorer",
    "📋 Trade Ledger Log",
    "⚔️ Strategy Comparison",
    "🔍 Raw Data Explorer"
])

# Plotly dark template helper
plotly_template = "plotly_dark"
chart_bgcolor = "#0A0F1D"
paper_bgcolor = "#121A2E"

# ----------------- TAB 1: DASHBOARD SUMMARY -----------------
with tab_dash:
    # Key Metrics Grid
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        # Total Return
        strat_tr = metrics['total_return']
        bench_tr = benchmark_metrics['total_return']
        st.metric(
            label="Total Return",
            value=f"{strat_tr:.2f}%",
            delta=f"vs. Benchmark: {bench_tr:.2f}%"
        )
    with c2:
        # CAGR
        strat_cagr = metrics['cagr']
        bench_cagr = benchmark_metrics['cagr']
        st.metric(
            label="Annualized CAGR",
            value=f"{strat_cagr:.2f}%",
            delta=f"vs. Benchmark: {bench_cagr:.2f}%"
        )
    with c3:
        # Sharpe Ratio
        strat_sharpe = metrics['sharpe_ratio']
        bench_sharpe = benchmark_metrics['sharpe_ratio']
        st.metric(
            label="Sharpe Ratio",
            value=f"{strat_sharpe:.2f}",
            delta=f"vs. Benchmark: {bench_sharpe:.2f}",
            delta_color="normal"
        )
    with c4:
        # Maximum Drawdown
        strat_dd = metrics['max_drawdown']
        bench_dd = benchmark_metrics['max_drawdown']
        st.metric(
            label="Max Drawdown",
            value=f"{strat_dd:.2f}%",
            delta=f"vs. Benchmark: {bench_dd:.2f}%",
            delta_color="inverse"
        )

    st.markdown("---")

    # Equity Curve Plotly Chart
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve['Equity'],
        mode='lines', name=f'Strategy: {strategy.name}',
        line=dict(color='#00E5FF', width=2.5)
    ))
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve['Benchmark'],
        mode='lines', name=f'Benchmark (Buy & Hold {ticker})',
        line=dict(color='#64748B', width=1.5, dash='dash')
    ))
    
    fig_equity.update_layout(
        title=dict(text=f"Portfolio Value Over Time (Initial: ${initial_capital:,.2f})", font=dict(size=18, color='#E2E8F0')),
        xaxis_title="Date",
        yaxis_title="Portfolio Equity ($)",
        template=plotly_template,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=chart_bgcolor,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor='#1E293B'),
        yaxis=dict(showgrid=True, gridcolor='#1E293B')
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # Drawdown Chart
    cum_max = equity_curve['Equity'].cummax()
    dd_pct = ((equity_curve['Equity'] / cum_max) - 1.0) * 100.0
    
    cum_max_bench = equity_curve['Benchmark'].cummax()
    dd_pct_bench = ((equity_curve['Benchmark'] / cum_max_bench) - 1.0) * 100.0

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=equity_curve.index, y=dd_pct,
        mode='lines', fill='tozeroy', name='Strategy Drawdown',
        line=dict(color='#EF4444', width=1.5),
        fillcolor='rgba(239, 68, 68, 0.15)'
    ))
    fig_dd.add_trace(go.Scatter(
        x=equity_curve.index, y=dd_pct_bench,
        mode='lines', name='Benchmark Drawdown',
        line=dict(color='#64748B', width=1, dash='dot')
    ))

    fig_dd.update_layout(
        title=dict(text="Underwater Equity Drawdown (%)", font=dict(size=18, color='#E2E8F0')),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template=plotly_template,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=chart_bgcolor,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor='#1E293B'),
        yaxis=dict(showgrid=True, gridcolor='#1E293B')
    )
    st.plotly_chart(fig_dd, use_container_width=True)


# ----------------- TAB 2: SIGNALS & PRICE CHART -----------------
with tab_signals:
    st.subheader(f"Price Chart & Trade Execution Markers ({ticker})")
    st.markdown("Inspect specific entry and exit trigger markers relative to strategy rules.")

    fig_price = go.Figure()
    
    # Base Price Close line
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', name='Asset Price',
        line=dict(color='#E2E8F0', width=1.5)
    ))
    
    # Overlay strategy components if selected
    if strat_type == "Moving Average Crossover":
        # Add SMA/EMA fast/slow lines
        if ma_type == "EMA":
            fast_ma = df['Close'].ewm(span=fast_window, adjust=False).mean()
            slow_ma = df['Close'].ewm(span=slow_window, adjust=False).mean()
        else:
            fast_ma = df['Close'].rolling(window=fast_window).mean()
            slow_ma = df['Close'].rolling(window=slow_window).mean()
            
        fig_price.add_trace(go.Scatter(
            x=df.index, y=fast_ma,
            mode='lines', name=f'Fast MA ({fast_window}d)',
            line=dict(color='#38BDF8', width=1.2)
        ))
        fig_price.add_trace(go.Scatter(
            x=df.index, y=slow_ma,
            mode='lines', name=f'Slow MA ({slow_window}d)',
            line=dict(color='#F59E0B', width=1.2)
        ))
        
    elif strat_type == "Mean Reversion (Bollinger Bands)":
        rolling_mean = df['Close'].rolling(window=bb_window).mean()
        rolling_std = df['Close'].rolling(window=bb_window).std()
        upper_band = rolling_mean + (bb_std * rolling_std)
        lower_band = rolling_mean - (bb_std * rolling_std)
        
        fig_price.add_trace(go.Scatter(
            x=df.index, y=rolling_mean,
            mode='lines', name='MA (Mid Band)',
            line=dict(color='#F59E0B', width=1, dash='dash')
        ))
        fig_price.add_trace(go.Scatter(
            x=df.index, y=upper_band,
            mode='lines', name='Upper Band',
            line=dict(color='#34D399', width=1)
        ))
        fig_price.add_trace(go.Scatter(
            x=df.index, y=lower_band,
            mode='lines', name='Lower Band',
            line=dict(color='#EF4444', width=1)
        ))
        
    # Plot trade signals entries and exits
    if not trades_df.empty:
        # Separate Long entry, Long exit, Short entry, Short exit
        long_entries = trades_df[trades_df['direction'] == 'LONG']
        short_entries = trades_df[trades_df['direction'] == 'SHORT']
        
        fig_price.add_trace(go.Scatter(
            x=long_entries['entry_date'], y=long_entries['entry_price'],
            mode='markers', name='Buy (Long Entry)',
            marker=dict(symbol='triangle-up', size=12, color='#10B981', line=dict(width=1, color='#022C22'))
        ))
        fig_price.add_trace(go.Scatter(
            x=long_entries['exit_date'], y=long_entries['exit_price'],
            mode='markers', name='Sell (Long Exit)',
            marker=dict(symbol='triangle-down', size=12, color='#EF4444', line=dict(width=1, color='#450A0A'))
        ))
        
        if allow_short:
            fig_price.add_trace(go.Scatter(
                x=short_entries['entry_date'], y=short_entries['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=12, color='#EC4899', line=dict(width=1, color='#500724'))
            ))
            fig_price.add_trace(go.Scatter(
                x=short_entries['exit_date'], y=short_entries['exit_price'],
                mode='markers', name='Cover Short (Exit)',
                marker=dict(symbol='triangle-up', size=12, color='#3B82F6', line=dict(width=1, color='#172554'))
            ))
            
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Asset Price ($)",
        template=plotly_template,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=chart_bgcolor,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor='#1E293B'),
        yaxis=dict(showgrid=True, gridcolor='#1E293B')
    )
    st.plotly_chart(fig_price, use_container_width=True)


# ----------------- TAB 3: TRADE LOG -----------------
with tab_trades:
    st.subheader("📋 Executed Trades Ledger")
    
    if trades_df.empty:
        st.info("No trades were executed during the backtest timeframe.")
    else:
        # Display general stats for trade entries
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Total Trades", f"{metrics['total_trades']}")
        tc2.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        tc3.metric("Profit Factor", f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "N/A (No Losses)")
        tc4.metric("Avg. PnL per Trade", f"{metrics['avg_trade_pnl_pct']:.2f}%")
        
        st.markdown("---")
        
        # Format columns for display
        display_trades = trades_df.copy()
        display_trades['entry_price'] = display_trades['entry_price'].map('${:,.2f}'.format)
        display_trades['exit_price'] = display_trades['exit_price'].map('${:,.2f}'.format)
        display_trades['pnl'] = display_trades['pnl'].map('${:,.2f}'.format)
        display_trades['pnl_pct'] = display_trades['pnl_pct'].map('{:.2f}%'.format)
        display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d')
        display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d')
        display_trades['size'] = display_trades['size'].map('{:,.4f}'.format)
        
        st.dataframe(display_trades, use_container_width=True)
        
        # PnL distribution chart
        st.subheader("📊 Trade Return Distribution")
        fig_hist = px.histogram(
            trades_df, x="pnl_pct", nbins=20,
            title="Histogram of Trade Returns (%)",
            labels={'pnl_pct': 'Return (%)', 'count': 'Frequency'},
            color_discrete_sequence=['#00E5FF']
        )
        fig_hist.update_layout(
            template=plotly_template,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=chart_bgcolor,
            xaxis=dict(showgrid=True, gridcolor='#1E293B'),
            yaxis=dict(showgrid=True, gridcolor='#1E293B')
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ----------------- TAB 4: STRATEGY COMPARISON -----------------
with tab_compare:
    st.subheader("⚔️ Side-by-Side Strategy Crossover Analysis")
    st.markdown("Run a multi-strategy benchmark evaluation on the current asset.")
    
    if st.button("🚀 Run Comparative Analysis"):
        with st.spinner("Processing benchmark comparisons..."):
            # We will run default versions of the three strategy classes
            comp_strategies = {
                "MA Crossover (50/200)": MACrossoverStrategy(fast_window=50, slow_window=200, allow_short=allow_short),
                "Momentum (RSI 14)": MomentumStrategy(lookback_period=14, indicator="RSI", allow_short=allow_short),
                "Momentum (ROC 14)": MomentumStrategy(lookback_period=14, indicator="ROC", allow_short=allow_short),
                "Mean Reversion (BB 20/2.0)": MeanReversionStrategy(window=20, num_std=2.0, allow_short=allow_short),
                "Buy & Hold Benchmark": None
            }
            
            comparison_results = []
            
            for name, strat in comp_strategies.items():
                if name == "Buy & Hold Benchmark":
                    # Use benchmark curve
                    eq_curve = equity_curve['Benchmark']
                    c_metrics = benchmark_metrics
                else:
                    c_signals = strat.generate_signals(df)
                    c_backtester = Backtester(
                        df=df,
                        signals=c_signals,
                        initial_capital=initial_capital,
                        commission_pct=commission_pct,
                        slippage_pct=slippage_pct,
                        sizing_type=sizing_type,
                        sizing_value=sizing_value,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct
                    )
                    c_eq, c_tr = c_backtester.run()
                    c_metrics = calculate_metrics(c_eq['Equity'], c_tr)
                
                comparison_results.append({
                    "Strategy": name,
                    "Total Return (%)": c_metrics['total_return'],
                    "CAGR (%)": c_metrics['cagr'],
                    "Sharpe Ratio": c_metrics['sharpe_ratio'],
                    "Sortino Ratio": c_metrics['sortino_ratio'],
                    "Max Drawdown (%)": c_metrics['max_drawdown'],
                    "Max DD Duration (Days)": c_metrics['max_drawdown_duration_days'],
                    "Total Trades": c_metrics.get('total_trades', 0),
                    "Win Rate (%)": c_metrics.get('win_rate', 0.0)
                })
                
            comp_df = pd.DataFrame(comparison_results)
            
            # Format display
            formatted_comp_df = comp_df.copy()
            for col in ["Total Return (%)", "CAGR (%)", "Max Drawdown (%)", "Win Rate (%)"]:
                formatted_comp_df[col] = formatted_comp_df[col].map('{:.2f}%'.format)
            for col in ["Sharpe Ratio", "Sortino Ratio"]:
                formatted_comp_df[col] = formatted_comp_df[col].map('{:.2f}'.format)
                
            st.dataframe(formatted_comp_df, use_container_width=True, hide_index=True)
            
            # Comparative Bar Charts
            st.subheader("Visual Performance Metric Comparisons")
            col_bar1, col_bar2 = st.columns(2)
            
            with col_bar1:
                fig_comp_tr = px.bar(
                    comp_df, x="Strategy", y="Total Return (%)",
                    title="Total Return by Strategy (%)",
                    color="Total Return (%)",
                    color_continuous_scale=px.colors.sequential.Teal
                )
                fig_comp_tr.update_layout(
                    template=plotly_template, paper_bgcolor=paper_bgcolor, plot_bgcolor=chart_bgcolor,
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#1E293B')
                )
                st.plotly_chart(fig_comp_tr, use_container_width=True)
                
            with col_bar2:
                fig_comp_sr = px.bar(
                    comp_df, x="Strategy", y="Sharpe Ratio",
                    title="Sharpe Ratio by Strategy",
                    color="Sharpe Ratio",
                    color_continuous_scale=px.colors.sequential.Electric
                )
                fig_comp_sr.update_layout(
                    template=plotly_template, paper_bgcolor=paper_bgcolor, plot_bgcolor=chart_bgcolor,
                    xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#1E293B')
                )
                st.plotly_chart(fig_comp_sr, use_container_width=True)


# ----------------- TAB 5: RAW DATA EXPLORER -----------------
with tab_data:
    st.subheader(f"Raw Pricing & Feature Columns ({ticker})")
    
    # Allow downloading data as CSV
    csv_data = df.to_csv()
    st.download_button(
        label="📥 Download Dataset as CSV",
        data=csv_data,
        file_name=f"{ticker}_market_data.csv",
        mime="text/csv"
    )
    
    st.markdown("Below are the first 100 rows of the processed market price dataframe including the generated trading signals:")
    
    # Merge signals to DF for raw view
    raw_display = df.copy()
    raw_display['Strategy_Signal'] = signals
    st.dataframe(raw_display.head(100), use_container_width=True)
