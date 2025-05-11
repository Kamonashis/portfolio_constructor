# pages/2_Backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from data_fetcher import fetch_data
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Backtesting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Backtesting")

st.write("""
Test your portfolio strategy against historical data to evaluate its performance.
""")

# Check if portfolio weights are available
if 'portfolio_weights' not in st.session_state:
    st.warning("Please construct a portfolio first in the Portfolio Construction page.")
    st.stop()

# Get portfolio weights from session state
weights_df = st.session_state.portfolio_weights

# --- Backtesting Parameters ---
st.header("1. Backtesting Parameters")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        pd.to_datetime("2018-01-01")
    )
with col2:
    end_date = st.date_input(
        "End Date",
        pd.to_datetime("today")
    )

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
    st.stop()

# --- Rebalancing Options ---
st.header("2. Rebalancing Strategy")

rebalance_frequency = st.selectbox(
    "Rebalancing Frequency",
    ["Monthly", "Quarterly", "Semi-Annually", "Annually", "No Rebalancing"]
)

transaction_cost = st.number_input(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.1,
    step=0.1
) / 100

# --- Run Backtest ---
if st.button("Run Backtest"):
    # Fetch historical data
    data = fetch_data(weights_df['Asset'].tolist(), start_date, end_date)
    
    if data is None or data.empty:
        st.error("Could not fetch historical data. Please check your tickers and date range.")
        st.stop()
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Initialize portfolio value
    initial_value = 1000000  # $1M initial investment
    portfolio_value = initial_value
    portfolio_history = pd.Series(index=returns.index, dtype=float)
    portfolio_history.iloc[0] = initial_value
    
    # Calculate rebalancing dates
    if rebalance_frequency == "Monthly":
        rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='M')
    elif rebalance_frequency == "Quarterly":
        rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='Q')
    elif rebalance_frequency == "Semi-Annually":
        rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='6M')
    elif rebalance_frequency == "Annually":
        rebalance_dates = pd.date_range(start=returns.index[0], end=returns.index[-1], freq='Y')
    else:  # No Rebalancing
        rebalance_dates = pd.DatetimeIndex([])
    
    # Current weights
    current_weights = weights_df['Weight'].values
    
    # Run backtest
    for i in range(1, len(returns)):
        date = returns.index[i]
        
        # Check if rebalancing is needed
        if date in rebalance_dates:
            # Calculate transaction costs
            weight_changes = np.abs(current_weights - weights_df['Weight'].values)
            transaction_cost_amount = portfolio_value * np.sum(weight_changes) * transaction_cost
            portfolio_value -= transaction_cost_amount
            
            # Update weights
            current_weights = weights_df['Weight'].values
        
        # Calculate daily return
        daily_return = np.sum(returns.iloc[i].values * current_weights)
        portfolio_value *= (1 + daily_return)
        portfolio_history.iloc[i] = portfolio_value
    
    # Calculate performance metrics
    total_return = (portfolio_value - initial_value) / initial_value
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    daily_returns = portfolio_history.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / volatility  # Assuming 2% risk-free rate
    
    # Calculate drawdown
    rolling_max = portfolio_history.expanding().max()
    drawdowns = (portfolio_history - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Display results
    st.header("3. Backtest Results")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{total_return:.2%}")
    with col2:
        st.metric("Annual Return", f"{annual_return:.2%}")
    with col3:
        st.metric("Volatility", f"{volatility:.2%}")
    with col4:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Portfolio value chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_history.index,
        y=portfolio_history,
        mode='lines',
        name='Portfolio Value'
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red')
    ))
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    window = 252  # 1 year
    rolling_returns = daily_returns.rolling(window=window).mean() * 252
    rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_returns - 0.02) / rolling_vol
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    fig.add_trace(go.Scatter(
        x=rolling_returns.index,
        y=rolling_returns,
        name='Rolling Returns'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        name='Rolling Volatility'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        name='Rolling Sharpe Ratio'
    ), row=3, col=1)
    
    fig.update_layout(height=800, title_text="Rolling Portfolio Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns heatmap
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_returns = monthly_returns.to_frame()
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_returns = monthly_returns.pivot(index='Year', columns='Month', values=0)
    
    fig = px.imshow(
        monthly_returns,
        labels=dict(x="Month", y="Year", color="Return"),
        title="Monthly Returns Heatmap",
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export results
    st.header("4. Export Results")
    
    if st.button("Export to CSV"):
        results_df = pd.DataFrame({
            'Date': portfolio_history.index,
            'Portfolio Value': portfolio_history,
            'Daily Return': daily_returns,
            'Drawdown': drawdowns
        })
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="backtest_results.csv",
            mime="text/csv"
        ) 