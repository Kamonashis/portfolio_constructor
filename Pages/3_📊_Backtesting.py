# pages/3_📊_Backtesting.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Backtesting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Backtesting")

st.write("""
Test your portfolio strategies against historical data to evaluate their performance.
""")

# --- Portfolio Input Section ---
st.header("1. Portfolio Configuration")

# Get portfolio weights from session state or allow manual input
if 'portfolio_weights' in st.session_state:
    st.info("Using portfolio weights from Portfolio Construction page")
    weights_df = st.session_state.portfolio_weights
    tickers = weights_df['Asset'].tolist()
    weights = weights_df['Weight'].tolist()
else:
    st.warning("No portfolio weights found. Please configure a portfolio in the Portfolio Construction page or enter weights manually.")
    
    # Manual portfolio input
    ticker_input = st.text_area(
        "Enter ticker symbols separated by commas (e.g., AAPL, MSFT, GOOG, AMZN):",
        "AAPL, MSFT, GOOG, AMZN"
    )
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
    
    if tickers:
        weights = []
        for ticker in tickers:
            weight = st.number_input(f"Weight for {ticker} (%)", 0.0, 100.0, 100.0/len(tickers)) / 100
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight != 0:
            weights = [w/total_weight for w in weights]
            weights_df = pd.DataFrame({'Asset': tickers, 'Weight': weights})
        else:
            st.error("Total weight must be greater than 0")
            st.stop()
    else:
        st.error("Please enter at least one ticker symbol")
        st.stop()

# --- Backtesting Parameters ---
st.header("2. Backtesting Parameters")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        datetime.now() - timedelta(days=365*2)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        datetime.now()
    )

# Additional parameters
rebalance_frequency = st.selectbox(
    "Rebalancing Frequency",
    ["Daily", "Weekly", "Monthly", "Quarterly", "Annually", "No Rebalancing"]
)

initial_investment = st.number_input(
    "Initial Investment ($)",
    min_value=1000.0,
    max_value=1000000.0,
    value=10000.0,
    step=1000.0
)

# --- Data Fetching and Processing ---
@st.cache_data
def fetch_historical_data(tickers, start_date, end_date):
    """Fetch historical price data for the portfolio"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error("No data available for the specified date range")
            return None
        
        # Get adjusted close prices
        if isinstance(data.columns, pd.MultiIndex):
            price_data = data['Adj Close']
        else:
            price_data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
        return price_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# --- Backtesting Function ---
def run_backtest(price_data, weights, rebalance_freq, initial_investment):
    """Run the backtest simulation"""
    # Calculate daily returns
    returns = price_data.pct_change()
    
    # Initialize portfolio value
    portfolio_value = initial_investment
    portfolio_history = []
    current_weights = weights.copy()
    
    # Determine rebalancing dates
    if rebalance_freq == "Daily":
        rebalance_dates = returns.index
    elif rebalance_freq == "Weekly":
        rebalance_dates = returns.index[::5]
    elif rebalance_freq == "Monthly":
        rebalance_dates = returns.index[::21]
    elif rebalance_freq == "Quarterly":
        rebalance_dates = returns.index[::63]
    elif rebalance_freq == "Annually":
        rebalance_dates = returns.index[::252]
    else:  # No Rebalancing
        rebalance_dates = [returns.index[0]]
    
    # Run simulation
    for date in returns.index:
        if date in rebalance_dates:
            current_weights = weights.copy()
        
        # Calculate daily return
        daily_return = (returns.loc[date] * current_weights).sum()
        portfolio_value *= (1 + daily_return)
        portfolio_history.append(portfolio_value)
    
    return pd.Series(portfolio_history, index=returns.index)

# --- Run Backtest Button ---
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        # Fetch data
        price_data = fetch_historical_data(tickers, start_date, end_date)
        if price_data is None:
            st.stop()
        
        # Run backtest
        portfolio_history = run_backtest(price_data, weights, rebalance_frequency, initial_investment)
        
        # Calculate performance metrics
        total_return = (portfolio_history[-1] / initial_investment - 1) * 100
        annual_return = ((1 + total_return/100) ** (252/len(portfolio_history)) - 1) * 100
        daily_returns = portfolio_history.pct_change()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annual_return - 2) / volatility  # Assuming 2% risk-free rate
        
        # Display results
        st.header("3. Backtest Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("Annual Return", f"{annual_return:.2f}%")
        with col3:
            st.metric("Volatility", f"{volatility:.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Portfolio value chart
        st.subheader("Portfolio Value Over Time")
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
        
        # Drawdown analysis
        st.subheader("Drawdown Analysis")
        rolling_max = portfolio_history.expanding().max()
        drawdowns = (portfolio_history - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns,
            mode='lines',
            name='Drawdown',
            fill='tozeroy'
        ))
        fig.update_layout(
            title=f"Portfolio Drawdown (Max: {max_drawdown:.2f}%)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        st.subheader("Monthly Returns Heatmap")
        monthly_returns = portfolio_history.resample('M').last().pct_change() * 100
        monthly_returns_matrix = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(monthly_returns_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title("Monthly Returns (%)")
        st.pyplot(fig)
        
        # Risk metrics
        st.subheader("Risk Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            st.metric("Value at Risk (95%)", f"{daily_returns.quantile(0.05)*100:.2f}%")
        with col2:
            st.metric("Positive Months", f"{(monthly_returns > 0).sum()}")
            st.metric("Negative Months", f"{(monthly_returns < 0).sum()}")
        
        # Export results
        st.subheader("Export Results")
        if st.button("Download Backtest Results"):
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

# --- Future Enhancements ---
st.header("Future Enhancements")
st.write("""
* Add transaction costs and slippage modeling
* Implement multiple portfolio strategies comparison
* Add factor analysis and attribution
* Include stress testing scenarios
* Add custom benchmark comparison
* Implement Monte Carlo simulation
""")