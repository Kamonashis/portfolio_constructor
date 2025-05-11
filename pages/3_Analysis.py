# pages/3_Analysis.py
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
    page_title="Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analysis")

st.write("""
Analyze your portfolio's performance and risk metrics in detail.
""")

# Check if portfolio weights are available
if 'portfolio_weights' not in st.session_state:
    st.warning("Please construct a portfolio first in the Portfolio Construction page.")
    st.stop()

# Get portfolio weights from session state
weights_df = st.session_state.portfolio_weights

# --- Analysis Parameters ---
st.header("1. Analysis Parameters")

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

# --- Run Analysis ---
if st.button("Run Analysis"):
    # Fetch historical data
    data = fetch_data(weights_df['Asset'].tolist(), start_date, end_date)
    
    if data is None or data.empty:
        st.error("Could not fetch historical data. Please check your tickers and date range.")
        st.stop()
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = returns @ weights_df['Weight']
    
    # Calculate performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / volatility  # Assuming 2% risk-free rate
    
    # Calculate drawdown
    portfolio_value = (1 + portfolio_returns).cumprod()
    rolling_max = portfolio_value.expanding().max()
    drawdowns = (portfolio_value - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Display results
    st.header("2. Performance Analysis")
    
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
        x=portfolio_value.index,
        y=portfolio_value,
        mode='lines',
        name='Portfolio Value'
    ))
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
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
    rolling_returns = portfolio_returns.rolling(window=window).mean() * 252
    rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
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
    monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
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
    st.header("3. Export Results")
    
    if st.button("Export to CSV"):
        results_df = pd.DataFrame({
            'Date': portfolio_value.index,
            'Portfolio Value': portfolio_value,
            'Daily Return': portfolio_returns,
            'Drawdown': drawdowns
        })
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        ) 