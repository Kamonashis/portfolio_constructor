# pages/1_Portfolio_Construction.py
import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import norm
import networkx as nx
from scipy import stats
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import io
from scipy.optimize import minimize
import warnings
from data_fetcher import fetch_data
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Construction",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Portfolio Construction")

st.write("""
Build and optimize your investment portfolio with advanced optimization techniques.
""")

# --- User Inputs ---
st.header("1. Select Assets")
ticker_input = st.text_area(
    "Enter ticker symbols separated by commas (e.g., AAPL, MSFT, GOOG, AMZN):",
    "AAPL, MSFT, GOOG, AMZN"
)

tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]

if not tickers:
    st.warning("Please enter at least one ticker symbol.")
    st.stop()

st.header("2. Select Data Range")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("today"))

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
    st.stop()

# --- Risk-Free Rate Input ---
st.header("3. Risk Parameters")
risk_free_rate = st.number_input(
    "Risk-Free Rate (Annual %):",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
) / 100  # Convert to decimal

# --- Portfolio Constraints ---
st.header("4. Portfolio Constraints")
col1, col2 = st.columns(2)
with col1:
    min_weight = st.number_input("Minimum Weight per Asset (%)", 0.0, 100.0, 0.0) / 100
    max_weight = st.number_input("Maximum Weight per Asset (%)", 0.0, 100.0, 100.0) / 100
with col2:
    target_return = st.number_input("Target Annual Return (%)", 0.0, 100.0, 10.0) / 100
    rebalance_frequency = st.selectbox(
        "Rebalancing Frequency",
        ["Monthly", "Quarterly", "Semi-Annually", "Annually"]
    )

# --- Data Fetching ---
data = fetch_data(tickers, start_date, end_date)

if data is None or data.empty:
    st.stop()

data.dropna(axis=1, how='all', inplace=True)
data.dropna(axis=0, inplace=True)

if data.empty:
    st.error("No valid data available after cleaning. Please check tickers or date range.")
    st.stop()

st.subheader("Adjusted Close Prices (Sample)")
st.write(data.head())

# --- Calculate Returns and Covariance ---
returns = data.pct_change().dropna()
if returns.empty:
    st.error("Not enough data to calculate returns. Please select a longer date range.")
    st.stop()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

st.subheader("Annualized Mean Returns")
st.write(mean_returns)

st.subheader("Annualized Covariance Matrix")
st.write(cov_matrix)

# --- Optimization Options ---
st.header("5. Select Optimization Model and Objective")

# User chooses what to optimize for
optimization_goal = st.selectbox(
    "What do you want to optimize your portfolio for?",
    [
        "Maximize Sharpe Ratio",
        "Minimize Volatility",
        "Maximize Return",
        "Target Return",
        "Risk Parity",
        "Maximum Diversification",
        "Equal Risk Contribution",
        "Black-Litterman"
    ]
)

# Map optimization_goal to strategy and objective
if optimization_goal in ["Maximize Sharpe Ratio", "Minimize Volatility", "Maximize Return", "Target Return"]:
    optimization_strategy = "Mean-Variance Optimization"
    if optimization_goal == "Target Return":
        optimization_objective = "Target Return"
    elif optimization_goal == "Maximize Sharpe Ratio":
        optimization_objective = "Maximize Sharpe Ratio"
    elif optimization_goal == "Minimize Volatility":
        optimization_objective = "Minimize Volatility"
    else:
        optimization_objective = "Maximize Return"
elif optimization_goal == "Risk Parity":
    optimization_strategy = "Risk Parity"
elif optimization_goal == "Maximum Diversification":
    optimization_strategy = "Maximum Diversification"
elif optimization_goal == "Equal Risk Contribution":
    optimization_strategy = "Equal Risk Contribution"
elif optimization_goal == "Black-Litterman":
    optimization_strategy = "Black-Litterman"

# Strategy-specific parameters
if optimization_strategy == "Mean-Variance Optimization":
    optimization_objective = st.selectbox(
        "Select Optimization Objective:",
        [
            "Maximize Sharpe Ratio",
            "Minimize Volatility",
            "Target Return",
            "Maximize Return"
        ]
    )
    
    if optimization_objective == "Target Return":
        target_return = st.number_input(
            "Target Annual Return (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1
        ) / 100

elif optimization_strategy == "Risk Parity":
    st.info("Risk Parity aims to equalize the risk contribution of each asset in the portfolio.")
    risk_measure = st.selectbox(
        "Risk Measure:",
        ["Variance", "Semi-Variance", "Expected Shortfall"]
    )

elif optimization_strategy == "Maximum Diversification":
    st.info("Maximum Diversification aims to maximize the portfolio's diversification ratio.")
    diversification_measure = st.selectbox(
        "Diversification Measure:",
        ["Correlation-based", "Variance-based"]
    )

elif optimization_strategy == "Equal Risk Contribution":
    st.info("Equal Risk Contribution aims to make each asset contribute equally to portfolio risk.")
    risk_contribution_measure = st.selectbox(
        "Risk Contribution Measure:",
        ["Standard Deviation", "Variance", "Expected Shortfall"]
    )

elif optimization_strategy == "Black-Litterman":
    st.info("Black-Litterman combines market equilibrium with investor views.")
    market_cap_weighted = st.checkbox("Use Market Cap Weighted Prior", value=True)
    if not market_cap_weighted:
        st.warning("Market cap data will be fetched for the prior. This may take a moment.")
    
    # Add view configuration
    st.subheader("Add Your Views")
    num_views = st.number_input("Number of Views", min_value=0, max_value=len(tickers), value=0)
    
    views = []
    for i in range(num_views):
        col1, col2, col3 = st.columns(3)
        with col1:
            asset = st.selectbox(f"Asset {i+1}", tickers, key=f"view_asset_{i}")
        with col2:
            view_type = st.selectbox("View Type", ["Outperform", "Underperform"], key=f"view_type_{i}")
        with col3:
            confidence = st.slider("Confidence", 0.0, 1.0, 0.5, key=f"view_conf_{i}")
        views.append({"asset": asset, "type": view_type, "confidence": confidence})

# Common constraints
st.subheader("Portfolio Constraints")

# Weight constraints
col1, col2 = st.columns(2)
with col1:
    min_weight = st.number_input("Minimum Weight per Asset (%)", 0.0, 100.0, 0.0) / 100
with col2:
    max_weight = st.number_input("Maximum Weight per Asset (%)", 0.0, 100.0, 100.0) / 100

# --- Portfolio Optimization ---
st.header("6. Optimized Portfolio Weights")

if st.button("Run Optimization"):
    num_assets = len(tickers)
    if num_assets == 0:
        st.warning("No valid assets to optimize.")
        st.stop()

    problem = None
    weights = cp.Variable(num_assets)
    
    # Common constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= min_weight,
        weights <= max_weight
    ]

    # Strategy-specific optimization
    if optimization_strategy == "Mean-Variance Optimization":
        if optimization_objective == "Maximize Sharpe Ratio":
            objective = cp.Maximize((mean_returns.values @ weights - risk_free_rate) / 
                                  cp.sqrt(cp.quad_form(weights, cov_matrix.values)))
        elif optimization_objective == "Minimize Volatility":
            objective = cp.Minimize(cp.sqrt(cp.quad_form(weights, cov_matrix.values)))
        elif optimization_objective == "Target Return":
            constraints.append(mean_returns.values @ weights >= target_return)
            objective = cp.Minimize(cp.sqrt(cp.quad_form(weights, cov_matrix.values)))
        else:  # Maximize Return
            objective = cp.Maximize(mean_returns.values @ weights)

    elif optimization_strategy == "Risk Parity":
        risk_contributions = cp.multiply(weights, cp.sqrt(cp.quad_form(weights, cov_matrix.values)))
        objective = cp.Minimize(cp.sum_squares(risk_contributions - cp.sum(risk_contributions) / num_assets))

    elif optimization_strategy == "Maximum Diversification":
        if diversification_measure == "Correlation-based":
            # Use correlation matrix for diversification
            corr_matrix = returns.corr()
            objective = cp.Maximize(cp.sum(weights) / cp.sqrt(cp.quad_form(weights, corr_matrix.values)))
        else:
            # Use variance for diversification
            objective = cp.Maximize(cp.sum(weights) / cp.sqrt(cp.quad_form(weights, cov_matrix.values)))

    elif optimization_strategy == "Equal Risk Contribution":
        risk_contributions = cp.multiply(weights, cp.sqrt(cp.quad_form(weights, cov_matrix.values)))
        objective = cp.Minimize(cp.sum_squares(risk_contributions - cp.sum(risk_contributions) / num_assets))

    elif optimization_strategy == "Black-Litterman":
        # Implement Black-Litterman optimization
        if market_cap_weighted:
            try:
                market_caps = []
                for ticker in tickers:
                    try:
                        market_cap = yf.Ticker(ticker).info.get('marketCap', 0)
                        market_caps.append(market_cap)
                    except:
                        market_caps.append(0)
                
                # Normalize market caps to get prior weights
                market_caps = np.array(market_caps)
                prior_weights = market_caps / np.sum(market_caps)
                
                # Adjust returns based on views
                adjusted_returns = mean_returns.copy()
                for view in views:
                    asset_idx = tickers.index(view['asset'])
                    if view['type'] == "Outperform":
                        adjusted_returns[asset_idx] *= (1 + view['confidence'])
                    else:
                        adjusted_returns[asset_idx] *= (1 - view['confidence'])
                
                # Use adjusted returns for optimization
                objective = cp.Maximize(adjusted_returns.values @ weights)
            except:
                st.warning("Could not fetch market cap data. Using equal weights as prior.")
                objective = cp.Maximize(mean_returns.values @ weights)
        else:
            objective = cp.Maximize(mean_returns.values @ weights)

        problem = cp.Problem(objective, constraints)

    # Add input validation
    def validate_inputs():
        if len(tickers) < 2:
            st.error("Please select at least 2 assets for portfolio optimization.")
            return False
        
        if min_weight > max_weight:
            st.error("Minimum weight cannot be greater than maximum weight.")
            return False
        
        if target_return > mean_returns.max():
            st.warning(f"Target return ({target_return:.2%}) is higher than the maximum possible return ({mean_returns.max():.2%}).")
            return False
        
        if target_return < mean_returns.min():
            st.warning(f"Target return ({target_return:.2%}) is lower than the minimum possible return ({mean_returns.min():.2%}).")
            return False
        
        return True

    # Enhanced efficient frontier calculation
    def calculate_efficient_frontier(num_portfolios=100, risk_free_rate=0.02):
        try:
            # Generate random portfolios for better frontier estimation
            portfolio_returns = []
            portfolio_volatilities = []
            portfolio_weights = []
            
            for _ in range(num_portfolios):
                # Generate random weights
                weights = np.random.random(len(tickers))
                weights = weights / np.sum(weights)
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(mean_returns.values * weights)
                portfolio_volatility = np.sqrt(weights.T @ cov_matrix.values @ weights)
                
                portfolio_returns.append(portfolio_return)
                portfolio_volatilities.append(portfolio_volatility)
                portfolio_weights.append(weights)
            
            # Convert to numpy arrays
            portfolio_returns = np.array(portfolio_returns)
            portfolio_volatilities = np.array(portfolio_volatilities)
            portfolio_weights = np.array(portfolio_weights)
            
            # Find the efficient frontier
            efficient_indices = []
            min_vol_idx = np.argmin(portfolio_volatilities)
            max_sharpe_idx = np.argmax((portfolio_returns - risk_free_rate) / portfolio_volatilities)
            
            # Sort portfolios by volatility
            sorted_indices = np.argsort(portfolio_volatilities)
            sorted_returns = portfolio_returns[sorted_indices]
            sorted_volatilities = portfolio_volatilities[sorted_indices]
            
            # Find the efficient frontier using the convex hull
            current_max_return = sorted_returns[0]
            for i in range(len(sorted_indices)):
                if sorted_returns[i] >= current_max_return:
                    efficient_indices.append(sorted_indices[i])
                    current_max_return = sorted_returns[i]
            
            return {
                'returns': portfolio_returns[efficient_indices],
                'volatilities': portfolio_volatilities[efficient_indices],
                'weights': portfolio_weights[efficient_indices],
                'min_vol_portfolio': {
                    'return': portfolio_returns[min_vol_idx],
                    'volatility': portfolio_volatilities[min_vol_idx],
                    'weights': portfolio_weights[min_vol_idx]
                },
                'max_sharpe_portfolio': {
                    'return': portfolio_returns[max_sharpe_idx],
                    'volatility': portfolio_volatilities[max_sharpe_idx],
                    'weights': portfolio_weights[max_sharpe_idx]
                }
            }
        except Exception as e:
            st.error(f"Error calculating efficient frontier: {str(e)}")
            return None

    # Add error handling for optimization
    try:
        if not validate_inputs():
            st.stop()

        problem.solve()

        if problem.status in ["optimal", "optimal_near"]:
            optimized_weights = weights.value
            weights_df = pd.DataFrame({
                'Asset': tickers,
                'Weight': optimized_weights
            })
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: max(0, x))
            weights_df['Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()

            # Calculate portfolio metrics with error handling
            try:
                optimized_port_return = mean_returns.values @ weights_df['Weight']
                optimized_port_volatility = np.sqrt(weights_df['Weight'].values @ cov_matrix.values @ weights_df['Weight'].values)
                optimized_sharpe_ratio = (optimized_port_return - risk_free_rate) / optimized_port_volatility

                # Calculate daily returns for additional metrics
                daily_returns = returns @ weights_df['Weight']
                annual_return = daily_returns.mean() * 252

                # Calculate maximum drawdown with error handling
                try:
                    portfolio_history = (1 + daily_returns).cumprod()
                    rolling_max = portfolio_history.expanding().max()
                    drawdowns = (portfolio_history - rolling_max) / rolling_max
                    max_drawdown = drawdowns.min()
                except Exception as e:
                    st.warning(f"Error calculating drawdown: {str(e)}")
                    max_drawdown = 0

                # Calculate additional risk metrics
                try:
                    var_95 = np.percentile(daily_returns, 5)
                    cvar_95 = daily_returns[daily_returns <= var_95].mean()
                    sortino_ratio = (annual_return - risk_free_rate) / (daily_returns[daily_returns < 0].std() * np.sqrt(252))
                    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
                except Exception as e:
                    st.warning(f"Error calculating risk metrics: {str(e)}")
                    var_95 = cvar_95 = sortino_ratio = calmar_ratio = 0

                # Calculate efficient frontier
                frontier_data = calculate_efficient_frontier(num_portfolios=100, risk_free_rate=risk_free_rate)
                
                if frontier_data:
                    # Store weights in session state for backtesting
                    st.session_state.portfolio_weights = weights_df

                    # Display results with enhanced metrics
                    st.subheader("Optimization Results")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Portfolio Allocation", 
                        "Risk Analysis", 
                        "Correlation Analysis", 
                        "Performance Metrics",
                        "Advanced Analytics"
                    ])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Interactive pie chart with donut
                            fig = go.Figure(data=[go.Pie(
                                labels=weights_df['Asset'],
                                values=weights_df['Weight'],
                                hole=.3,
                                textinfo='label+percent',
                                insidetextorientation='radial',
                                pull=[0.1 if w == max(weights_df['Weight']) else 0 for w in weights_df['Weight']]
                            )])
                            fig.update_layout(
                                title="Portfolio Allocation",
                                annotations=[dict(text='Weights', x=0.5, y=0.5, font_size=20, showarrow=False)]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Weight distribution with cumulative line
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=weights_df['Asset'],
                                y=weights_df['Weight'],
                                name='Weight'
                            ))
                            fig.add_trace(go.Scatter(
                                x=weights_df['Asset'],
                                y=weights_df['Weight'].cumsum(),
                                name='Cumulative',
                                yaxis='y2'
                            ))
                            fig.update_layout(
                                title="Asset Weights Distribution",
                                yaxis=dict(title="Weight"),
                                yaxis2=dict(title="Cumulative Weight", overlaying='y', side='right'),
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Risk decomposition with contribution percentage
                            risk_contributions = np.multiply(weights_df['Weight'], 
                                np.sqrt(np.diag(cov_matrix.values)))
                            total_risk = np.sum(risk_contributions)
                            risk_df = pd.DataFrame({
                                'Asset': weights_df['Asset'],
                                'Risk Contribution': risk_contributions,
                                'Risk %': risk_contributions / total_risk * 100
                            })
                            
                            fig = px.bar(risk_df, x='Asset', y='Risk %',
                                       title="Risk Contribution by Asset (%)",
                                       labels={'Risk %': 'Risk Contribution (%)', 'Asset': 'Asset'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Risk-return scatter with size and color
                            risk_return_df = pd.DataFrame({
                                'Asset': weights_df['Asset'],
                                'Return': mean_returns.values,
                                'Risk': np.sqrt(np.diag(cov_matrix.values)),
                                'Weight': weights_df['Weight'],
                                'Sharpe': (mean_returns.values - risk_free_rate) / np.sqrt(np.diag(cov_matrix.values))
                            })
                            
                            fig = px.scatter(risk_return_df, 
                                           x='Risk', y='Return',
                                           size='Weight',
                                           color='Sharpe',
                                           hover_data=['Asset', 'Weight', 'Sharpe'],
                                           title="Risk-Return Profile",
                                           color_continuous_scale='RdYlGn')
                            st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Enhanced correlation heatmap
                            corr_matrix = returns.corr()
                            fig = px.imshow(corr_matrix,
                                          labels=dict(color="Correlation"),
                                          title="Asset Correlation Matrix",
                                          color_continuous_scale='RdBu')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Network graph with correlation strength
                            G = nx.Graph()
                            for i in range(len(tickers)):
                                for j in range(i+1, len(tickers)):
                                    corr = corr_matrix.iloc[i,j]
                                    if abs(corr) > 0.3:  # Only show significant correlations
                                        G.add_edge(tickers[i], tickers[j], 
                                                 weight=abs(corr),
                                                 color='red' if corr < 0 else 'green')
                            
                            pos = nx.spring_layout(G)
                            edge_trace = go.Scatter(
                                x=[], y=[], line=dict(width=0.5, color='#888'),
                                hoverinfo='none', mode='lines')
                            
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                edge_trace['x'] += tuple([x0, x1, None])
                                edge_trace['y'] += tuple([y0, y1, None])
                            
                            node_trace = go.Scatter(
                                x=[], y=[], text=[], mode='markers+text',
                                hoverinfo='text', textposition="top center",
                                marker=dict(size=20, color='lightblue'))
                            
                            for node in G.nodes():
                                x, y = pos[node]
                                node_trace['x'] += tuple([x])
                                node_trace['y'] += tuple([y])
                                node_trace['text'] += tuple([node])
                            
                            fig = go.Figure(data=[edge_trace, node_trace],
                                          layout=go.Layout(
                                              title='Asset Correlation Network',
                                              showlegend=False,
                                              hovermode='closest',
                                              margin=dict(b=20,l=5,r=5,t=40)))
                            st.plotly_chart(fig, use_container_width=True)

                    with tab4:
                        # Performance metrics with more details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Return", f"{optimized_port_return:.2%}")
                            st.metric("Annual Return", f"{annual_return:.2%}")
                            st.metric("Sharpe Ratio", f"{optimized_sharpe_ratio:.2f}")
                            st.metric("Sortino Ratio", 
                                     f"{sortino_ratio:.2f}")
                        with col2:
                            st.metric("Portfolio Volatility", f"{optimized_port_volatility:.2%}")
                            st.metric("Value at Risk (95%)", 
                                     f"{var_95:.2%}")
                            st.metric("Calmar Ratio", 
                                     f"{calmar_ratio:.2f}")
                        
                        # Rolling metrics with confidence intervals
                        window = 252  # 1 year
                        rolling_returns = portfolio_history.pct_change().rolling(window=window).mean() * 252
                        rolling_vol = portfolio_history.pct_change().rolling(window=window).std() * np.sqrt(252)
                        rolling_sharpe = (rolling_returns - risk_free_rate) / rolling_vol
                        
                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                          vertical_spacing=0.05)
                        
                        # Add confidence intervals
                        for metric, row in zip([rolling_returns, rolling_vol, rolling_sharpe], [1, 2, 3]):
                            mean = metric.mean()
                            std = metric.std()
                            fig.add_trace(go.Scatter(
                                x=metric.index,
                                y=mean + 2*std,
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,100,80,0.2)',
                                name='Upper Bound'
                            ), row=row, col=1)
                            fig.add_trace(go.Scatter(
                                x=metric.index,
                                y=mean - 2*std,
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,100,80,0.2)',
                                name='Lower Bound'
                            ), row=row, col=1)
                            fig.add_trace(go.Scatter(
                                x=metric.index,
                                y=metric,
                                name=metric.name,
                                line=dict(color='blue')
                            ), row=row, col=1)
                        
                        fig.update_layout(height=800, title_text="Rolling Portfolio Metrics with Confidence Intervals")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab5:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Return distribution
                            returns_hist = portfolio_history.pct_change().dropna()
                            fig = ff.create_distplot(
                                [returns_hist],
                                ['Portfolio Returns'],
                                bin_size=0.001,
                                show_rug=True
                            )
                            fig.update_layout(title="Return Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Drawdown analysis
                            rolling_max = portfolio_history.expanding().max()
                            drawdowns = (portfolio_history - rolling_max) / rolling_max * 100
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=drawdowns.index,
                                y=drawdowns,
                                fill='tozeroy',
                                name='Drawdown',
                                line=dict(color='red')
                            ))
                            fig.update_layout(
                                title="Portfolio Drawdown Analysis",
                                yaxis_title="Drawdown (%)",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Enhanced Efficient Frontier Analysis
                    st.subheader("Efficient Frontier Analysis")
                    
                    fig = go.Figure()
                    
                    # Plot efficient frontier
                    fig.add_trace(go.Scatter(
                        x=frontier_data['volatilities'],
                        y=frontier_data['returns'],
                        mode='lines',
                        name='Efficient Frontier',
                        line=dict(color='blue')
                    ))
                    
                    # Plot individual assets
                    fig.add_trace(go.Scatter(
                        x=np.sqrt(np.diag(cov_matrix.values)),
                        y=mean_returns.values,
                        mode='markers+text',
                        name='Individual Assets',
                        text=tickers,
                        textposition="top center",
                        marker=dict(size=10, color='red')
                    ))
                    
                    # Plot optimal portfolio
                    fig.add_trace(go.Scatter(
                        x=[optimized_port_volatility],
                        y=[optimized_port_return],
                        mode='markers',
                        name='Optimal Portfolio',
                        marker=dict(size=15, color='green', symbol='star')
                    ))
                    
                    # Plot minimum volatility portfolio
                    fig.add_trace(go.Scatter(
                        x=[frontier_data['min_vol_portfolio']['volatility']],
                        y=[frontier_data['min_vol_portfolio']['return']],
                        mode='markers',
                        name='Minimum Volatility Portfolio',
                        marker=dict(size=15, color='purple', symbol='diamond')
                    ))
                    
                    # Plot maximum Sharpe ratio portfolio
                    fig.add_trace(go.Scatter(
                        x=[frontier_data['max_sharpe_portfolio']['volatility']],
                        y=[frontier_data['max_sharpe_portfolio']['return']],
                        mode='markers',
                        name='Maximum Sharpe Ratio Portfolio',
                        marker=dict(size=15, color='orange', symbol='square')
                    ))
                    
                    fig.update_layout(
                        title='Efficient Frontier with Portfolio Options',
                        xaxis_title='Portfolio Volatility',
                        yaxis_title='Portfolio Return',
                        hovermode='closest',
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display portfolio comparison
                    st.subheader("Portfolio Comparison")
                    comparison_data = {
                        'Metric': ['Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
                        'Optimal Portfolio': [
                            f"{optimized_port_return:.2%}",
                            f"{optimized_port_volatility:.2%}",
                            f"{optimized_sharpe_ratio:.2f}",
                            f"{sortino_ratio:.2f}",
                            f"{calmar_ratio:.2f}",
                            f"{var_95:.2%}",
                            f"{cvar_95:.2%}",
                            f"{max_drawdown:.2%}"
                        ],
                        'Min Volatility Portfolio': [
                            f"{frontier_data['min_vol_portfolio']['return']:.2%}",
                            f"{frontier_data['min_vol_portfolio']['volatility']:.2%}",
                            f"{(frontier_data['min_vol_portfolio']['return'] - risk_free_rate) / frontier_data['min_vol_portfolio']['volatility']:.2f}",
                            "N/A", "N/A", "N/A", "N/A", "N/A"
                        ],
                        'Max Sharpe Portfolio': [
                            f"{frontier_data['max_sharpe_portfolio']['return']:.2%}",
                            f"{frontier_data['max_sharpe_portfolio']['volatility']:.2%}",
                            f"{(frontier_data['max_sharpe_portfolio']['return'] - risk_free_rate) / frontier_data['max_sharpe_portfolio']['volatility']:.2f}",
                            "N/A", "N/A", "N/A", "N/A", "N/A"
                        ]
                    }
                    st.dataframe(pd.DataFrame(comparison_data))

            except Exception as e:
                st.error(f"Error calculating portfolio metrics: {str(e)}")
                st.stop()

        else:
            st.error(f"Optimization failed. Status: {problem.status}")
            st.write("This might happen if the problem is infeasible or unbounded.")

    except Exception as e:
        st.error(f"An error occurred during optimization: {str(e)}")
        st.write("Please check your inputs and try again.")