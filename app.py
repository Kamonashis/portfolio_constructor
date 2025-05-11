# app.py
import streamlit as st

# This is the main file that Streamlit runs.
# It doesn't need much content itself, as the pages in the 'pages' directory
# will automatically create the sidebar navigation.

# Set page configuration
st.set_page_config(
    page_title="Portfolio Constructor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to style the sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Main page content
st.write("""
# Welcome to Portfolio Constructor! 🚀

Your comprehensive tool for portfolio construction, optimization, and analysis.

## Features

### 🏗️ Portfolio Construction
- Build optimized portfolios using various strategies
- Customize risk parameters and constraints
- Visualize portfolio allocations and risk metrics

### 📊 Backtesting
- Test portfolio strategies against historical data
- Analyze performance metrics and risk measures
- Compare different portfolio configurations

### 📈 Analysis
- Deep dive into portfolio performance
- Generate detailed reports and visualizations
- Export results for further analysis

## Getting Started
1. Use the sidebar to navigate between features
2. Start with Portfolio Construction to build your portfolio
3. Use Backtesting to validate your strategy
4. Analyze results to make informed decisions

**Disclaimer:** This app is for educational purposes only and should not be considered financial advice.
""")