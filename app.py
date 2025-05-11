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
st.title("Portfolio Constructor")
st.write("Welcome to the Portfolio Constructor app! Use the sidebar to navigate between different sections.")

# Add some helpful information
st.markdown("""
### Available Features:
- **🏗️ Portfolio Construction**: Build and optimize your investment portfolio
- **📊 Backtesting**: Test your portfolio strategy against historical data
- **📈 Analysis**: Analyze portfolio performance and risk metrics

**Note**: Use the sidebar to navigate between these features.
""")