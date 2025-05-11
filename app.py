# app.py
import streamlit as st

# This is the main file that Streamlit runs.
# It doesn't need much content itself, as the pages in the 'pages' directory
# will automatically create the sidebar navigation.

st.set_page_config(
    page_title="Portfolio Lab App",
    page_icon="📈",
    layout="wide"
)

st.sidebar.title("Navigation")

# Streamlit automatically finds and displays pages from the 'pages' directory.
# The content of the selected page will be displayed in the main area.

st.write("""
# Welcome to the Portfolio Lab App!

Use the sidebar to navigate through the different sections of the application.

* **Home:** Introduction to the app.
* **Portfolio Construction:** Build and optimize your investment portfolio.
* **Backtesting:** (Coming Soon) Test your portfolio strategies against historical data.
* **Analysis:** (Coming Soon) Analyze portfolio performance and characteristics.
""")

# You can add some general information or a logo here if you like.