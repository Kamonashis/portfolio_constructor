# app.py
import streamlit as st

# This is the main file that Streamlit runs.
# It doesn't need much content itself, as the pages in the 'pages' directory
# will automatically create the sidebar navigation.

st.set_page_config(
    page_title="Portfolio Constructor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Portfolio Constructor")
st.write("Welcome to the Portfolio Constructor app! Use the sidebar to navigate between different sections.")