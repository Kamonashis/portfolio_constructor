import requests
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
import streamlit as st
from config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL, MAX_REQUESTS_PER_MINUTE

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Wait until we can make another request
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(time.time())

# Create a global rate limiter
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, 60)  # 5 requests per minute

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch daily stock data from Alpha Vantage
    """
    try:
        rate_limiter.wait_if_needed()
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Error Message' in data:
            raise Exception(data['Error Message'])
        
        if 'Time Series (Daily)' not in data:
            st.error(f"No data found for {symbol}. Full API response: {data}")
            raise Exception(f"No data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        
        # Rename columns
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Filter date range
        df = df.loc[start_date:end_date]
        
        # Select and rename relevant columns
        if '5. adjusted close' in df.columns:
            df = df[['5. adjusted close']].rename(columns={'5. adjusted close': 'Adj Close'})
        elif 'Adj Close' in df.columns:
            df = df[['Adj Close']]
        elif 'adjusted close' in df.columns:
            df = df[['adjusted close']].rename(columns={'adjusted close': 'Adj Close'})
        else:
            st.error(f"Expected '5. adjusted close' or 'adjusted close' column not found for {symbol}. Columns returned: {list(df.columns)}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data
def fetch_data(tickers, start_date, end_date):
    """
    Enhanced data fetching with rate limiting and retries
    """
    try:
        st.info(f"Fetching data for tickers: {', '.join(tickers)}")
        st.info(f"Date range: {start_date} to {end_date}")
        
        # Validate tickers with rate limiting
        valid_tickers = []
        invalid_tickers = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                # Quick validation check with rate limiting
                rate_limiter.wait_if_needed()
                
                # Try to fetch one day of data to validate ticker
                test_data = fetch_stock_data(ticker, 
                                           datetime.now() - timedelta(days=1),
                                           datetime.now())
                
                if test_data is not None and not test_data.empty:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
                    st.warning(f"Could not validate ticker {ticker}: No market data available")
                
                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(0.5)  # Additional delay between requests
                
            except Exception as e:
                invalid_tickers.append(ticker)
                st.warning(f"Could not validate ticker {ticker}: {str(e)}")
                time.sleep(1)  # Wait longer after an error

        if invalid_tickers:
            st.warning(f"Invalid or unavailable tickers: {', '.join(invalid_tickers)}")
            if not valid_tickers:
                st.error("No valid tickers to fetch data for.")
                return None
            tickers = valid_tickers

        # Fetch data with rate limiting and retries
        all_data = {}
        for ticker in valid_tickers:
            for attempt in range(3):  # Try up to 3 times
                try:
                    data = fetch_stock_data(ticker, start_date, end_date)
                    if data is not None and not data.empty:
                        all_data[ticker] = data
                        break
                    else:
                        if attempt < 2:  # Don't wait on last attempt
                            st.warning(f"Retrying {ticker} (attempt {attempt + 1}/3)")
                            time.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    if attempt < 2:
                        st.warning(f"Error fetching {ticker}, retrying: {str(e)}")
                        time.sleep(2 ** attempt)
                    else:
                        st.error(f"Failed to fetch data for {ticker} after 3 attempts")

        if not all_data:
            st.error("Could not fetch data for any tickers.")
            return None

        # Combine all data
        price_data = pd.DataFrame()
        for ticker, data in all_data.items():
            price_data[ticker] = data['Adj Close']

        if price_data.empty:
            st.error("No valid price data available.")
            return None

        # Validate the data
        st.success(f"Successfully fetched data for {len(price_data.columns)} assets")
        st.write(f"Data points: {len(price_data)}")
        st.write(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
        
        # Check for missing values
        missing_values = price_data.isnull().sum()
        if missing_values.any():
            st.warning("Missing values detected:")
            st.write(missing_values[missing_values > 0])

        return price_data

    except Exception as e:
        st.error(f"Error in data fetching process: {str(e)}")
        st.error("Please try the following:")
        st.error("1. Check your internet connection")
        st.error("2. Verify the ticker symbols are correct")
        st.error("3. Try a different date range")
        st.error("4. If the problem persists, try again in a few minutes")
        return None