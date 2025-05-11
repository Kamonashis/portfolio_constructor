import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv('JAOM7U1B9PMG2WB7', 'demo')  # Use 'demo' as fallback

# API endpoints
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 5  # Alpha Vantage free tier limit 