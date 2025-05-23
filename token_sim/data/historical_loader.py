import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import requests
from datetime import datetime, timedelta
import sys
import os

# Get API key from environment variable
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set. Please set it before running the simulation.")

class HistoricalDataLoader:
    """Loads and processes historical cryptocurrency data."""
    
    # Mapping of cryptocurrencies to their consensus mechanisms
    CONSENSUS_MAPPING = {
        'BTC': 'PoW',    # Bitcoin uses Proof of Work
        'ETH': 'PoS',    # Ethereum uses Proof of Stake (after merge)
        'TRX': 'DPoS',   # TRON uses Delegated Proof of Stake
        'EOS': 'DPoS',   # EOS uses Delegated Proof of Stake
        'ADA': 'PoS',    # Cardano uses Proof of Stake
        'SOL': 'PoS',    # Solana uses Proof of Stake
        'DOT': 'PoS',    # Polkadot uses Proof of Stake
        'AVAX': 'PoS',   # Avalanche uses Proof of Stake
        'LTC': 'PoW',    # Litecoin uses Proof of Work
        'BCH': 'PoW',    # Bitcoin Cash uses Proof of Work
        'XRP': 'PoS',    # Ripple uses Proof of Stake
        'BNB': 'PoS',    # Binance Coin uses Proof of Stake
    }
    
    def __init__(self, symbol: str = "BTC", start_date: str = None, end_date: str = None):
        """
        Initialize the data loader.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.consensus = self.CONSENSUS_MAPPING.get(symbol, 'Unknown')
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage."""
        try:
            url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={self.symbol}&market=USD&apikey={self.api_key}&outputsize=full"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Digital Currency Daily)" not in data:
                print("Error: No data received from Alpha Vantage")
                print("API Response:", data)  # Print the response for debugging
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Digital Currency Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            
            # Print column names for debugging
            print("Available columns:", df.columns.tolist())
            
            # Rename columns based on the actual API response
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            
            # Rename only the columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col].astype(float)
            
            # Filter date range
            mask = (df.index >= self.start_date) & (df.index <= self.end_date)
            self.data = df[mask].copy()
            
            # Add consensus mechanism information
            self.data['Consensus'] = self.consensus
            
            # Process data
            self.data = self._process_data()
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Full error details:", str(e))
            return pd.DataFrame()
    
    def _process_data(self) -> pd.DataFrame:
        """Process raw data and calculate additional metrics."""
        df = self.data.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate volatility (20-day rolling)
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate volume metrics
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Calculate technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
        
        return df
    
    def get_simulation_data(self) -> Dict[str, List[float]]:
        """Convert processed data into simulation-compatible format."""
        if self.data is None:
            self.fetch_data()
            
        return {
            'price_history': self.data['Close'].tolist(),
            'volume_history': self.data['Volume'].tolist(),
            'volatility_history': self.data['Volatility'].tolist(),
            'rsi_history': self.data['RSI'].tolist(),
            'macd_history': self.data['MACD'].tolist(),
            'signal_line_history': self.data['Signal_Line'].tolist(),
            'bb_upper_history': self.data['BB_upper'].tolist(),
            'bb_lower_history': self.data['BB_lower'].tolist()
        }
    
    def get_market_metrics(self) -> Dict[str, float]:
        """Calculate market metrics for simulation initialization."""
        if self.data is None:
            self.fetch_data()
            
        last_price = self.data['Close'].iloc[-1]
        avg_volume = self.data['Volume'].mean()
        volatility = self.data['Volatility'].iloc[-1]
        
        return {
            'initial_price': last_price,
            'avg_daily_volume': avg_volume,
            'volatility': volatility,
            'price_range': {
                'min': self.data['Close'].min(),
                'max': self.data['Close'].max(),
                'mean': self.data['Close'].mean()
            }
        } 