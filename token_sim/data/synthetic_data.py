import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

class SyntheticDataGenerator:
    """Generates synthetic cryptocurrency data based on historical patterns."""
    
    def __init__(self, 
                 initial_price: float = 50000.0,
                 volatility: float = 0.02,
                 trend: float = 0.0001,
                 volume_mean: float = 1000000.0,
                 volume_std: float = 200000.0,
                 days: int = 30):
        """
        Initialize the synthetic data generator.
        
        Args:
            initial_price: Starting price
            volatility: Daily price volatility
            trend: Long-term price trend
            volume_mean: Average daily volume
            volume_std: Volume standard deviation
            days: Number of days to generate
        """
        self.initial_price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.volume_mean = volume_mean
        self.volume_std = volume_std
        self.days = days
        self.data = None
        
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic market data."""
        # Generate dates
        dates = [datetime.now() - timedelta(days=i) for i in range(self.days)]
        dates.reverse()
        
        # Generate price movements
        returns = np.random.normal(self.trend, self.volatility, self.days)
        price = self.initial_price * np.exp(np.cumsum(returns))
        
        # Create DataFrame first for rolling calculations
        df = pd.DataFrame({
            'Date': dates,
            'Returns': returns,
            'Close': price
        })
        df.set_index('Date', inplace=True)
        
        # Add some realistic patterns
        # 1. Mean reversion
        df['Close'] = df['Close'] + (self.initial_price - df['Close']) * 0.1
        
        # 2. Momentum effects
        momentum = np.convolve(returns, np.ones(5)/5, mode='same')
        df['Close'] = df['Close'] * (1 + momentum * 0.5)
        
        # 3. Volatility clustering
        df['Vol_Cluster'] = df['Returns'].abs().rolling(window=5, min_periods=1).mean()
        df['Close'] = df['Close'] * (1 + df['Vol_Cluster'] * np.random.normal(0, 0.5, self.days))
        
        # Generate volume with correlation to absolute returns
        df['Volume'] = np.random.normal(self.volume_mean, self.volume_std, self.days)
        df['Volume'] = df['Volume'] * (1 + df['Returns'].abs() * 5)  # Volume increases with volatility
        
        # Calculate additional metrics
        df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std() * np.sqrt(252)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
        
        self.data = df
        return df
    
    def get_simulation_data(self) -> Dict[str, List[float]]:
        """Get data in simulation-compatible format."""
        if self.data is None:
            self.generate_data()
            
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
        """Get market metrics for simulation initialization."""
        if self.data is None:
            self.generate_data()
            
        return {
            'initial_price': self.data['Close'].iloc[-1],
            'avg_daily_volume': self.data['Volume'].mean(),
            'volatility': self.data['Volatility'].iloc[-1],
            'price_range': {
                'min': self.data['Close'].min(),
                'max': self.data['Close'].max(),
                'mean': self.data['Close'].mean()
            }
        } 