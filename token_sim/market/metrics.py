from typing import Dict, Any, List
import pandas as pd
import numpy as np
import talib
from scipy import stats

class MarketMetricsCalculator:
    """Calculator for comprehensive market metrics and analysis."""
    
    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive market metrics from historical data.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Dictionary of calculated market metrics
        """
        # Convert price data to numpy arrays for technical analysis
        close_prices = data['Close'].values
        high_prices = data['High'].values
        low_prices = data['Low'].values
        volume = data['Volume'].values
        
        return {
            # Volume analysis
            'volume_profile': self._calculate_volume_profile(volume),
            'volume_distribution': self._calculate_volume_distribution(volume),
            
            # Market microstructure
            'bid_ask_spread': self._calculate_bid_ask_spread(data),
            'market_depth': self._calculate_market_depth(data),
            'liquidity_metrics': self._calculate_liquidity_metrics(data),
            'order_book_metrics': self._calculate_order_book_metrics(data),
            
            # Technical indicators
            'rsi': self._calculate_rsi(close_prices),
            'macd': self._calculate_macd(close_prices),
            'bollinger_bands': self._calculate_bollinger_bands(close_prices),
            'moving_averages': self._calculate_moving_averages(close_prices),
            
            # Risk metrics
            'var_95': self._calculate_var(close_prices),
            'expected_shortfall': self._calculate_expected_shortfall(close_prices),
            'max_drawdown': self._calculate_max_drawdown(close_prices),
            'volatility_clustering': self._calculate_volatility_clustering(close_prices)
        }
    
    def _calculate_volume_profile(self, volume: np.ndarray) -> Dict[str, float]:
        """Calculate volume profile metrics."""
        return {
            'avg_volume': np.mean(volume),
            'volume_std': np.std(volume),
            'volume_skew': stats.skew(volume),
            'volume_kurtosis': stats.kurtosis(volume)
        }
    
    def _calculate_volume_distribution(self, volume: np.ndarray) -> Dict[str, float]:
        """Calculate volume distribution metrics."""
        percentiles = np.percentile(volume, [25, 50, 75, 90, 95, 99])
        return {
            'p25': percentiles[0],
            'median': percentiles[1],
            'p75': percentiles[2],
            'p90': percentiles[3],
            'p95': percentiles[4],
            'p99': percentiles[5]
        }
    
    def _calculate_bid_ask_spread(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate bid-ask spread metrics."""
        # Simulate bid-ask spread if not available
        spread = (data['High'] - data['Low']) * 0.001  # 0.1% of price range
        return {
            'avg_spread': np.mean(spread),
            'spread_std': np.std(spread),
            'min_spread': np.min(spread),
            'max_spread': np.max(spread)
        }
    
    def _calculate_market_depth(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market depth metrics."""
        # Simulate market depth based on volume
        depth = data['Volume'] * 0.1  # 10% of volume as market depth
        return {
            'avg_depth': np.mean(depth),
            'depth_std': np.std(depth),
            'min_depth': np.min(depth),
            'max_depth': np.max(depth)
        }
    
    def _calculate_liquidity_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate liquidity metrics."""
        # Calculate Amihud illiquidity ratio
        returns = np.diff(np.log(data['Close']))
        volume = data['Volume'].values[1:]
        amihud = np.abs(returns) / volume
        
        return {
            'amihud_ratio': np.mean(amihud),
            'liquidity_ratio': 1 / np.mean(amihud),
            'turnover_ratio': np.mean(data['Volume'] / data['Close'])
        }
    
    def _calculate_order_book_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate order book metrics."""
        # Simulate order book metrics
        spread = (data['High'] - data['Low']) * 0.001
        depth = data['Volume'] * 0.1
        
        return {
            'order_imbalance': np.random.normal(0, 0.1),  # Simulated
            'book_depth': np.mean(depth),
            'spread_impact': np.mean(spread / data['Close'])
        }
    
    def _calculate_rsi(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate RSI metrics."""
        rsi = talib.RSI(prices)
        return {
            'current': rsi[-1],
            'avg': np.mean(rsi[~np.isnan(rsi)]),
            'std': np.std(rsi[~np.isnan(rsi)])
        }
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD metrics."""
        macd, signal, hist = talib.MACD(prices)
        return {
            'macd': macd[-1],
            'signal': signal[-1],
            'histogram': hist[-1]
        }
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate Bollinger Bands metrics."""
        upper, middle, lower = talib.BBANDS(prices)
        return {
            'upper': upper[-1],
            'middle': middle[-1],
            'lower': lower[-1],
            'band_width': (upper[-1] - lower[-1]) / middle[-1]
        }
    
    def _calculate_moving_averages(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate moving averages metrics."""
        return {
            'sma_20': talib.SMA(prices, timeperiod=20)[-1],
            'sma_50': talib.SMA(prices, timeperiod=50)[-1],
            'sma_200': talib.SMA(prices, timeperiod=200)[-1],
            'ema_20': talib.EMA(prices, timeperiod=20)[-1],
            'ema_50': talib.EMA(prices, timeperiod=50)[-1]
        }
    
    def _calculate_var(self, prices: np.ndarray) -> float:
        """Calculate 95% Value at Risk."""
        returns = np.diff(np.log(prices))
        return np.percentile(returns, 5)
    
    def _calculate_expected_shortfall(self, prices: np.ndarray) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        returns = np.diff(np.log(prices))
        var_95 = np.percentile(returns, 5)
        return np.mean(returns[returns <= var_95])
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        return np.max(drawdown)
    
    def _calculate_volatility_clustering(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate volatility clustering metrics."""
        returns = np.diff(np.log(prices))
        volatility = np.abs(returns)
        
        # Calculate autocorrelation of volatility
        acf = np.correlate(volatility, volatility, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]  # Normalize
        
        return {
            'volatility_autocorr': acf[1],  # First lag autocorrelation
            'volatility_persistence': np.sum(acf[1:6]) / 5  # Average of first 5 lags
        } 