from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class AnalysisType(Enum):
    """Types of analysis to perform."""
    MARKET = "market"
    NETWORK = "network"
    AGENT = "agent"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"

@dataclass
class AnalysisMetrics:
    """Metrics for analysis results."""
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    quantiles: Dict[str, float]
    correlation: float
    r_squared: float
    mse: float
    mae: float

class ResultsAnalyzer:
    """Analyzes simulation results comprehensively."""
    
    def __init__(self):
        """Initialize the results analyzer."""
        self.analysis_results: Dict[str, AnalysisMetrics] = {}
        self.plots: Dict[str, plt.Figure] = {}
    
    def analyze_market_results(self, market_data: pd.DataFrame) -> AnalysisMetrics:
        """Analyze market simulation results.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            AnalysisMetrics object with market analysis results
        """
        # Calculate basic statistics
        price_stats = self._calculate_basic_stats(market_data['price'])
        volume_stats = self._calculate_basic_stats(market_data['volume'])
        
        # Calculate market microstructure metrics
        spread_stats = self._calculate_basic_stats(market_data['spread'])
        depth_stats = self._calculate_basic_stats(market_data['market_depth'])
        
        # Calculate trading metrics
        returns = market_data['price'].pct_change().dropna()
        volatility = returns.std()
        sharpe_ratio = returns.mean() / volatility * np.sqrt(252)
        
        metrics = AnalysisMetrics(
            mean=price_stats['mean'],
            median=price_stats['median'],
            std=price_stats['std'],
            skewness=price_stats['skewness'],
            kurtosis=price_stats['kurtosis'],
            min=price_stats['min'],
            max=price_stats['max'],
            quantiles=price_stats['quantiles'],
            correlation=self._calculate_correlation(returns, market_data['volume']),
            r_squared=self._calculate_r_squared(returns, market_data['volume']),
            mse=self._calculate_mse(returns, market_data['volume']),
            mae=self._calculate_mae(returns, market_data['volume'])
        )
        
        self.analysis_results['market'] = metrics
        return metrics
    
    def analyze_network_results(self, network_data: pd.DataFrame) -> AnalysisMetrics:
        """Analyze network simulation results.
        
        Args:
            network_data: DataFrame with network data
            
        Returns:
            AnalysisMetrics object with network analysis results
        """
        # Calculate network statistics
        hashrate_stats = self._calculate_basic_stats(network_data['hashrate'])
        difficulty_stats = self._calculate_basic_stats(network_data['difficulty'])
        
        # Calculate network health metrics
        block_time_stats = self._calculate_basic_stats(network_data['block_time'])
        active_addresses_stats = self._calculate_basic_stats(network_data['active_addresses'])
        
        # Calculate network efficiency metrics
        tps = network_data['transaction_count'] / network_data['block_time']
        tps_stats = self._calculate_basic_stats(tps)
        
        metrics = AnalysisMetrics(
            mean=hashrate_stats['mean'],
            median=hashrate_stats['median'],
            std=hashrate_stats['std'],
            skewness=hashrate_stats['skewness'],
            kurtosis=hashrate_stats['kurtosis'],
            min=hashrate_stats['min'],
            max=hashrate_stats['max'],
            quantiles=hashrate_stats['quantiles'],
            correlation=self._calculate_correlation(network_data['hashrate'], network_data['difficulty']),
            r_squared=self._calculate_r_squared(network_data['hashrate'], network_data['difficulty']),
            mse=self._calculate_mse(network_data['hashrate'], network_data['difficulty']),
            mae=self._calculate_mae(network_data['hashrate'], network_data['difficulty'])
        )
        
        self.analysis_results['network'] = metrics
        return metrics
    
    def analyze_agent_results(self, agent_data: Dict[str, Any]) -> AnalysisMetrics:
        """Analyze agent simulation results.
        
        Args:
            agent_data: Dictionary with agent performance data
            
        Returns:
            AnalysisMetrics object with agent analysis results
        """
        # Calculate agent performance statistics
        pnl_stats = self._calculate_basic_stats(pd.Series(agent_data['pnl_history']))
        position_stats = self._calculate_basic_stats(pd.Series(agent_data['position_history']))
        
        # Calculate trading metrics
        win_rate = self._calculate_win_rate(agent_data['pnl_history'])
        profit_factor = self._calculate_profit_factor(agent_data['pnl_history'])
        
        metrics = AnalysisMetrics(
            mean=pnl_stats['mean'],
            median=pnl_stats['median'],
            std=pnl_stats['std'],
            skewness=pnl_stats['skewness'],
            kurtosis=pnl_stats['kurtosis'],
            min=pnl_stats['min'],
            max=pnl_stats['max'],
            quantiles=pnl_stats['quantiles'],
            correlation=self._calculate_correlation(pd.Series(agent_data['pnl_history']), pd.Series(agent_data['position_history'])),
            r_squared=self._calculate_r_squared(pd.Series(agent_data['pnl_history']), pd.Series(agent_data['position_history'])),
            mse=self._calculate_mse(pd.Series(agent_data['pnl_history']), pd.Series(agent_data['position_history'])),
            mae=self._calculate_mae(pd.Series(agent_data['pnl_history']), pd.Series(agent_data['position_history']))
        )
        
        self.analysis_results['agent'] = metrics
        return metrics
    
    def analyze_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between different metrics.
        
        Args:
            data: DataFrame with multiple metrics
            
        Returns:
            DataFrame with correlation matrix
        """
        correlation_matrix = data.corr()
        self.analysis_results['correlation'] = correlation_matrix
        return correlation_matrix
    
    def perform_time_series_analysis(self, data: pd.Series) -> Dict[str, Any]:
        """Perform time series analysis on data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with time series analysis results
        """
        # Calculate time series metrics
        acf = self._calculate_acf(data)
        pacf = self._calculate_pacf(data)
        stationarity = self._test_stationarity(data)
        
        results = {
            'acf': acf,
            'pacf': pacf,
            'stationarity': stationarity,
            'trend': self._detect_trend(data),
            'seasonality': self._detect_seasonality(data)
        }
        
        self.analysis_results['time_series'] = results
        return results
    
    def _calculate_basic_stats(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate basic statistical metrics.
        
        Args:
            data: Data series
            
        Returns:
            Dictionary with statistical metrics
        """
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis(),
            'min': data.min(),
            'max': data.max(),
            'quantiles': {
                '25%': data.quantile(0.25),
                '50%': data.quantile(0.50),
                '75%': data.quantile(0.75)
            }
        }
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate correlation between two series.
        
        Args:
            x: First series
            y: Second series
            
        Returns:
            Correlation coefficient
        """
        return x.corr(y)
    
    def _calculate_r_squared(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate R-squared value.
        
        Args:
            x: Independent variable
            y: Dependent variable
            
        Returns:
            R-squared value
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return r_value ** 2
    
    def _calculate_mse(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Mean Squared Error.
        
        Args:
            x: Predicted values
            y: Actual values
            
        Returns:
            MSE value
        """
        return mean_squared_error(x, y)
    
    def _calculate_mae(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Mean Absolute Error.
        
        Args:
            x: Predicted values
            y: Actual values
            
        Returns:
            MAE value
        """
        return mean_absolute_error(x, y)
    
    def _calculate_win_rate(self, pnl_history: List[float]) -> float:
        """Calculate win rate from PnL history.
        
        Args:
            pnl_history: List of PnL values
            
        Returns:
            Win rate
        """
        winning_trades = sum(1 for pnl in pnl_history if pnl > 0)
        return winning_trades / len(pnl_history) if pnl_history else 0
    
    def _calculate_profit_factor(self, pnl_history: List[float]) -> float:
        """Calculate profit factor from PnL history.
        
        Args:
            pnl_history: List of PnL values
            
        Returns:
            Profit factor
        """
        gross_profit = sum(pnl for pnl in pnl_history if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_history if pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_acf(self, data: pd.Series, nlags: int = 40) -> pd.Series:
        """Calculate Autocorrelation Function.
        
        Args:
            data: Time series data
            nlags: Number of lags
            
        Returns:
            Series with ACF values
        """
        return pd.Series(stats.acf(data, nlags=nlags))
    
    def _calculate_pacf(self, data: pd.Series, nlags: int = 40) -> pd.Series:
        """Calculate Partial Autocorrelation Function.
        
        Args:
            data: Time series data
            nlags: Number of lags
            
        Returns:
            Series with PACF values
        """
        return pd.Series(stats.pacf(data, nlags=nlags))
    
    def _test_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Test time series stationarity.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        # Perform Augmented Dickey-Fuller test
        adf_result = stats.adfuller(data)
        
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05
        }
    
    def _detect_trend(self, data: pd.Series) -> Dict[str, Any]:
        """Detect trend in time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with trend analysis results
        """
        # Calculate linear regression
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'has_trend': p_value < 0.05
        }
    
    def _detect_seasonality(self, data: pd.Series) -> Dict[str, Any]:
        """Detect seasonality in time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with seasonality analysis results
        """
        # Calculate periodogram
        f, pxx = stats.periodogram(data)
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(pxx)
        dominant_freq = f[dominant_freq_idx]
        
        return {
            'dominant_frequency': dominant_freq,
            'period': 1 / dominant_freq if dominant_freq > 0 else None,
            'has_seasonality': dominant_freq > 0
        }
    
    def plot_results(self, data: pd.DataFrame, plot_type: str) -> None:
        """Create plots for analysis results.
        
        Args:
            data: Data to plot
            plot_type: Type of plot to create
        """
        if plot_type == 'market':
            self._plot_market_analysis(data)
        elif plot_type == 'network':
            self._plot_network_analysis(data)
        elif plot_type == 'correlation':
            self._plot_correlation_matrix(data)
        elif plot_type == 'time_series':
            self._plot_time_series_analysis(data)
    
    def _plot_market_analysis(self, data: pd.DataFrame) -> None:
        """Create market analysis plots.
        
        Args:
            data: Market data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price plot
        axes[0, 0].plot(data.index, data['price'])
        axes[0, 0].set_title('Price')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price')
        
        # Volume plot
        axes[0, 1].bar(data.index, data['volume'])
        axes[0, 1].set_title('Volume')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Volume')
        
        # Returns distribution
        returns = data['price'].pct_change().dropna()
        sns.histplot(returns, ax=axes[1, 0], kde=True)
        axes[1, 0].set_title('Returns Distribution')
        
        # Market depth
        axes[1, 1].plot(data.index, data['market_depth'])
        axes[1, 1].set_title('Market Depth')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Depth')
        
        plt.tight_layout()
        self.plots['market'] = fig
    
    def _plot_network_analysis(self, data: pd.DataFrame) -> None:
        """Create network analysis plots.
        
        Args:
            data: Network data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hashrate plot
        axes[0, 0].plot(data.index, data['hashrate'])
        axes[0, 0].set_title('Hashrate')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Hashrate')
        
        # Difficulty plot
        axes[0, 1].plot(data.index, data['difficulty'])
        axes[0, 1].set_title('Difficulty')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Difficulty')
        
        # Block time distribution
        sns.histplot(data['block_time'], ax=axes[1, 0], kde=True)
        axes[1, 0].set_title('Block Time Distribution')
        
        # Active addresses
        axes[1, 1].plot(data.index, data['active_addresses'])
        axes[1, 1].set_title('Active Addresses')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        self.plots['network'] = fig
    
    def _plot_correlation_matrix(self, data: pd.DataFrame) -> None:
        """Create correlation matrix plot.
        
        Args:
            data: Data for correlation analysis
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        self.plots['correlation'] = fig
    
    def _plot_time_series_analysis(self, data: pd.Series) -> None:
        """Create time series analysis plots.
        
        Args:
            data: Time series data
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0, 0].plot(data.index, data)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        
        # ACF plot
        acf = self._calculate_acf(data)
        axes[0, 1].stem(range(len(acf)), acf)
        axes[0, 1].set_title('Autocorrelation Function')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('ACF')
        
        # PACF plot
        pacf = self._calculate_pacf(data)
        axes[1, 0].stem(range(len(pacf)), pacf)
        axes[1, 0].set_title('Partial Autocorrelation Function')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('PACF')
        
        # Distribution plot
        sns.histplot(data, ax=axes[1, 1], kde=True)
        axes[1, 1].set_title('Distribution')
        
        plt.tight_layout()
        self.plots['time_series'] = fig
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report.
        
        Returns:
            Dictionary containing analysis results and plots
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_results': {
                key: {
                    'mean': metrics.mean,
                    'median': metrics.median,
                    'std': metrics.std,
                    'skewness': metrics.skewness,
                    'kurtosis': metrics.kurtosis,
                    'min': metrics.min,
                    'max': metrics.max,
                    'quantiles': metrics.quantiles,
                    'correlation': metrics.correlation,
                    'r_squared': metrics.r_squared,
                    'mse': metrics.mse,
                    'mae': metrics.mae
                }
                for key, metrics in self.analysis_results.items()
                if isinstance(metrics, AnalysisMetrics)
            }
        }
        
        return report 