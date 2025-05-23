from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    market_metrics: Dict[str, float]
    network_metrics: Dict[str, float]
    interaction_metrics: Dict[str, float]
    statistical_tests: Dict[str, Tuple[float, float]]  # (statistic, p-value)

class SimulationValidator:
    """Framework for validating simulation results."""
    
    def __init__(self, historical_data: pd.DataFrame, simulation_results: Dict[str, pd.DataFrame]):
        """Initialize the validator.
        
        Args:
            historical_data: DataFrame containing historical market and network data
            simulation_results: Dictionary containing simulation results
        """
        self.historical_data = historical_data
        self.simulation_results = simulation_results
        self.market_df = simulation_results['market']
        self.network_df = simulation_results['network']
        self.interactions_df = simulation_results['interactions']
    
    def validate_market_metrics(self) -> Dict[str, float]:
        """Validate market-related metrics against historical data."""
        metrics = {}
        
        # Price validation
        historical_returns = np.diff(np.log(self.historical_data['price']))
        simulated_returns = np.diff(np.log(self.market_df['price']))
        
        metrics['price_volatility_match'] = self._compare_volatility(
            historical_returns, simulated_returns
        )
        metrics['price_distribution_match'] = self._compare_distributions(
            historical_returns, simulated_returns
        )
        
        # Volume validation
        metrics['volume_correlation'] = self._calculate_correlation(
            self.historical_data['volume'],
            self.market_df['volume']
        )
        
        # Spread validation
        metrics['spread_realism'] = self._validate_spread_realism()
        
        # Market depth validation
        metrics['depth_liquidity'] = self._validate_market_depth()
        
        return metrics
    
    def validate_network_metrics(self) -> Dict[str, float]:
        """Validate network-related metrics against historical data."""
        metrics = {}
        
        # Hashrate validation
        if 'hashrate' in self.historical_data.columns:
            metrics['hashrate_correlation'] = self._calculate_correlation(
                self.historical_data['hashrate'],
                self.network_df['hashrate']
            )
        
        # Block time validation
        if 'block_time' in self.historical_data.columns:
            metrics['block_time_realism'] = self._validate_block_time()
        
        # Participant count validation
        if 'participant_count' in self.historical_data.columns:
            metrics['participant_growth'] = self._validate_participant_growth()
        
        # Stake distribution validation
        if 'total_stake' in self.historical_data.columns:
            metrics['stake_distribution'] = self._validate_stake_distribution()
        
        return metrics
    
    def validate_interactions(self) -> Dict[str, float]:
        """Validate market-network interactions."""
        metrics = {}
        
        # Price impact validation
        metrics['price_impact_realism'] = self._validate_price_impact()
        
        # Volume impact validation
        metrics['volume_impact_realism'] = self._validate_volume_impact()
        
        # Network growth impact validation
        metrics['network_growth_realism'] = self._validate_network_growth()
        
        return metrics
    
    def run_statistical_tests(self) -> Dict[str, Tuple[float, float]]:
        """Run statistical tests on simulation results."""
        tests = {}
        
        # Test for stationarity
        tests['price_stationarity'] = self._test_stationarity(self.market_df['price'])
        tests['volume_stationarity'] = self._test_stationarity(self.market_df['volume'])
        
        # Test for autocorrelation
        tests['price_autocorrelation'] = self._test_autocorrelation(self.market_df['price'])
        tests['volume_autocorrelation'] = self._test_autocorrelation(self.market_df['volume'])
        
        # Test for normality
        tests['returns_normality'] = self._test_normality(
            np.diff(np.log(self.market_df['price']))
        )
        
        return tests
    
    def validate_all(self) -> ValidationMetrics:
        """Run all validation checks and return comprehensive metrics."""
        return ValidationMetrics(
            market_metrics=self.validate_market_metrics(),
            network_metrics=self.validate_network_metrics(),
            interaction_metrics=self.validate_interactions(),
            statistical_tests=self.run_statistical_tests()
        )
    
    def _compare_volatility(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Compare volatility between two series."""
        vol1 = np.std(series1)
        vol2 = np.std(series2)
        return 1 - abs(vol1 - vol2) / max(vol1, vol2)
    
    def _compare_distributions(self, series1: np.ndarray, series2: np.ndarray) -> float:
        """Compare distributions using Kolmogorov-Smirnov test."""
        ks_stat, p_value = stats.ks_2samp(series1, series2)
        return 1 - ks_stat  # Convert to similarity score
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two series."""
        return series1.corr(series2)
    
    def _validate_spread_realism(self) -> float:
        """Validate if spread behavior is realistic."""
        spread = self.market_df['spread']
        price = self.market_df['price']
        
        # Check if spread is proportional to price
        spread_ratio = spread / price
        return 1 - np.std(spread_ratio)  # Lower std = more realistic
    
    def _validate_market_depth(self) -> float:
        """Validate market depth and liquidity."""
        depth = self.market_df['depth']
        
        # Calculate bid-ask imbalance
        bid_depth = depth.apply(lambda x: sum(q for _, q in x['bids']))
        ask_depth = depth.apply(lambda x: sum(q for _, q in x['asks']))
        
        # Check if depth is reasonably balanced
        imbalance = abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
        return 1 - np.mean(imbalance)
    
    def _validate_block_time(self) -> float:
        """Validate if block time is realistic."""
        block_time = self.network_df['block_time']
        
        # Check if block time is relatively stable
        return 1 - np.std(block_time) / np.mean(block_time)
    
    def _validate_participant_growth(self) -> float:
        """Validate participant growth patterns."""
        participants = self.network_df['participant_count']
        
        # Check if growth is smooth and positive
        growth_rate = np.diff(participants) / participants[:-1]
        return 1 - np.std(growth_rate)
    
    def _validate_stake_distribution(self) -> float:
        """Validate stake distribution patterns."""
        total_stake = self.network_df['total_stake']
        
        # Check if stake growth is reasonable
        stake_growth = np.diff(total_stake) / total_stake[:-1]
        return 1 - np.std(stake_growth)
    
    def _validate_price_impact(self) -> float:
        """Validate price impact patterns."""
        price_impact = self.interactions_df['price_impact']
        
        # Check if price impact is reasonable
        return 1 - np.std(price_impact)
    
    def _validate_volume_impact(self) -> float:
        """Validate volume impact patterns."""
        volume_impact = self.interactions_df['volume_impact']
        
        # Check if volume impact is reasonable
        return 1 - np.std(volume_impact)
    
    def _validate_network_growth(self) -> float:
        """Validate network growth patterns."""
        network_growth = self.interactions_df['network_growth']
        
        # Check if network growth is reasonable
        return 1 - np.std(network_growth)
    
    def _test_stationarity(self, series: pd.Series) -> Tuple[float, float]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series)
        return result[0], result[1]  # test statistic, p-value
    
    def _test_autocorrelation(self, series: pd.Series) -> Tuple[float, float]:
        """Test for autocorrelation using Ljung-Box test."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        result = acorr_ljungbox(series, lags=10)
        return result[0][0], result[1][0]  # test statistic, p-value
    
    def _test_normality(self, series: np.ndarray) -> Tuple[float, float]:
        """Test for normality using Jarque-Bera test."""
        from statsmodels.stats.stattools import jarque_bera
        result = jarque_bera(series)
        return result[0], result[1]  # test statistic, p-value
    
    def generate_validation_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive validation report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report as a string
        """
        metrics = self.validate_all()
        
        report = []
        report.append("Simulation Validation Report")
        report.append("=" * 50)
        
        # Market metrics
        report.append("\nMarket Metrics Validation:")
        report.append("-" * 30)
        for metric, value in metrics.market_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Network metrics
        report.append("\nNetwork Metrics Validation:")
        report.append("-" * 30)
        for metric, value in metrics.network_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Interaction metrics
        report.append("\nInteraction Metrics Validation:")
        report.append("-" * 30)
        for metric, value in metrics.interaction_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        
        # Statistical tests
        report.append("\nStatistical Tests:")
        report.append("-" * 30)
        for test, (stat, pval) in metrics.statistical_tests.items():
            report.append(f"{test}:")
            report.append(f"  Statistic: {stat:.4f}")
            report.append(f"  p-value: {pval:.4f}")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str 