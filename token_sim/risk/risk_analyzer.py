from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class RiskType(Enum):
    """Types of risks to analyze."""
    MARKET = "market"
    NETWORK = "network"
    AGENT = "agent"
    SYSTEMIC = "systemic"

@dataclass
class RiskMetrics:
    """Risk metrics for a specific risk type."""
    value_at_risk: float  # Value at Risk (VaR)
    expected_shortfall: float  # Expected Shortfall (ES)
    volatility: float  # Volatility
    sharpe_ratio: float  # Sharpe Ratio
    max_drawdown: float  # Maximum Drawdown
    correlation: float  # Correlation with market
    beta: float  # Beta coefficient
    gini_coefficient: float  # Gini coefficient for distribution
    concentration_risk: float  # Concentration risk
    liquidity_risk: float  # Liquidity risk

class RiskAnalyzer:
    """Analyzes various risks in the cryptocurrency simulation."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize the risk analyzer.
        
        Args:
            confidence_level: Confidence level for VaR and ES calculations
        """
        self.confidence_level = confidence_level
        self.risk_metrics: Dict[RiskType, RiskMetrics] = {}
    
    def analyze_market_risk(self, market_data: pd.DataFrame) -> RiskMetrics:
        """Analyze market risk metrics.
        
        Args:
            market_data: DataFrame with market data (price, volume, etc.)
            
        Returns:
            RiskMetrics object with market risk metrics
        """
        returns = market_data['price'].pct_change().dropna()
        
        # Calculate basic risk metrics
        volatility = returns.std()
        var = self._calculate_var(returns)
        es = self._calculate_expected_shortfall(returns)
        sharpe = self._calculate_sharpe_ratio(returns)
        max_dd = self._calculate_max_drawdown(market_data['price'])
        
        # Calculate market microstructure metrics
        liquidity_risk = self._calculate_liquidity_risk(market_data)
        concentration_risk = self._calculate_concentration_risk(market_data)
        
        # Calculate distribution metrics
        gini = self._calculate_gini_coefficient(returns)
        
        metrics = RiskMetrics(
            value_at_risk=var,
            expected_shortfall=es,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            correlation=1.0,  # Market correlation with itself is 1
            beta=1.0,  # Market beta is 1
            gini_coefficient=gini,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk
        )
        
        self.risk_metrics[RiskType.MARKET] = metrics
        return metrics
    
    def analyze_network_risk(self, network_data: pd.DataFrame) -> RiskMetrics:
        """Analyze network risk metrics.
        
        Args:
            network_data: DataFrame with network data (hashrate, difficulty, etc.)
            
        Returns:
            RiskMetrics object with network risk metrics
        """
        # Calculate network stability metrics
        hashrate_volatility = network_data['hashrate'].pct_change().std()
        difficulty_volatility = network_data['difficulty'].pct_change().std()
        
        # Calculate network health metrics
        network_health = self._calculate_network_health(network_data)
        security_risk = self._calculate_security_risk(network_data)
        
        # Calculate distribution metrics
        gini = self._calculate_gini_coefficient(network_data['hashrate'])
        
        metrics = RiskMetrics(
            value_at_risk=self._calculate_var(network_data['hashrate'].pct_change()),
            expected_shortfall=self._calculate_expected_shortfall(network_data['hashrate'].pct_change()),
            volatility=hashrate_volatility,
            sharpe_ratio=self._calculate_sharpe_ratio(network_data['hashrate'].pct_change()),
            max_drawdown=self._calculate_max_drawdown(network_data['hashrate']),
            correlation=self._calculate_correlation(network_data['hashrate'], network_data['difficulty']),
            beta=self._calculate_beta(network_data['hashrate'], network_data['difficulty']),
            gini_coefficient=gini,
            concentration_risk=security_risk,
            liquidity_risk=network_health
        )
        
        self.risk_metrics[RiskType.NETWORK] = metrics
        return metrics
    
    def analyze_agent_risk(self, agent_data: Dict[str, Any], market_data: pd.DataFrame) -> RiskMetrics:
        """Analyze agent-specific risk metrics.
        
        Args:
            agent_data: Dictionary with agent performance data
            market_data: DataFrame with market data for correlation analysis
            
        Returns:
            RiskMetrics object with agent risk metrics
        """
        # Calculate agent performance metrics
        returns = pd.Series(agent_data['pnl_history']).pct_change()
        position_volatility = pd.Series(agent_data['position_history']).std()
        
        # Calculate agent-specific risks
        strategy_risk = self._calculate_strategy_risk(agent_data)
        execution_risk = self._calculate_execution_risk(agent_data)
        
        # Calculate correlation with market
        market_returns = market_data['price'].pct_change()
        correlation = self._calculate_correlation(returns, market_returns)
        beta = self._calculate_beta(returns, market_returns)
        
        metrics = RiskMetrics(
            value_at_risk=self._calculate_var(returns),
            expected_shortfall=self._calculate_expected_shortfall(returns),
            volatility=position_volatility,
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(pd.Series(agent_data['pnl_history'])),
            correlation=correlation,
            beta=beta,
            gini_coefficient=self._calculate_gini_coefficient(returns),
            concentration_risk=strategy_risk,
            liquidity_risk=execution_risk
        )
        
        self.risk_metrics[RiskType.AGENT] = metrics
        return metrics
    
    def analyze_systemic_risk(self, market_data: pd.DataFrame, network_data: pd.DataFrame) -> RiskMetrics:
        """Analyze systemic risk metrics.
        
        Args:
            market_data: DataFrame with market data
            network_data: DataFrame with network data
            
        Returns:
            RiskMetrics object with systemic risk metrics
        """
        # Calculate systemic risk indicators
        market_volatility = market_data['price'].pct_change().std()
        network_volatility = network_data['hashrate'].pct_change().std()
        
        # Calculate cross-market impact
        market_impact = self._calculate_market_impact(market_data, network_data)
        network_impact = self._calculate_network_impact(market_data, network_data)
        
        # Calculate systemic risk metrics
        systemic_risk = self._calculate_systemic_risk(market_data, network_data)
        
        metrics = RiskMetrics(
            value_at_risk=self._calculate_var(market_data['price'].pct_change()),
            expected_shortfall=self._calculate_expected_shortfall(market_data['price'].pct_change()),
            volatility=market_volatility,
            sharpe_ratio=self._calculate_sharpe_ratio(market_data['price'].pct_change()),
            max_drawdown=self._calculate_max_drawdown(market_data['price']),
            correlation=self._calculate_correlation(market_data['price'], network_data['hashrate']),
            beta=self._calculate_beta(market_data['price'], network_data['hashrate']),
            gini_coefficient=self._calculate_gini_coefficient(market_data['price'].pct_change()),
            concentration_risk=market_impact,
            liquidity_risk=network_impact
        )
        
        self.risk_metrics[RiskType.SYSTEMIC] = metrics
        return metrics
    
    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - self.confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, returns: pd.Series) -> float:
        """Calculate Expected Shortfall."""
        var = self._calculate_var(returns)
        return returns[returns <= var].mean()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe Ratio."""
        return returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate correlation between two series."""
        return x.corr(y)
    
    def _calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient."""
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * values).sum() / (n * values.sum())
    
    def _calculate_liquidity_risk(self, market_data: pd.DataFrame) -> float:
        """Calculate liquidity risk based on volume and spread."""
        volume_volatility = market_data['volume'].pct_change().std()
        spread_volatility = market_data['spread'].pct_change().std()
        return (volume_volatility + spread_volatility) / 2
    
    def _calculate_concentration_risk(self, market_data: pd.DataFrame) -> float:
        """Calculate concentration risk based on volume distribution."""
        volume_ratio = market_data['volume'].max() / market_data['volume'].mean()
        return 1 / (1 + volume_ratio)
    
    def _calculate_network_health(self, network_data: pd.DataFrame) -> float:
        """Calculate network health score."""
        hashrate_stability = 1 / (1 + network_data['hashrate'].pct_change().std())
        difficulty_stability = 1 / (1 + network_data['difficulty'].pct_change().std())
        return (hashrate_stability + difficulty_stability) / 2
    
    def _calculate_security_risk(self, network_data: pd.DataFrame) -> float:
        """Calculate security risk based on hashrate distribution."""
        hashrate_gini = self._calculate_gini_coefficient(network_data['hashrate'])
        return hashrate_gini
    
    def _calculate_strategy_risk(self, agent_data: Dict[str, Any]) -> float:
        """Calculate strategy risk based on position changes."""
        position_changes = np.diff(agent_data['position_history'])
        return np.std(position_changes) / np.mean(np.abs(position_changes))
    
    def _calculate_execution_risk(self, agent_data: Dict[str, Any]) -> float:
        """Calculate execution risk based on slippage and fill rates."""
        slippage = agent_data.get('slippage', 0)
        fill_rate = agent_data.get('fill_rate', 1)
        return (slippage + (1 - fill_rate)) / 2
    
    def _calculate_market_impact(self, market_data: pd.DataFrame, network_data: pd.DataFrame) -> float:
        """Calculate market impact of network changes."""
        market_returns = market_data['price'].pct_change()
        network_returns = network_data['hashrate'].pct_change()
        return abs(market_returns.corr(network_returns))
    
    def _calculate_network_impact(self, market_data: pd.DataFrame, network_data: pd.DataFrame) -> float:
        """Calculate network impact of market changes."""
        market_volatility = market_data['price'].pct_change().std()
        network_volatility = network_data['hashrate'].pct_change().std()
        return market_volatility / network_volatility
    
    def _calculate_systemic_risk(self, market_data: pd.DataFrame, network_data: pd.DataFrame) -> float:
        """Calculate systemic risk score."""
        market_risk = self.analyze_market_risk(market_data)
        network_risk = self.analyze_network_risk(network_data)
        
        # Combine risk metrics
        systemic_risk = (
            market_risk.volatility +
            network_risk.volatility +
            market_risk.concentration_risk +
            network_risk.concentration_risk
        ) / 4
        
        return systemic_risk
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate a comprehensive risk report.
        
        Returns:
            Dictionary containing risk metrics and analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'confidence_level': self.confidence_level,
            'risk_metrics': {
                risk_type.value: {
                    'value_at_risk': metrics.value_at_risk,
                    'expected_shortfall': metrics.expected_shortfall,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'correlation': metrics.correlation,
                    'beta': metrics.beta,
                    'gini_coefficient': metrics.gini_coefficient,
                    'concentration_risk': metrics.concentration_risk,
                    'liquidity_risk': metrics.liquidity_risk
                }
                for risk_type, metrics in self.risk_metrics.items()
            }
        }
        
        return report 