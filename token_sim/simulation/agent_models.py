from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
from datetime import datetime

class AgentType(Enum):
    """Types of agents in the simulation."""
    TRADER = "trader"
    MINER = "miner"
    VALIDATOR = "validator"
    HOLDER = "holder"
    MARKET_MAKER = "market_maker"

class TradingStrategy(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"

@dataclass
class AgentState:
    """Current state of an agent."""
    balance: float  # Token balance
    fiat_balance: float  # Fiat currency balance
    position: float  # Net position (positive = long, negative = short)
    last_action_time: datetime
    strategy: TradingStrategy
    risk_tolerance: float  # 0 to 1, where 1 is most risk-tolerant
    performance_metrics: Dict[str, float]

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, initial_balance: float):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent
            initial_balance: Initial token balance
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(
            balance=initial_balance,
            fiat_balance=0.0,
            position=0.0,
            last_action_time=datetime.now(),
            strategy=TradingStrategy.MOMENTUM,
            risk_tolerance=0.5,
            performance_metrics={}
        )
    
    def update_state(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> None:
        """Update agent state based on market and network data."""
        raise NotImplementedError
    
    def decide_action(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on next action based on current state and market conditions."""
        raise NotImplementedError
    
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute the decided action."""
        raise NotImplementedError
    
    def update_metrics(self) -> None:
        """Update performance metrics."""
        raise NotImplementedError

class TraderAgent(BaseAgent):
    """Agent that implements trading strategies."""
    
    def __init__(self, agent_id: str, strategy: TradingStrategy, initial_balance: float):
        """Initialize the trader agent.
        
        Args:
            agent_id: Unique identifier for the agent
            strategy: Trading strategy to use
            initial_balance: Initial token balance
        """
        super().__init__(agent_id, AgentType.TRADER, initial_balance)
        self.state.strategy = strategy
        self.position_history = []
        self.trade_history = []
    
    def update_state(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> None:
        """Update trader state based on market data."""
        # Update position history
        self.position_history.append(self.state.position)
        
        # Update performance metrics
        if len(self.position_history) > 1:
            price_change = market_data['price'] / market_data['prev_price'] - 1
            position_pnl = self.state.position * price_change
            self.state.performance_metrics['pnl'] = position_pnl
    
    def decide_action(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decide trading action based on strategy."""
        action = {'type': 'no_action', 'quantity': 0}
        
        if self.state.strategy == TradingStrategy.MOMENTUM:
            action = self._momentum_strategy(market_data)
        elif self.state.strategy == TradingStrategy.MEAN_REVERSION:
            action = self._mean_reversion_strategy(market_data)
        elif self.state.strategy == TradingStrategy.TREND_FOLLOWING:
            action = self._trend_following_strategy(market_data)
        elif self.state.strategy == TradingStrategy.ARBITRAGE:
            action = self._arbitrage_strategy(market_data)
        
        return action
    
    def _momentum_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement momentum trading strategy."""
        returns = market_data['returns']
        volatility = market_data['volatility']
        
        # Calculate position size based on momentum and risk tolerance
        position_size = returns * self.state.risk_tolerance / volatility
        
        return {
            'type': 'trade',
            'quantity': position_size,
            'price': market_data['price']
        }
    
    def _mean_reversion_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement mean reversion trading strategy."""
        price = market_data['price']
        ma = market_data['moving_average']
        std = market_data['price_std']
        
        # Calculate position size based on deviation from mean
        deviation = (price - ma) / std
        position_size = -deviation * self.state.risk_tolerance
        
        return {
            'type': 'trade',
            'quantity': position_size,
            'price': price
        }
    
    def _trend_following_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement trend following trading strategy."""
        trend = market_data['trend']
        strength = market_data['trend_strength']
        
        # Calculate position size based on trend strength
        position_size = trend * strength * self.state.risk_tolerance
        
        return {
            'type': 'trade',
            'quantity': position_size,
            'price': market_data['price']
        }
    
    def _arbitrage_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement arbitrage trading strategy."""
        price_diffs = market_data['price_differences']
        
        # Find best arbitrage opportunity
        best_opp = max(price_diffs.items(), key=lambda x: abs(x[1]))
        
        return {
            'type': 'arbitrage',
            'quantity': self.state.risk_tolerance,
            'price': best_opp[1]
        }
    
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute trading action."""
        if action['type'] == 'trade':
            self.state.position += action['quantity']
            self.trade_history.append(action)
        elif action['type'] == 'arbitrage':
            # Implement arbitrage execution logic
            pass

class MinerAgent(BaseAgent):
    """Agent that implements mining behavior."""
    
    def __init__(self, agent_id: str, hashrate: float, initial_balance: float):
        """Initialize the miner agent.
        
        Args:
            agent_id: Unique identifier for the agent
            hashrate: Mining hashrate
            initial_balance: Initial token balance
        """
        super().__init__(agent_id, AgentType.MINER, initial_balance)
        self.hashrate = hashrate
        self.rewards_history = []
    
    def update_state(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> None:
        """Update miner state based on market and network data."""
        # Update rewards history
        if 'block_reward' in network_data:
            self.rewards_history.append(network_data['block_reward'])
        
        # Update performance metrics
        self.state.performance_metrics['mining_rewards'] = sum(self.rewards_history)
        self.state.performance_metrics['hashrate'] = self.hashrate
    
    def decide_action(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decide mining action based on profitability."""
        electricity_cost = network_data['electricity_cost']
        token_price = market_data['price']
        
        # Calculate mining profitability
        daily_revenue = self.hashrate * network_data['block_reward'] * token_price
        daily_cost = self.hashrate * electricity_cost
        
        if daily_revenue > daily_cost:
            return {'type': 'mine', 'hashrate': self.hashrate}
        else:
            return {'type': 'stop_mining'}
    
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute mining action."""
        if action['type'] == 'mine':
            # Implement mining logic
            pass
        elif action['type'] == 'stop_mining':
            self.hashrate = 0

class ValidatorAgent(BaseAgent):
    """Agent that implements staking and validation behavior."""
    
    def __init__(self, agent_id: str, stake: float, initial_balance: float):
        """Initialize the validator agent.
        
        Args:
            agent_id: Unique identifier for the agent
            stake: Amount of tokens staked
            initial_balance: Initial token balance
        """
        super().__init__(agent_id, AgentType.VALIDATOR, initial_balance)
        self.stake = stake
        self.rewards_history = []
    
    def update_state(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> None:
        """Update validator state based on market and network data."""
        # Update rewards history
        if 'staking_reward' in network_data:
            self.rewards_history.append(network_data['staking_reward'])
        
        # Update performance metrics
        self.state.performance_metrics['staking_rewards'] = sum(self.rewards_history)
        self.state.performance_metrics['stake'] = self.stake
    
    def decide_action(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decide staking action based on rewards and risks."""
        staking_apr = network_data['staking_apr']
        token_price = market_data['price']
        
        # Calculate staking profitability
        annual_revenue = self.stake * staking_apr * token_price
        
        if annual_revenue > 0:
            return {'type': 'stake', 'amount': self.stake}
        else:
            return {'type': 'unstake', 'amount': self.stake}
    
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute staking action."""
        if action['type'] == 'stake':
            # Implement staking logic
            pass
        elif action['type'] == 'unstake':
            self.stake = 0

class MarketMakerAgent(BaseAgent):
    """Agent that implements market making behavior."""
    
    def __init__(self, agent_id: str, initial_balance: float):
        """Initialize the market maker agent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_balance: Initial token balance
        """
        super().__init__(agent_id, AgentType.MARKET_MAKER, initial_balance)
        self.state.strategy = TradingStrategy.MARKET_MAKING
        self.spread_history = []
    
    def update_state(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> None:
        """Update market maker state based on market data."""
        # Update spread history
        if 'spread' in market_data:
            self.spread_history.append(market_data['spread'])
        
        # Update performance metrics
        self.state.performance_metrics['avg_spread'] = np.mean(self.spread_history)
    
    def decide_action(self, market_data: Dict[str, Any], network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decide market making action based on market conditions."""
        volatility = market_data['volatility']
        volume = market_data['volume']
        
        # Calculate optimal spread based on volatility and volume
        spread = volatility * (1 + 1/volume) * self.state.risk_tolerance
        
        return {
            'type': 'update_quotes',
            'bid_price': market_data['price'] * (1 - spread/2),
            'ask_price': market_data['price'] * (1 + spread/2),
            'quantity': volume * 0.1  # Provide 10% of current volume
        }
    
    def execute_action(self, action: Dict[str, Any]) -> None:
        """Execute market making action."""
        if action['type'] == 'update_quotes':
            # Implement quote update logic
            pass 