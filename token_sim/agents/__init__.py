from abc import ABC, abstractmethod
from typing import Dict, Any

class Agent(ABC):
    """Base class for all agents in the simulation."""
    
    @abstractmethod
    def initialize(self, initial_balance: float = 0.0) -> None:
        """Initialize the agent with its initial state."""
        pass
    
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an action based on the current state.
        
        Args:
            state: Current state of the simulation
            
        Returns:
            Dict containing the agent's actions
        """
        pass
    
    @abstractmethod
    def update(self, reward: float, new_state: Dict[str, Any]) -> None:
        """Update agent's state based on received reward and new state.
        
        Args:
            reward: Reward received from the last action
            new_state: New state of the simulation
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        pass

    def get_performance_metric(self) -> float:
        """Get a performance metric for the agent."""
        # Default implementation returns total profit
        return self.state.get('total_profit', 0.0) 