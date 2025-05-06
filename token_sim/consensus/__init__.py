from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class ConsensusMechanism(ABC):
    """Base class for all consensus mechanisms in the simulation."""
    
    @abstractmethod
    def initialize_participants(self, num_participants: int, initial_stake: float = 0) -> None:
        """Initialize the consensus participants with their initial states."""
        pass
    
    @abstractmethod
    def perform_consensus_step(self) -> Tuple[float, Dict[str, float]]:
        """Perform one step of the consensus mechanism.
        
        Returns:
            Tuple[float, Dict[str, float]]: (total_rewards, rewards_distribution)
        """
        pass
    
    @abstractmethod
    def get_rewards_distribution(self) -> Dict[str, float]:
        """Get the current rewards distribution across participants."""
        pass
    
    @abstractmethod
    def get_active_participants(self) -> List[str]:
        """Get list of currently active participants."""
        pass
    
    @abstractmethod
    def get_participant_stats(self, participant_id: str) -> Dict:
        """Get current statistics for a specific participant."""
        pass 