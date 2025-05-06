from abc import ABC, abstractmethod
from typing import Dict, List, Any

class Governance(ABC):
    """Base class for governance mechanisms in the simulation."""
    
    @abstractmethod
    def submit_proposal(self, proposal: Dict[str, Any]) -> str:
        """Submit a new governance proposal.
        
        Args:
            proposal: Dictionary containing proposal details
            
        Returns:
            str: Unique proposal ID
        """
        pass
    
    @abstractmethod
    def vote(self, proposal_id: str, voter_id: str, vote: bool) -> None:
        """Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of the proposal to vote on
            voter_id: ID of the voter
            vote: True for yes, False for no
        """
        pass
    
    @abstractmethod
    def get_proposal_state(self, proposal_id: str) -> Dict[str, Any]:
        """Get the current state of a proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            Dict containing proposal state
        """
        pass
    
    @abstractmethod
    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get list of currently active proposals.
        
        Returns:
            List of proposal dictionaries
        """
        pass 