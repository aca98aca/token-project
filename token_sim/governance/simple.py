import uuid
from typing import Dict, List, Any
from . import Governance

class SimpleGovernance(Governance):
    """Simple governance mechanism implementation."""
    
    def __init__(self,
                 voting_period: int = 30,  # 30 blocks
                 quorum_threshold: float = 0.4,  # 40% participation required
                 approval_threshold: float = 0.6):  # 60% approval required
        self.voting_period = voting_period
        self.quorum_threshold = quorum_threshold
        self.approval_threshold = approval_threshold
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.active_proposals: List[str] = []
        self.passed_proposals: List[str] = []
        self.current_block = 0
        self.total_voting_power = 100  # Default total voting power
        
    def submit_proposal(self, proposal: Dict[str, Any]) -> str:
        """Submit a new governance proposal."""
        proposal_id = str(uuid.uuid4())
        self.proposals[proposal_id] = {
            'details': proposal,
            'votes': {},
            'status': 'active',
            'start_block': self.current_block,
            'end_block': self.current_block + self.voting_period,
            'total_votes': 0,
            'yes_votes': 0,
            'no_votes': 0
        }
        self.active_proposals.append(proposal_id)
        return proposal_id
    
    def vote(self, proposal_id: str, voter_id: str, vote: bool) -> None:
        """Cast a vote on a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
            
        proposal = self.proposals[proposal_id]
        
        if proposal['status'] != 'active':
            raise ValueError(f"Proposal {proposal_id} is not active")
            
        if self.current_block > proposal['end_block']:
            raise ValueError(f"Voting period for proposal {proposal_id} has ended")
            
        # Record vote
        if voter_id not in proposal['votes']:
            proposal['total_votes'] += 1
            if vote:
                proposal['yes_votes'] += 1
            else:
                proposal['no_votes'] += 1
        elif proposal['votes'][voter_id] != vote:
            # Change vote
            if vote:
                proposal['yes_votes'] += 1
                proposal['no_votes'] -= 1
            else:
                proposal['yes_votes'] -= 1
                proposal['no_votes'] += 1
                
        proposal['votes'][voter_id] = vote
    
    def get_proposal_state(self, proposal_id: str) -> Dict[str, Any]:
        """Get the current state of a proposal."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
            
        proposal = self.proposals[proposal_id]
        total_eligible_votes = 100  # TODO: Get from token holders
        
        # Calculate metrics
        participation_rate = proposal['total_votes'] / total_eligible_votes
        approval_rate = (proposal['yes_votes'] / proposal['total_votes'] 
                        if proposal['total_votes'] > 0 else 0)
        
        return {
            'id': proposal_id,
            'status': proposal['status'],
            'total_votes': proposal['total_votes'],
            'yes_votes': proposal['yes_votes'],
            'no_votes': proposal['no_votes'],
            'participation_rate': participation_rate,
            'approval_rate': approval_rate,
            'blocks_remaining': max(0, proposal['end_block'] - self.current_block)
        }
    
    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get list of currently active proposals."""
        return [
            {
                'id': pid,
                **self.get_proposal_state(pid)
            }
            for pid, p in self.proposals.items()
            if p['status'] == 'active'
        ]
    
    def step(self) -> None:
        """Process one block of governance activity."""
        self.current_block += 1
        
        # Check for proposals that need to be finalized
        for proposal_id, proposal in self.proposals.items():
            if (proposal['status'] == 'active' and 
                self.current_block > proposal['end_block']):
                self._finalize_proposal(proposal_id)
    
    def _finalize_proposal(self, proposal_id: str) -> None:
        """Finalize a proposal after voting period ends."""
        proposal = self.proposals[proposal_id]
        
        # Calculate final metrics
        participation_rate = proposal['total_votes'] / self.total_voting_power
        approval_rate = (proposal['yes_votes'] / proposal['total_votes'] 
                        if proposal['total_votes'] > 0 else 0)
        
        # Check if proposal passed
        if (participation_rate >= self.quorum_threshold and 
            approval_rate >= self.approval_threshold):
            proposal['status'] = 'passed'
            # Move from active to passed
            if proposal_id in self.active_proposals:
                self.active_proposals.remove(proposal_id)
            self.passed_proposals.append(proposal_id)
        else:
            proposal['status'] = 'rejected'
            # Remove from active
            if proposal_id in self.active_proposals:
                self.active_proposals.remove(proposal_id)
    
    def force_proposal_pass(self, proposal_id: str) -> None:
        """Force a proposal to pass (for testing purposes)."""
        if proposal_id not in self.proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        proposal = self.proposals[proposal_id]
        
        # Set votes to pass threshold
        total_votes_needed = self.total_voting_power * self.quorum_threshold
        approval_votes_needed = total_votes_needed * self.approval_threshold
        
        # Set sufficient votes to pass
        proposal['yes_votes'] = approval_votes_needed
        proposal['no_votes'] = 0
        proposal['total_votes'] = proposal['yes_votes']
        proposal['status'] = 'passed'
        
        # Update proposal in the active list
        self.proposals[proposal_id] = proposal
        
        # Move from active to passed
        if proposal_id in self.active_proposals:
            self.active_proposals.remove(proposal_id)
        self.passed_proposals.append(proposal_id)
    
    def reset(self) -> None:
        """Reset governance state."""
        self.proposals = {}
        self.active_proposals = []
        self.passed_proposals = []
        self.current_block = 0 