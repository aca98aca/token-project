import random
from typing import Dict, List, Tuple
from . import ConsensusMechanism

class DelegatedProofOfStake(ConsensusMechanism):
    """Delegated Proof of Stake consensus mechanism implementation."""
    
    def __init__(self, 
                 block_reward: float = 1.0,
                 min_stake: float = 100.0,
                 num_delegates: int = 21,
                 staking_apy: float = 0.05):  # 5% APY
        self.block_reward = block_reward
        self.min_stake = min_stake
        self.num_delegates = num_delegates
        self.staking_apy = staking_apy
        self.participants: Dict[str, Dict] = {}
        self.delegates: List[str] = []
        self.total_stake = 0.0
        self.current_block = 0  # Add block counter
        
    def initialize_participants(self, num_participants: int, initial_stake: float = 1000.0) -> None:
        """Initialize validators and delegators."""
        # Create validators (potential delegates)
        for i in range(num_participants):
            participant_id = f"validator_{i}"
            # Random stake between min_stake and 10x min_stake
            stake = random.uniform(self.min_stake, self.min_stake * 10)
            self.participants[participant_id] = {
                'stake': stake,
                'rewards': 0.0,
                'blocks_produced': 0,
                'active': True,
                'stake_rewards': 0.0,
                'delegated_stake': 0.0,
                'is_delegate': False
            }
            self.total_stake += stake
        
        # Select top validators as delegates
        self._select_delegates()
    
    def perform_consensus_step(self) -> Tuple[float, Dict[str, float]]:
        """Simulate one block production step."""
        rewards_distribution = {}
        total_rewards = 0.0
        
        # Only delegates can produce blocks
        for delegate_id in self.delegates:
            if not self.participants[delegate_id]['active']:
                continue
                
            # Each delegate gets equal chance to produce a block
            if random.random() < 1.0 / self.num_delegates:
                # Block reward
                reward = self.block_reward
                rewards_distribution[delegate_id] = reward
                total_rewards += reward
                self.participants[delegate_id]['rewards'] += reward
                self.participants[delegate_id]['blocks_produced'] += 1
                self.current_block += 1  # Increment block counter
        
        # Calculate and distribute staking rewards
        for participant_id, stats in self.participants.items():
            if not stats['active']:
                continue
                
            # Calculate staking rewards for own stake
            own_stake_reward = stats['stake'] * (self.staking_apy / 365)
            stats['stake_rewards'] += own_stake_reward
            stats['stake'] += own_stake_reward
            
            # Calculate rewards for delegated stake
            if stats['is_delegate']:
                delegate_reward = stats['delegated_stake'] * (self.staking_apy / 365)
                stats['rewards'] += delegate_reward
            
            self.total_stake += own_stake_reward
            
        return total_rewards, rewards_distribution
    
    def get_rewards_distribution(self) -> Dict[str, float]:
        """Get current rewards distribution."""
        return {participant_id: stats['rewards'] + stats['stake_rewards']
                for participant_id, stats in self.participants.items()}
    
    def get_active_participants(self) -> List[str]:
        """Get list of active participants."""
        return [participant_id for participant_id, stats in self.participants.items() 
                if stats['active']]
    
    def get_participant_stats(self, participant_id: str) -> Dict:
        """Get current statistics for a specific participant."""
        if participant_id not in self.participants:
            raise ValueError(f"Unknown participant: {participant_id}")
        return self.participants[participant_id].copy()
    
    def delegate_stake(self, delegator_id: str, delegate_id: str, amount: float) -> None:
        """Delegate stake to a validator."""
        if delegator_id not in self.participants or delegate_id not in self.participants:
            raise ValueError("Unknown participant")
            
        if not self.participants[delegate_id]['is_delegate']:
            raise ValueError("Target is not a delegate")
            
        if amount > self.participants[delegator_id]['stake']:
            raise ValueError("Insufficient stake")
            
        self.participants[delegator_id]['stake'] -= amount
        self.participants[delegate_id]['delegated_stake'] += amount
    
    def _select_delegates(self) -> None:
        """Select top validators as delegates based on total stake."""
        # Sort participants by total stake (own stake + delegated stake)
        sorted_participants = sorted(
            self.participants.items(),
            key=lambda x: x[1]['stake'] + x[1]['delegated_stake'],
            reverse=True
        )
        
        # Select top validators as delegates
        self.delegates = [p[0] for p in sorted_participants[:self.num_delegates]]
        for delegate_id in self.delegates:
            self.participants[delegate_id]['is_delegate'] = True 
    
    @property
    def current_height(self) -> int:
        """Get current block height."""
        return self.current_block 