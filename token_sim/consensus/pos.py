import random
from typing import Dict, List, Tuple
from . import ConsensusMechanism

class ProofOfStake(ConsensusMechanism):
    """Proof of Stake consensus mechanism implementation."""
    
    def __init__(self, 
                 block_reward: float = 1.0,
                 min_stake: float = 100.0,
                 staking_apy: float = 0.05):  # 5% APY
        self.block_reward = block_reward
        self.min_stake = min_stake
        self.staking_apy = staking_apy
        self.participants: Dict[str, Dict] = {}
        self.total_stake = 0.0
        
    def initialize_participants(self, num_participants: int, initial_stake: float = 1000.0) -> None:
        """Initialize validators with random stake amounts."""
        for i in range(num_participants):
            validator_id = f"validator_{i}"
            # Random stake between min_stake and 10x min_stake
            stake = random.uniform(self.min_stake, self.min_stake * 10)
            self.participants[validator_id] = {
                'stake': stake,
                'rewards': 0.0,
                'blocks_produced': 0,
                'active': True,
                'stake_rewards': 0.0  # Rewards from staking
            }
            self.total_stake += stake
    
    def perform_consensus_step(self) -> Tuple[float, Dict[str, float]]:
        """Simulate one block production step."""
        rewards_distribution = {}
        total_rewards = 0.0
        
        # Calculate probability of producing block for each validator
        for validator_id, stats in self.participants.items():
            if not stats['active'] or stats['stake'] < self.min_stake:
                continue
                
            # Probability proportional to stake
            probability = stats['stake'] / self.total_stake
            
            # Simulate block production
            if random.random() < probability:
                # Block reward
                reward = self.block_reward
                rewards_distribution[validator_id] = reward
                total_rewards += reward
                stats['rewards'] += reward
                stats['blocks_produced'] += 1
            
            # Calculate and distribute staking rewards
            staking_reward = stats['stake'] * (self.staking_apy / 365)  # Daily staking reward
            stats['stake_rewards'] += staking_reward
            stats['stake'] += staking_reward
            self.total_stake += staking_reward
            
        return total_rewards, rewards_distribution
    
    def get_rewards_distribution(self) -> Dict[str, float]:
        """Get current rewards distribution."""
        return {validator_id: stats['rewards'] + stats['stake_rewards']
                for validator_id, stats in self.participants.items()}
    
    def get_active_participants(self) -> List[str]:
        """Get list of active validators."""
        return [validator_id for validator_id, stats in self.participants.items() 
                if stats['active'] and stats['stake'] >= self.min_stake]
    
    def get_participant_stats(self, participant_id: str) -> Dict:
        """Get current statistics for a specific validator."""
        if participant_id not in self.participants:
            raise ValueError(f"Unknown participant: {participant_id}")
        return self.participants[participant_id].copy()
    
    def update_stake(self, validator_id: str, stake_change: float) -> None:
        """Update a validator's stake amount."""
        if validator_id not in self.participants:
            raise ValueError(f"Unknown validator: {validator_id}")
            
        stats = self.participants[validator_id]
        new_stake = stats['stake'] + stake_change
        
        if new_stake < 0:
            raise ValueError("Stake cannot be negative")
            
        self.total_stake = self.total_stake - stats['stake'] + new_stake
        stats['stake'] = new_stake 