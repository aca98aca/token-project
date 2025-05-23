from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ConsensusType(Enum):
    """Types of consensus mechanisms."""
    POW = "PoW"
    POS = "PoS"
    DPOS = "DPoS"

@dataclass
class NetworkParticipant:
    """Represents a network participant (miner/validator)."""
    id: str
    type: str  # 'miner', 'validator', 'delegate'
    stake: float
    hashrate: float
    voting_power: float
    rewards: float
    uptime: float
    last_active: int

class EnhancedNetworkModel:
    """Enhanced network model with sophisticated consensus dynamics."""
    
    def __init__(self, consensus_type: ConsensusType, initial_params: Dict[str, Any]):
        """Initialize the network model.
        
        Args:
            consensus_type: Type of consensus mechanism
            initial_params: Initial network parameters
        """
        self.consensus_type = consensus_type
        self.participants = {}
        self.block_height = 0
        self.total_stake = 0.0
        self.total_hashrate = 0.0
        self.total_voting_power = 0.0
        
        # Network parameters
        self.block_time = initial_params.get('block_time', 600)  # 10 minutes
        self.difficulty = initial_params.get('difficulty', 1000000)
        self.min_stake = initial_params.get('min_stake', 1000.0)
        self.num_delegates = initial_params.get('num_delegates', 21)
        
        # Initialize participants based on consensus type
        self._initialize_participants(initial_params)
        
        # History tracking
        self.hashrate_history = [self.total_hashrate]
        self.stake_history = [self.total_stake]
        self.block_time_history = [self.block_time]
        self.participant_count_history = [len(self.participants)]
    
    def _initialize_participants(self, params: Dict[str, Any]):
        """Initialize network participants based on consensus type."""
        if self.consensus_type == ConsensusType.POW:
            self._initialize_pow_participants(params)
        elif self.consensus_type == ConsensusType.POS:
            self._initialize_pos_participants(params)
        else:  # DPoS
            self._initialize_dpos_participants(params)
    
    def _initialize_pow_participants(self, params: Dict[str, Any]):
        """Initialize PoW miners."""
        num_miners = params.get('num_miners', 100)
        base_hashrate = params.get('base_hashrate', 1000.0)
        
        for i in range(num_miners):
            # Create miner with random hashrate
            hashrate = base_hashrate * np.random.lognormal(mean=0, sigma=0.5)
            miner = NetworkParticipant(
                id=f"miner_{i}",
                type='miner',
                stake=0.0,
                hashrate=hashrate,
                voting_power=0.0,
                rewards=0.0,
                uptime=0.95 + np.random.random() * 0.05,  # 95-100% uptime
                last_active=0
            )
            self.participants[miner.id] = miner
            self.total_hashrate += hashrate
    
    def _initialize_pos_participants(self, params: Dict[str, Any]):
        """Initialize PoS validators."""
        num_validators = params.get('num_validators', 100)
        base_stake = params.get('base_stake', 10000.0)
        
        for i in range(num_validators):
            # Create validator with random stake
            stake = base_stake * np.random.lognormal(mean=0, sigma=0.5)
            validator = NetworkParticipant(
                id=f"validator_{i}",
                type='validator',
                stake=stake,
                hashrate=0.0,
                voting_power=stake,
                rewards=0.0,
                uptime=0.98 + np.random.random() * 0.02,  # 98-100% uptime
                last_active=0
            )
            self.participants[validator.id] = validator
            self.total_stake += stake
            self.total_voting_power += stake
    
    def _initialize_dpos_participants(self, params: Dict[str, Any]):
        """Initialize DPoS delegates and voters."""
        num_delegates = params.get('num_delegates', 21)
        num_voters = params.get('num_voters', 1000)
        base_stake = params.get('base_stake', 1000.0)
        
        # Create delegates
        for i in range(num_delegates):
            stake = base_stake * 10 * np.random.lognormal(mean=0, sigma=0.3)
            delegate = NetworkParticipant(
                id=f"delegate_{i}",
                type='delegate',
                stake=stake,
                hashrate=0.0,
                voting_power=stake,
                rewards=0.0,
                uptime=0.99 + np.random.random() * 0.01,  # 99-100% uptime
                last_active=0
            )
            self.participants[delegate.id] = delegate
            self.total_stake += stake
            self.total_voting_power += stake
        
        # Create voters
        for i in range(num_voters):
            stake = base_stake * np.random.lognormal(mean=0, sigma=0.5)
            voter = NetworkParticipant(
                id=f"voter_{i}",
                type='voter',
                stake=stake,
                hashrate=0.0,
                voting_power=stake,
                rewards=0.0,
                uptime=0.9 + np.random.random() * 0.1,  # 90-100% uptime
                last_active=0
            )
            self.participants[voter.id] = voter
            self.total_stake += stake
    
    def update(self, time_step: int, market_conditions: Dict[str, Any]):
        """Update the network state.
        
        Args:
            time_step: Current time step
            market_conditions: Current market conditions
        """
        # Update participant states
        self._update_participants(time_step, market_conditions)
        
        # Update consensus-specific metrics
        if self.consensus_type == ConsensusType.POW:
            self._update_pow_metrics(time_step)
        elif self.consensus_type == ConsensusType.POS:
            self._update_pos_metrics(time_step)
        else:  # DPoS
            self._update_dpos_metrics(time_step)
        
        # Update history
        self.hashrate_history.append(self.total_hashrate)
        self.stake_history.append(self.total_stake)
        self.block_time_history.append(self.block_time)
        self.participant_count_history.append(len(self.participants))
    
    def _update_participants(self, time_step: int, market_conditions: Dict[str, Any]):
        """Update all network participants."""
        for participant in self.participants.values():
            # Update uptime
            if np.random.random() > participant.uptime:
                participant.last_active = time_step
                continue
            
            # Update rewards
            if self.consensus_type == ConsensusType.POW:
                # PoW rewards based on hashrate
                reward = self._calculate_pow_reward(participant)
            else:
                # PoS/DPoS rewards based on stake
                reward = self._calculate_stake_reward(participant)
            
            participant.rewards += reward
    
    def _update_pow_metrics(self, time_step: int):
        """Update PoW-specific metrics."""
        # Adjust difficulty based on block time
        target_blocks = time_step / self.block_time
        actual_blocks = self.block_height
        
        if actual_blocks > 0:
            difficulty_adjustment = (target_blocks / actual_blocks) ** 0.25
            self.difficulty *= difficulty_adjustment
        
        # Update block time
        self.block_time = self.difficulty / self.total_hashrate
    
    def _update_pos_metrics(self, time_step: int):
        """Update PoS-specific metrics."""
        # Update total stake
        self.total_stake = sum(p.stake for p in self.participants.values())
        
        # Update voting power
        self.total_voting_power = sum(p.voting_power for p in self.participants.values())
        
        # Update block time based on stake
        self.block_time = 600 * (1 - self.total_stake / (self.total_stake * 2))  # 10 minutes base
    
    def _update_dpos_metrics(self, time_step: int):
        """Update DPoS-specific metrics."""
        # Update delegate rankings
        delegates = [p for p in self.participants.values() if p.type == 'delegate']
        delegates.sort(key=lambda x: x.voting_power, reverse=True)
        
        # Update active delegates
        active_delegates = delegates[:self.num_delegates]
        
        # Update block time based on number of active delegates
        self.block_time = 600 / len(active_delegates)  # 10 minutes / num_delegates
    
    def _calculate_pow_reward(self, participant: NetworkParticipant) -> float:
        """Calculate PoW mining reward."""
        if participant.type != 'miner':
            return 0.0
        
        # Reward based on hashrate share
        hashrate_share = participant.hashrate / self.total_hashrate
        block_reward = 6.25  # Current Bitcoin block reward
        
        return block_reward * hashrate_share
    
    def _calculate_stake_reward(self, participant: NetworkParticipant) -> float:
        """Calculate staking reward."""
        if participant.type not in ['validator', 'delegate']:
            return 0.0
        
        # Reward based on stake share
        stake_share = participant.stake / self.total_stake
        annual_reward_rate = 0.1  # 10% APY
        
        return stake_share * annual_reward_rate / 365  # Daily reward
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get current network metrics."""
        metrics = {
            'consensus_type': self.consensus_type.value,
            'block_height': self.block_height,
            'block_time': self.block_time,
            'total_hashrate': self.total_hashrate,
            'total_stake': self.total_stake,
            'total_voting_power': self.total_voting_power,
            'participant_count': len(self.participants),
            'hashrate_history': self.hashrate_history,
            'stake_history': self.stake_history,
            'block_time_history': self.block_time_history,
            'participant_count_history': self.participant_count_history
        }
        
        # Add consensus-specific metrics
        if self.consensus_type == ConsensusType.POW:
            metrics.update({
                'difficulty': self.difficulty,
                'miner_distribution': self._get_miner_distribution()
            })
        elif self.consensus_type == ConsensusType.POS:
            metrics.update({
                'staking_ratio': self.total_stake / (self.total_stake * 2),
                'validator_distribution': self._get_validator_distribution()
            })
        else:  # DPoS
            metrics.update({
                'delegate_distribution': self._get_delegate_distribution(),
                'voting_power_distribution': self._get_voting_power_distribution()
            })
        
        return metrics
    
    def _get_miner_distribution(self) -> Dict[str, float]:
        """Get mining power distribution metrics."""
        miners = [p for p in self.participants.values() if p.type == 'miner']
        hashrates = [m.hashrate for m in miners]
        
        return {
            'gini_coefficient': self._calculate_gini(hashrates),
            'top_10_percent': np.sum(np.sort(hashrates)[-len(hashrates)//10:]) / self.total_hashrate
        }
    
    def _get_validator_distribution(self) -> Dict[str, float]:
        """Get validator stake distribution metrics."""
        validators = [p for p in self.participants.values() if p.type == 'validator']
        stakes = [v.stake for v in validators]
        
        return {
            'gini_coefficient': self._calculate_gini(stakes),
            'top_10_percent': np.sum(np.sort(stakes)[-len(stakes)//10:]) / self.total_stake
        }
    
    def _get_delegate_distribution(self) -> Dict[str, float]:
        """Get delegate voting power distribution metrics."""
        delegates = [p for p in self.participants.values() if p.type == 'delegate']
        voting_powers = [d.voting_power for d in delegates]
        
        return {
            'gini_coefficient': self._calculate_gini(voting_powers),
            'top_10_percent': np.sum(np.sort(voting_powers)[-len(voting_powers)//10:]) / self.total_voting_power
        }
    
    def _get_voting_power_distribution(self) -> Dict[str, float]:
        """Get overall voting power distribution metrics."""
        voting_powers = [p.voting_power for p in self.participants.values()]
        
        return {
            'gini_coefficient': self._calculate_gini(voting_powers),
            'participation_rate': len([p for p in self.participants.values() if p.voting_power > 0]) / len(self.participants)
        }
    
    def _calculate_gini(self, x: List[float]) -> float:
        """Calculate Gini coefficient for distribution."""
        # Sort values
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x)
        # Calculate Gini coefficient
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n 