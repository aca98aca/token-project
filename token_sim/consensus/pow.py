import random
from typing import Dict, List, Tuple
from . import ConsensusMechanism

class ProofOfWork(ConsensusMechanism):
    """Proof of Work consensus mechanism implementation."""
    
    def __init__(self, 
                 block_reward: float = 1.0,
                 difficulty_adjustment_blocks: int = 2016,
                 target_block_time: float = 600.0):  # 10 minutes in seconds
        self.block_reward = block_reward
        self.difficulty_adjustment_blocks = difficulty_adjustment_blocks
        self.target_block_time = target_block_time
        self.participants: Dict[str, Dict] = {}
        self.current_difficulty = 1.0
        self.blocks_since_adjustment = 0
        self.total_hashrate = 0.0
        self.initial_difficulty = 1.0
        self.current_block = 0
        self.total_rewards = 0.0
        self.rewards_distribution = {}
        self.last_block_time = 0.0
        self.block_times = []
        self.network_security_score = 0.0
        self.target_hashrate = 1000.0  # Initial target hashrate
        self.total_storage = 0.0  # Storage capacity of the network
        
    def initialize_participants(self, num_participants: int, initial_stake: float = 0) -> None:
        """Initialize miners with random hash rates."""
        for i in range(num_participants):
            miner_id = f"miner_{i}"
            # Random hash rate between 10 and 100 TH/s
            hashrate = random.uniform(10, 100)
            self.participants[miner_id] = {
                'hashrate': hashrate,
                'rewards': 0.0,
                'blocks_found': 0,
                'active': True,
                'last_reward_time': 0.0
            }
            self.total_hashrate += hashrate
            
        # Set initial target hashrate based on number of participants
        self.target_hashrate = self.total_hashrate * 1.5  # Target 50% more than initial
        
        # Calculate initial network security score
        active_miners = len([m for m in self.participants.values() if m['active']])
        if active_miners > 0:
            # Normalize components
            hashrate_score = min(self.total_hashrate / self.target_hashrate, 1.0)
            difficulty_score = min(self.current_difficulty / 100.0, 1.0)
            decentralization_score = min(active_miners / 25.0, 1.0)  # Target 25 active miners
            
            # Calculate weighted geometric mean
            self.network_security_score = (
                hashrate_score ** 0.4 *      # 40% weight on hashrate
                difficulty_score ** 0.3 *    # 30% weight on difficulty
                decentralization_score ** 0.3  # 30% weight on decentralization
            )
        else:
            self.network_security_score = 0.0
        
    def perform_consensus_step(self) -> Tuple[float, Dict[str, float]]:
        """Simulate one block mining step."""
        rewards_distribution = {}
        total_rewards = 0.0
        
        # If in recovery, process recovery first
        if hasattr(self, 'recovery_state'):
            if not self.is_healthy():
                self._advance_recovery()
                # During recovery, return no rewards
                return 0.0, {}
        
        # Update total hashrate from active miners
        self.total_hashrate = sum(stats['hashrate'] for stats in self.participants.values() 
                                if stats['active'])
        
        # Update total storage (1TB per 100 TH/s of hashrate as a rough approximation)
        self.total_storage = self.total_hashrate * 0.01  # TB
        
        if self.total_hashrate > 0 and self.current_difficulty != float('inf'):
            # Calculate probability of finding block for each miner
            for miner_id, stats in self.participants.items():
                if not stats['active'] or stats['hashrate'] <= 0:
                    continue
                    
                # Probability proportional to hash rate
                probability = stats['hashrate'] / self.total_hashrate
                
                # Simulate block finding with difficulty adjustment
                if random.random() < probability * (1.0 / self.current_difficulty):
                    reward = self.block_reward
                    rewards_distribution[miner_id] = reward
                    total_rewards += reward
                    stats['rewards'] += reward
                    stats['blocks_found'] += 1
                    stats['last_reward_time'] = self.current_block
                    self.current_block += 1
                    
                    # Record block time
                    current_time = self.current_block * self.target_block_time
                    if self.last_block_time > 0:
                        self.block_times.append(current_time - self.last_block_time)
                    self.last_block_time = current_time
        
        # Update difficulty if needed and not in recovery
        if not hasattr(self, 'recovery_state'):
            self.blocks_since_adjustment += 1
            if self.blocks_since_adjustment >= self.difficulty_adjustment_blocks:
                self._adjust_difficulty()
                self.blocks_since_adjustment = 0
            
        # Calculate network security score
        active_miners = len([m for m in self.participants.values() if m['active'] and m['hashrate'] > 0])
        if active_miners > 0:
            # Normalize components
            hashrate_score = min(self.total_hashrate / self.target_hashrate, 1.0)
            difficulty_score = min(self.current_difficulty / 100.0, 1.0) if self.current_difficulty != float('inf') else 0.0
            decentralization_score = min(active_miners / 25.0, 1.0)  # Target 25 active miners
            
            # Calculate weighted geometric mean
            self.network_security_score = (
                hashrate_score ** 0.4 *      # 40% weight on hashrate
                difficulty_score ** 0.3 *    # 30% weight on difficulty
                decentralization_score ** 0.3  # 30% weight on decentralization
            )
            
            # Update target hashrate based on current state
            self.target_hashrate = max(
                self.total_hashrate * 1.5,  # Target 50% more than current
                self.target_hashrate * 0.9  # Don't decrease too quickly
            )
        else:
            self.network_security_score = 0.0
            
        return total_rewards, rewards_distribution
    
    def get_rewards_distribution(self) -> Dict[str, float]:
        """Get current rewards distribution."""
        return {miner_id: stats['rewards'] 
                for miner_id, stats in self.participants.items()}
    
    def get_active_participants(self) -> List[str]:
        """Get list of active miners."""
        # During recovery or failure, consider all miners inactive
        if hasattr(self, 'recovery_state') and self.recovery_state['phase'] == 0:
            return []
            
        # Otherwise return active miners with positive hashrate
        return [miner_id for miner_id, stats in self.participants.items() 
                if stats['active'] and stats['hashrate'] > 0]
    
    def get_participant_stats(self, participant_id: str) -> Dict:
        """Get current statistics for a specific miner."""
        if participant_id not in self.participants:
            raise ValueError(f"Unknown participant: {participant_id}")
        return self.participants[participant_id].copy()
    
    def _adjust_difficulty(self) -> None:
        """Adjust mining difficulty based on network hashrate and block times."""
        if not self.block_times:
            return
            
        # Calculate average block time
        avg_block_time = sum(self.block_times) / len(self.block_times)
        
        # Adjust difficulty to target block time
        if avg_block_time > 0:
            adjustment_factor = self.target_block_time / avg_block_time
            self.current_difficulty *= adjustment_factor
            
            # Clamp difficulty to reasonable bounds
            self.current_difficulty = max(0.1, min(self.current_difficulty, 100.0))
        
        # Clear block times for next adjustment period
        self.block_times = []
    
    def reset(self):
        """Reset consensus state."""
        self.current_difficulty = self.initial_difficulty
        self.current_block = 0
        self.total_rewards = 0.0
        self.rewards_distribution = {}
        self.blocks_since_adjustment = 0
        self.total_hashrate = 0.0
        self.last_block_time = 0.0
        self.block_times = []
        
        # Reset participant stats
        for stats in self.participants.values():
            stats['rewards'] = 0.0
            stats['blocks_found'] = 0
            stats['active'] = True
            stats['last_reward_time'] = 0.0 
    
    def get_network_security_score(self) -> float:
        """Get the current network security score."""
        return self.network_security_score 
    
    def force_failure(self) -> None:
        """Force a consensus failure for testing purposes."""
        # Store previous state for recovery
        self._previous_state = {
            'difficulty': self.initial_difficulty,
            'block_reward': self.block_reward,
            'participants': {pid: stats.copy() for pid, stats in self.participants.items()},
            'total_hashrate': self.total_hashrate,
            'network_security_score': self.network_security_score,
            'block_times': self.block_times.copy() if self.block_times else [],
            'blocks_since_adjustment': self.blocks_since_adjustment
        }
        
        # Disable all participants and clear their state
        for stats in self.participants.values():
            stats['active'] = False
            stats['hashrate'] = 0.0  # Set hashrate to 0
            stats['last_reward_time'] = self.current_block  # Mark last reward time
        
        # Update network state
        self.total_hashrate = 0.0
        self.current_difficulty = float('inf')  # Make mining impossible
        self.block_reward = 0.0  # Remove rewards
        self.network_security_score = 0.0  # Network is insecure
        self.block_times = []  # Clear block times during failure
        
        # Initialize recovery state
        self.recovery_state = {
            'timer': 3,  # Recovery takes 3 steps
            'phase': 0,  # Current recovery phase
            'participants_restored': 0,  # Track restored participants
            'failure_block': self.current_block  # Record when failure occurred
        }

    def is_healthy(self) -> bool:
        """Check if consensus mechanism is healthy."""
        # Check for recovery in progress
        if hasattr(self, 'recovery_state'):
            if self.recovery_state['timer'] <= 0:
                # Recovery complete
                self._complete_recovery()
                return True
            else:
                self._advance_recovery()
                return False
        
        # Consensus is healthy if:
        # 1. There are active participants
        # 2. Network hashrate is above minimum threshold
        # 3. Block production is happening
        # 4. Difficulty is within reasonable bounds
        # 5. Block times are within acceptable range
        active_miners = len([p for p in self.participants.values() if p['active']])
        min_hashrate_threshold = self.target_hashrate * 0.1  # 10% of target
        max_difficulty = 100.0  # Maximum reasonable difficulty
        
        # Check block times if we have enough data
        block_time_healthy = True
        if len(self.block_times) >= 3:
            avg_block_time = sum(self.block_times[-3:]) / 3
            block_time_healthy = 0.5 * self.target_block_time <= avg_block_time <= 2 * self.target_block_time
        
        return (
            active_miners > 0 and
            self.total_hashrate > min_hashrate_threshold and
            self.current_block > 0 and
            self.current_difficulty <= max_difficulty and
            block_time_healthy
        )

    def _advance_recovery(self) -> None:
        """Advance the recovery process by one step."""
        self.recovery_state['timer'] -= 1
        
        # Phase 0: Start restoring participants
        if self.recovery_state['phase'] == 0:
            # Restore 1/3 of participants each step
            total_participants = len(self.participants)
            participants_to_restore = total_participants // 3
            
            for i in range(participants_to_restore):
                if self.recovery_state['participants_restored'] < total_participants:
                    participant_id = list(self.participants.keys())[self.recovery_state['participants_restored']]
                    # Restore participant with original hashrate
                    self.participants[participant_id]['active'] = True
                    self.participants[participant_id]['hashrate'] = self._previous_state['participants'][participant_id]['hashrate']
                    self.total_hashrate += self.participants[participant_id]['hashrate']
                    self.recovery_state['participants_restored'] += 1
            
            if self.recovery_state['participants_restored'] >= total_participants:
                self.recovery_state['phase'] = 1
        
        # Phase 1: Gradually restore difficulty and rewards
        elif self.recovery_state['phase'] == 1:
            # Calculate progress (0 to 1)
            progress = 1 - (self.recovery_state['timer'] / 3)
            
            # Gradually restore difficulty from infinity to original
            if self.current_difficulty == float('inf'):
                self.current_difficulty = self._previous_state['difficulty'] * 10  # Start with high but finite difficulty
            else:
                # Exponential decrease towards target
                self.current_difficulty = self._previous_state['difficulty'] + \
                    (self.current_difficulty - self._previous_state['difficulty']) * (1 - progress)
            
            # Gradually restore block reward
            self.block_reward = self._previous_state['block_reward'] * progress

    def _complete_recovery(self) -> None:
        """Complete the recovery process and clean up recovery state."""
        # Restore all participants to their original state
        self.total_hashrate = 0.0
        for participant_id, stats in self.participants.items():
            stats['active'] = True
            stats['hashrate'] = self._previous_state['participants'][participant_id]['hashrate']
            self.total_hashrate += stats['hashrate']
        
        # Restore difficulty and block reward exactly to previous values
        self.current_difficulty = self._previous_state['difficulty']
        self.block_reward = self._previous_state['block_reward']
        
        # Reset block times to avoid incorrect difficulty adjustments
        self.block_times = []
        self.blocks_since_adjustment = 0
        
        # Clean up recovery state
        delattr(self, 'recovery_state')
        delattr(self, '_previous_state')

    @property
    def current_height(self) -> int:
        """Get current block height (alias for current_block)."""
        return self.current_block