from typing import Dict, List, Any
import numpy as np
from token_sim.consensus import ConsensusMechanism
from token_sim.agents import Agent
from token_sim.agents.miner import Miner
from token_sim.agents.staker import Staker
from token_sim.agents.trader import Trader
from token_sim.agents.holder import Holder
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.market.market_maker import MarketMaker

class TokenSimulation:
    """Main simulation class that coordinates all components."""
    
    def __init__(self,
                 consensus: Any,
                 price_discovery: PriceDiscovery,
                 agents: List[Agent],
                 initial_supply: float = 1000000.0,
                 market_depth: float = 1000000.0,
                 initial_token_price: float = 1.0,
                 time_steps: int = 1000,
                 governance: Any = None):
        self.consensus = consensus
        self.price_discovery = price_discovery
        self.market = price_discovery  # Alias for backward compatibility
        self.agents = agents
        self.initial_supply = initial_supply
        self.current_supply = initial_supply
        self.time_steps = time_steps
        self.current_step = 0
        self.market_depth = market_depth
        self.governance = governance
        self.transaction_fee = 0.001  # Add transaction fee
        
        # Store parameters for reference
        self.params = {
            'initial_supply': initial_supply,
            'market_depth': market_depth,
            'initial_token_price': initial_token_price,
            'time_steps': time_steps,
            'simulation_months': time_steps,  # Alias for backward compatibility
            'transaction_fee': self.transaction_fee
        }
        
        # Initialize market maker with market depth and initial price
        self.market_maker = MarketMaker(
            unique_id="market_maker",
            model=self,
            liquidity_fiat=market_depth,
            liquidity_tokens=market_depth / initial_token_price,  # Adjust token liquidity based on price
            fee_rate=0.001,
            price_volatility=0.05
        )
        
        # Initialize history
        self.history = {
            'price': [],
            'volume': [],
            'rewards': [],
            'active_miners': [],
            'network_hashrate': [],
            'rewards_distribution': [],
            'trades': [],
            'network_security': []
        }
        
        # Initialize agents
        for agent in agents:
            agent.initialize()
    
    def run(self) -> Dict[str, List]:
        """Run the simulation for the specified number of time steps."""
        for _ in range(self.time_steps):
            self._run_step()
            self.current_step += 1
        return self.history
    
    def _run_step(self) -> None:
        """Run a single simulation step."""
        # Get consensus rewards
        total_rewards, rewards_distribution = self.consensus.perform_consensus_step()
        
        # Sync market with consensus block height
        self.price_discovery.last_block_height = self.consensus.current_height
        
        # Sync agent balances with market
        self.price_discovery.update_balances(self.agents)
        
        # Get current state
        current_state = {
            'price': self.price_discovery.current_price,
            'price_history': self.history['price'],
            'volume': self.history['volume'][-1] if self.history['volume'] else 0,
            'token_price': self.price_discovery.current_price,
            'active_miners': len([a for a in self.agents if a.state.get('active', False) and 
                                (isinstance(a, Miner) or isinstance(a, Staker))]),
            'price_stats': self.price_discovery.get_price_stats(),
            'staking_apy': getattr(self.consensus, 'staking_apy', 0.0),
            'active_validators': self.consensus.get_active_participants(),
            'validator_stats': {pid: self.consensus.get_participant_stats(pid) 
                              for pid in self.consensus.get_active_participants()},
            'block_reward': self.consensus.block_reward,
            'network_difficulty': getattr(self.consensus, 'current_difficulty', 1.0),
            'network_hashrate': self._get_network_hashrate(),
            'total_stake': getattr(self.consensus, 'total_stake', 0.0),
            'market_depth': self.market_depth,
            'current_supply': self.current_supply,
            'market_maker_liquidity': {
                'fiat': self.market_maker.liquidity_fiat,
                'tokens': self.market_maker.liquidity_tokens
            },
            'current_step': self.current_step,
            'network_security_score': getattr(self.consensus, 'get_network_security_score', lambda: 0.0)()
        }
        
        # Update agents and collect their actions
        trades_this_step = []
        total_volume = 0.0
        market_sentiment = 0.0
        trader_profits = 0.0
        holder_returns = 0.0
        
        for agent in self.agents:
            if not agent.state['active']:
                continue
            
            # Get agent's action
            action = agent.act(current_state)
            
            # Record agent action for price impact
            self.price_discovery.record_agent_action(agent.agent_id, action.get('type', 'unknown'), action)
            
            # Process trading action
            if action.get('trade'):
                trade_result = self.market_maker.execute_trade(
                    agent,
                    action['type'],
                    action['amount']
                )
                if trade_result:
                    trades_this_step.append(trade_result)
                    total_volume += trade_result['volume']
                    # Update market sentiment based on trade direction
                    if action['type'] == 'buy':
                        market_sentiment += 0.1
                    else:
                        market_sentiment -= 0.1
                    
                    # Track profits/returns
                    if isinstance(agent, Trader):
                        trader_profits += trade_result.get('profit', 0.0)
                    elif isinstance(agent, Holder):
                        holder_returns += trade_result.get('return', 0.0)
            
            # Process mining action
            if action.get('mining'):
                if isinstance(action['mining'], dict):  # Ensure mining action is a dictionary
                    self.price_discovery.record_agent_action(agent.agent_id, 'mining', action['mining'])
                else:
                    # Handle legacy format or invalid format
                    self.price_discovery.record_agent_action(agent.agent_id, 'mining', {
                        'participate': bool(action['mining']),
                        'hashrate': agent.state.get('hashrate', 0.0)
                    })
            
            # Process staking action
            if action.get('staking'):
                self.price_discovery.record_agent_action(agent.agent_id, 'staking', action['staking'])
            
            # Update agent state
            reward = rewards_distribution.get(agent.agent_id, 0.0)
            agent.update(reward, current_state)
        
        # Normalize market sentiment
        if len(self.agents) > 0:
            market_sentiment /= len(self.agents)
        
        # Update price discovery with trading volume and market sentiment
        self.price_discovery.update_price(
            volume=total_volume,
            market_sentiment=market_sentiment,
            time_step=self.current_step
        )
        
        # Update history
        self.history['price'].append(self.price_discovery.current_price)
        self.history['volume'].append(total_volume)
        self.history['rewards'].append(total_rewards)
        self.history['active_miners'].append(current_state['active_miners'])
        self.history['network_hashrate'].append(current_state['network_hashrate'])
        self.history['rewards_distribution'].append(rewards_distribution)
        self.history['trades'].append(trades_this_step)
        self.history['network_security'].append(current_state['network_security_score'])
        self.history['trader_profits'] = self.history.get('trader_profits', []) + [trader_profits]
        self.history['holder_returns'] = self.history.get('holder_returns', []) + [holder_returns]
        
        # Update supply
        self.current_supply += total_rewards
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the simulation."""
        return {
            'price': self.price_discovery.current_price,
            'supply': self.current_supply,
            'active_miners': len(self.consensus.get_active_participants()),
            'price_stats': self.price_discovery.get_price_stats(),
            'network_hashrate': sum(stats['hashrate'] for pid, stats in self.consensus.participants.items() 
                                  if stats['active']),
            'market_depth': self.market_depth,
            'volume': self.history['volume'][-1] if self.history['volume'] else 0,
            'market_maker_liquidity': {
                'fiat': self.market_maker.liquidity_fiat,
                'tokens': self.market_maker.liquidity_tokens
            },
            'current_step': self.current_step
        }

    def reset(self):
        """Reset simulation state."""
        self.current_step = 0
        self.current_supply = self.initial_supply
        
        # Reset components
        for agent in self.agents:
            agent.reset()
        self.consensus.reset()
        self.price_discovery.reset()
        self.market_maker.reset()
        
        # Reset history
        self.history = {
            'price': [],
            'volume': [],
            'rewards': [],
            'active_miners': [],
            'network_hashrate': [],
            'rewards_distribution': [],
            'trades': [],
            'network_security': []
        }
        
        return self.get_current_state()

    def step(self) -> None:
        """Run a single simulation step."""
        # Process governance proposals first
        if self.governance:
            # Process governance step
            self.governance.step()
            
            # Apply any passed proposals
            self._apply_passed_proposals()
        
        # Run the main simulation step
        self._run_step()
        self.current_step += 1
    
    def _apply_passed_proposals(self) -> None:
        """Apply passed governance proposals."""
        if not self.governance:
            return
            
        for proposal_id in self.governance.passed_proposals:
            proposal = self.governance.proposals[proposal_id]
            
            # Skip already applied proposals
            if proposal.get('applied', False):
                continue
                
            # Process parameter updates
            if proposal['details']['type'] == 'parameter_update':
                param_name = proposal['details']['parameter']
                new_value = proposal['details']['new_value']
                
                # Update market parameters
                if param_name == 'market_fee_rate':
                    self.market.fee_rate = new_value
                    self.market_maker.fee_rate = new_value
                elif param_name == 'market_depth':
                    self.market_depth = new_value
                    self.market.market_depth = new_value
                elif param_name == 'market_price_volatility':
                    self.market.volatility = new_value
                    self.market_maker.price_volatility = new_value
                
                # Mark as applied
                proposal['applied'] = True
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the simulation."""
        return {
            'price': self.price_discovery.current_price,
            'supply': self.current_supply,
            'active_miners': len(self.consensus.get_active_participants()),
            'price_stats': self.price_discovery.get_price_stats(),
            'network_hashrate': sum(stats['hashrate'] for pid, stats in self.consensus.participants.items() 
                                  if stats['active']),
            'market_depth': self.market_depth,
            'volume': self.history['volume'][-1] if self.history['volume'] else 0,
            'market_maker_liquidity': {
                'fiat': self.market_maker.liquidity_fiat,
                'tokens': self.market_maker.liquidity_tokens
            },
            'current_step': self.current_step,
            'last_block_height': self.consensus.current_block
        }

    def get_parameter(self, parameter_name: str) -> Any:
        """Get a parameter value from the simulation."""
        if parameter_name in self.params:
            return self.params[parameter_name]
        elif hasattr(self.market, parameter_name):
            return getattr(self.market, parameter_name)
        elif hasattr(self.consensus, parameter_name):
            return getattr(self.consensus, parameter_name)
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

    def get_balance(self, agent_id: str) -> float:
        """Get the token balance for an agent."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.state['token_balance']
        raise ValueError(f"Unknown agent: {agent_id}")

    def _get_network_hashrate(self) -> float:
        """Get network hashrate or equivalent metric based on consensus type."""
        if hasattr(self.consensus, 'total_stake'):
            # For PoS and DPoS, use total stake as a proxy for network security
            return self.consensus.total_stake
        else:
            # For PoW, use actual hashrate
            return sum(stats['hashrate'] for pid, stats in self.consensus.participants.items() 
                      if stats.get('active', False) and 'hashrate' in stats) 