from token_sim.simulation import TokenSimulation
from token_sim.consensus.pow import ProofOfWork
from token_sim.consensus.pos import ProofOfStake
from token_sim.consensus.dpos import DelegatedProofOfStake
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.agents.miner import Miner
from token_sim.agents.trader import Trader
from token_sim.agents.holder import Holder
from token_sim.ai.agent_learning import TokenAgent, MarketPredictor, TokenEnvironment
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

def load_trained_models():
    """Load the trained AI models."""
    # Create a temporary simulation for the agent
    temp_consensus = ProofOfWork(
        block_reward=50,
        difficulty_adjustment_blocks=2016,
        target_block_time=600
    )
    temp_price_discovery = PriceDiscovery(
        initial_price=1.0,
        volatility=0.1,
        market_depth=1000000.0
    )
    temp_simulation = TokenSimulation(
        consensus=temp_consensus,
        price_discovery=temp_price_discovery,
        agents=[],
        initial_supply=1000000.0
    )
    
    # Load trading agent
    trading_agent = TokenAgent(temp_simulation)
    if os.path.exists('models/trading_agent_100.pkl'):
        trading_agent.load('models/trading_agent_100.pkl')
    
    # Load price predictor
    price_predictor = MarketPredictor()
    if os.path.exists('models/price_predictor.pth'):
        price_predictor.load_state_dict(torch.load('models/price_predictor.pth'))
        price_predictor.eval()
    
    return trading_agent, price_predictor

def run_tokenomics_analysis(scenario_name, params, trading_agent, price_predictor):
    """Run a tokenomics analysis with specific parameters using trained AI models."""
    print(f"\nRunning analysis for scenario: {scenario_name}")
    
    # Create consensus mechanism based on scenario
    if scenario_name == 'balanced':
        consensus = ProofOfWork(
            block_reward=params['block_reward'],
            difficulty_adjustment_blocks=params['difficulty_adjustment_blocks'],
            target_block_time=params['target_block_time']
        )
    elif scenario_name == 'high_inflation':
        consensus = ProofOfStake(
            block_reward=params['block_reward'],
            min_stake=params['min_stake'],
            staking_apy=params['staking_apy']
        )
    else:  # low_liquidity
        consensus = DelegatedProofOfStake(
            block_reward=params['block_reward'],
            min_stake=params['min_stake'],
            num_delegates=params['num_delegates'],
            staking_apy=params['staking_apy']
        )
    
    # Create price discovery mechanism
    price_discovery = PriceDiscovery(
        initial_price=params['initial_price'],
        volatility=params['volatility'],
        market_depth=params['market_depth']
    )
    
    # Create agents with AI-powered decision making
    agents = []
    
    # Add miners/validators based on consensus type
    if isinstance(consensus, ProofOfWork):
        for i in range(params['num_miners']):
            miner = Miner(
                agent_id=f"miner_{i}",
                strategy='ai',
                initial_hashrate=random.uniform(50, 200),
                electricity_cost=random.uniform(0.03, 0.08),
                initial_balance=random.uniform(1000, 5000)
            )
            miner.initialize()
            agents.append(miner)
    else:  # PoS or DPoS
        for i in range(params['num_validators']):
            staker = Staker(
                agent_id=f"validator_{i}",
                strategy='ai',
                initial_balance=random.uniform(10000, 50000)
            )
            staker.initialize()
            agents.append(staker)
    
    # Add traders with AI strategy
    for i in range(params['num_traders']):
        trader = Trader(
            agent_id=f"trader_{i}",
            strategy='ai',
            initial_balance=random.uniform(1000, 5000)
        )
        trader.initialize()
        agents.append(trader)
    
    # Add holders with AI strategy
    for i in range(params['num_holders']):
        holder = Holder(
            agent_id=f"holder_{i}",
            strategy='ai',
            initial_balance=random.uniform(100, 1000)
        )
        holder.initialize()
        agents.append(holder)
    
    # Initialize simulation
    simulation = TokenSimulation(
        consensus=consensus,
        price_discovery=price_discovery,
        agents=agents,
        initial_supply=params['initial_supply']
    )
    
    # Create environment for AI agents
    env = TokenEnvironment(simulation)
    
    # Run simulation
    results = []
    print(f"\nRunning simulation for {params['simulation_steps']} steps...")
    
    for step in range(params['simulation_steps']):
        if step % 100 == 0:
            print(f"Step {step}/{params['simulation_steps']}")
        
        # Get current state
        state = simulation.get_current_state()
        
        # Print state keys for debugging
        if step == 0:
            print("Available state keys:", state.keys())
            print("Price stats:", state.get('price_stats', {}))
            print("Initial price:", state.get('price', params['initial_price']))
            print("Initial supply:", state.get('supply', params['initial_supply']))
        
        # Extract price statistics
        price_stats = state.get('price_stats', {})
        
        # Convert state to tensor for price predictor
        state_features = [
            state.get('price', params['initial_price']),
            state.get('supply', params['initial_supply']),
            state.get('active_miners', 0),
            state.get('network_hashrate', 0),
            price_stats.get('price_change_24h', 0.0),
            price_stats.get('volatility', params['volatility']),
            price_stats.get('volume_24h', 0.0),
            params['initial_price'],
            params['market_depth'],
            params['volatility']
        ]
        
        # Create a sequence of length 10
        state_sequence = [state_features] * 10
        state_tensor = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)
        
        # Use price predictor to forecast price
        with torch.no_grad():
            price_forecast = price_predictor.predict(state_tensor).item()
        
        # Update state with AI predictions
        state['price_forecast'] = price_forecast
        
        # Let AI agents make decisions
        for agent in agents:
            if hasattr(agent, 'strategy') and agent.strategy == 'ai':
                action = trading_agent.predict(state)
                agent_action = {
                    'trade': True,
                    'type': 'buy' if action[0] > 0 else 'sell',
                    'amount': abs(action[0]) * action[1] * agent.state['balance']
                }
                
                if agent_action['trade']:
                    simulation.market_maker.execute_trade(
                        agent,
                        agent_action['type'],
                        agent_action['amount']
                    )
                    
                    agent.update(0.0, {
                        'trade_type': agent_action['type'],
                        'trade_amount': agent_action['amount'],
                        'token_price': state.get('price', params['initial_price'])
                    })
        
        # Step simulation
        simulation._run_step()
        
        # Record results
        results.append({
            'step': step,
            'price': state.get('price', params['initial_price']),
            'price_forecast': price_forecast,
            'supply': state.get('supply', params['initial_supply']),
            'active_miners': state.get('active_miners', 0),
            'network_hashrate': state.get('network_hashrate', 0),
            'price_mean': price_stats.get('mean', params['initial_price']),
            'price_std': price_stats.get('std', params['volatility']),
            'price_min': price_stats.get('min', params['initial_price']),
            'price_max': price_stats.get('max', params['initial_price']),
            'volume': price_stats.get('volume_24h', 0),
            'volatility': price_stats.get('volatility', params['volatility']),
            'miner_rewards': sum(agent.state.get('total_rewards', 0) for agent in agents if isinstance(agent, Miner)),
            'trader_profits': sum(agent.state.get('total_profit', 0) for agent in agents if isinstance(agent, Trader)),
            'holder_returns': sum(agent.state.get('total_profit', 0) for agent in agents if isinstance(agent, Holder))
        })
    
    return results

def main():
    # Load trained models
    trading_agent, price_predictor = load_trained_models()
    
    # Define scenarios
    scenarios = {
        'balanced': {  # PoW
            'block_reward': 50,
            'difficulty_adjustment_blocks': 2016,
            'target_block_time': 600,
            'initial_price': 1.0,
            'volatility': 0.1,
            'market_depth': 1000000.0,
            'num_miners': 100,
            'num_traders': 50,
            'num_holders': 100,
            'initial_supply': 1000000.0,
            'simulation_steps': 1000
        },
        'high_inflation': {  # PoS
            'block_reward': 100,
            'min_stake': 1000,
            'staking_apy': 0.05,
            'initial_price': 1.0,
            'volatility': 0.1,
            'market_depth': 1000000.0,
            'num_validators': 50,
            'num_traders': 50,
            'num_holders': 100,
            'initial_supply': 1000000.0,
            'simulation_steps': 1000
        },
        'low_liquidity': {  # DPoS
            'block_reward': 50,
            'min_stake': 1000,
            'num_delegates': 21,
            'staking_apy': 0.05,
            'initial_price': 1.0,
            'volatility': 0.1,
            'market_depth': 500000.0,
            'num_validators': 21,
            'num_traders': 25,
            'num_holders': 100,
            'initial_supply': 1000000.0,
            'simulation_steps': 1000
        }
    }
    
    # Run analysis for each scenario
    results = {}
    for scenario_name, params in scenarios.items():
        metrics = run_tokenomics_analysis(scenario_name, params, trading_agent, price_predictor)
        results[scenario_name] = metrics
    
    # Compare scenarios
    print("\nScenario Comparison:")
    print("-" * 100)
    print(f"{'Metric':<20} {'Balanced (PoW)':<15} {'High Inflation (PoS)':<15} {'Low Liquidity (DPoS)':<15}")
    print("-" * 100)
    
    metrics = ['price_stability', 'volume_stability', 'miner_activity', 
              'network_health', 'price_trend', 'volatility',
              'forecast_accuracy', 'miner_sustainability',
              'trader_success', 'holder_satisfaction']
    
    for metric in metrics:
        print(f"{metric:<20}", end='')
        for scenario in scenarios.keys():
            print(f"{results[scenario][metric]:<15.4f}", end='')
        print()

if __name__ == "__main__":
    main() 