from token_sim.simulation import TokenSimulation
from token_sim.consensus.pow import ProofOfWork
from token_sim.consensus.pos import ProofOfStake
from token_sim.consensus.dpos import DelegatedProofOfStake
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader
from token_sim.agents.staker import Staker
from token_sim.market.price_discovery import PriceDiscovery
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
import talib
import shutil

def cleanup_simulation_results():
    """Clean up the simulation_results directory."""
    if os.path.exists('simulation_results'):
        shutil.rmtree('simulation_results')
    os.makedirs('simulation_results')

def run_market_simulation(scenario_name, params):
    """Run a market simulation with specific parameters using technical analysis."""
    print(f"\nRunning simulation for scenario: {scenario_name}")
    
    # Initialize price discovery with volatility parameter
    price_discovery = PriceDiscovery(
        initial_price=params['initial_price'],
        market_depth=params['market_depth'],
        volatility=params['volatility']
    )
    
    # Initialize consensus mechanism based on scenario
    if scenario_name == 'PoW':
        consensus = ProofOfWork(
            block_reward=params['block_reward'],
            difficulty_adjustment_blocks=params['difficulty_adjustment_blocks'],
            target_block_time=params['target_block_time']
        )
        consensus.initialize_participants(params['num_miners'])
    elif scenario_name == 'PoS':
        consensus = ProofOfStake(
            block_reward=params['block_reward'] * 2,  # Higher rewards
            min_stake=1000.0,  # Minimum stake requirement
            staking_apy=0.1  # 10% APY for higher rewards
        )
        consensus.initialize_participants(params['num_miners'], initial_stake=2000.0)
    else:  # DPoS
        consensus = DelegatedProofOfStake(
            block_reward=params['block_reward'],
            num_delegates=21
        )
        consensus.initialize_participants(params['num_miners'], initial_stake=1500.0)
    
    # Initialize agents with strategy
    agents = []
    # Add miners/validators with different strategies
    strategies = ['passive', 'aggressive', 'adaptive']
    for i in range(params['num_miners']):
        strategy = strategies[i % len(strategies)]  # Distribute strategies evenly
        
        if scenario_name == 'PoW':  # PoW
            agent = Miner(f"miner_{i}", strategy=strategy, initial_balance=1000.0)
            agent.initialize()
            agent.state['hashrate'] = consensus.participants[f"miner_{i}"]['hashrate']
            agent.state['mining'] = {
                'participate': True,
                'hashrate': agent.state['hashrate']
            }
            agent.state['fiat_balance'] = 10000.0  # Initial fiat balance
            agent.state['token_balance'] = 1000.0  # Initial token balance
        elif scenario_name == 'PoS':  # PoS
            agent = Staker(f"validator_{i}", strategy=strategy, initial_balance=2000.0)
            agent.initialize()
            agent.state['stake'] = consensus.participants[f"validator_{i}"]['stake']
            agent.state['mining'] = {
                'participate': True,
                'stake': agent.state['stake']
            }
            agent.state['fiat_balance'] = 20000.0  # Higher initial fiat balance for PoS
            agent.state['token_balance'] = 2000.0  # Higher initial token balance for PoS
        else:  # DPoS
            agent = Staker(f"validator_{i}", strategy=strategy, initial_balance=1500.0)
            agent.initialize()
            agent.state['stake'] = consensus.participants[f"validator_{i}"]['stake']
            agent.state['mining'] = {
                'participate': True,
                'stake': agent.state['stake']
            }
            agent.state['fiat_balance'] = 15000.0  # Initial fiat balance for DPoS
            agent.state['token_balance'] = 1500.0  # Initial token balance for DPoS
        
        agents.append(agent)
    
    # Add traders with different strategies
    for i in range(params['num_traders']):
        strategy = 'technical' if i < params['num_traders'] // 2 else 'momentum'
        trader = Trader(f"trader_{i}", strategy=strategy, initial_balance=10000.0)
        trader.initialize()  # Initialize trader
        trader.state['fiat_balance'] = 50000.0  # Higher initial fiat balance for traders
        trader.state['token_balance'] = 5000.0  # Initial token balance for traders
        agents.append(trader)
    
    # Add holders with different strategies
    for i in range(params['num_holders']):
        strategy = 'long_term' if i < params['num_holders'] // 2 else 'dynamic'
        holder = Holder(f"holder_{i}", strategy=strategy, initial_balance=5000.0)
        holder.initialize()  # Initialize holder
        holder.state['fiat_balance'] = 25000.0  # Initial fiat balance for holders
        holder.state['token_balance'] = 10000.0  # Higher initial token balance for holders
        agents.append(holder)
    
    # Initialize simulation
    simulation = TokenSimulation(
        consensus=consensus,
        price_discovery=price_discovery,
        agents=agents,
        initial_supply=params['initial_supply'],
        market_depth=params['market_depth'],
        initial_token_price=params['initial_price'],
        time_steps=params['simulation_steps']
    )
    
    # Run simulation
    history = simulation.run()
    
    # Convert history to list of dictionaries for plotting
    results = []
    for i in range(len(history['price'])):
        results.append({
            'price': history['price'][i],
            'volume': history['volume'][i],
            'active_miners': history['active_miners'][i],
            'network_hashrate': history['network_hashrate'][i],
            'total_trading_volume': sum(history['volume'][:i+1]),
            'average_holding_time': i,  # Simplified holding time metric
            'miner_rewards': sum(sum(r.values()) for r in history['rewards_distribution'][:i+1]),
            'trader_profits': 0.0,  # Would need more complex calculation
            'holder_returns': 0.0,  # Would need more complex calculation
            'rsi': 50.0,  # Placeholder for technical indicators
            'macd': 0.0,
            'stoch_k': 50.0,
            'stoch_d': 50.0,
            'adx': 50.0,
            'ema_20': history['price'][i],
            'ema_50': history['price'][i]
        })
        
        # Calculate technical indicators if enough data points
        if i >= 14:
            prices = np.array(history['price'][:i+1])
            results[i]['rsi'] = talib.RSI(prices)[-1]
            macd, signal, hist = talib.MACD(prices)
            results[i]['macd'] = macd[-1]
            stoch_k, stoch_d = talib.STOCH(prices, prices, prices)
            results[i]['stoch_k'] = stoch_k[-1]
            results[i]['stoch_d'] = stoch_d[-1]
            results[i]['adx'] = talib.ADX(prices, prices, prices)[-1]
            if i >= 20:
                results[i]['ema_20'] = talib.EMA(prices, timeperiod=20)[-1]
            if i >= 50:
                results[i]['ema_50'] = talib.EMA(prices, timeperiod=50)[-1]
    
    return results

def plot_simulation_results(results, scenario_name, save_path=None):
    """Plot key simulation results."""
    plt.figure(figsize=(15, 10))
    
    # Price and Volume
    plt.subplot(2, 2, 1)
    plt.plot([r['price'] for r in results], label='Price', color='blue')
    plt.title(f'{scenario_name} - Price Movement')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot([r['volume'] for r in results], label='Volume', color='green')
    plt.title('Trading Volume')
    plt.xlabel('Time Step')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    
    # Network Metrics
    plt.subplot(2, 2, 3)
    plt.plot([r['network_hashrate'] for r in results], label='Hashrate', color='red')
    plt.title('Network Hashrate')
    plt.xlabel('Time Step')
    plt.ylabel('Hashrate')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot([r['active_miners'] for r in results], label='Active Miners', color='purple')
    plt.title('Active Miners')
    plt.xlabel('Time Step')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_trading_signals(results, scenario_name, save_path=None):
    """Plot trading signals and indicators."""
    plt.figure(figsize=(15, 10))
    
    # Price and Trading Signals
    plt.subplot(2, 2, 1)
    plt.plot([r['price'] for r in results], label='Price', color='blue')
    plt.title(f'{scenario_name} - Price and Signals')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # RSI
    plt.subplot(2, 2, 2)
    plt.plot([r['rsi'] for r in results], label='RSI', color='orange')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI Indicator')
    plt.xlabel('Time Step')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # MACD
    plt.subplot(2, 2, 3)
    plt.plot([r['macd'] for r in results], label='MACD', color='purple')
    plt.title('MACD')
    plt.xlabel('Time Step')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    
    # Stochastic
    plt.subplot(2, 2, 4)
    plt.plot([r['stoch_k'] for r in results], label='Stoch K', color='blue')
    plt.plot([r['stoch_d'] for r in results], label='Stoch D', color='red')
    plt.axhline(y=80, color='r', linestyle='--')
    plt.axhline(y=20, color='g', linestyle='--')
    plt.title('Stochastic Oscillator')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_performance_metrics(results, scenario_name, save_path=None):
    """Plot performance metrics."""
    plt.figure(figsize=(15, 10))
    
    # Trading Performance
    plt.subplot(2, 2, 1)
    plt.plot([r['trader_profits'] for r in results], label='Trader Profits', color='green')
    plt.title('Trader Profits')
    plt.xlabel('Time Step')
    plt.ylabel('Profits')
    plt.legend()
    plt.grid(True)
    
    # Miner Rewards
    plt.subplot(2, 2, 2)
    plt.plot([r['miner_rewards'] for r in results], label='Miner Rewards', color='orange')
    plt.title('Miner Rewards')
    plt.xlabel('Time Step')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)
    
    # Holder Returns
    plt.subplot(2, 2, 3)
    plt.plot([r['holder_returns'] for r in results], label='Holder Returns', color='blue')
    plt.title('Holder Returns')
    plt.xlabel('Time Step')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    
    # Network Metrics
    plt.subplot(2, 2, 4)
    plt.plot([r['network_hashrate'] for r in results], label='Network Hashrate', color='red')
    plt.plot([r['active_miners'] for r in results], label='Active Miners', color='purple')
    plt.title('Network Metrics')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_scenario_comparison(results_dict, save_path=None):
    """Create comparison plots for all scenarios."""
    plt.figure(figsize=(20, 15))
    
    # Price Comparison
    plt.subplot(3, 2, 1)
    for scenario, results in results_dict.items():
        plt.plot([r['price'] for r in results], label=scenario)
    plt.title('Price Movement Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Volume Comparison
    plt.subplot(3, 2, 2)
    for scenario, results in results_dict.items():
        plt.plot([r['volume'] for r in results], label=scenario)
    plt.title('Trading Volume Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    
    # Network Hashrate Comparison
    plt.subplot(3, 2, 3)
    for scenario, results in results_dict.items():
        plt.plot([r['network_hashrate'] for r in results], label=scenario)
    plt.title('Network Hashrate Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Hashrate')
    plt.legend()
    plt.grid(True)
    
    # Active Miners Comparison
    plt.subplot(3, 2, 4)
    for scenario, results in results_dict.items():
        plt.plot([r['active_miners'] for r in results], label=scenario)
    plt.title('Active Miners Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Miner Rewards Comparison
    plt.subplot(3, 2, 5)
    for scenario, results in results_dict.items():
        plt.plot([r['miner_rewards'] for r in results], label=scenario)
    plt.title('Miner Rewards Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Rewards')
    plt.legend()
    plt.grid(True)
    
    # RSI Comparison
    plt.subplot(3, 2, 6)
    for scenario, results in results_dict.items():
        plt.plot([r['rsi'] for r in results], label=scenario)
    plt.title('RSI Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_summary_table(results_dict):
    """Create a summary table comparing key metrics across scenarios."""
    summary_data = []
    
    for scenario, results in results_dict.items():
        # Calculate key metrics
        final_price = results[-1]['price']
        avg_volume = np.mean([r['volume'] for r in results])
        max_price = max([r['price'] for r in results])
        min_price = min([r['price'] for r in results])
        price_volatility = np.std([r['price'] for r in results]) / np.mean([r['price'] for r in results])
        total_miner_rewards = results[-1]['miner_rewards']
        avg_active_miners = np.mean([r['active_miners'] for r in results])
        avg_network_hashrate = np.mean([r['network_hashrate'] for r in results])
        
        summary_data.append({
            'Scenario': scenario,
            'Final Price': f"${final_price:.2f}",
            'Average Volume': f"${avg_volume:.2f}",
            'Max Price': f"${max_price:.2f}",
            'Min Price': f"${min_price:.2f}",
            'Price Volatility': f"{price_volatility:.2%}",
            'Total Miner Rewards': f"${total_miner_rewards:.2f}",
            'Avg Active Miners': f"{avg_active_miners:.1f}",
            'Avg Network Hashrate': f"{avg_network_hashrate:.1f}"
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv('simulation_results/scenario_comparison.csv', index=False)
    
    # Create a formatted table for display
    print("\nScenario Comparison Summary:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    return df

def main():
    # Clean up previous results
    cleanup_simulation_results()
    
    # Define scenarios
    scenarios = {
        'PoW': {  # Proof of Work
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
        'PoS': {  # Proof of Stake
            'block_reward': 100,  # Double the reward
            'difficulty_adjustment_blocks': 2016,
            'target_block_time': 600,
            'initial_price': 1.0,
            'volatility': 0.15,  # Higher volatility
            'market_depth': 1000000.0,
            'num_miners': 100,
            'num_traders': 50,
            'num_holders': 100,
            'initial_supply': 1000000.0,
            'simulation_steps': 1000
        },
        'DPoS': {  # Delegated Proof of Stake
            'block_reward': 50,
            'difficulty_adjustment_blocks': 2016,
            'target_block_time': 600,
            'initial_price': 1.0,
            'volatility': 0.2,  # Higher volatility
            'market_depth': 500000.0,  # Half the market depth
            'num_miners': 100,
            'num_traders': 50,
            'num_holders': 100,
            'initial_supply': 1000000.0,
            'simulation_steps': 1000
        }
    }
    
    # Run analysis for each scenario
    results = {}
    for scenario_name, params in scenarios.items():
        print(f"\nRunning simulation for {scenario_name} scenario...")
        metrics = run_market_simulation(scenario_name, params)
        results[scenario_name] = metrics
        
        # Plot individual scenario results
        plot_simulation_results(metrics, scenario_name, f'simulation_results/{scenario_name}_indicators.png')
        plot_trading_signals(metrics, scenario_name, f'simulation_results/{scenario_name}_signals.png')
        plot_performance_metrics(metrics, scenario_name, f'simulation_results/{scenario_name}_performance.png')
    
    # Create comparison plots and summary table
    plot_scenario_comparison(results, 'simulation_results/scenario_comparison.png')
    create_summary_table(results)

if __name__ == "__main__":
    main() 