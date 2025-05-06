import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from token_sim.simulation import TokenSimulation
from token_sim.consensus.pow import ProofOfWork
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.governance.simple import SimpleGovernance
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader

def run_single_simulation():
    """Run a single simulation and return its results."""
    # Initialize components
    consensus = ProofOfWork(
        block_reward=100,
        difficulty_adjustment_blocks=10,
        target_block_time=1.0
    )
    market = PriceDiscovery(
        initial_price=1.0,
        volatility=0.1,
        market_depth=1000.0
    )
    governance = SimpleGovernance(
        voting_period=5,
        quorum_threshold=0.4,
        approval_threshold=0.6
    )

    # Create agents
    miners = [
        Miner(
            agent_id=f"miner_{i}",
            strategy='passive',
            initial_hashrate=10.0,
            electricity_cost=0.01,
            initial_balance=100.0
        ) for i in range(10)
    ]
    holders = [
        Holder(
            agent_id=f"holder_{i}",
            strategy='long_term',
            initial_balance=100.0,
            initial_tokens=10.0
        ) for i in range(5)
    ]
    traders = [
        Trader(
            agent_id=f"trader_{i}",
            strategy='momentum',
            initial_balance=100.0,
            initial_tokens=10.0
        ) for i in range(5)
    ]
    agents = miners + holders + traders

    # Initialize consensus participants and simulation
    consensus.initialize_participants(len(miners))
    sim = TokenSimulation(
        consensus=consensus,
        price_discovery=market,
        governance=governance,
        agents=agents,
        initial_supply=10000.0,
        market_depth=1000.0,
        initial_token_price=1.0,
        time_steps=20
    )

    # Run simulation
    history = sim.run()

    # Calculate metrics
    metrics = {
        'avg_price': np.mean(history['price']),
        'price_volatility': np.std(history['price']),
        'total_volume': sum(history['volume']),
        'avg_volume': np.mean(history['volume']),
        'final_security_score': history.get('network_security', [0])[-1],
        'miner_tokens': sum(agent.state['token_balance'] for agent in miners),
        'holder_tokens': sum(agent.state['token_balance'] for agent in holders),
        'trader_tokens': sum(agent.state['token_balance'] for agent in traders),
        'price_history': history['price'],
        'volume_history': history['volume']
    }
    
    return metrics

def run_multiple_simulations(num_simulations=100):
    """Run multiple simulations and aggregate results."""
    print(f"Running {num_simulations} simulations...")
    
    # Store results
    results = []
    price_histories = []
    volume_histories = []
    
    for i in range(num_simulations):
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} simulations...")
        metrics = run_single_simulation()
        results.append(metrics)
        price_histories.append(metrics['price_history'])
        volume_histories.append(metrics['volume_history'])
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create visualizations
    plt.style.use('ggplot')
    
    # 1. Price Distribution
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.hist(df['avg_price'], bins=30, alpha=0.7)
    plt.title('Distribution of Average Prices')
    plt.xlabel('Average Price')
    plt.ylabel('Frequency')
    
    # 2. Volume Distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['total_volume'], bins=30, alpha=0.7)
    plt.title('Distribution of Total Trading Volume')
    plt.xlabel('Total Volume')
    plt.ylabel('Frequency')
    
    # 3. Token Distribution
    plt.subplot(2, 2, 3)
    token_data = pd.DataFrame({
        'Miner': df['miner_tokens'],
        'Holder': df['holder_tokens'],
        'Trader': df['trader_tokens']
    })
    token_data.boxplot()
    plt.title('Token Distribution by Agent Type')
    plt.ylabel('Total Tokens')
    
    # 4. Price Volatility vs Volume
    plt.subplot(2, 2, 4)
    plt.scatter(df['price_volatility'], df['total_volume'], alpha=0.5)
    plt.title('Price Volatility vs Trading Volume')
    plt.xlabel('Price Volatility')
    plt.ylabel('Total Volume')
    
    plt.tight_layout()
    plt.savefig('simulation_analysis.png')
    plt.close()
    
    # Print summary statistics
    print("\nSimulation Summary Statistics:")
    print("\nPrice Statistics:")
    print(f"Average Price: {df['avg_price'].mean():.2f} ± {df['avg_price'].std():.2f}")
    print(f"Price Volatility: {df['price_volatility'].mean():.2f} ± {df['price_volatility'].std():.2f}")
    
    print("\nVolume Statistics:")
    print(f"Average Total Volume: {df['total_volume'].mean():.2f} ± {df['total_volume'].std():.2f}")
    print(f"Average Volume per Step: {df['avg_volume'].mean():.2f} ± {df['avg_volume'].std():.2f}")
    
    print("\nToken Distribution:")
    print(f"Average Miner Tokens: {df['miner_tokens'].mean():.2f} ± {df['miner_tokens'].std():.2f}")
    print(f"Average Holder Tokens: {df['holder_tokens'].mean():.2f} ± {df['holder_tokens'].std():.2f}")
    print(f"Average Trader Tokens: {df['trader_tokens'].mean():.2f} ± {df['trader_tokens'].std():.2f}")
    
    print("\nNetwork Security:")
    print(f"Average Security Score: {df['final_security_score'].mean():.2f} ± {df['final_security_score'].std():.2f}")

if __name__ == "__main__":
    run_multiple_simulations(100) 