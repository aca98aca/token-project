from token_sim.ai.agent_learning import TokenAgent, MarketPredictor, TokenEnvironment
from token_sim.ai.consensus_optimizer import ConsensusOptimizer
from token_sim.simulation import TokenSimulation
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import os
import random

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 1. Train Trading Agent
def train_trading_agent(simulation, agent_index=0, total_timesteps=100000):
    print(f"Training trading agent {agent_index}...")
    agent = TokenAgent(simulation, agent_index)
    
    # Create tensorboard directory
    os.makedirs('tensorboard_logs', exist_ok=True)
    
    # Train with progress bar
    agent.train(total_timesteps=total_timesteps)
    
    # Save the model
    agent.save(f'models/trading_agent_{agent_index}.pkl')
    return agent

# 2. Train Price Predictor
def train_price_predictor(historical_data, sequence_length=10):
    print("Training price predictor...")
    # Prepare data
    df = pd.read_csv('tokenomics_simulation_results.csv')
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[['price', 'volume', 'price_volatility']].values[i:i+sequence_length])
        y.append(df['price'].values[i+sequence_length])
    
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    # Initialize and train model
    predictor = MarketPredictor(input_size=3, hidden_size=64)
    for epoch in range(100):
        loss = predictor.train(X, y)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    predictor.save('models/price_predictor.pth')
    return predictor

# 3. Optimize Consensus Parameters
def optimize_consensus(simulation):
    print("Optimizing consensus parameters...")
    param_bounds = {
        'block_reward': (10, 100),
        'min_stake': (100, 10000),
        'staking_apy': (0.01, 0.2),
        'difficulty_adjustment_blocks': (1000, 3000),
        'target_block_time': (300, 900)
    }
    
    optimizer = ConsensusOptimizer(simulation, param_bounds)
    best_params = optimizer.optimize(n_iterations=50)
    print("Best parameters found:", best_params)
    return best_params

def evaluate_trading_agent(agent, simulation, num_episodes=10):
    """Evaluate the trained trading agent."""
    print("\nEvaluating trading agent...")
    
    # Create environment for evaluation
    env = TokenEnvironment(simulation, agent_index=agent.agent_index)
    
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")
    
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward

def evaluate_price_predictor(predictor, data_path):
    """Evaluate price predictor performance."""
    # Load test data
    df = pd.read_csv(data_path)
    X_test = torch.FloatTensor(df[['price', 'volume', 'price_volatility']].values[-10:])
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((predictions.numpy() - df['price'].values[-1]) ** 2)
    mae = np.mean(np.abs(predictions.numpy() - df['price'].values[-1]))
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

def calculate_sharpe_ratio(prices, risk_free_rate=0.02):
    """Calculate Sharpe ratio for price series."""
    returns = np.diff(prices) / prices[:-1]
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown for price series."""
    peak = prices[0]
    max_dd = 0
    
    for price in prices:
        if price > peak:
            peak = price
        dd = (peak - price) / peak
        max_dd = max(max_dd, dd)
    
    return max_dd

def main():
    # Initialize simulation components
    from token_sim.consensus.pow import ProofOfWork
    from token_sim.market.price_discovery import PriceDiscovery
    from token_sim.agents.miner import Miner
    from token_sim.agents.trader import Trader
    from token_sim.agents.holder import Holder
    
    # Create consensus mechanism
    consensus = ProofOfWork(
        block_reward=50,
        difficulty_adjustment_blocks=2016,
        target_block_time=600
    )
    
    # Create price discovery mechanism
    price_discovery = PriceDiscovery(
        initial_price=1.0,
        volatility=0.1,
        market_depth=1000000.0
    )
    
    # Create agents
    agents = []
    
    # Add miners
    for i in range(100):
        agents.append(Miner(
            agent_id=f"miner_{i}",
            strategy=random.choice(['efficient', 'aggressive', 'passive']),
            initial_hashrate=random.uniform(50, 200),
            electricity_cost=random.uniform(0.03, 0.08),
            initial_balance=random.uniform(1000, 5000)
        ))
    
    # Add traders
    for i in range(50):
        agents.append(Trader(
            agent_id=f"trader_{i}",
            strategy=random.choice(['momentum', 'mean_reversion', 'random']),
            initial_balance=random.uniform(1000, 5000)
        ))
    
    # Add holders
    for i in range(100):
        agents.append(Holder(
            agent_id=f"holder_{i}",
            strategy=random.choice(['long_term', 'medium_term', 'short_term']),
            initial_balance=random.uniform(100, 1000)
        ))
    
    # Initialize simulation
    simulation = TokenSimulation(
        consensus=consensus,
        price_discovery=price_discovery,
        agents=agents,
        initial_supply=1000000.0
    )
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # 1. Train trading agent (using the first trader)
    trader_index = 100  # Index of the first trader in the agents list
    agent = train_trading_agent(simulation, agent_index=trader_index)
    
    # 2. Evaluate trading agent
    evaluate_trading_agent(agent, simulation)
    
    # 3. Train price predictor
    predictor = train_price_predictor('tokenomics_simulation_results.csv')
    
    # 4. Optimize consensus parameters
    best_params = optimize_consensus(simulation)
    
    # Save results
    results = {
        'best_consensus_params': best_params,
        'trading_agent_path': f'models/trading_agent_{trader_index}.pkl',
        'price_predictor_path': 'models/price_predictor.pth'
    }
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {results}")

    # Evaluate models
    print("\nModel Evaluation:")
    print("1. Trading Agent Performance:")
    evaluate_trading_agent(agent, simulation)
    
    print("\n2. Price Predictor Performance:")
    evaluate_price_predictor(predictor, 'tokenomics_simulation_results.csv')
    
    print("\n3. Consensus Parameters:")
    print(f"Optimized parameters: {best_params}")

if __name__ == "__main__":
    main() 