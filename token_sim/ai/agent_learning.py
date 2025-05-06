import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import gym
from token_sim.simulation import TokenSimulation

class TokenEnvironment(gym.Env):
    """Environment for training trading agents."""
    
    def __init__(self, simulation: TokenSimulation, agent_index: int = 0):
        super().__init__()
        self.simulation = simulation
        self.agent_index = agent_index
        
        # Define action space (continuous)
        # [action_type, amount] where action_type: -1 (sell) to 1 (buy)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space (continuous)
        # [price, volume, balance, token_balance, market_depth, volatility, liquidity_ratio, trading_frequency, price_momentum, market_sentiment]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Convert state dictionary to observation array."""
        state = self.simulation.get_current_state()
        agent_state = self.simulation.agents[self.agent_index].get_state()
        
        # Normalize values
        price = state.get('token_price', 0.0) / 100.0  # Normalize price
        volume = state.get('current_volume', 0.0) / 1000.0  # Normalize volume
        balance = agent_state.get('balance', 0.0) / agent_state.get('initial_balance', 1.0)  # Normalize balance
        token_balance = agent_state.get('token_balance', 0.0) / 100.0  # Normalize token balance
        
        # Get market metrics
        market_depth = state.get('market_depth', 0.0) / 1000.0  # Normalize market depth
        volatility = state.get('volatility', 0.0)  # Already normalized
        
        # Get optional metrics with defaults
        liquidity_ratio = state.get('liquidity_ratio', 0.5)
        trading_frequency = state.get('trading_frequency', 0.5)
        price_momentum = state.get('price_momentum', 0.5)
        market_sentiment = state.get('market_sentiment', 0.5)
        
        # Return as numpy array
        return np.array([
            price,
            volume,
            balance,
            token_balance,
            market_depth,
            volatility,
            liquidity_ratio,
            trading_frequency,
            price_momentum,
            market_sentiment
        ], dtype=np.float32)
    
    def reset(self):
        """Reset the environment."""
        self.simulation.reset()
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        # Convert action to trade
        action_type = action[0]  # -1 to 1
        amount = action[1]  # 0 to 1
        
        # Store initial state for reward calculation
        initial_balance = self.simulation.agents[self.agent_index].state['balance']
        initial_token_balance = self.simulation.agents[self.agent_index].state['token_balance']
        
        # Determine trade type and amount
        trade_type = 'buy' if action_type > 0 else 'sell'
        trade_amount = abs(action_type) * amount * initial_balance
        
        # Execute trade using market maker
        self.simulation.market_maker.execute_trade(
            self.simulation.agents[self.agent_index],
            trade_type,
            trade_amount
        )
        
        # Calculate reward based on state changes
        current_state = self.simulation.get_current_state()
        agent_state = self.simulation.agents[self.agent_index].state
        
        # Calculate profit/loss
        balance_change = agent_state['balance'] - initial_balance
        token_value_change = (agent_state['token_balance'] - initial_token_balance) * current_state.get('token_price', 0.0)
        profit = balance_change + token_value_change
        
        # Calculate reward components
        volatility = current_state.get('volatility', 0.0)
        risk_penalty = -abs(profit) * volatility
        
        liquidity_ratio = current_state.get('liquidity_ratio', 0.5)
        liquidity_reward = profit * liquidity_ratio
        
        trading_frequency = current_state.get('trading_frequency', 0.5)
        frequency_penalty = -abs(profit) * (1 - trading_frequency)
        
        # Combine rewards
        reward = profit + risk_penalty + liquidity_reward + frequency_penalty
        
        # Get new state
        observation = self._get_observation()
        done = False  # You might want to add episode termination conditions
        
        return observation, reward, done, {}
    
    def render(self, mode='human'):
        pass

class CustomPolicy(BaseFeaturesExtractor):
    """Custom policy network for the trading agent."""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64, **kwargs):
        super().__init__(observation_space, features_dim)
        
        n_input = np.prod(observation_space.shape)
        
        self.network = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

class TokenAgent:
    """AI-powered agent using PPO for decision making."""
    
    def __init__(self, simulation, agent_index=0):
        self.simulation = simulation
        self.agent_index = agent_index
        self.env = DummyVecEnv([lambda: TokenEnvironment(simulation, agent_index)])
        
        # Initialize PPO model with custom policy
        self.model = PPO(
            "MlpPolicy",  # Use default MlpPolicy instead of custom policy for now
            self.env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
    
    def train(self, total_timesteps: int = 100000):
        """Train the agent."""
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
    
    def predict(self, observation):
        """Make a prediction based on current state."""
        # Convert dictionary observation to numpy array if needed
        if isinstance(observation, dict):
            # Get state values
            state = observation
            
            # Normalize values
            price = state.get('price', 0.0) / 100.0  # Normalize price
            volume = state.get('volume_24h', 0.0) / 1000.0  # Normalize volume
            supply = state.get('supply', 0.0) / 1000000.0  # Normalize supply
            active_miners = state.get('active_miners', 0) / 100.0  # Normalize active miners
            network_hashrate = state.get('network_hashrate', 0) / 1000.0  # Normalize hashrate
            
            # Get price stats
            price_stats = state.get('price_stats', {})
            price_change = price_stats.get('price_change_24h', 0.0) / 100.0  # Normalize price change
            volatility = price_stats.get('volatility', 0.0)  # Already normalized
            volume_24h = price_stats.get('volume_24h', 0.0) / 1000.0  # Normalize volume
            
            # Convert to numpy array
            observation = np.array([
                price,
                volume,
                supply,
                active_miners,
                network_hashrate,
                price_change,
                volatility,
                volume_24h,
                0.5,  # Default liquidity ratio
                0.5   # Default market sentiment
            ], dtype=np.float32)
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        """Save the trained model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model."""
        self.model = PPO.load(path, env=self.env)

class MarketPredictor:
    """LSTM-based price prediction model."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.optimizer = optim.Adam(list(self.lstm.parameters()) + list(self.linear.parameters()))
        self.criterion = nn.MSELoss()
    
    def forward(self, X: torch.Tensor):
        """Forward pass through the model."""
        lstm_out, _ = self.lstm(X)
        return self.linear(lstm_out[:, -1, :])  # Take last timestep output
    
    def train(self, X: torch.Tensor, y: torch.Tensor):
        """Train the model."""
        self.lstm.train()
        self.linear.train()
        self.optimizer.zero_grad()
        output = self.forward(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict(self, X: torch.Tensor):
        """Make predictions."""
        self.lstm.eval()
        self.linear.eval()
        with torch.no_grad():
            return self.forward(X)
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'lstm_state_dict': self.lstm.state_dict(),
            'linear_state_dict': self.linear.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path)
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.linear.load_state_dict(checkpoint['linear_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 