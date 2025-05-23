import numpy as np
import matplotlib.pyplot as plt
from token_sim.market.price_discovery import PriceDiscovery

class SimplePrice:
    """A simple price discovery model with basic random walk."""
    def __init__(self, initial_price=1.0, volatility=0.1):
        self.current_price = initial_price
        self.volatility = volatility
        self.price_history = [initial_price]
    
    def update_price(self, volume=0.0, market_sentiment=0.0, time_step=1):
        # Simple random walk with drift
        price_change = np.random.normal(0, self.volatility)
        self.current_price *= (1 + price_change)
        self.price_history.append(self.current_price)
        return self.current_price

class ModifiedPriceDiscovery(PriceDiscovery):
    """Modified version of PriceDiscovery to test different parameters."""
    def __init__(self, initial_price=1.0, volatility=0.1, market_depth=1000000.0,
                 mean_reversion_strength=0.1, volume_impact_factor=0.0001,
                 price_bounds=(0.001, 100)):
        super().__init__(initial_price, volatility, market_depth)
        self.mean_reversion_strength = mean_reversion_strength
        self.volume_impact_factor = volume_impact_factor
        self.price_bounds = price_bounds
    
    def update_price(self, volume=0.0, market_sentiment=0.0, time_step=1):
        # Update current volume
        self.current_volume = volume
        
        # Calculate price impact from volume
        volume_impact = (volume / self.market_depth) * self.volume_impact_factor
        
        # Generate market noise with realistic patterns
        noise_factor = max(0.1, 1.0 - (volume / self.market_depth))
        
        # Mean reversion component
        mean_reversion = -self.mean_reversion_strength * (self.current_price - self.initial_price) / self.initial_price
        
        # Volatility clustering
        if len(self.price_history) > 1:
            last_return = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            volatility_factor = 1.0 + abs(last_return) * 2.0
        else:
            volatility_factor = 1.0
            
        # Volume-based noise
        volume_noise = np.random.normal(0, self.volatility * noise_factor * volatility_factor)
        
        # Market sentiment impact
        sentiment_weight = min(1.0, volume / self.market_depth)
        sentiment_impact = market_sentiment * self.volatility * 0.5 * sentiment_weight
        
        # Combine all components
        price_change = (
            volume_impact +
            mean_reversion +
            volume_noise +
            sentiment_impact
        )
        
        # Update price with bounds
        self.current_price *= (1 + price_change)
        self.current_price = max(self.price_bounds[0], min(self.current_price, self.price_bounds[1]))
        
        # Update histories
        self._update_histories(volume)
        
        # Update order book
        self._update_order_book()
        
        self.last_update_time = time_step
        return self.current_price

def run_simulation(price_model, steps=1000, seed=42):
    """Run a simulation with the given price model."""
    np.random.seed(seed)
    prices = []
    volumes = []
    
    # Generate some random volumes with occasional spikes
    base_volumes = np.random.normal(1000000, 200000, steps)
    volume_spikes = np.random.randint(0, steps, size=5)  # 5 random volume spikes
    for spike in volume_spikes:
        base_volumes[spike] *= 5  # 5x volume spike
    volumes = base_volumes
    
    # Generate sentiment with occasional strong shifts
    sentiments = np.random.normal(0, 0.5, steps)
    sentiment_shifts = np.random.randint(0, steps, size=3)  # 3 major sentiment shifts
    for shift in sentiment_shifts:
        sentiments[shift:shift+10] += np.random.choice([-1, 1])  # Sustained sentiment shift
    
    # Run simulation
    for i in range(steps):
        price = price_model.update_price(
            volume=volumes[i],
            market_sentiment=sentiments[i],
            time_step=i
        )
        prices.append(price)
    
    return prices, volumes, sentiments

def analyze_price_stability(prices, window=50):
    """Analyze price stability using rolling statistics."""
    prices = np.array(prices)
    rolling_mean = np.convolve(prices, np.ones(window)/window, mode='valid')
    rolling_std = np.array([np.std(prices[i:i+window]) for i in range(len(prices)-window+1)])
    
    # Calculate stability metrics
    mean_reversion = np.corrcoef(prices[:-1], prices[1:])[0,1]  # Autocorrelation at lag 1
    volatility_ratio = np.std(rolling_std) / np.mean(rolling_std)  # Volatility of volatility
    
    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'mean_reversion': mean_reversion,
        'volatility_ratio': volatility_ratio
    }

def main():
    # Test different parameter combinations
    parameter_sets = [
        {
            'name': 'Default',
            'mean_reversion': 0.1,
            'volume_impact': 0.0001,
            'price_bounds': (0.001, 100)
        },
        {
            'name': 'Weak Mean Reversion',
            'mean_reversion': 0.01,
            'volume_impact': 0.0001,
            'price_bounds': (0.001, 100)
        },
        {
            'name': 'Strong Volume Impact',
            'mean_reversion': 0.1,
            'volume_impact': 0.001,
            'price_bounds': (0.001, 100)
        },
        {
            'name': 'Tighter Bounds',
            'mean_reversion': 0.1,
            'volume_impact': 0.0001,
            'price_bounds': (0.5, 2)
        }
    ]
    
    # Run simulations for each parameter set
    results = {}
    for params in parameter_sets:
        model = ModifiedPriceDiscovery(
            initial_price=1.0,
            volatility=0.1,
            market_depth=1000000.0,
            mean_reversion_strength=params['mean_reversion'],
            volume_impact_factor=params['volume_impact'],
            price_bounds=params['price_bounds']
        )
        
        prices, volumes, sentiments = run_simulation(model)
        stability = analyze_price_stability(prices)
        results[params['name']] = {
            'prices': prices,
            'stability': stability
        }
    
    # Create plots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Price Evolution
    ax1 = plt.subplot(3, 1, 1)
    for name, result in results.items():
        ax1.plot(result['prices'], label=name, alpha=0.7)
    ax1.set_title('Price Evolution Comparison (Different Parameters)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Rolling Statistics
    ax2 = plt.subplot(3, 1, 2)
    for name, result in results.items():
        ax2.plot(result['stability']['rolling_std'], label=f'{name} Volatility', alpha=0.7)
    ax2.set_title('Rolling Volatility (50-period window)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Volatility')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Stability Metrics
    metrics = ['mean_reversion', 'volatility_ratio']
    ax3 = plt.subplot(3, 1, 3)
    x = np.arange(len(parameter_sets))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        values = [results[params['name']]['stability'][metric] for params in parameter_sets]
        ax3.bar(x + i*width, values, width, label=metric)
    
    ax3.set_title('Stability Metrics Comparison')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels([params['name'] for params in parameter_sets])
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('price_stability_analysis.png')
    plt.close()
    
    # Print detailed statistics
    print("\nPrice Stability Analysis:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Mean Price: {np.mean(result['prices']):.4f}")
        print(f"Price Std: {np.std(result['prices']):.4f}")
        print(f"Mean Reversion: {result['stability']['mean_reversion']:.4f}")
        print(f"Volatility Ratio: {result['stability']['volatility_ratio']:.4f}")
        print(f"Max Drawdown: {np.max(np.maximum.accumulate(result['prices']) - result['prices']):.4f}")

if __name__ == "__main__":
    main() 