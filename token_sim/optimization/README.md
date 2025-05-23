# Tokenomics Simulation Optimization

This module contains optimization algorithms for tokenomics simulations. These tools help find optimal parameter configurations for token economic systems.

## Core Components

### Performance Optimization

- **Parallel Processing**: Multi-threaded and multi-process execution for CPU-intensive tasks
- **Vectorized Operations**: NumPy-based vectorization for numerical computations
- **Caching**: Result caching for repeated function calls
- **Batch Processing**: Efficient batch processing of large datasets
- **Memory Management**: Automatic memory cleanup and monitoring
- **Performance Monitoring**: Real-time performance metrics tracking

### Optimization Algorithms

- **Bayesian Optimizer**: Uses Gaussian Process Regression to efficiently search the parameter space
- **Genetic Algorithm**: Optimize complex parameter combinations using evolutionary methods
- **Hybrid Optimizer**: Combines Bayesian and Genetic approaches for robust optimization
- **Reinforcement Learning**: (Coming soon) Dynamic parameter adjustment based on simulation feedback
- **Particle Swarm Optimization**: (Coming soon) Multi-agent optimization for continuous parameter spaces
- **Neural Network Optimization**: (Coming soon) Deep learning-based optimization for complex parameter relationships

### Evaluation Metrics

- **Price Stability**: Measures token price volatility
- **Network Security**: Estimates security levels based on miner/validator participation
- **Market Liquidity**: Evaluates trading volumes and market depth
- **Energy Efficiency**: Assesses consensus mechanism energy requirements
- **Performance Metrics**: Tracks execution time, memory usage, CPU usage, throughput, and latency

## Usage

### Performance Optimization

```python
from token_sim.optimization import PerformanceOptimizer, OptimizationStrategy

# Initialize optimizer
optimizer = PerformanceOptimizer(max_workers=4)

# Start performance monitoring
optimizer.monitor_performance(interval=60)  # Log metrics every 60 seconds

# Optimize a function
@optimizer.optimize_function(strategy=OptimizationStrategy.PARALLEL)
def my_function(data):
    # Your code here
    pass
```

### Bayesian Optimization for Core Parameters

```bash
# Run with default settings (PoW consensus)
python -m token_sim.optimization.optimize_core_params

# Run with PoS consensus
python -m token_sim.optimization.optimize_core_params --consensus pos

# Run with more iterations
python -m token_sim.optimization.optimize_core_params --iterations 50

# Run with more initial random points
python -m token_sim.optimization.optimize_core_params --initial-points 10
```

### Parameter Optimization Details

The core parameters being optimized:

- **Block Reward**: Token amount given for block validation (0.1-10.0)
- **Initial Supply**: Initial token circulation amount (1M-100M)
- **Market Depth**: Market liquidity parameter (100K-10M)
- **Initial Price**: Starting token price in USD (0.1-10.0)

## Future Improvements

- Integration with multiple consensus mechanisms
- Advanced parameter dependency handling
- Distributed optimization across multiple machines
- Real-time parameter adjustment based on network conditions
- Machine learning-based optimization strategies 