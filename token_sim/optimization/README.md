# Tokenomics Simulation Optimization

This module contains optimization algorithms for tokenomics simulations. These tools help find optimal parameter configurations for token economic systems.

## Core Components

### Algorithms

- **Bayesian Optimizer**: Uses Gaussian Process Regression to efficiently search the parameter space.
- **Genetic Algorithm**: (Coming soon) Optimize complex parameter combinations using evolutionary methods.
- **Reinforcement Learning**: (Coming soon) Dynamic parameter adjustment based on simulation feedback.
- **Particle Swarm Optimization**: (Coming soon) Multi-agent optimization for continuous parameter spaces.
- **Neural Network Optimization**: (Coming soon) Deep learning-based optimization for complex parameter relationships.

### Evaluation Metrics

- **Price Stability**: Measures token price volatility
- **Network Security**: Estimates security levels based on miner/validator participation
- **Market Liquidity**: Evaluates trading volumes and market depth
- **Energy Efficiency**: Assesses consensus mechanism energy requirements

## Usage

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
- Multi-objective optimization
- Visualization dashboard
- Distributed optimization processing 