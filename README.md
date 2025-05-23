# Tokenomics Simulation Framework

A comprehensive framework for simulating token economics, market dynamics, and network behavior across different consensus mechanisms.

## Features

- Multiple consensus mechanisms (PoW, PoS, DPoS)
- Dynamic price discovery with market impact
- Multiple agent types with different strategies:
  - Miners/Validators
  - Traders
  - Holders
  - Stakers
- Technical analysis integration
- Comprehensive metrics tracking and visualization
- Historical data simulation capabilities
- Performance optimization tools

## Project Structure

```
token_sim/
├── agents/           # Agent implementations (miners, traders, holders)
├── consensus/        # Consensus mechanisms (PoW, PoS, DPoS)
├── market/          # Market dynamics and price discovery
├── analysis/        # Analysis tools and metrics
├── optimization/    # Performance optimization
├── visualization/   # Data visualization tools
├── tests/          # Test suite
└── examples/       # Example simulations and usage
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Alpha Vantage API key:
```bash
# On macOS/Linux
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# On Windows (Command Prompt)
set ALPHA_VANTAGE_API_KEY=your_api_key_here

# On Windows (PowerShell)
$env:ALPHA_VANTAGE_API_KEY="your_api_key_here"
```

You can get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

Note: TA-Lib installation might require additional steps:
- On macOS: `brew install ta-lib`
- On Linux: `sudo apt-get install ta-lib`
- On Windows: Download and install from [TA-Lib website](https://ta-lib.org/hdr_dw.html)

## Running the Simulation

To run a basic simulation:

```bash
python examples/run_basic_simulation.py
```

For historical data simulation:

```bash
python historical_simulation.py
```

## Generated Files

The simulation generates several output files in the `results/` directory:
- HTML visualizations of metrics and analysis
- CSV files with detailed simulation data
- Summary statistics and reports

These files are not tracked in Git as they are generated during simulation runs. Each time you run the simulation, new files will be created in the `results/` directory.

## Components

### Consensus Mechanisms
- Proof of Work (PoW)
- Proof of Stake (PoS)
- Delegated Proof of Stake (DPoS)

### Agents
- Miner/Validator agents with strategies:
  - Passive
  - Aggressive
  - Adaptive
- Trader agents with strategies:
  - Technical analysis
  - Momentum
- Holder agents with strategies:
  - Long-term
  - Dynamic

### Market
- Price discovery with market impact
- Volume tracking
- Technical indicators (RSI, MACD, Stochastic, ADX)
- Price statistics and analysis

### Analysis
- Performance metrics
- Network health indicators
- Trading signals
- Risk assessment

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking

## License

MIT License 