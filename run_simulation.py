import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Tokenomics_Sim_Run_For_Optuna import run_tokenomics_simulation
from TokenEconModel import TokenEconomyModel
from Market_Maker import MarketMaker
from Token_Holders import TokenHolder
from Miner_Agent import StorageMiner
from Simulation_Parameters import plot_simulation_results

# Define simulation parameters (as an object to match the model's expectations)
class SimulationParams:
    def __init__(self):
        self.initial_token_price = 1.0
        self.monthly_token_distribution = 1000000
        self.simulation_months = 12
        self.initial_miners = 100
        self.operating_cost_per_miner = 0.1
        self.initial_fiat_per_miner = 1000
        self.market_maker_liquidity = 1000000
        self.market_fee_rate = 0.001
        self.market_price_volatility = 0.1
        self.holder_strategies = ['hodl', 'trader', 'swing'] * 10  # 30 holders with different strategies
        self.verbose = True

params = SimulationParams()

# Run the simulation
print("Starting tokenomics simulation...")
results = run_tokenomics_simulation(params)

# Print key results
print("\nSimulation Results:")
print(f"Final Token Price: ${results['final_price']:.2f}")
print(f"Price Change: {results['price_change']:.2f}%")
print(f"Final Network Storage: {results['network_storage']:.2f}")
print(f"Active Miners: {results['active_miners']}")

# Plot the results
print("\nGenerating plots...")
plot_simulation_results(results['model_data'])

# Additional analysis
model_data = results['model_data']
print("\nDetailed Analysis:")
print("\nPrice Statistics:")
print(model_data['TokenPrice'].describe())
print("\nNetwork Storage Statistics:")
print(model_data['TotalNetworkStorage'].describe())
print("\nActive Miners Statistics:")
print(model_data['ActiveMiners'].describe()) 