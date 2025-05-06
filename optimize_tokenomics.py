import optuna
import numpy as np
from Tokenomics_Sim_Run_For_Optuna import run_tokenomics_simulation
from dataclasses import dataclass

@dataclass
class SimulationParams:
    initial_token_price: float
    monthly_token_distribution: float
    simulation_months: int
    initial_miners: int
    operating_cost_per_miner: float
    initial_fiat_per_miner: float
    market_maker_liquidity: float
    market_fee_rate: float
    market_price_volatility: float
    holder_strategies: list
    verbose: bool = False

def objective(trial):
    # Define the parameter search space
    params = SimulationParams(
        initial_token_price=trial.suggest_float('initial_token_price', 0.1, 10.0),
        monthly_token_distribution=trial.suggest_float('monthly_token_distribution', 100000, 1000000),
        simulation_months=trial.suggest_int('simulation_months', 12, 36),
        initial_miners=trial.suggest_int('initial_miners', 10, 100),
        operating_cost_per_miner=trial.suggest_float('operating_cost_per_miner', 100, 1000),
        initial_fiat_per_miner=trial.suggest_float('initial_fiat_per_miner', 1000, 10000),
        market_maker_liquidity=trial.suggest_float('market_maker_liquidity', 100000, 1000000),
        market_fee_rate=trial.suggest_float('market_fee_rate', 0.001, 0.05),
        market_price_volatility=trial.suggest_float('market_price_volatility', 0.01, 0.2),
        holder_strategies=['hold', 'trade', 'stake']  # Fixed strategies for now
    )

    # Run the simulation
    results = run_tokenomics_simulation(params)
    
    # Define the objective function components
    price_stability = -abs(results['price_change'])  # Negative because we want to minimize price change
    network_growth = results['network_storage']  # Positive because we want to maximize storage
    miner_retention = results['active_miners'] / params.initial_miners  # Ratio of active miners
    
    # Combine objectives with weights
    objective_value = (
        0.4 * price_stability +  # 40% weight on price stability
        0.4 * network_growth +   # 40% weight on network growth
        0.2 * miner_retention    # 20% weight on miner retention
    )
    
    return objective_value

def optimize_tokenomics(n_trials=100):
    # Create a study object
    study = optuna.create_study(direction='maximize')
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print the best parameters and results
    print("\nBest trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study

if __name__ == "__main__":
    # Run the optimization
    study = optimize_tokenomics(n_trials=50)
    
    # Run a final simulation with the best parameters
    best_params = SimulationParams(
        initial_token_price=study.best_params['initial_token_price'],
        monthly_token_distribution=study.best_params['monthly_token_distribution'],
        simulation_months=study.best_params['simulation_months'],
        initial_miners=study.best_params['initial_miners'],
        operating_cost_per_miner=study.best_params['operating_cost_per_miner'],
        initial_fiat_per_miner=study.best_params['initial_fiat_per_miner'],
        market_maker_liquidity=study.best_params['market_maker_liquidity'],
        market_fee_rate=study.best_params['market_fee_rate'],
        market_price_volatility=study.best_params['market_price_volatility'],
        holder_strategies=['hold', 'trade', 'stake'],
        verbose=True
    )
    
    print("\nRunning final simulation with best parameters...")
    final_results = run_tokenomics_simulation(best_params)
    
    print("\nFinal Results:")
    print(f"Final Token Price: ${final_results['final_price']:.4f}")
    print(f"Price Change: {final_results['price_change']:.2f}%")
    print(f"Network Storage: {final_results['network_storage']:.2f} TB")
    print(f"Active Miners: {final_results['active_miners']}") 