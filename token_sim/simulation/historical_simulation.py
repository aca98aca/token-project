from token_sim.data.historical_loader import HistoricalDataLoader
from token_market_simulator import run_market_simulation, plot_scenario_comparison
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os

def run_historical_simulation(symbol: str = "BTC", days: int = 30) -> Dict[str, Any]:
    """Run simulation using historical data as T0 for different consensus mechanisms."""
    
    print(f"\nLoading historical data for {symbol} for {days} days...")
    
    # Load historical data
    loader = HistoricalDataLoader(symbol=symbol)
    historical_data = loader.fetch_data()
    market_metrics = loader.get_market_metrics()
    
    # Get the actual consensus mechanism of the cryptocurrency
    actual_consensus = loader.consensus
    print(f"\nActual consensus mechanism for {symbol}: {actual_consensus}")
    
    print("\nHistorical data summary:")
    print(f"Initial Price: ${market_metrics['initial_price']:,.2f}")
    print(f"Average Daily Volume: ${market_metrics['avg_daily_volume']:,.2f}")
    print(f"Volatility: {market_metrics['volatility']*100:.2f}%")
    print(f"Price Range: ${market_metrics['price_range']['min']:,.2f} - ${market_metrics['price_range']['max']:,.2f}")
    
    # Configure simulation parameters based on historical data
    initial_supply = 21000000  # Initial token supply
    target_block_time = 600    # Target block time in seconds (10 minutes)
    blocks_per_day = 24 * 60 * 60 // target_block_time  # Number of blocks per day
    simulation_steps = days * blocks_per_day  # Total simulation steps
    
    # Define volatility parameters based on historical data
    base_volatility = {
        'base_volatility': market_metrics['volatility'],
        'vol_of_vol': 0.1,  # Volatility of volatility
        'mean_reversion': 0.1,  # Speed of mean reversion
        'long_term_vol': market_metrics['volatility'],  # Long-term volatility level
        'correlation': {  # Correlation between different factors
            'returns_volume': 0.6,
            'returns_volatility': 0.3,
            'volume_volatility': 0.4
        }
    }
    
    base_params = {
        'initial_price': market_metrics['initial_price'],
        'block_reward': market_metrics['initial_price'] * 0.001,  # 0.1% of price as block reward
        'block_time': target_block_time,
        'num_miners': 50,
        'num_traders': 25,
        'num_holders': 25,
        'initial_liquidity': market_metrics['avg_daily_volume'] * 0.1,  # 10% of daily volume
        'price_volatility': market_metrics['volatility'],
        'historical_data': historical_data,
        'market_depth': market_metrics['avg_daily_volume'] * 0.05,  # 5% of daily volume as market depth
        'market_impact': 0.1,  # 10% price impact for trades equal to market depth
        'min_market_depth': market_metrics['avg_daily_volume'] * 0.01,  # 1% of daily volume as minimum depth
        'initial_supply': initial_supply,
        'max_supply': initial_supply * 1.1,  # 10% more than initial supply
        'difficulty_adjustment_blocks': 2016,  # Number of blocks for difficulty adjustment
        'target_block_time': target_block_time,
        'difficulty_adjustment_factor': 0.25,  # Maximum difficulty adjustment per period
        'simulation_steps': simulation_steps,
        'blocks_per_day': blocks_per_day,
        'volatility': market_metrics['volatility']  # Use the scalar volatility value directly
    }
    
    # Run simulations for each consensus mechanism
    results = {}
    scenarios = ['PoW', 'PoS', 'DPoS']
    
    for scenario in scenarios:
        print(f"\nRunning simulation for {scenario}...")
        scenario_params = base_params.copy()
        
        # Adjust parameters based on consensus mechanism
        if scenario == 'PoW':
            scenario_params.update({
                'mining_difficulty': 1000000,
                'hashrate_growth_rate': 0.1,
                'market_depth': base_params['market_depth'] * 1.2,  # Higher liquidity for PoW
                'difficulty_adjustment_blocks': 2016,  # ~2 weeks at 10-minute blocks
                'initial_hashrate': 1000000,  # Initial network hashrate
                'max_hashrate_change': 0.3,  # Maximum 30% hashrate change per period
                'volatility': market_metrics['volatility'] * 0.9  # Lower volatility for PoW
            })
        elif scenario == 'PoS':
            scenario_params.update({
                'min_stake': market_metrics['initial_price'] * 100,
                'staking_apy': 0.1,
                'market_depth': base_params['market_depth'] * 0.8,  # Lower liquidity for PoS
                'initial_staked': initial_supply * 0.5,  # 50% of supply staked initially
                'max_stake_change': 0.2,  # Maximum 20% stake change per period
                'unstaking_period': 14 * blocks_per_day,  # 14 days worth of blocks
                'volatility': market_metrics['volatility'] * 1.1  # Higher volatility for PoS
            })
        else:  # DPoS
            scenario_params.update({
                'num_delegates': 21,
                'min_stake': market_metrics['initial_price'] * 50,
                'delegation_reward_share': 0.5,
                'market_depth': base_params['market_depth'] * 0.9,  # Medium liquidity for DPoS
                'initial_staked': initial_supply * 0.3,  # 30% of supply staked initially
                'max_stake_change': 0.15,  # Maximum 15% stake change per period
                'voting_period': 7 * blocks_per_day,  # 7 days worth of blocks
                'volatility': market_metrics['volatility']  # Baseline volatility for DPoS
            })
        
        try:
            results[scenario] = run_market_simulation(scenario, scenario_params)
            print(f"{scenario} simulation completed successfully.")
        except Exception as e:
            print(f"Error running {scenario} simulation: {e}")
            continue
    
    if results:
        try:
            # Create results directory if it doesn't exist
            os.makedirs('simulation_results', exist_ok=True)
            
            # Plot comparison with historical data
            plot_historical_comparison(results, historical_data, symbol, actual_consensus)
            print(f"\nPlots generated successfully in simulation_results/{symbol}_comparison.png")
            
            # Save results to CSV
            save_historical_results_to_csv(results, historical_data, symbol, actual_consensus)
            print(f"Results saved to simulation_results/{symbol}_results.csv")
            
            # Print summary statistics
            print("\nSimulation Results Summary:")
            for scenario, result in results.items():
                final_price = result['price_history'][-1]
                avg_volume = np.mean(result['volume_history'])
                volatility = np.std(np.diff(np.log(result['price_history']))) * np.sqrt(252)
                print(f"\n{scenario}:")
                print(f"  Final Price: ${final_price:,.2f}")
                print(f"  Avg Daily Volume: ${avg_volume:,.2f}")
                print(f"  Annualized Volatility: {volatility*100:.2f}%")
                if scenario == 'PoW':
                    print(f"  Final Network Hashrate: {result['network_hashrate'][-1]:,.0f}")
                else:
                    print(f"  Final Active Validators: {result['active_validators'][-1]:,.0f}")
        except Exception as e:
            print(f"Error generating plots or saving results: {e}")
    
    return results

def plot_historical_comparison(simulation_results: Dict[str, Any], historical_data: pd.DataFrame, symbol: str, actual_consensus: str):
    """Plot comparison between simulated and historical data."""
    plt.style.use('default')  # Use default style instead of seaborn
    fig = plt.figure(figsize=(15, 10))
    
    # Convert historical data to numeric
    historical_data['Close'] = pd.to_numeric(historical_data['Close'])
    historical_data['Volume'] = pd.to_numeric(historical_data['Volume'])
    historical_data['Volatility'] = pd.to_numeric(historical_data['Volatility'])
    
    # Price comparison
    ax1 = plt.subplot(2, 2, 1)
    historical_data['Close'].plot(label=f'{symbol} ({actual_consensus})', color='black', alpha=0.7, ax=ax1)
    for scenario, results in simulation_results.items():
        price_series = pd.Series(
            results['price_history'],
            index=historical_data.index[:len(results['price_history'])]
        )
        price_series.plot(label=f'{scenario} Simulated', alpha=0.7, ax=ax1)
    ax1.set_title(f'{symbol} Price Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Volume comparison
    ax2 = plt.subplot(2, 2, 2)
    historical_data['Volume'].plot(label=f'{symbol} ({actual_consensus})', color='black', alpha=0.7, ax=ax2)
    for scenario, results in simulation_results.items():
        volume_series = pd.Series(
            results['volume_history'],
            index=historical_data.index[:len(results['volume_history'])]
        )
        volume_series.plot(label=f'{scenario} Simulated', alpha=0.7, ax=ax2)
    ax2.set_title('Trading Volume Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Volatility comparison
    ax3 = plt.subplot(2, 2, 3)
    historical_data['Volatility'].plot(label=f'{symbol} ({actual_consensus})', color='black', alpha=0.7, ax=ax3)
    for scenario, results in simulation_results.items():
        volatility_series = pd.Series(
            results['volatility_history'],
            index=historical_data.index[:len(results['volatility_history'])]
        )
        volatility_series.plot(label=f'{scenario} Simulated', alpha=0.7, ax=ax3)
    ax3.set_title('Volatility Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # Network metrics
    ax4 = plt.subplot(2, 2, 4)
    for scenario, results in simulation_results.items():
        if scenario == 'PoW':
            metric = results.get('network_hashrate', [])
            label = 'Hashrate'
        else:
            metric = results.get('active_validators', [])
            label = 'Validators'
            
        if metric:
            metric_series = pd.Series(
                metric,
                index=historical_data.index[:len(metric)]
            )
            metric_series.plot(label=f'{scenario} {label}', alpha=0.7, ax=ax4)
    ax4.set_title('Network Metrics')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'simulation_results/{symbol}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_historical_results_to_csv(simulation_results: Dict[str, Any], historical_data: pd.DataFrame, symbol: str, actual_consensus: str):
    """Save simulation results to CSV for further analysis."""
    results_df = pd.DataFrame()
    
    # Add historical data
    results_df[f'{symbol}_Price'] = historical_data['Close']
    results_df[f'{symbol}_Volume'] = historical_data['Volume']
    results_df[f'{symbol}_Volatility'] = historical_data['Volatility']
    results_df[f'{symbol}_Consensus'] = actual_consensus
    
    # Add simulation results
    for scenario, results in simulation_results.items():
        results_df[f'{scenario}_Price'] = pd.Series(results['price_history'])
        results_df[f'{scenario}_Volume'] = pd.Series(results['volume_history'])
        results_df[f'{scenario}_Volatility'] = pd.Series(results['volatility_history'])
        
        if scenario == 'PoW':
            results_df[f'{scenario}_Hashrate'] = pd.Series(results.get('network_hashrate', []))
        else:
            results_df[f'{scenario}_Validators'] = pd.Series(results.get('active_validators', []))
    
    results_df.to_csv(f'simulation_results/{symbol}_results.csv')

if __name__ == "__main__":
    # Run simulation with BTC historical data for 30 days
    results = run_historical_simulation(symbol="BTC", days=30) 