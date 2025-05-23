from token_market_simulator import run_market_simulation, plot_simulation_results, plot_trading_signals, plot_performance_metrics
import os

def test_balanced_scenario():
    """Test the balanced (PoW) scenario with optimized parameters."""
    print("\nTesting Balanced (PoW) Scenario...")
    
    # Create output directory
    os.makedirs('test_results', exist_ok=True)
    
    # Define balanced scenario parameters
    params = {
        'block_reward': 50,
        'difficulty_adjustment_blocks': 2016,
        'target_block_time': 600,
        'initial_price': 1.0,
        'volatility': 0.1,
        'market_depth': 1000000.0,
        'num_miners': 50,  # Reduced for testing
        'num_traders': 25,  # Reduced for testing
        'num_holders': 50,  # Reduced for testing
        'initial_supply': 1000000.0,
        'simulation_steps': 200  # Reduced for testing
    }
    
    # Run simulation
    results = run_market_simulation('balanced', params)
    
    # Generate plots
    plot_simulation_results(results, 'Balanced (PoW)', 
                          'test_results/balanced_indicators.png')
    plot_trading_signals(results, 'Balanced (PoW)', 
                        'test_results/balanced_signals.png')
    plot_performance_metrics(results, 'Balanced (PoW)', 
                           'test_results/balanced_performance.png')
    
    # Print key metrics
    print("\nKey Metrics:")
    print("-" * 50)
    print(f"Final Price: ${results[-1]['price']:.4f}")
    print(f"Price Change: {((results[-1]['price'] - params['initial_price']) / params['initial_price'] * 100):.2f}%")
    print(f"Total Volume: ${sum(r['volume'] for r in results):,.2f}")
    print(f"Active Miners: {results[-1]['active_miners']}")
    print(f"Network Hashrate: {results[-1]['network_hashrate']:,.2f}")
    print(f"Total Trading Volume: {results[-1]['total_trading_volume']:,.0f}")
    print(f"Average Holding Time: {results[-1]['average_holding_time']:.2f} steps")
    print(f"Miner Rewards: ${results[-1]['miner_rewards']:,.2f}")
    print(f"Trader Profits: ${results[-1]['trader_profits']:,.2f}")
    print(f"Holder Returns: ${results[-1]['holder_returns']:,.2f}")
    
    # Print technical indicators
    print("\nTechnical Indicators (Final):")
    print("-" * 50)
    print(f"RSI: {results[-1]['rsi']:.2f}")
    print(f"MACD: {results[-1]['macd']:.4f}")
    print(f"Stochastic K: {results[-1]['stoch_k']:.2f}")
    print(f"Stochastic D: {results[-1]['stoch_d']:.2f}")
    print(f"ADX: {results[-1]['adx']:.2f}")
    print(f"EMA 20: {results[-1]['ema_20']:.4f}")
    print(f"EMA 50: {results[-1]['ema_50']:.4f}")

if __name__ == "__main__":
    test_balanced_scenario() 