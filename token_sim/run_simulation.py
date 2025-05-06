from token_sim.consensus.pow import ProofOfWork
from token_sim.consensus.pos import ProofOfStake
from token_sim.consensus.dpos import DelegatedProofOfStake
from token_sim.agents.miner import Miner
from token_sim.agents.trader import Trader
from token_sim.agents.staker import Staker
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.simulation import TokenSimulation
import random

def create_consensus(mechanism: str = 'pow', **kwargs):
    """Create consensus mechanism based on type."""
    if mechanism == 'pow':
        return ProofOfWork(**kwargs)
    elif mechanism == 'pos':
        return ProofOfStake(**kwargs)
    elif mechanism == 'dpos':
        return DelegatedProofOfStake(**kwargs)
    else:
        raise ValueError(f"Unknown consensus mechanism: {mechanism}")

def main():
    # Initialize consensus mechanism (choose one)
    consensus_type = 'pow'  # or 'pos' or 'dpos'
    consensus = create_consensus(
        mechanism=consensus_type,
        block_reward=1.0,
        difficulty_adjustment_blocks=2016,
        target_block_time=600.0
    )
    
    price_discovery = PriceDiscovery(
        initial_price=2.0,
        volatility=0.1,
        market_depth=1000000.0,
        price_impact_factor=0.0001
    )
    
    # Create agents
    agents = []
    
    # Add miners/validators
    miner_strategies = ['passive', 'aggressive', 'opportunistic']
    for i in range(50):  # 50 miners
        strategy = miner_strategies[i % len(miner_strategies)]
        miner = Miner(
            agent_id=f"miner_{i}",
            strategy=strategy,
            initial_hashrate=50.0,
            electricity_cost=0.05
        )
        agents.append(miner)
    
    # Add traders
    trader_strategies = ['momentum', 'mean_reversion', 'random']
    for i in range(30):  # 30 traders
        strategy = trader_strategies[i % len(trader_strategies)]
        trader = Trader(
            agent_id=f"trader_{i}",
            strategy=strategy,
            initial_balance=10000.0,
            risk_tolerance=random.uniform(0.3, 0.7)
        )
        agents.append(trader)
    
    # Add stakers
    staker_strategies = ['long_term', 'dynamic', 'validator_hopping']
    for i in range(20):  # 20 stakers
        strategy = staker_strategies[i % len(staker_strategies)]
        staker = Staker(
            agent_id=f"staker_{i}",
            strategy=strategy,
            initial_balance=10000.0,
            min_stake_duration=30
        )
        agents.append(staker)
    
    # Initialize consensus with participants
    consensus.initialize_participants(len(agents))
    
    # Create and run simulation
    simulation = TokenSimulation(
        consensus=consensus,
        price_discovery=price_discovery,
        agents=agents,
        initial_supply=1000000.0,
        time_steps=1000
    )
    
    # Run simulation
    history = simulation.run()
    
    # Print results
    final_state = simulation.get_current_state()
    print(f"\nSimulation Results ({consensus_type.upper()}):")
    print(f"Final Price: ${final_state['price']:.4f}")
    print(f"Price Change: {((final_state['price'] - 2.0) / 2.0 * 100):.2f}%")
    print(f"Final Supply: {final_state['supply']:,.0f}")
    print(f"Active Participants: {final_state['active_miners']}")
    
    if consensus_type == 'pow':
        print(f"Network Hashrate: {final_state['network_hashrate']:,.2f} TH/s")
    else:
        print(f"Total Stake: {consensus.total_stake:,.2f}")
    
    # Print price statistics
    price_stats = final_state['price_stats']
    print("\nPrice Statistics:")
    print(f"24h Price Change: {price_stats['price_change_24h']:.2f}%")
    print(f"Volatility: {price_stats['volatility']:.4f}")
    print(f"24h Volume: {price_stats['volume_24h']:,.2f}")
    
    # Print agent statistics
    print("\nAgent Statistics:")
    miners = [a for a in agents if isinstance(a, Miner)]
    traders = [a for a in agents if isinstance(a, Trader)]
    stakers = [a for a in agents if isinstance(a, Staker)]
    
    print(f"Miners: {len(miners)}")
    print(f"Traders: {len(traders)}")
    print(f"Stakers: {len(stakers)}")
    
    # Calculate average profits
    miner_profits = sum(m.state['total_rewards'] - m.state['total_costs'] for m in miners) / len(miners)
    trader_profits = sum(t.state['total_profit'] for t in traders) / len(traders)
    staker_profits = sum(s.state['total_rewards'] for s in stakers) / len(stakers)
    
    print(f"\nAverage Profits:")
    print(f"Miners: ${miner_profits:,.2f}")
    print(f"Traders: ${trader_profits:,.2f}")
    print(f"Stakers: ${staker_profits:,.2f}")

if __name__ == "__main__":
    main() 