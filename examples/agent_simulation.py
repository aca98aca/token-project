from token_sim.simulation import TokenSimulation
from token_sim.consensus import ProofOfWork
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.agents import Miner, Trader, Holder, Staker

def run_agent_simulation():
    # Initialize consensus mechanism
    consensus = ProofOfWork(
        initial_difficulty=1.0,
        block_time=600,  # 10 minutes
        block_reward=1.0,
        halving_interval=210000
    )
    
    # Initialize price discovery
    price_discovery = PriceDiscovery(
        initial_price=1.0,
        volatility=0.1,
        market_depth=1000000.0
    )
    
    # Create agents
    agents = [
        # Miners
        Miner(agent_id="miner1", strategy="aggressive", initial_hashrate=100.0),
        Miner(agent_id="miner2", strategy="passive", initial_hashrate=50.0),
        
        # Traders
        Trader(agent_id="trader1", strategy="momentum", initial_balance=10000.0),
        Trader(agent_id="trader2", strategy="mean_reversion", initial_balance=5000.0),
        
        # Holders
        Holder(agent_id="holder1", initial_balance=1000.0),
        Holder(agent_id="holder2", initial_balance=2000.0),
        
        # Stakers
        Staker(agent_id="staker1", initial_balance=5000.0),
        Staker(agent_id="staker2", initial_balance=3000.0)
    ]
    
    # Create simulation
    simulation = TokenSimulation(
        consensus=consensus,
        price_discovery=price_discovery,
        agents=agents,
        initial_supply=1000000.0,
        market_depth=1000000.0,
        initial_token_price=1.0,
        time_steps=1000
    )
    
    # Run simulation
    history = simulation.run()
    
    # Print results
    print("\nSimulation Results:")
    print(f"Final Price: ${history['price'][-1]:.2f}")
    print(f"Total Volume: ${sum(history['volume']):.2f}")
    print(f"Active Miners: {history['active_miners'][-1]}")
    print(f"Network Hashrate: {history['network_hashrate'][-1]:.2f}")
    
    # Print agent performance
    print("\nAgent Performance:")
    for agent in agents:
        state = agent.get_state()
        print(f"\n{agent.agent_id}:")
        print(f"Total Profit: ${state['total_profit']:.2f}")
        print(f"Token Balance: {state['token_balance']:.2f}")
        if hasattr(agent, 'blocks_found'):
            print(f"Blocks Found: {state['blocks_found']}")
        if hasattr(agent, 'trades'):
            print(f"Trades Made: {state['trades']}")

if __name__ == "__main__":
    run_agent_simulation() 