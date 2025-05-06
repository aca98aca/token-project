import pprint
from token_sim.simulation import TokenSimulation
from token_sim.consensus.pow import ProofOfWork
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.governance.simple import SimpleGovernance
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader

# Initialize components
consensus = ProofOfWork(
    block_reward=100,
    difficulty_adjustment_blocks=10,
    target_block_time=1.0
)
market = PriceDiscovery(
    initial_price=1.0,
    volatility=0.1,
    market_depth=1000.0
)
governance = SimpleGovernance(
    voting_period=5,
    quorum_threshold=0.4,
    approval_threshold=0.6
)

# Create agents
miners = [
    Miner(
        agent_id=f"miner_{i}",
        strategy='passive',
        initial_hashrate=10.0,
        electricity_cost=0.01,
        initial_balance=100.0
    ) for i in range(10)
]
holders = [
    Holder(
        agent_id=f"holder_{i}",
        strategy='long_term',
        initial_balance=100.0,
        initial_tokens=10.0
    ) for i in range(5)
]
traders = [
    Trader(
        agent_id=f"trader_{i}",
        strategy='momentum',
        initial_balance=100.0,
        initial_tokens=10.0
    ) for i in range(5)
]
agents = miners + holders + traders

# Initialize consensus participants and simulation
consensus.initialize_participants(len(miners))
sim = TokenSimulation(
    consensus=consensus,
    price_discovery=market,
    governance=governance,
    agents=agents,
    initial_supply=10000.0,
    market_depth=1000.0,
    initial_token_price=1.0,
    time_steps=20
)

# Run simulation
history = sim.run()

# Print outcomes
print("Price history:")
pprint.pprint(history['price'])
print("Volume history:")
pprint.pprint(history['volume'])
print("Final agent token balances:")
balances = {agent.id: agent.state['token_balance'] for agent in agents}
pprint.pprint(balances) 