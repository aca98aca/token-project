import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from token_sim.simulation import TokenSimulation
from token_sim.consensus.pow import ProofOfWork
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.governance.simple import SimpleGovernance
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader
from token_sim.ai.agent_learning import TokenAgent, MarketPredictor

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
    ) for i in range(5)
]

# Add AI-powered miners
ai_miners = [
    Miner(
        agent_id=f"ai_miner_{i}",
        strategy='ai',
        initial_hashrate=15.0,
        electricity_cost=0.008,
        initial_balance=150.0
    ) for i in range(3)
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
    ) for i in range(3)
]

# Add AI-powered traders
ai_traders = [
    Trader(
        agent_id=f"ai_trader_{i}",
        strategy='ai',
        initial_balance=150.0,
        initial_tokens=15.0
    ) for i in range(2)
]

agents = miners + ai_miners + holders + traders + ai_traders

# Initialize consensus participants and simulation
consensus.initialize_participants(len(miners) + len(ai_miners))
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

# Initialize AI components
market_predictor = MarketPredictor()
ai_agents = []

# Initialize AI agents for miners and traders
for agent in agents:
    if agent.strategy == 'ai':
        ai_agent = TokenAgent(sim, agent_index=agents.index(agent))
        ai_agents.append(ai_agent)

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

# Create visualizations
plt.style.use('ggplot')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Price history plot
ax1.plot(history['price'], label='Token Price')
ax1.set_title('Token Price History')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Price')
ax1.grid(True)

# Volume history plot
ax2.plot(history['volume'], label='Trading Volume', color='orange')
ax2.set_title('Trading Volume History')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Volume')
ax2.grid(True)

# Token distribution by agent type
agent_types = ['Miner', 'AI Miner', 'Holder', 'Trader', 'AI Trader']
token_distribution = [
    sum(balances[agent.id] for agent in agents if agent.strategy == 'passive' and isinstance(agent, Miner)),
    sum(balances[agent.id] for agent in agents if agent.strategy == 'ai' and isinstance(agent, Miner)),
    sum(balances[agent.id] for agent in agents if isinstance(agent, Holder)),
    sum(balances[agent.id] for agent in agents if agent.strategy == 'momentum' and isinstance(agent, Trader)),
    sum(balances[agent.id] for agent in agents if agent.strategy == 'ai' and isinstance(agent, Trader))
]
ax3.bar(agent_types, token_distribution)
ax3.set_title('Token Distribution by Agent Type')
ax3.set_ylabel('Total Tokens')
ax3.grid(True)

# Network security score
security_scores = history.get('network_security', [])
if security_scores:
    ax4.plot(security_scores, label='Network Security', color='green')
    ax4.set_title('Network Security Score')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Security Score')
    ax4.grid(True)

plt.tight_layout()
plt.savefig('simulation_results.png')
plt.close()

# Calculate and print statistics
print("\nSimulation Statistics:")
print(f"Average Price: {np.mean(history['price']):.2f}")
print(f"Price Volatility: {np.std(history['price']):.2f}")
print(f"Total Trading Volume: {sum(history['volume']):.2f}")
print(f"Average Trading Volume: {np.mean(history['volume']):.2f}")
print(f"Final Network Security Score: {security_scores[-1] if security_scores else 'N/A'}")

# Print AI agent performance
print("\nAI Agent Performance:")
for ai_agent in ai_agents:
    agent = agents[ai_agent.agent_index]
    print(f"\n{agent.id}:")
    print(f"Final Token Balance: {agent.state['token_balance']:.2f}")
    print(f"Total Profit: {agent.state.get('total_profit', 0):.2f}") 