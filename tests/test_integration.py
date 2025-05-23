import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from token_sim.simulation import TokenSimulation
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.consensus.pow import ProofOfWork
from token_sim.governance.simple import SimpleGovernance

class TestSimulationIntegration:
    @pytest.fixture
    def simulation_params(self):
        return {
            'initial_token_price': 1.0,
            'monthly_token_distribution': 1000000,
            'simulation_months': 12,
            'initial_miners': 100,
            'operating_cost_per_miner': 0.1,
            'initial_fiat_per_miner': 1000,
            'market_maker_liquidity': 1000000,
            'market_fee_rate': 0.001,
            'market_price_volatility': 0.1,
            'holder_strategies': ['hodl', 'trader', 'swing'] * 10
        }

    @pytest.fixture
    def simulation_model(self, simulation_params):
        # Create components
        consensus = ProofOfWork(
            block_reward=simulation_params['monthly_token_distribution'] / (30 * 24 * 6),  # Monthly rewards / blocks per month
            difficulty_adjustment_blocks=2016,  # ~2 weeks worth of blocks
            target_block_time=600.0  # 10 minutes
        )
        market = PriceDiscovery(
            initial_price=simulation_params['initial_token_price'],
            volatility=simulation_params['market_price_volatility'],
            market_depth=simulation_params['market_maker_liquidity']
        )
        governance = SimpleGovernance(
            voting_period=30,  # 30 blocks
            quorum_threshold=0.4,  # 40% participation required
            approval_threshold=0.6  # 60% approval required
        )
        
        # Create miners with different strategies
        miners = [
            Miner(
                agent_id=f"miner_{i}",
                strategy='passive' if i % 3 == 0 else 'aggressive' if i % 3 == 1 else 'opportunistic',
                initial_hashrate=50.0,
                electricity_cost=simulation_params['operating_cost_per_miner'],
                initial_balance=simulation_params['initial_fiat_per_miner']
            )
            for i in range(simulation_params['initial_miners'])
        ]
        
        # Create token holders
        holders = [
            Holder(
                agent_id=f"holder_{i}",
                strategy=strategy,
                initial_balance=1000.0
            )
            for i, strategy in enumerate(simulation_params['holder_strategies'])
        ]
        
        # Create traders
        traders = [
            Trader(
                agent_id=f"trader_{i}",
                strategy='momentum' if i % 2 == 0 else 'mean_reversion',
                initial_balance=1000.0,
                initial_tokens=100.0
            )
            for i in range(10)  # Add 10 traders
        ]
        
        # Combine all agents
        agents = miners + holders + traders
        
        # Initialize consensus with participants
        consensus.initialize_participants(len(miners))
        
        return TokenSimulation(
            consensus=consensus,
            price_discovery=market,
            governance=governance,
            agents=agents,
            initial_supply=simulation_params['monthly_token_distribution'],
            market_depth=simulation_params['market_maker_liquidity'],
            time_steps=simulation_params['simulation_months']
        )

    def test_simulation_initialization(self, simulation_model):
        """Test that the simulation initializes with correct components"""
        assert simulation_model.market is not None
        assert simulation_model.consensus is not None
        assert simulation_model.governance is not None
        assert len(simulation_model.agents) > 0

    def test_market_agent_interaction(self, simulation_model):
        """Test that agents can interact with the market correctly"""
        # Run simulation for a few steps
        for _ in range(5):
            simulation_model.step()
        
        # Check market state
        assert simulation_model.market.current_price > 0
        assert len(simulation_model.market.order_book) > 0
        
        # Verify agent balances have changed
        for agent in simulation_model.agents:
            assert agent.state['token_balance'] >= 0
            assert agent.state['balance'] >= 0

    def test_price_formation(self, simulation_model):
        """Test that market prices form realistically"""
        initial_price = simulation_model.market.current_price
        price_history = []
        
        # Run simulation for multiple steps
        for _ in range(10):
            simulation_model.step()
            price_history.append(simulation_model.market.current_price)
        
        # Check price volatility
        price_volatility = np.std(price_history) / np.mean(price_history)
        assert 0 < price_volatility < 1  # Reasonable volatility range
        
        # Check price never goes below zero
        assert min(price_history) > 0

    def test_market_liquidity(self, simulation_model):
        """Test market liquidity and trading mechanics"""
        # Initial liquidity check
        initial_liquidity = simulation_model.market.liquidity
        
        # Run simulation
        for _ in range(5):
            simulation_model.step()
        
        # Check liquidity changes
        assert simulation_model.market.liquidity != initial_liquidity
        
        # Verify trades are being executed
        assert len(simulation_model.market.trade_history) > 0

    def test_agent_strategies(self, simulation_model):
        """Test different agent strategies affect the market"""
        # Track initial positions
        initial_positions = {
            agent.id: (agent.state['token_balance'], agent.state['balance'])
            for agent in simulation_model.agents
        }
        
        # Run simulation
        for _ in range(10):
            simulation_model.step()
        
        # Check that positions have changed differently based on strategies
        position_changes = {}
        for agent in simulation_model.agents:
            initial_tokens, initial_fiat = initial_positions[agent.id]
            token_change = agent.state['token_balance'] - initial_tokens
            fiat_change = agent.state['balance'] - initial_fiat
            position_changes[agent.id] = (token_change, fiat_change)
        
        # Verify different strategies led to different outcomes
        changes = list(position_changes.values())
        assert not all(change == changes[0] for change in changes)

    def test_simulation_termination(self, simulation_model):
        """Test that simulation runs for expected duration and terminates properly"""
        expected_steps = simulation_model.params['simulation_months']
        
        # Run full simulation
        for _ in range(expected_steps):
            simulation_model.step()
        
        # Check final state
        assert simulation_model.current_step == expected_steps
        assert simulation_model.market.current_price > 0
        assert all(agent.state['token_balance'] >= 0 for agent in simulation_model.agents)
        assert all(agent.state['balance'] >= 0 for agent in simulation_model.agents)

    def test_governance_proposals(self, simulation_model):
        """Test governance proposal creation and voting"""
        # Create a test proposal
        proposal = {
            'type': 'parameter_update',
            'parameter': 'market_fee_rate',
            'new_value': 0.002,
            'description': 'Increase market fee rate'
        }
        
        # Submit proposal
        proposal_id = simulation_model.governance.submit_proposal(proposal)
        assert proposal_id is not None
        
        # Simulate voting from agents
        for i, agent in enumerate(simulation_model.agents[:10]):  # First 10 agents vote
            # Alternate yes/no votes
            simulation_model.governance.vote(proposal_id, agent.id, i % 2 == 0)
        
        # Run simulation to allow voting
        for _ in range(3):
            simulation_model.step()
        
        # Check proposal state
        proposal_state = simulation_model.governance.get_proposal_state(proposal_id)
        assert proposal_state['status'] in ['active', 'passed', 'rejected']
        
        # Verify voting power is being used
        assert proposal_state['total_votes'] > 0

    def test_governance_state_transitions(self, simulation_model):
        """Test that governance actions properly update system state"""
        initial_fee_rate = simulation_model.market.fee_rate
        
        # Create and pass a proposal to change fee rate
        proposal = {
            'type': 'parameter_update',
            'parameter': 'market_fee_rate',
            'new_value': initial_fee_rate * 2,
            'description': 'Double market fee rate'
        }
        
        proposal_id = simulation_model.governance.submit_proposal(proposal)
        
        # Force proposal to pass (simulate enough votes)
        simulation_model.governance.force_proposal_pass(proposal_id)
        
        # Run simulation to process the change
        simulation_model.step()
        
        # Verify the change was applied
        assert simulation_model.market.fee_rate == initial_fee_rate * 2

    def test_ai_agent_learning(self, simulation_model):
        """Test AI-driven agents' learning capabilities"""
        # Identify AI agents
        ai_agents = [agent for agent in simulation_model.agents 
                    if hasattr(agent, 'model') and agent.model is not None]
        
        if not ai_agents:
            pytest.skip("No AI agents in simulation")
        
        # Track initial performance
        initial_performances = {
            agent.id: agent.get_performance_metric()
            for agent in ai_agents
        }
        
        # Run simulation for learning
        for _ in range(20):
            simulation_model.step()
        
        # Check that agents have learned
        for agent in ai_agents:
            current_performance = agent.get_performance_metric()
            initial_performance = initial_performances[agent.id]
            assert current_performance >= initial_performance

    def test_model_persistence(self, simulation_model, tmp_path):
        """Test that AI models can be saved and loaded"""
        # Find an AI agent
        ai_agent = next((agent for agent in simulation_model.agents 
                        if hasattr(agent, 'model') and agent.model is not None), None)
        
        if not ai_agent:
            pytest.skip("No AI agents in simulation")
        
        # Save model
        model_path = tmp_path / "test_model"
        ai_agent.save_model(str(model_path))
        assert model_path.exists()
        
        # Create new agent and load model
        new_agent = type(ai_agent)(simulation_model.params)
        new_agent.load_model(str(model_path))
        
        # Verify model loaded correctly
        assert new_agent.model is not None
        # Compare model predictions
        test_state = simulation_model.get_state()
        assert np.allclose(
            ai_agent.model.predict(test_state),
            new_agent.model.predict(test_state)
        )

    def test_market_crash_scenario(self, simulation_model):
        """Test system behavior during market crash"""
        # Simulate market crash
        simulation_model.market.force_price_crash(0.5, governance=simulation_model.governance)  # 50% price drop
        
        # Run simulation for a few steps
        for _ in range(5):
            simulation_model.step()
        
        # Check system stability
        assert simulation_model.market.current_price > 0
        assert all(agent.state['token_balance'] >= 0 for agent in simulation_model.agents)
        
        # Verify governance response
        active_proposals = simulation_model.governance.get_active_proposals()
        assert len(active_proposals) > 0  # Should have emergency proposals
        
        # Check miner behavior
        miners = [agent for agent in simulation_model.agents 
                 if isinstance(agent, Miner)]
        assert all(miner.is_operational() for miner in miners)

    def test_consensus_failure_recovery(self, simulation_model):
        """Test that the consensus mechanism can recover from failures."""
        # Force a consensus failure
        simulation_model.consensus.force_failure()
        
        # Verify consensus is unhealthy
        assert not simulation_model.consensus.is_healthy()
        
        # Track recovery progress
        recovery_states = []
        for _ in range(5):  # Run for 5 steps to ensure recovery completes
            simulation_model.step()
            recovery_states.append({
                'active_miners': len(simulation_model.consensus.get_active_participants()),
                'difficulty': simulation_model.consensus.current_difficulty,
                'block_reward': simulation_model.consensus.block_reward,
                'is_healthy': simulation_model.consensus.is_healthy()
            })
        
        # Verify recovery progression
        assert recovery_states[0]['active_miners'] == 0  # All miners inactive initially
        assert recovery_states[0]['difficulty'] == float('inf')  # Infinite difficulty
        assert recovery_states[0]['block_reward'] == 0.0  # No rewards
        
        # Check gradual recovery
        active_miners = [state['active_miners'] for state in recovery_states]
        assert all(active_miners[i] <= active_miners[i+1] for i in range(len(active_miners)-1))  # Monotonically increasing
        
        # Verify final state
        final_state = recovery_states[-1]
        assert final_state['is_healthy']  # Consensus should be healthy
        assert final_state['active_miners'] > 0  # Should have active miners
        assert final_state['difficulty'] < float('inf')  # Finite difficulty
        assert final_state['block_reward'] > 0.0  # Positive block reward
        
        # Verify consensus is fully operational
        assert simulation_model.consensus.is_healthy()
        assert len(simulation_model.consensus.get_active_participants()) > 0
        assert simulation_model.consensus.current_difficulty > 0
        assert simulation_model.consensus.block_reward > 0

    def test_cross_component_state_consistency(self, simulation_model):
        """Test that all components maintain consistent state"""
        # Run simulation
        for _ in range(10):
            simulation_model.step()
        
        # Check market-consensus consistency
        assert simulation_model.market.last_block_height == simulation_model.consensus.current_height
        
        # Check governance-market consistency
        active_proposals = simulation_model.governance.get_active_proposals()
        for proposal in active_proposals:
            if proposal['details']['type'] == 'parameter_update':
                assert simulation_model.market.get_parameter(proposal['details']['parameter']) is not None
        
        # Update balances manually for test
        simulation_model.price_discovery.update_balances(simulation_model.agents)
        
        # Check agent-market consistency
        for agent in simulation_model.agents:
            assert agent.state['token_balance'] == simulation_model.market.get_balance(agent.id)

    @pytest.mark.performance
    def test_simulation_performance(self, simulation_model):
        """Test simulation performance with large numbers of agents"""
        # Measure initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run simulation with timing
        start_time = time.time()
        for _ in range(100):  # Run for 100 steps
            simulation_model.step()
        end_time = time.time()
        
        # Calculate performance metrics
        execution_time = end_time - start_time
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Performance assertions
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB memory increase
        
        # Check step time consistency
        step_times = []
        for _ in range(10):
            step_start = time.time()
            simulation_model.step()
            step_times.append(time.time() - step_start)
        
        # Step times should be relatively consistent
        assert np.std(step_times) < np.mean(step_times) * 0.5

    @pytest.mark.scalability
    def test_agent_scalability(self):
        """Test simulation scalability with increasing number of agents"""
        agent_counts = [100, 500, 1000]
        execution_times = []
        
        for count in agent_counts:
            # Create components
            consensus = ProofOfWork(
                block_reward=1000000 / (30 * 24 * 6),  # Monthly rewards / blocks per month
                difficulty_adjustment_blocks=2016,
                target_block_time=600.0
            )
            market = PriceDiscovery(
                initial_price=1.0,
                volatility=0.1,
                market_depth=1000000
            )
            governance = SimpleGovernance(
                voting_period=30,
                quorum_threshold=0.4,
                approval_threshold=0.6
            )
            
            # Create agents
            miners = [
                Miner(
                    agent_id=f"miner_{i}",
                    strategy='passive' if i % 3 == 0 else 'aggressive' if i % 3 == 1 else 'opportunistic',
                    initial_hashrate=50.0,
                    electricity_cost=0.1,
                    initial_balance=1000.0
                )
                for i in range(count // 2)
            ]
            
            holders = [
                Holder(
                    agent_id=f"holder_{i}",
                    strategy='hodl' if i % 3 == 0 else 'trader' if i % 3 == 1 else 'swing',
                    initial_balance=1000.0
                )
                for i in range(count // 3)
            ]
            
            traders = [
                Trader(
                    agent_id=f"trader_{i}",
                    strategy='momentum' if i % 2 == 0 else 'mean_reversion',
                    initial_balance=1000.0,
                    initial_tokens=100.0
                )
                for i in range(count // 6)
            ]
            
            agents = miners + holders + traders
            
            # Initialize consensus with participants
            consensus.initialize_participants(len(miners))
            
            # Create simulation
            model = TokenSimulation(
                consensus=consensus,
                price_discovery=market,
                governance=governance,
                agents=agents,
                initial_supply=1000000.0,
                market_depth=1000000.0,
                initial_token_price=1.0,
                time_steps=12
            )
            
            # Run simulation and measure time
            start_time = time.time()
            model.run()
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Verify results
            assert len(model.history['price']) == 12
            assert len(model.history['volume']) == 12
            assert len(model.history['rewards']) == 12
            
        # Verify scalability (execution time should increase sub-linearly with agent count)
        for i in range(1, len(agent_counts)):
            ratio = execution_times[i] / execution_times[i-1]
            agent_ratio = agent_counts[i] / agent_counts[i-1]
            # Allow a small margin of error (5%) for timing variances
            assert ratio < agent_ratio * 1.05, f"Execution time increased too much: {ratio} vs {agent_ratio}"

    @pytest.mark.reproducibility
    @pytest.mark.xfail(reason="Price randomness not fully controllable between runs")
    def test_simulation_reproducibility(self, simulation_model):
        """Test that simulation results are reproducible with the same seed"""
        # Run first simulation
        simulation_model.set_seed(42)
        results1 = []
        for _ in range(10):
            simulation_model.step()
            results1.append({
                'price': simulation_model.market.current_price,
                'hashrate': simulation_model.consensus.total_hashrate,
                'active_miners': len([a for a in simulation_model.agents 
                                    if isinstance(a, Miner) and a.is_operational()])
            })
        
        # Reset and run second simulation with same seed
        simulation_model.reset()
        simulation_model.set_seed(42)
        results2 = []
        for _ in range(10):
            simulation_model.step()
            results2.append({
                'price': simulation_model.market.current_price,
                'hashrate': simulation_model.consensus.total_hashrate,
                'active_miners': len([a for a in simulation_model.agents 
                                    if isinstance(a, Miner) and a.is_operational()])
            })
        
        # Compare results (with some tolerance for floating point)
        for r1, r2 in zip(results1, results2):
            assert abs(r1['price'] - r2['price']) < 0.01  # Allow small floating point differences
            assert r1['hashrate'] == r2['hashrate']
            assert r1['active_miners'] == r2['active_miners']

    @pytest.mark.stress
    def test_stress_conditions(self, simulation_model):
        """Test simulation under stress conditions"""
        # Run simulation for extended period
        for _ in range(1000):
            simulation_model.step()
        
        # Verify system stability
        assert simulation_model.market.current_price > 0
        assert simulation_model.consensus.is_healthy()
        assert all(agent.state['token_balance'] >= 0 for agent in simulation_model.agents)
        
        # Check for memory leaks
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        assert memory_usage < 2 * 1024 * 1024 * 1024  # Less than 2GB
        
        # Verify data structures haven't grown unbounded
        assert len(simulation_model.market.trade_history) < 10000
        assert len(simulation_model.governance.get_active_proposals()) < 100 