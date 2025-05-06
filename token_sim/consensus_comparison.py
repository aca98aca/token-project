import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import random
from token_sim.consensus.pow import ProofOfWork
from token_sim.consensus.pos import ProofOfStake
from token_sim.consensus.dpos import DelegatedProofOfStake
from token_sim.agents.miner import Miner
from token_sim.agents.holder import Holder
from token_sim.agents.trader import Trader
from token_sim.agents.staker import Staker
from token_sim.market.price_discovery import PriceDiscovery
from token_sim.simulation import TokenSimulation

class ConsensusComparison:
    """Compare different consensus mechanisms and tokenomics parameters."""
    
    def __init__(self, num_simulations: int = 20):
        self.num_simulations = num_simulations
        self.results = {
            'pow': [],
            'pos': [],
            'dpos': []
        }
        
        # Tokenomics parameters to test
        self.tokenomics_params = {
            'initial_price': [0.1, 1.0, 10.0],
            'initial_supply': [1000000, 10000000, 100000000],
            'launch_type': ['ico', 'ido', 'fair_launch'],
            'vesting_schedule': [
                {'linear': 12},  # 12 months linear
                {'cliff': 6, 'linear': 12},  # 6 months cliff + 12 months linear
                {'tranches': [25, 25, 25, 25]}  # 4 equal tranches
            ],
            'dao_mechanism': ['token_weighted', 'quadratic', 'conviction']
        }
    
    def _create_agents(self, consensus_type: str) -> List[Any]:
        """Create agents based on consensus type."""
        agents = []
        
        # Create miners/validators
        if consensus_type == 'pow':
            for i in range(100):
                agents.append(Miner(
                    agent_id=f"miner_{i}",
                    strategy=random.choice(['efficient', 'aggressive', 'passive']),
                    initial_hashrate=random.uniform(50, 200),
                    electricity_cost=random.uniform(0.03, 0.08)
                ))
        elif consensus_type == 'pos':
            for i in range(50):
                agents.append(Staker(
                    agent_id=f"validator_{i}",
                    strategy=random.choice(['long_term', 'active', 'passive']),
                    initial_balance=random.uniform(10000, 100000)
                ))
        else:  # dpos
            for i in range(21):  # 21 delegates
                agents.append(Staker(
                    agent_id=f"delegate_{i}",
                    strategy='active',
                    initial_balance=random.uniform(50000, 200000)
                ))
        
        # Add traders and holders
        for i in range(50):
            agents.append(Trader(
                agent_id=f"trader_{i}",
                strategy=random.choice(['momentum', 'mean_reversion', 'random']),
                initial_balance=random.uniform(1000, 5000)
            ))
        
        for i in range(100):
            agents.append(Holder(
                agent_id=f"holder_{i}",
                strategy=random.choice(['long_term', 'medium_term', 'short_term']),
                initial_balance=random.uniform(100, 1000)
            ))
        
        return agents
    
    def _run_single_simulation(self,
                             consensus_type: str,
                             tokenomics_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single simulation with specific parameters."""
        # Create consensus mechanism
        if consensus_type == 'pow':
            consensus = ProofOfWork(
                block_reward=tokenomics_params.get('block_reward', 50),
                difficulty_adjustment_blocks=tokenomics_params.get('difficulty_adjustment_blocks', 2016),
                target_block_time=tokenomics_params.get('target_block_time', 600)
            )
        elif consensus_type == 'pos':
            consensus = ProofOfStake(
                block_reward=tokenomics_params.get('block_reward', 50),
                min_stake=tokenomics_params.get('min_stake', 1000),
                staking_apy=tokenomics_params.get('staking_apy', 0.05)
            )
        else:  # dpos
            consensus = DelegatedProofOfStake(
                block_reward=tokenomics_params.get('block_reward', 50),
                min_stake=tokenomics_params.get('min_stake', 1000),
                num_delegates=tokenomics_params.get('num_delegates', 21),
                staking_apy=tokenomics_params.get('staking_apy', 0.05)
            )
        
        # Create price discovery mechanism
        price_discovery = PriceDiscovery(
            initial_price=tokenomics_params.get('initial_price', 1.0),
            volatility=tokenomics_params.get('volatility', 0.1),
            market_depth=tokenomics_params.get('market_depth', 1000000.0)
        )
        
        # Create agents
        agents = self._create_agents(consensus_type)
        
        # Create and run simulation
        simulation = TokenSimulation(
            consensus=consensus,
            price_discovery=price_discovery,
            agents=agents,
            initial_supply=tokenomics_params.get('initial_supply', 1000000.0)
        )
        
        history = simulation.run()
        
        # Calculate metrics
        final_price = history['price'][-1]
        price_change = ((final_price - history['price'][0]) / history['price'][0]) * 100
        avg_volume = np.mean(history['volume'])
        price_volatility = np.std(history['price']) / np.mean(history['price'])
        
        # Calculate Gini coefficient for token distribution
        final_balances = []
        for agent in agents:
            if isinstance(agent, (Holder, Trader)):
                final_balances.append(agent.state['token_balance'])
            elif isinstance(agent, Staker):
                final_balances.append(agent.state['staked_amount'])
            elif isinstance(agent, Miner):
                final_balances.append(agent.state['total_rewards'])
        
        gini = self._calculate_gini(final_balances)
        
        # Calculate network security score
        security_score = self._calculate_security_score(consensus, history)
        
        # Convert vesting schedule dict to string for groupby
        vesting_schedule = tokenomics_params.get('vesting_schedule', {'linear': 12})
        vesting_schedule_str = str(vesting_schedule)
        
        return {
            'consensus_type': consensus_type,
            'initial_price': tokenomics_params.get('initial_price', 1.0),
            'initial_supply': tokenomics_params.get('initial_supply', 1000000.0),
            'launch_type': tokenomics_params.get('launch_type', 'fair_launch'),
            'vesting_schedule': vesting_schedule_str,
            'dao_mechanism': tokenomics_params.get('dao_mechanism', 'token_weighted'),
            'final_price': final_price,
            'price_change': price_change,
            'avg_volume': avg_volume,
            'price_volatility': price_volatility,
            'gini_coefficient': gini,
            'security_score': security_score
        }
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for token distribution."""
        values = sorted(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return ((2 * np.sum(index * values)) / (n * np.sum(values))) - (n + 1) / n
    
    def _calculate_security_score(self, consensus: Any, history: Dict[str, List]) -> float:
        """Calculate network security score based on consensus type."""
        if isinstance(consensus, ProofOfWork):
            # For PoW: based on hashrate and difficulty
            hashrate = sum(agent.state['hashrate'] for agent in consensus.get_active_participants())
            difficulty = consensus.current_difficulty
            return min(1.0, (hashrate / difficulty) * 100) if difficulty > 0 else 0.0
        else:
            # For PoS/DPoS: based on total stake and number of validators
            total_stake = sum(agent.state['stake'] for agent in consensus.get_active_participants())
            num_validators = len(consensus.get_active_participants())
            min_stake = consensus.min_stake
            
            # Safety checks
            if num_validators == 0 or min_stake == 0:
                return 0.0
                
            return min(1.0, (total_stake / (min_stake * num_validators)) * 100)
    
    def run_comparison(self) -> None:
        """Run comparison simulations for all consensus types."""
        print("\nRunning comprehensive tokenomics simulations...")
        
        for consensus_type in ['pow', 'pos', 'dpos']:
            print(f"\nRunning simulations for {consensus_type.upper()}...")
            
            for i in range(self.num_simulations):
                print(f"Simulation {i+1}/{self.num_simulations}")
                
                # Generate random tokenomics parameters
                params = {
                    'initial_price': random.choice(self.tokenomics_params['initial_price']),
                    'initial_supply': random.choice(self.tokenomics_params['initial_supply']),
                    'launch_type': random.choice(self.tokenomics_params['launch_type']),
                    'vesting_schedule': random.choice(self.tokenomics_params['vesting_schedule']),
                    'dao_mechanism': random.choice(self.tokenomics_params['dao_mechanism'])
                }
                
                result = self._run_single_simulation(consensus_type, params)
                self.results[consensus_type].append(result)
        
        self._analyze_results()
    
    def _analyze_results(self) -> None:
        """Analyze and visualize simulation results."""
        # Convert results to DataFrame
        all_results = []
        for consensus_type, results in self.results.items():
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        
        # Save results
        df.to_csv('tokenomics_simulation_results.csv', index=False)
        
        # Generate visualizations
        self._create_visualizations(df)
        
        # Generate recommendations
        self._generate_recommendations(df)
    
    def _create_visualizations(self, df: pd.DataFrame) -> None:
        """Create visualizations of simulation results."""
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 10))
        
        # Price stability comparison
        plt.subplot(2, 2, 1)
        sns.boxplot(x='consensus_type', y='price_volatility', data=df)
        plt.title('Price Stability Comparison')
        plt.xlabel('Consensus Type')
        plt.ylabel('Price Volatility')
        
        # Token distribution
        plt.subplot(2, 2, 2)
        sns.boxplot(x='consensus_type', y='gini_coefficient', data=df)
        plt.title('Token Distribution (Gini Coefficient)')
        plt.xlabel('Consensus Type')
        plt.ylabel('Gini Coefficient')
        
        # Network security
        plt.subplot(2, 2, 3)
        sns.boxplot(x='consensus_type', y='security_score', data=df)
        plt.title('Network Security Score')
        plt.xlabel('Consensus Type')
        plt.ylabel('Security Score')
        
        # Trading volume
        plt.subplot(2, 2, 4)
        sns.boxplot(x='consensus_type', y='avg_volume', data=df)
        plt.title('Average Trading Volume')
        plt.xlabel('Consensus Type')
        plt.ylabel('Volume')
        
        plt.tight_layout()
        plt.savefig('tokenomics_comparison.png')
        plt.close()
    
    def _generate_recommendations(self, df: pd.DataFrame) -> None:
        """Generate comprehensive tokenomics recommendations."""
        print("\n=== Tokenomics Recommendations ===")
        
        # Consensus mechanism recommendations
        print("\n1. Consensus Mechanism:")
        consensus_metrics = df.groupby('consensus_type').agg({
            'price_volatility': 'mean',
            'security_score': 'mean',
            'gini_coefficient': 'mean',
            'avg_volume': 'mean'
        }).round(4)
        
        print("\nConsensus Mechanism Metrics:")
        print(consensus_metrics)
        
        # Launch mechanism recommendations
        print("\n2. Launch Mechanism:")
        launch_metrics = df.groupby('launch_type').agg({
            'price_volatility': 'mean',
            'gini_coefficient': 'mean',
            'avg_volume': 'mean'
        }).round(4)
        
        print("\nLaunch Mechanism Metrics:")
        print(launch_metrics)
        
        # Vesting schedule recommendations
        print("\n3. Vesting Schedule:")
        vesting_metrics = df.groupby('vesting_schedule').agg({
            'price_volatility': 'mean',
            'gini_coefficient': 'mean'
        }).round(4)
        
        print("\nVesting Schedule Metrics:")
        print(vesting_metrics)
        
        # DAO mechanism recommendations
        print("\n4. DAO Mechanism:")
        dao_metrics = df.groupby('dao_mechanism').agg({
            'gini_coefficient': 'mean',
            'security_score': 'mean'
        }).round(4)
        
        print("\nDAO Mechanism Metrics:")
        print(dao_metrics)
        
        # Initial parameters recommendations
        print("\n5. Initial Parameters:")
        print("\nRecommended Initial Price Range:", 
              df[df['price_volatility'] < df['price_volatility'].quantile(0.25)]['initial_price'].mean())
        print("Recommended Initial Supply Range:", 
              df[df['gini_coefficient'] < df['gini_coefficient'].quantile(0.25)]['initial_supply'].mean())
        
        # Save recommendations to file
        with open('tokenomics_recommendations.txt', 'w') as f:
            f.write("=== Tokenomics Recommendations ===\n\n")
            f.write("1. Consensus Mechanism:\n")
            f.write(str(consensus_metrics) + "\n\n")
            f.write("2. Launch Mechanism:\n")
            f.write(str(launch_metrics) + "\n\n")
            f.write("3. Vesting Schedule:\n")
            f.write(str(vesting_metrics) + "\n\n")
            f.write("4. DAO Mechanism:\n")
            f.write(str(dao_metrics) + "\n\n")
            f.write("5. Initial Parameters:\n")
            f.write(f"Recommended Initial Price Range: {df[df['price_volatility'] < df['price_volatility'].quantile(0.25)]['initial_price'].mean()}\n")
            f.write(f"Recommended Initial Supply Range: {df[df['gini_coefficient'] < df['gini_coefficient'].quantile(0.25)]['initial_supply'].mean()}\n")

def main():
    comparison = ConsensusComparison(num_simulations=20)
    comparison.run_comparison()

if __name__ == "__main__":
    main() 