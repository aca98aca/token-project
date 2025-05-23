from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path

from .simulation_controller import SimulationConfig
from ..network.enhanced_network_model import ConsensusType

@dataclass
class MarketScenario:
    """Market scenario configuration."""
    initial_price: float
    market_depth: float
    volatility: float
    liquidity_profile: Dict[str, float]  # e.g., {'spread': 0.001, 'depth': 1000}
    trading_activity: Dict[str, float]  # e.g., {'base_volume': 1000, 'volume_volatility': 0.2}

@dataclass
class NetworkScenario:
    """Network scenario configuration."""
    consensus_type: ConsensusType
    num_miners: int
    base_hashrate: float
    block_time: int
    difficulty: float
    min_stake: float
    network_growth: Dict[str, float]  # e.g., {'growth_rate': 0.1, 'volatility': 0.05}

@dataclass
class SimulationScenario:
    """Complete simulation scenario configuration."""
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    time_step: timedelta
    market: MarketScenario
    network: NetworkScenario
    random_seed: Optional[int] = None

class ScenarioManager:
    """Manager for simulation scenarios."""
    
    def __init__(self, scenarios_dir: str = "scenarios"):
        """Initialize the scenario manager.
        
        Args:
            scenarios_dir: Directory to store scenario configurations
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios: Dict[str, SimulationScenario] = {}
    
    def create_scenario(self, scenario: SimulationScenario) -> None:
        """Create a new simulation scenario.
        
        Args:
            scenario: Simulation scenario configuration
        """
        self.scenarios[scenario.name] = scenario
        self._save_scenario(scenario)
    
    def load_scenario(self, name: str) -> SimulationScenario:
        """Load a simulation scenario by name.
        
        Args:
            name: Name of the scenario to load
            
        Returns:
            Loaded simulation scenario
        """
        if name not in self.scenarios:
            self._load_scenario(name)
        return self.scenarios[name]
    
    def list_scenarios(self) -> List[str]:
        """List all available scenarios.
        
        Returns:
            List of scenario names
        """
        return list(self.scenarios.keys())
    
    def delete_scenario(self, name: str) -> None:
        """Delete a simulation scenario.
        
        Args:
            name: Name of the scenario to delete
        """
        if name in self.scenarios:
            del self.scenarios[name]
            scenario_file = self.scenarios_dir / f"{name}.yaml"
            if scenario_file.exists():
                scenario_file.unlink()
    
    def to_simulation_config(self, scenario: SimulationScenario) -> SimulationConfig:
        """Convert a scenario to a simulation configuration.
        
        Args:
            scenario: Simulation scenario to convert
            
        Returns:
            Simulation configuration
        """
        return SimulationConfig(
            start_time=scenario.start_time,
            end_time=scenario.end_time,
            time_step=scenario.time_step,
            initial_price=scenario.market.initial_price,
            market_depth=scenario.market.market_depth,
            volatility=scenario.market.volatility,
            consensus_type=scenario.network.consensus_type,
            num_miners=scenario.network.num_miners,
            base_hashrate=scenario.network.base_hashrate,
            block_time=scenario.network.block_time,
            difficulty=scenario.network.difficulty,
            min_stake=scenario.network.min_stake,
            random_seed=scenario.random_seed
        )
    
    def _save_scenario(self, scenario: SimulationScenario) -> None:
        """Save a scenario to disk.
        
        Args:
            scenario: Scenario to save
        """
        scenario_data = {
            'name': scenario.name,
            'description': scenario.description,
            'start_time': scenario.start_time.isoformat(),
            'end_time': scenario.end_time.isoformat(),
            'time_step': scenario.time_step,
            'market': {
                'initial_price': scenario.market.initial_price,
                'market_depth': scenario.market.market_depth,
                'volatility': scenario.market.volatility,
                'liquidity_profile': scenario.market.liquidity_profile,
                'trading_activity': scenario.market.trading_activity
            },
            'network': {
                'consensus_type': scenario.network.consensus_type.value,
                'num_miners': scenario.network.num_miners,
                'base_hashrate': scenario.network.base_hashrate,
                'block_time': scenario.network.block_time,
                'difficulty': scenario.network.difficulty,
                'min_stake': scenario.network.min_stake,
                'network_growth': scenario.network.network_growth
            },
            'random_seed': scenario.random_seed
        }
        
        scenario_file = self.scenarios_dir / f"{scenario.name}.yaml"
        with open(scenario_file, 'w') as f:
            yaml.dump(scenario_data, f)
    
    def _load_scenario(self, name: str) -> None:
        """Load a scenario from disk.
        
        Args:
            name: Name of the scenario to load
        """
        scenario_file = self.scenarios_dir / f"{name}.yaml"
        if not scenario_file.exists():
            raise ValueError(f"Scenario '{name}' not found")
        
        with open(scenario_file, 'r') as f:
            scenario_data = yaml.safe_load(f)
        
        scenario = SimulationScenario(
            name=scenario_data['name'],
            description=scenario_data['description'],
            start_time=datetime.fromisoformat(scenario_data['start_time']),
            end_time=datetime.fromisoformat(scenario_data['end_time']),
            time_step=timedelta(seconds=scenario_data['time_step']),
            market=MarketScenario(
                initial_price=scenario_data['market']['initial_price'],
                market_depth=scenario_data['market']['market_depth'],
                volatility=scenario_data['market']['volatility'],
                liquidity_profile=scenario_data['market']['liquidity_profile'],
                trading_activity=scenario_data['market']['trading_activity']
            ),
            network=NetworkScenario(
                consensus_type=ConsensusType(scenario_data['network']['consensus_type']),
                num_miners=scenario_data['network']['num_miners'],
                base_hashrate=scenario_data['network']['base_hashrate'],
                block_time=scenario_data['network']['block_time'],
                difficulty=scenario_data['network']['difficulty'],
                min_stake=scenario_data['network']['min_stake'],
                network_growth=scenario_data['network']['network_growth']
            ),
            random_seed=scenario_data.get('random_seed')
        )
        
        self.scenarios[name] = scenario
    
    def create_default_scenarios(self) -> None:
        """Create a set of default scenarios for common use cases."""
        # Bull market scenario
        bull_market = SimulationScenario(
            name="bull_market",
            description="Simulation of a bull market with high trading activity",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=30),
            time_step=timedelta(hours=1),
            market=MarketScenario(
                initial_price=50000.0,
                market_depth=1000000.0,
                volatility=0.02,
                liquidity_profile={'spread': 0.001, 'depth': 2000},
                trading_activity={'base_volume': 2000, 'volume_volatility': 0.3}
            ),
            network=NetworkScenario(
                consensus_type=ConsensusType.POW,
                num_miners=1000,
                base_hashrate=100.0,
                block_time=600,
                difficulty=1000000.0,
                min_stake=0.0,
                network_growth={'growth_rate': 0.15, 'volatility': 0.1}
            )
        )
        self.create_scenario(bull_market)
        
        # Bear market scenario
        bear_market = SimulationScenario(
            name="bear_market",
            description="Simulation of a bear market with low trading activity",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=30),
            time_step=timedelta(hours=1),
            market=MarketScenario(
                initial_price=30000.0,
                market_depth=500000.0,
                volatility=0.03,
                liquidity_profile={'spread': 0.002, 'depth': 1000},
                trading_activity={'base_volume': 1000, 'volume_volatility': 0.4}
            ),
            network=NetworkScenario(
                consensus_type=ConsensusType.POW,
                num_miners=800,
                base_hashrate=80.0,
                block_time=600,
                difficulty=800000.0,
                min_stake=0.0,
                network_growth={'growth_rate': 0.05, 'volatility': 0.15}
            )
        )
        self.create_scenario(bear_market)
        
        # PoS transition scenario
        pos_transition = SimulationScenario(
            name="pos_transition",
            description="Simulation of transition from PoW to PoS",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=30),
            time_step=timedelta(hours=1),
            market=MarketScenario(
                initial_price=40000.0,
                market_depth=750000.0,
                volatility=0.025,
                liquidity_profile={'spread': 0.0015, 'depth': 1500},
                trading_activity={'base_volume': 1500, 'volume_volatility': 0.35}
            ),
            network=NetworkScenario(
                consensus_type=ConsensusType.POS,
                num_miners=500,
                base_hashrate=0.0,
                block_time=300,
                difficulty=0.0,
                min_stake=1000.0,
                network_growth={'growth_rate': 0.1, 'volatility': 0.12}
            )
        )
        self.create_scenario(pos_transition) 