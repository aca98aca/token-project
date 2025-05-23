import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from token_sim.data.enhanced_historical_loader import EnhancedHistoricalLoader
from token_sim.market.enhanced_market_model import EnhancedMarketModel
from token_sim.network.enhanced_network_model import EnhancedNetworkModel, ConsensusType
from token_sim.simulation.simulation_controller import SimulationController, SimulationConfig
from token_sim.simulation.scenario_manager import ScenarioManager, SimulationScenario, MarketScenario, NetworkScenario
from token_sim.simulation.agent_models import AgentType, TradingStrategy, AgentState
from token_sim.risk.risk_analyzer import RiskAnalyzer, RiskType, RiskMetrics
from token_sim.optimization.performance_optimizer import PerformanceOptimizer, OptimizationStrategy
from token_sim.analysis.results_analyzer import ResultsAnalyzer, AnalysisType
from token_sim.visualization.simulation_visualizer import SimulationVisualizer

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('simulation.log')
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Token Simulation System')
    parser.add_argument('--scenario', type=str, default='default', help='Scenario name to run')
    parser.add_argument('--duration', type=int, default=30, help='Simulation duration in days')
    parser.add_argument('--consensus', type=str, default='PoW', choices=['PoW', 'PoS', 'DPoS'], help='Consensus mechanism')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    return parser.parse_args()

def create_default_scenario(
    duration: int = 30,
    consensus_type: str = "PoW"
) -> SimulationScenario:
    """Create a default simulation scenario.
    
    Args:
        duration: Simulation duration in days
        consensus_type: Consensus mechanism type (PoW or PoS)
        
    Returns:
        SimulationScenario: Default scenario configuration
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(days=duration)
    
    return SimulationScenario(
        name="default",
        description="Default simulation scenario",
        start_time=start_time,
        end_time=end_time,
        time_step=timedelta(hours=1),
        market=MarketScenario(
            initial_price=50000.0,
            market_depth=1000000.0,
            volatility=0.02,
            liquidity_profile={'spread': 0.001, 'depth': 2000},
            trading_activity={'base_volume': 2000, 'volume_volatility': 0.3}
        ),
        network=NetworkScenario(
            consensus_type=ConsensusType[consensus_type.upper()],
            num_miners=1000,
            base_hashrate=100.0,
            block_time=600,
            difficulty=1000000.0,
            min_stake=0.0,
            network_growth={'growth_rate': 0.15, 'volatility': 0.1}
        )
    )

def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(description="Token Economics Simulation")
    parser.add_argument("--scenario", type=str, default="default", help="Scenario name to run")
    parser.add_argument("--duration", type=int, default=30, help="Simulation duration in days")
    parser.add_argument("--consensus", type=str, default="PoW", choices=["PoW", "PoS"], help="Consensus mechanism type")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scenario manager
        scenario_manager = ScenarioManager()
        
        # Create default scenario if it doesn't exist
        if args.scenario == "default":
            scenario = create_default_scenario(args.duration, args.consensus)
            scenario_manager.create_scenario(scenario)
        else:
            scenario = scenario_manager.load_scenario(args.scenario)
        
        # Convert scenario to simulation config
        config = scenario_manager.to_simulation_config(scenario)
        
        # Initialize and run simulation
        controller = SimulationController(config)
        results = controller.run_simulation()
        
        # Save results
        results_file = output_dir / f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists before saving
        controller.save_results(str(output_dir))
        logger.info(f"Results saved to {output_dir}")
        
        # Format results for visualization
        formatted_results = {
            'market': pd.DataFrame(results['market']),
            'network': pd.DataFrame(results['network']),
            'interactions': pd.DataFrame(results['interactions'])
        }
        
        # Create visualizations
        visualizer = SimulationVisualizer(formatted_results)
        visualizer.create_dashboard(str(output_dir))
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 