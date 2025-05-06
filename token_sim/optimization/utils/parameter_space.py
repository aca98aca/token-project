from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

class ConsensusType(Enum):
    POW = "pow"
    POS = "pos"
    DPOS = "dpos"

@dataclass
class ParameterBounds:
    """Define bounds and constraints for parameters."""
    min_value: float
    max_value: float
    default_value: float
    is_integer: bool = False
    is_enum: bool = False
    enum_values: List = None

class ParameterSpace:
    """Define the parameter space for tokenomics optimization."""
    
    def __init__(self):
        # Core Parameters (Level 1)
        self.core_params = {
            'consensus_type': ParameterBounds(
                min_value=0,
                max_value=2,
                default_value=0,
                is_enum=True,
                enum_values=[ConsensusType.POW, ConsensusType.POS, ConsensusType.DPOS]
            ),
            'block_reward': ParameterBounds(
                min_value=1.0,
                max_value=100.0,
                default_value=50.0,
                is_integer=False
            ),
            'initial_supply': ParameterBounds(
                min_value=100000.0,
                max_value=10000000.0,
                default_value=1000000.0,
                is_integer=False
            ),
            'market_depth': ParameterBounds(
                min_value=100000.0,
                max_value=10000000.0,
                default_value=1000000.0,
                is_integer=False
            )
        }
        
        # Network Parameters (Level 2)
        self.network_params = {
            'num_participants': ParameterBounds(
                min_value=10,
                max_value=1000,
                default_value=100,
                is_integer=True
            ),
            'network_capacity': ParameterBounds(
                min_value=1000.0,
                max_value=100000.0,
                default_value=10000.0,
                is_integer=False
            ),
            'block_time': ParameterBounds(
                min_value=60.0,
                max_value=600.0,
                default_value=300.0,
                is_integer=False
            ),
            'transaction_fee': ParameterBounds(
                min_value=0.0001,
                max_value=0.01,
                default_value=0.001,
                is_integer=False
            )
        }
        
        # Economic Parameters (Level 3)
        self.economic_params = {
            'price_volatility': ParameterBounds(
                min_value=0.01,
                max_value=0.5,
                default_value=0.1,
                is_integer=False
            ),
            'trading_fee': ParameterBounds(
                min_value=0.0001,
                max_value=0.01,
                default_value=0.001,
                is_integer=False
            ),
            'liquidity_ratio': ParameterBounds(
                min_value=0.1,
                max_value=0.9,
                default_value=0.5,
                is_integer=False
            ),
            'staking_apy': ParameterBounds(
                min_value=0.01,
                max_value=0.2,
                default_value=0.05,
                is_integer=False
            )
        }
        
        # Agent Parameters (Level 4)
        self.agent_params = {
            'initial_balance': ParameterBounds(
                min_value=100.0,
                max_value=10000.0,
                default_value=1000.0,
                is_integer=False
            ),
            'risk_tolerance': ParameterBounds(
                min_value=0.1,
                max_value=0.9,
                default_value=0.5,
                is_integer=False
            ),
            'learning_rate': ParameterBounds(
                min_value=0.001,
                max_value=0.1,
                default_value=0.01,
                is_integer=False
            )
        }
    
    def get_all_params(self) -> Dict[str, ParameterBounds]:
        """Get all parameters with their bounds."""
        return {
            **self.core_params,
            **self.network_params,
            **self.economic_params,
            **self.agent_params
        }
    
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds in format required by optimizers."""
        return {
            name: (param.min_value, param.max_value)
            for name, param in self.get_all_params().items()
        }
    
    def get_default_params(self) -> Dict[str, float]:
        """Get default parameter values."""
        return {
            name: param.default_value
            for name, param in self.get_all_params().items()
        }
    
    def validate_params(self, params: Dict[str, float]) -> bool:
        """Validate parameter values against bounds."""
        all_params = self.get_all_params()
        
        for name, value in params.items():
            if name not in all_params:
                raise ValueError(f"Unknown parameter: {name}")
            
            param = all_params[name]
            
            # Check bounds
            if value < param.min_value or value > param.max_value:
                return False
            
            # Check integer constraint
            if param.is_integer and not float(value).is_integer():
                return False
            
            # Check enum constraint
            if param.is_enum and value not in param.enum_values:
                return False
        
        return True
    
    def get_param_dependencies(self) -> Dict[str, List[str]]:
        """Get parameter dependencies."""
        return {
            'consensus_type': ['block_reward', 'staking_apy'],
            'block_reward': ['num_participants'],
            'market_depth': ['liquidity_ratio'],
            'price_volatility': ['risk_tolerance'],
            'staking_apy': ['initial_balance']
        } 