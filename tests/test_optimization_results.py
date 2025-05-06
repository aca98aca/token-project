import pytest
import json
import os
from datetime import datetime
from token_sim.optimization.utils.parameter_space import ParameterSpace

def test_save_optimization_results():
    """Test saving optimization results to a file."""
    # Create sample optimization results
    results = {
        'best_params': {
            'param1': 1.0,
            'param2': 2.0
        },
        'best_score': 0.5,
        'history': [
            {'params': {'param1': 1.1, 'param2': 2.1}, 'score': 0.6},
            {'params': {'param1': 1.0, 'param2': 2.0}, 'score': 0.5}
        ]
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'optimization_results_{timestamp}.json'
    
    # Save results
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Verify file exists
    assert os.path.exists(filename)
    
    # Load and verify results
    with open(filename, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results['best_params'] == results['best_params']
    assert loaded_results['best_score'] == results['best_score']
    assert len(loaded_results['history']) == len(results['history'])
    
    # Clean up
    os.remove(filename)

def test_load_optimization_results():
    """Test loading optimization results from a file."""
    # Create sample optimization results
    results = {
        'best_params': {
            'param1': 1.0,
            'param2': 2.0
        },
        'best_score': 0.5,
        'history': [
            {'params': {'param1': 1.1, 'param2': 2.1}, 'score': 0.6},
            {'params': {'param1': 1.0, 'param2': 2.0}, 'score': 0.5}
        ]
    }
    
    # Save results to a temporary file
    filename = 'temp_optimization_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Load results
    with open(filename, 'r') as f:
        loaded_results = json.load(f)
    
    # Verify loaded results
    assert loaded_results['best_params'] == results['best_params']
    assert loaded_results['best_score'] == results['best_score']
    assert len(loaded_results['history']) == len(results['history'])
    
    # Verify history entries
    for i, entry in enumerate(loaded_results['history']):
        assert entry['params'] == results['history'][i]['params']
        assert entry['score'] == results['history'][i]['score']
    
    # Clean up
    os.remove(filename)

def test_optimization_results_validation():
    """Test validation of optimization results structure."""
    # Create invalid results (missing required fields)
    invalid_results = {
        'best_params': {
            'param1': 1.0,
            'param2': 2.0
        }
        # Missing best_score and history
    }
    
    # Save invalid results
    filename = 'temp_invalid_results.json'
    with open(filename, 'w') as f:
        json.dump(invalid_results, f, indent=4)
    
    # Try to load and validate results
    with pytest.raises(KeyError):
        with open(filename, 'r') as f:
            loaded_results = json.load(f)
            # This should raise KeyError when trying to access missing fields
            score = loaded_results['best_score']
            history = loaded_results['history']
    
    # Clean up
    os.remove(filename) 