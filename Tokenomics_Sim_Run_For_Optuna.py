from TokenEconModel import TokenEconomyModel

# @title Run Tokenomics Simulation (for Optuna integration)
def run_tokenomics_simulation(params):
    """Runs the tokenomics simulation for a given parameter set and returns the model and final stats."""
    model = TokenEconomyModel(params)

    while model.running:
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    # Compute outputs of interest
    try:
        final_token_price = model_data['TokenPrice'].iloc[-1]
        price_change = (final_token_price / model_data['TokenPrice'].iloc[0] - 1) * 100
        final_storage = model_data['TotalNetworkStorage'].iloc[-1]
        final_miners = model_data['ActiveMiners'].iloc[-1]

        return {
            'final_price': final_token_price,
            'price_change': price_change,
            'network_storage': final_storage,
            'active_miners': final_miners,
            'model_data': model_data,
            'agent_data': agent_data,
            'model': model
        }
    except Exception as e:
        print(f"Error computing final results: {str(e)}")
        return {
            'final_price': 0,
            'price_change': -100,
            'network_storage': 0,
            'active_miners': 0
        }
