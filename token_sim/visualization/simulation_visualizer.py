from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SimulationVisualizer:
    """Visualization tools for simulation results."""
    
    def __init__(self, results: Dict[str, pd.DataFrame]):
        """Initialize the visualizer with simulation results.
        
        Args:
            results: Dictionary containing market, network, and interaction DataFrames
        """
        self.results = results
        self.market_df = results['market']
        self.network_df = results['network']
        self.interactions_df = results['interactions']
        
        # Remove style setting
        # plt.style.use('seaborn')
        # sns.set_palette("husl")
    
    def plot_market_metrics(self, save_path: Optional[str] = None):
        """Plot key market metrics over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price', 'Volume', 'Spread', 'Market Depth')
        )
        
        # Price plot
        fig.add_trace(
            go.Scatter(
                x=self.market_df['timestamp'],
                y=self.market_df['price'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume plot
        fig.add_trace(
            go.Scatter(
                x=self.market_df['timestamp'],
                y=self.market_df['volume'],
                name='Volume'
            ),
            row=1, col=2
        )
        
        # Spread plot
        fig.add_trace(
            go.Scatter(
                x=self.market_df['timestamp'],
                y=self.market_df['spread'],
                name='Spread'
            ),
            row=2, col=1
        )
        
        # Market depth plot
        fig.add_trace(
            go.Scatter(
                x=self.market_df['timestamp'],
                y=self.market_df['depth'].apply(lambda x: sum(q for _, q in x['bids'])),
                name='Bid Depth'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=self.market_df['timestamp'],
                y=self.market_df['depth'].apply(lambda x: sum(q for _, q in x['asks'])),
                name='Ask Depth'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Market Metrics Over Time',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/market_metrics.html")
        return fig
    
    def plot_network_metrics(self, save_path: Optional[str] = None):
        """Plot key network metrics over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hashrate', 'Block Time', 'Participant Count', 'Stake Distribution')
        )
        
        # Hashrate plot
        fig.add_trace(
            go.Scatter(
                x=self.network_df['timestamp'],
                y=self.network_df['hashrate'],
                name='Hashrate'
            ),
            row=1, col=1
        )
        
        # Block time plot
        fig.add_trace(
            go.Scatter(
                x=self.network_df['timestamp'],
                y=self.network_df['block_time'],
                name='Block Time'
            ),
            row=1, col=2
        )
        
        # Participant count plot
        fig.add_trace(
            go.Scatter(
                x=self.network_df['timestamp'],
                y=self.network_df['participant_count'],
                name='Participants'
            ),
            row=2, col=1
        )
        
        # Stake distribution plot
        fig.add_trace(
            go.Scatter(
                x=self.network_df['timestamp'],
                y=self.network_df['total_stake'],
                name='Total Stake'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Network Metrics Over Time',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/network_metrics.html")
        return fig
    
    def plot_interactions(self, save_path: Optional[str] = None):
        """Plot market-network interactions."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Price Impact vs Network Growth',
                'Volume Impact vs Network Growth',
                'Price Impact Distribution',
                'Network Growth Distribution'
            )
        )
        
        # Price impact vs network growth
        fig.add_trace(
            go.Scatter(
                x=self.interactions_df['network_growth'],
                y=self.interactions_df['price_impact'],
                mode='markers',
                name='Price Impact'
            ),
            row=1, col=1
        )
        
        # Volume impact vs network growth
        fig.add_trace(
            go.Scatter(
                x=self.interactions_df['network_growth'],
                y=self.interactions_df['volume_impact'],
                mode='markers',
                name='Volume Impact'
            ),
            row=1, col=2
        )
        
        # Price impact distribution
        fig.add_trace(
            go.Histogram(
                x=self.interactions_df['price_impact'],
                name='Price Impact'
            ),
            row=2, col=1
        )
        
        # Network growth distribution
        fig.add_trace(
            go.Histogram(
                x=self.interactions_df['network_growth'],
                name='Network Growth'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Market-Network Interactions',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/interactions.html")
        return fig
    
    def plot_correlation_matrix(self, save_path: Optional[str] = None):
        """Plot correlation matrix of key metrics."""
        # Combine relevant metrics
        metrics_df = pd.concat([
            self.market_df[['price', 'volume', 'spread']],
            self.network_df[['hashrate', 'block_time', 'participant_count', 'total_stake']],
            self.interactions_df[['price_impact', 'volume_impact', 'network_growth']]
        ], axis=1)
        
        # Calculate correlation matrix
        corr_matrix = metrics_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Key Metrics',
            height=800,
            width=1000
        )
        
        if save_path:
            fig.write_html(f"{save_path}/correlation_matrix.html")
        return fig
    
    def plot_consensus_specific_metrics(self, save_path: Optional[str] = None):
        """Plot consensus-specific metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mining Power Distribution (Gini)',
                'Staking Ratio',
                'Validator Distribution (Gini)',
                'Voting Power Distribution (Gini)'
            )
        )
        
        # Add relevant metrics based on available data
        if 'miner_gini' in self.network_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.network_df['timestamp'],
                    y=self.network_df['miner_gini'],
                    name='Mining Power Gini'
                ),
                row=1, col=1
            )
        
        if 'staking_ratio' in self.network_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.network_df['timestamp'],
                    y=self.network_df['staking_ratio'],
                    name='Staking Ratio'
                ),
                row=1, col=2
            )
        
        if 'validator_gini' in self.network_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.network_df['timestamp'],
                    y=self.network_df['validator_gini'],
                    name='Validator Gini'
                ),
                row=2, col=1
            )
        
        if 'voting_power_gini' in self.network_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.network_df['timestamp'],
                    y=self.network_df['voting_power_gini'],
                    name='Voting Power Gini'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Consensus-Specific Metrics',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/consensus_metrics.html")
        return fig
    
    def create_dashboard(self, save_path: str):
        """Create a comprehensive dashboard of all visualizations.
        
        Args:
            save_path: Directory to save the dashboard
        """
        # Create all plots
        self.plot_market_metrics(save_path)
        self.plot_network_metrics(save_path)
        self.plot_interactions(save_path)
        self.plot_correlation_matrix(save_path)
        self.plot_consensus_specific_metrics(save_path)
        
        # Create index.html to link all plots
        with open(f"{save_path}/index.html", 'w') as f:
            f.write("""
            <html>
            <head>
                <title>Simulation Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    .plot-container { margin: 20px 0; }
                    iframe { border: none; width: 100%; height: 800px; }
                </style>
            </head>
            <body>
                <h1>Simulation Dashboard</h1>
                <div class="plot-container">
                    <h2>Market Metrics</h2>
                    <iframe src="market_metrics.html"></iframe>
                </div>
                <div class="plot-container">
                    <h2>Network Metrics</h2>
                    <iframe src="network_metrics.html"></iframe>
                </div>
                <div class="plot-container">
                    <h2>Market-Network Interactions</h2>
                    <iframe src="interactions.html"></iframe>
                </div>
                <div class="plot-container">
                    <h2>Correlation Matrix</h2>
                    <iframe src="correlation_matrix.html"></iframe>
                </div>
                <div class="plot-container">
                    <h2>Consensus-Specific Metrics</h2>
                    <iframe src="consensus_metrics.html"></iframe>
                </div>
            </body>
            </html>
            """) 