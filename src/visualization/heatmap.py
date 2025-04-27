import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, List

class HeatmapVisualizer:
    def __init__(self, heatmap_data: pd.DataFrame, tasks: List[str], devices: List[str]):
        self.heatmap_data = heatmap_data
        self.tasks = tasks
        self.devices = devices
        
    def create_performance_heatmap(self) -> go.Figure:
        """Create a heatmap showing performance patterns across tasks and devices."""
        fig = go.Figure(data=go.Heatmap(
            z=self.heatmap_data.values,
            x=self.devices,
            y=self.tasks,
            colorscale='Viridis',
            text=[[f'{val:.1f}%' for val in row] for row in self.heatmap_data.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Performance Heatmap',
            xaxis_title='Devices',
            yaxis_title='Tasks',
            height=400
        )
        
        return fig
    
    def create_resource_utilization_heatmap(self, power_data: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing resource utilization patterns."""
        fig = go.Figure(data=go.Heatmap(
            z=power_data.values,
            x=power_data.columns,
            y=power_data.index,
            colorscale='RdBu',
            text=[[f'{val:.1f}mW' for val in row] for row in power_data.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Resource Utilization Heatmap',
            xaxis_title='Devices',
            yaxis_title='Tasks',
            height=400
        )
        
        return fig
    
    def create_compilation_time_heatmap(self, compilation_data: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing compilation time distribution."""
        fig = go.Figure(data=go.Heatmap(
            z=compilation_data.values,
            x=compilation_data.columns,
            y=compilation_data.index,
            colorscale='Plasma',
            text=[[f'{val/1e6:.1f}ms' for val in row] for row in compilation_data.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Compilation Time Distribution',
            xaxis_title='Devices',
            yaxis_title='Tasks',
            height=400
        )
        
        return fig 