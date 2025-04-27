import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Union

class MetricsVisualizer:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics

    def create_time_distribution_pie(self, data: Union[Dict[str, float], pd.DataFrame]) -> go.Figure:
        """Create a pie chart showing the distribution of time across different tasks."""
        if isinstance(data, pd.DataFrame):
            labels = data['task_graph'].tolist()
            if 'total_task_graph_time' in data.columns:
                values = data['total_task_graph_time'].tolist()
            else:
                values = [0] * len(labels)
        else:
            labels = list(data.keys())
            values = list(data.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3
        )])
        
        fig.update_layout(
            title="Time Distribution Across Tasks",
            showlegend=True
        )
        
        return fig

    def create_power_usage_chart(self, data: Union[Dict[str, float], pd.DataFrame]) -> go.Figure:
        """Create a bar chart showing power usage across different tasks."""
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return go.Figure()
            
            # Calculate average and max power usage
            avg_power = data['power_usage'].mean() if 'power_usage' in data.columns else 0
            max_power = data['power_usage'].max() if 'power_usage' in data.columns else 0
            
            fig = go.Figure(data=[
                go.Bar(name='Average Power', x=['Power Usage'], y=[avg_power]),
                go.Bar(name='Max Power', x=['Power Usage'], y=[max_power])
            ])
        else:
            fig = go.Figure(data=[
                go.Bar(name='Power Usage', x=list(data.keys()), y=list(data.values()))
            ])
        
        fig.update_layout(
            title="Power Usage Analysis",
            barmode='group',
            yaxis_title="Power (W)",
            showlegend=True
        )
        
        return fig

    def create_memory_usage_chart(self, data: Union[Dict[str, float], pd.DataFrame]) -> go.Figure:
        """Create a bar chart showing memory usage across different tasks."""
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return go.Figure()
            
            # Calculate total and average memory usage in MB
            total_memory = data['copy_in_bytes'].sum() / (1024 * 1024)
            avg_memory = total_memory / len(data)
            
            fig = go.Figure(data=[
                go.Bar(name='Total Memory', x=['Memory Usage'], y=[total_memory]),
                go.Bar(name='Average Memory', x=['Memory Usage'], y=[avg_memory])
            ])
        else:
            fig = go.Figure(data=[
                go.Bar(name='Memory Usage', x=list(data.keys()), y=list(data.values()))
            ])
        
        fig.update_layout(
            title="Memory Usage Analysis",
            barmode='group',
            yaxis_title="Memory (MB)",
            showlegend=True
        )
        
        return fig

    def create_performance_metrics_dashboard(self, data: Union[Dict[str, float], pd.DataFrame], time_unit: str = 'ms') -> go.Figure:
        """Create a dashboard showing key performance metrics with units and formatting."""
        # Helper for formatting
        def fmt(val, unit=None, decimals=2):
            if val is None or pd.isna(val):
                return "N/A"
            if unit == '%':
                return f"{val:.2f}%"
            if unit:
                return f"{val:,.{decimals}f} {unit}"
            return f"{val:,.{decimals}f}"

        if isinstance(data, pd.DataFrame):
            if data.empty:
                return go.Figure()
            # Calculate metrics from DataFrame
            total_time = data['total_task_graph_time'].sum()
            avg_kernel_time = data['total_kernel_time'].mean() if 'total_kernel_time' in data.columns else 0
            avg_dispatch_time = data['total_dispatch_time'].mean() if 'total_dispatch_time' in data.columns else 0
            efficiency = (avg_kernel_time / total_time * 100) if total_time > 0 else 0
            metrics = {
                'Total Execution Time': (total_time, time_unit),
                'Average Kernel Time': (avg_kernel_time, time_unit),
                'Average Dispatch Time': (avg_dispatch_time, time_unit),
                'Efficiency (%)': (efficiency, '%')
            }
        else:
            metrics = self.metrics

        fig = go.Figure()
        for i, (metric, (value, unit)) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode="number",
                value=value,
                number={'valueformat': ',.2f' if unit != '%' else ',.2f'},
                title={'text': f"{metric} ({unit})" if unit != '%' else metric},
                domain={'row': 0, 'column': i}
            ))
        fig.update_layout(
            grid={'rows': 1, 'columns': len(metrics), 'pattern': "independent"},
            title="Performance Metrics Dashboard"
        )
        return fig 