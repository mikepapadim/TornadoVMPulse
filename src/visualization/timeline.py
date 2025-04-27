import plotly.figure_factory as ff
import plotly.graph_objects as go
from typing import List, Dict, Any

class TimelineVisualizer:
    def __init__(self, timeline_data: List[Dict[str, Any]]):
        self.timeline_data = timeline_data
        
    def create_gantt_chart(self) -> go.Figure:
        """Create a Gantt chart visualization of task execution."""
        df = []
        colors = {}
        
        for task in self.timeline_data:
            df.append(dict(
                Task=task['task'],
                Start=task['start_time'],
                Finish=task['start_time'] + task['duration'],
                Resource=task['device']
            ))
            
            # Create a color mapping for devices
            if task['device'] not in colors:
                colors[task['device']] = f'rgb({hash(task["device"]) % 255}, {(hash(task["device"]) + 85) % 255}, {(hash(task["device"]) + 170) % 255})'
        
        fig = ff.create_gantt(
            df,
            colors=colors,
            index_col='Resource',
            show_colorbar=True,
            group_tasks=True,
            showgrid_x=True,
            showgrid_y=True,
            height=400,
            title='Task Execution Timeline'
        )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title='Time (ns)',
            yaxis_title='Tasks',
            showlegend=True,
            legend_title='Devices'
        )
        
        return fig
    
    def create_task_distribution_chart(self) -> go.Figure:
        """Create a stacked bar chart showing task time distribution."""
        tasks = {}
        for task in self.timeline_data:
            if task['task'] not in tasks:
                tasks[task['task']] = 0
            tasks[task['task']] += task['duration']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(tasks.keys()),
                y=list(tasks.values()),
                text=[f'{v/1e6:.2f}ms' for v in tasks.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Task Time Distribution',
            xaxis_title='Tasks',
            yaxis_title='Time (ns)',
            showlegend=False
        )
        
        return fig 