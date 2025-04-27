import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import plotly.express as px
from visualizer import MetricsVisualizer
import json
import io
import tempfile
import subprocess
import datetime
import plotly.graph_objects as go

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization modules
from visualization.timeline import TimelineVisualizer
from visualization.heatmap import HeatmapVisualizer

def convert_time_cols(df, cols, unit):
    if unit == "ms":
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = df[col] / 1e6
                df = df.rename(columns={col: col + " (ms)"})
    else:
        # Remove (ms) if present
        for col in cols:
            if col + " (ms)" in df.columns:
                df = df.rename(columns={col + " (ms)": col})
    return df

def process_log_file(uploaded_file):
    """Process a profiling log file and return a DataFrame."""
    content = uploaded_file.read()
    try:
        # Try to parse as JSON first
        data = json.loads(content)
        df = pd.DataFrame(data)
    except json.JSONDecodeError:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        if 'Task Graph' in df.columns and 'Metric' in df.columns:
            task_graphs = df['Task Graph'].unique()
            records = []
            for graph in task_graphs:
                record = {'task_graph': graph}
                # Helper to get value or 0 if missing
                def get_metric(metric):
                    row = df[(df['Task Graph'] == graph) & (df['Task'] == 'OVERALL') & (df['Metric'] == metric)]
                    if not row.empty:
                        return row['Mean'].iloc[0] * row['Count'].iloc[0]
                    return 0
                record['total_task_graph_time'] = get_metric('TOTAL_TASK_GRAPH_TIME')
                record['total_kernel_time'] = get_metric('TOTAL_KERNEL_TIME')
                record['total_dispatch_time'] = get_metric('TOTAL_DISPATCH_KERNEL_TIME')
                record['copy_in_bytes'] = get_metric('TOTAL_COPY_IN_SIZE_BYTES')
                # Power usage: average of all POWER_USAGE_mW for this graph (not just OVERALL)
                power_rows = df[(df['Task Graph'] == graph) & (df['Metric'] == 'POWER_USAGE_mW')]
                record['power_usage'] = power_rows['Mean'].mean() if not power_rows.empty else 0
                records.append(record)
            return pd.DataFrame(records), df
    return df, df

def convert_time_unit(series, from_unit, to_unit):
    factor = 1
    if from_unit == 'ns' and to_unit == 'ms':
        factor = 1e-6
    elif from_unit == 'ns' and to_unit == 'sec':
        factor = 1e-9
    elif from_unit == 'ms' and to_unit == 'ns':
        factor = 1e6
    elif from_unit == 'ms' and to_unit == 'sec':
        factor = 1e-3
    elif from_unit == 'sec' and to_unit == 'ns':
        factor = 1e9
    elif from_unit == 'sec' and to_unit == 'ms':
        factor = 1e3
    return series * factor

def process_uploaded_file(uploaded_file):
    """
    Accepts a Streamlit UploadedFile (raw log or CSV),
    runs perf_analy.py if needed, and returns (summary_df, raw_df, csv_path).
    """
    filename = uploaded_file.name
    lower_filename = filename.lower()
    if lower_filename.endswith('.csv'):
        uploaded_file.seek(0)
        return process_log_file(uploaded_file) + (None,)
    else:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_csv')
        os.makedirs(output_dir, exist_ok=True)
        # Generate a unique CSV filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(filename))[0]
        csv_filename = f'{base_name}_{timestamp}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        # Save raw file to temp, run perf_analy.py, write to csv_path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_raw:
            tmp_raw.write(uploaded_file.read())
            tmp_raw.flush()
            tmp_raw_path = tmp_raw.name
        try:
            subprocess.run([
                sys.executable, os.path.join(os.path.dirname(__file__), 'tornadovm_profiler_log_json_analyzer.py'),
                tmp_raw_path, csv_path
            ], check=True)
            with open(csv_path, 'rb') as f:
                return process_log_file(f) + (csv_path,)
        except Exception as e:
            st.error(f"Error running perf_analy.py: {e}")
            return pd.DataFrame(), pd.DataFrame(), None
        finally:
            os.remove(tmp_raw_path)

def main():
    st.set_page_config(
        page_title="TornadoVMPulse",
        page_icon="🌪️",
        layout="wide"
    )
    
    st.markdown("<h1 style='display: flex; align-items: center; gap: 0.5em;'>🌪️ TornadoVMPulse</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### How to Use This Profiler Dashboard
    - **Upload a profiling log file**: You can upload either a raw JSON/log file (from TornadoVM or similar) or a pre-processed CSV file.
    - If you upload a raw JSON or log file, it will be automatically converted to CSV using the `tornadovm_profiler_log_json_analyzer.py` tool. The generated CSV will be available for download.
    - After upload, select your preferred time unit and choose which metrics to display using the sidebar.
    - The dashboard will provide detailed breakdowns of execution time, memory, power, and data transfer metrics.
    """)
    st.markdown("""
    <div style='background-color: #23272f; border-radius: 8px; padding: 1.2em; margin-bottom: 1.5em;'>
        <h3 style='color:#4FC3F7;'>ℹ️ About the TornadoVM Profiler</h3>
        <ul>
            <li>
                <b>Official Documentation:</b>
                <a href="https://tornadovm.readthedocs.io/en/latest/profiler.html" target="_blank" style="color:#81D4FA;">
                    TornadoVM Profiler Guide
                </a>
            </li>
            <li>
                <b>Enable the Profiler:</b> Use <code>--enableProfiler console</code> or <code>silent</code> with the <code>tornado</code> command, or via the <code>ExecutionPlan</code> API.
            </li>
            <li>
                <b>Profiler Output:</b> Produces detailed JSON logs for each task-graph, including kernel, dispatch, copy-in/out, and power metrics (all in nanoseconds).
            </li>
            <li>
                <b>Supported Power Metrics:</b> NVIDIA NVML (for OpenCL/PTX) and oneAPI Level Zero SYSMAN (for SPIRV) are supported for power usage reporting.
            </li>
            <li>
                <b>Save Profiler Output:</b> Use <code>--dumpProfiler &lt;FILENAME&gt;</code> to store logs in a file for later analysis.
            </li>
        </ul>
        <p style='font-size:0.95em; color:#B0BEC5;'>
            For a full explanation of all metrics and advanced usage, see the
            <a href="https://tornadovm.readthedocs.io/en/latest/profiler.html" target="_blank" style="color:#81D4FA;">
                TornadoVM Profiler Documentation
            </a>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for metric selection
    st.sidebar.header("Select Metrics to Display")
    st.sidebar.markdown("""
    - **Task Time Distribution**: Shows the sum of kernel execution times for each task across the entire application.
    - **Performance Dashboard**: Key performance metrics for each task graph (total execution time, average kernel time, etc).
    - **Memory Usage**: Total and average memory usage (copy-in bytes) for each task graph.
    - **Power Usage**: Average power usage for each task graph.
    - **Summary Statistics**: Statistical summary of the selected metrics.
    - **Raw Data**: The raw input data as a DataFrame.
    """)
    show_time_dist = st.sidebar.checkbox("Task Time Distribution", value=True, help="Bar and pie charts of per-task kernel execution time.")
    show_performance = st.sidebar.checkbox("Performance Dashboard", value=True, help="Dashboard of key performance metrics for each task graph.")
    show_memory = st.sidebar.checkbox("Memory Usage", value=True, help="Total and average memory usage for each task graph.")
    show_power = st.sidebar.checkbox("Power Usage", value=True, help="Average power usage for each task graph.")
    show_summary = st.sidebar.checkbox("Summary Statistics", value=True, help="Statistical summary of the selected metrics.")
    show_raw = st.sidebar.checkbox("Raw Data", value=False, help="Show the raw input data as a DataFrame.")
    st.sidebar.markdown('---')
    time_unit = st.sidebar.selectbox("Select time unit:", ["ns", "ms", "sec"], index=1)

    uploaded_file = st.file_uploader("Upload a profiling log file (raw JSON/log or CSV)", type=['json', 'csv', 'log'])

    # Show info if a non-CSV file is uploaded
    if uploaded_file is not None and not uploaded_file.name.lower().endswith('.csv'):
        st.info("""
        **Note:** If you upload a raw JSON or log file, it will be automatically converted to CSV using the `tornadovm_profiler_log_json_analyzer.py` tool.\
        The generated CSV is saved in the `generated_csv` directory and is available for download after processing.
        """)

    if uploaded_file is not None:
        try:
            summary_df, raw_df, csv_path = process_uploaded_file(uploaded_file)
            if summary_df.empty or raw_df.empty:
                st.error("No data found in the uploaded file")
                return

            left_col, right_col = st.columns([2, 1])

            with left_col:
                # --- Per-task kernel execution time aggregation ---
                task_kernel = raw_df[raw_df['Metric'] == 'TASK_KERNEL_TIME'].copy()
                task_kernel['total_kernel_time'] = task_kernel['Mean'] * task_kernel['Count']
                per_task = task_kernel.groupby('Task').agg({'total_kernel_time': 'sum'}).reset_index()
                per_task['total_kernel_time'] = convert_time_unit(per_task['total_kernel_time'], 'ns', time_unit)
                pie_data = per_task.rename(columns={'Task': 'task_name', 'total_kernel_time': 'time'})

                visualizer = MetricsVisualizer({})
                for col in ['total_task_graph_time', 'total_kernel_time', 'total_dispatch_time']:
                    if col in summary_df.columns:
                        summary_df[col] = convert_time_unit(summary_df[col], 'ns', time_unit)

                if show_time_dist:
                    st.header("Task Time Distribution")
                    st.markdown("""
                    **What you see:**
                    - The bar chart below shows the **sum of kernel execution times** for each task (e.g., `layer.rms`, `layer.matmul1`, etc.) across the entire application (all task graphs, all runs in the input file).
                    - Each bar represents the total time spent in that task, summed over all executions and all task graph instances.
                    - This helps identify which tasks are the most time-consuming in your entire workload, not just within a single task graph.
                    """)
                    fig_bar = px.bar(per_task, x='Task', y='total_kernel_time', labels={'total_kernel_time': f'Task Time Distribution ({time_unit})'})
                    st.plotly_chart(fig_bar)
                    fig_pie = px.pie(pie_data, names='task_name', values='time', title=f'Task Time Distribution ({time_unit})')
                    st.plotly_chart(fig_pie)

                if show_performance:
                    st.header("Performance Dashboard")
                    st.plotly_chart(visualizer.create_performance_metrics_dashboard(summary_df, time_unit=time_unit))

                if show_memory:
                    st.header("Memory Usage")
                    st.plotly_chart(visualizer.create_memory_usage_chart(summary_df))

                if show_power:
                    st.header("Power Usage")
                    st.plotly_chart(visualizer.create_power_usage_chart(summary_df))

                # --- Copy-In/Copy-Out Analysis ---
                st.header("Copy-In/Copy-Out Analysis")
                st.markdown("""
                This section analyzes the time spent on data transfers into and out of the device (copy-in/copy-out),
                how many such operations occurred, and how much they contribute to the total execution time.
                """)
                # Aggregate copy-in/out times and counts
                copy_in = raw_df[raw_df['Metric'] == 'COPY_IN_TIME'].copy()
                copy_out = raw_df[raw_df['Metric'] == 'COPY_OUT_TIME'].copy()
                total_copy_in_time = (copy_in['Mean'] * copy_in['Count']).sum()
                total_copy_out_time = (copy_out['Mean'] * copy_out['Count']).sum()
                num_copy_in = copy_in['Count'].sum()
                num_copy_out = copy_out['Count'].sum()
                # Convert to selected time unit
                total_copy_in_time_conv = convert_time_unit(pd.Series([total_copy_in_time]), 'ns', time_unit).iloc[0]
                total_copy_out_time_conv = convert_time_unit(pd.Series([total_copy_out_time]), 'ns', time_unit).iloc[0]
                # Use only OVERALL rows for total exec time
                overall_exec_time = raw_df[(raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'TOTAL_TASK_GRAPH_TIME')]
                total_exec_time = (overall_exec_time['Mean'] * overall_exec_time['Count']).sum()
                percent_in = round((total_copy_in_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                percent_out = round((total_copy_out_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                # Table
                copy_table = pd.DataFrame({
                    'Type': ['Copy-In', 'Copy-Out'],
                    f'Total Time ({time_unit})': [total_copy_in_time_conv, total_copy_out_time_conv],
                    'Num Operations': [num_copy_in, num_copy_out],
                    '% of Total Exec Time': [percent_in, percent_out]
                })
                st.dataframe(copy_table)
                # Bar chart
                fig_copy = go.Figure(data=[
                    go.Bar(name='Total Time', x=['Copy-In', 'Copy-Out'], y=[total_copy_in_time_conv, total_copy_out_time_conv]),
                    go.Bar(name='Num Operations', x=['Copy-In', 'Copy-Out'], y=[num_copy_in, num_copy_out], yaxis='y2')
                ])
                fig_copy.update_layout(
                    title=f'Copy-In/Copy-Out Time and Operations',
                    yaxis=dict(title=f'Time ({time_unit})'),
                    yaxis2=dict(title='Num Operations', overlaying='y', side='right'),
                    barmode='group'
                )
                st.plotly_chart(fig_copy)

                # --- Total Execution Time Analysis ---
                st.header("Total Execution Time Analysis")
                st.markdown("""
                This section breaks down the total execution time of the application, showing how much each task graph and their actions (kernel, dispatch, copy-in, copy-out) contribute to it.
                """)
                # Calculate total execution time (OVERALL TOTAL_TASK_GRAPH_TIME)
                overall_exec_time = raw_df[(raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'TOTAL_TASK_GRAPH_TIME')]
                total_exec_time = (overall_exec_time['Mean'] * overall_exec_time['Count']).sum()
                total_exec_time_conv = convert_time_unit(pd.Series([total_exec_time]), 'ns', time_unit).iloc[0]
                # For each task graph, get times for each action
                task_graphs = overall_exec_time['Task Graph'].unique()
                breakdown_rows = []
                for graph in task_graphs:
                    row = {'Task Graph': graph}
                    # Total time for this graph
                    graph_total = (overall_exec_time[overall_exec_time['Task Graph'] == graph]['Mean'] * overall_exec_time[overall_exec_time['Task Graph'] == graph]['Count']).sum()
                    graph_total_conv = convert_time_unit(pd.Series([graph_total]), 'ns', time_unit).iloc[0]
                    row['Total Time'] = graph_total_conv
                    row['% of Total'] = round((graph_total / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                    # Kernel
                    kernel = raw_df[(raw_df['Task Graph'] == graph) & (raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'TOTAL_KERNEL_TIME')]
                    kernel_time = (kernel['Mean'] * kernel['Count']).sum()
                    kernel_time_conv = convert_time_unit(pd.Series([kernel_time]), 'ns', time_unit).iloc[0]
                    row['Kernel Time'] = kernel_time_conv
                    row['Kernel %'] = round((kernel_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                    # Dispatch
                    dispatch = raw_df[(raw_df['Task Graph'] == graph) & (raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'TOTAL_DISPATCH_KERNEL_TIME')]
                    dispatch_time = (dispatch['Mean'] * dispatch['Count']).sum()
                    dispatch_time_conv = convert_time_unit(pd.Series([dispatch_time]), 'ns', time_unit).iloc[0]
                    row['Dispatch Time'] = dispatch_time_conv
                    row['Dispatch %'] = round((dispatch_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                    # Copy-In
                    copyin = raw_df[(raw_df['Task Graph'] == graph) & (raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'COPY_IN_TIME')]
                    copyin_time = (copyin['Mean'] * copyin['Count']).sum()
                    copyin_time_conv = convert_time_unit(pd.Series([copyin_time]), 'ns', time_unit).iloc[0]
                    row['Copy-In Time'] = copyin_time_conv
                    row['Copy-In %'] = round((copyin_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                    # Copy-Out
                    copyout = raw_df[(raw_df['Task Graph'] == graph) & (raw_df['Task'] == 'OVERALL') & (raw_df['Metric'] == 'COPY_OUT_TIME')]
                    copyout_time = (copyout['Mean'] * copyout['Count']).sum()
                    copyout_time_conv = convert_time_unit(pd.Series([copyout_time]), 'ns', time_unit).iloc[0]
                    row['Copy-Out Time'] = copyout_time_conv
                    row['Copy-Out %'] = round((copyout_time / total_exec_time * 100), 2) if total_exec_time > 0 else 0
                    breakdown_rows.append(row)
                breakdown_df = pd.DataFrame(breakdown_rows)
                st.dataframe(breakdown_df)
                # Stacked bar chart
                fig_stack = go.Figure()
                fig_stack.add_trace(go.Bar(
                    name='Kernel', x=breakdown_df['Task Graph'], y=breakdown_df['Kernel Time'], marker_color='royalblue'))
                fig_stack.add_trace(go.Bar(
                    name='Dispatch', x=breakdown_df['Task Graph'], y=breakdown_df['Dispatch Time'], marker_color='orange'))
                fig_stack.add_trace(go.Bar(
                    name='Copy-In', x=breakdown_df['Task Graph'], y=breakdown_df['Copy-In Time'], marker_color='green'))
                fig_stack.add_trace(go.Bar(
                    name='Copy-Out', x=breakdown_df['Task Graph'], y=breakdown_df['Copy-Out Time'], marker_color='red'))
                fig_stack.update_layout(
                    barmode='stack',
                    title=f'Task Graph Execution Time Breakdown ({time_unit})',
                    yaxis_title=f'Time ({time_unit})',
                    xaxis_title='Task Graph'
                )
                st.plotly_chart(fig_stack)

            with right_col:
                # CSV download info
                if csv_path:
                    st.success(f"CSV generated and saved at: {csv_path}")
                    with open(csv_path, 'rb') as f:
                        st.download_button('Download generated CSV', f, file_name=os.path.basename(csv_path))

                # --- Format summary statistics for display ---
                summary_stats_df = summary_df.copy()
                for col in ['total_task_graph_time', 'total_kernel_time', 'total_dispatch_time']:
                    if col in summary_stats_df.columns:
                        summary_stats_df[col] = convert_time_unit(summary_stats_df[col], 'ns', time_unit)
                # Format bytes columns for readability
                def format_bytes(x):
                    if pd.isna(x):
                        return x
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if abs(x) < 1024.0:
                            return f"{x:,.0f} {unit}"
                        x /= 1024.0
                    return f"{x:,.0f} PB"
                if 'copy_in_bytes' in summary_stats_df.columns:
                    summary_stats_df['copy_in_bytes'] = summary_stats_df['copy_in_bytes'].apply(format_bytes)
                if show_summary:
                    st.subheader("Summary Statistics")
                    st.dataframe(summary_stats_df.describe())
                if show_raw:
                    st.subheader("Raw Data")
                    st.dataframe(raw_df)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Please make sure the file is in the correct format (CSV with Task Graph, Task, Metric columns, or raw JSON/log file)")

if __name__ == "__main__":
    main() 