import json
import math
import numpy as np
import sys
from collections import defaultdict

def parse_json_file(file_path):
    """
    Parse a file containing multiple JSON objects separated by newlines.
    Returns a list of parsed JSON objects.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Split by double newlines (each JSON object is separated by blank line)
    json_blocks = content.split('\n\n')

    parsed_data = []
    for block in json_blocks:
        block = block.strip()
        if not block:  # Skip empty blocks
            continue

        try:
            # Try to fix common JSON issues (trailing commas)
            block = fix_json_format(block)
            data = json.loads(block)
            parsed_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON block: {e}")
            print(f"Problematic block: {block[:100]}...")
            # Continue processing other blocks

    return parsed_data

def fix_json_format(json_str):
    """
    Fix common JSON format issues like trailing commas.
    """
    # Replace trailing commas before closing brackets
    json_str = json_str.replace(',\n}', '\n}')
    json_str = json_str.replace(',\n]', '\n]')
    json_str = json_str.replace(',}', '}')
    json_str = json_str.replace(',]', ']')

    return json_str

def extract_metrics(data_objects):
    """
    Extract and organize metrics by task graph type.
    Returns a nested dictionary structure.
    """
    # First level: task graph type (e.g., "layer", "updX")
    # Second level: instance index
    # Third level: metric name and value
    metrics_by_type = defaultdict(list)

    for obj in data_objects:
        for graph_type, metrics in obj.items():
            # Create a copy of metrics to avoid modifying the original
            metrics_copy = {}

            # Extract top-level metrics
            for key, value in metrics.items():
                if not isinstance(value, dict):
                    try:
                        # Convert string numbers to float if possible
                        metrics_copy[key] = float(value)
                    except (ValueError, TypeError):
                        metrics_copy[key] = value

            # Extract task-specific metrics
            tasks = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    task_name = key
                    task_metrics = {}

                    for task_key, task_value in value.items():
                        if task_key not in ["BACKEND", "METHOD", "DEVICE_ID", "DEVICE"]:
                            try:
                                task_metrics[task_key] = float(task_value)
                            except (ValueError, TypeError):
                                task_metrics[task_key] = task_value

                    # Also store method and device info
                    if "METHOD" in value:
                        task_metrics["METHOD"] = value["METHOD"]
                    if "DEVICE" in value:
                        task_metrics["DEVICE"] = value["DEVICE"]

                    tasks[task_name] = task_metrics

            metrics_copy["tasks"] = tasks
            metrics_by_type[graph_type].append(metrics_copy)

    return metrics_by_type

def calculate_statistics(values):
    """
    Calculate statistical metrics for a list of values.
    """
    if not values:
        return {
            "count": 0,
            "mean": float('nan'),
            "geomean": float('nan'),
            "median": float('nan'),
            "std": float('nan'),
            "min": float('nan'),
            "max": float('nan'),
            "cv": float('nan')
        }

    try:
        values = np.array(values, dtype=float)

        # Basic statistics
        count = len(values)
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Coefficient of variation
        cv = (std / mean) * 100 if mean != 0 else float('nan')

        # Geometric mean (handling zeros and negative values)
        geomean_values = values.copy()
        if np.any(geomean_values <= 0):
            # For geometric mean, replace zeros/negatives with NaN
            geomean_values[geomean_values <= 0] = np.nan

        if np.all(np.isnan(geomean_values)):
            geomean = float('nan')
        else:
            geomean = np.exp(np.nanmean(np.log(geomean_values)))

        return {
            "count": count,
            "mean": mean,
            "geomean": geomean,
            "median": median,
            "std": std,
            "min": min_val,
            "max": max_val,
            "cv": cv
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            "count": len(values) if isinstance(values, list) else 0,
            "mean": float('nan'),
            "geomean": float('nan'),
            "median": float('nan'),
            "std": float('nan'),
            "min": float('nan'),
            "max": float('nan'),
            "cv": float('nan')
        }

def analyze_metrics(metrics_by_type):
    """
    Calculate statistics for each metric across all instances of each task graph type.
    """
    results = {}

    for graph_type, instances in metrics_by_type.items():
        graph_results = {
            "overall": {},
            "tasks": {}
        }

        # Collect all values for each top-level metric
        all_metrics = defaultdict(list)
        for instance in instances:
            for key, value in instance.items():
                if key != "tasks" and isinstance(value, (int, float)):
                    all_metrics[key].append(value)

        # Calculate statistics for each top-level metric
        for metric, values in all_metrics.items():
            graph_results["overall"][metric] = calculate_statistics(values)

        # First collect task names and initialize the structure
        task_metrics = defaultdict(lambda: defaultdict(list))

        # Collect all values for each task and metric
        for instance in instances:
            for task_name, task_data in instance.get("tasks", {}).items():
                for metric, value in task_data.items():
                    if isinstance(value, (int, float)):
                        task_metrics[task_name][metric].append(value)
                    # Save non-numeric metadata
                    elif metric in ["METHOD", "DEVICE"] and task_name not in graph_results["tasks"]:
                        if task_name not in graph_results["tasks"]:
                            graph_results["tasks"][task_name] = {}
                        graph_results["tasks"][task_name][metric] = value

        # Now calculate statistics for each task metric
        for task_name, metrics in task_metrics.items():
            if task_name not in graph_results["tasks"]:
                graph_results["tasks"][task_name] = {}

            for metric, values in metrics.items():
                graph_results["tasks"][task_name][metric] = calculate_statistics(values)

        results[graph_type] = graph_results

    return results

def format_time_unit(nanoseconds):
    """Convert nanoseconds to appropriate time unit for display"""
    if nanoseconds < 1000:
        return f"{nanoseconds:.2f} ns"
    elif nanoseconds < 1000000:
        return f"{nanoseconds/1000:.2f} µs"
    elif nanoseconds < 1000000000:
        return f"{nanoseconds/1000000:.2f} ms"
    else:
        return f"{nanoseconds/1000000000:.2f} s"

def format_number(number):
    """Format numbers with appropriate units (K, M, G, etc.)"""
    if number is None or math.isnan(number):
        return "N/A"

    if abs(number) >= 1e9:
        return f"{number/1e9:.2f} G"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.2f} M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.2f} K"
    else:
        return f"{number:.2f}"

def print_report(results):
    """
    Print a formatted report of the analysis results.
    """
    # Calculate total number of instances across all task graph types
    total_instances = 0
    for graph_type, graph_results in results.items():
        # Get count from the first metric in overall stats
        for metric, stats in graph_results["overall"].items():
            total_instances += stats["count"]
            break

    print(f"\nAnalyzed a total of {total_instances} task graph instances")

    for graph_type, graph_results in results.items():
        # Get count from the first metric in overall stats
        instances_count = 0
        for metric, stats in graph_results["overall"].items():
            instances_count = stats["count"]
            break

        print(f"\n{'='*80}")
        print(f"TASK GRAPH: {graph_type} ({instances_count} instances)")
        print(f"{'='*80}")

        # Print overall metrics
        print("\nOVERALL METRICS:")
        print(f"{'Metric':<30} {'Mean':<15} {'GeoMean':<15} {'Median':<15} {'CV(%)':<10} {'Min':<15} {'Max':<15}")
        print(f"{'-'*100}")

        for metric, stats in graph_results["overall"].items():
            try:
                if "TIME" in metric or "time" in metric.lower():
                    mean = format_time_unit(stats["mean"])
                    geomean = format_time_unit(stats["geomean"])
                    median = format_time_unit(stats["median"])
                    min_val = format_time_unit(stats["min"])
                    max_val = format_time_unit(stats["max"])
                else:
                    mean = format_number(stats["mean"])
                    geomean = format_number(stats["geomean"])
                    median = format_number(stats["median"])
                    min_val = format_number(stats["min"])
                    max_val = format_number(stats["max"])

                print(f"{metric:<30} {mean:<15} {geomean:<15} {median:<15} {stats['cv']:.2f}% {min_val:<15} {max_val:<15}")
            except Exception as e:
                print(f"{metric:<30} Error calculating statistics: {e}")

        # Check if there are any tasks
        if not graph_results["tasks"]:
            print("\nNo task-specific metrics found.")
            continue

        # Print tasks ordered by average kernel time
        print("\nTASK METRICS (Ordered by Mean TASK_KERNEL_TIME):")

        try:
            # Sort tasks by mean TASK_KERNEL_TIME (descending)
            sorted_tasks = []
            for task_name, task_metrics in graph_results["tasks"].items():
                mean_time = 0
                if "TASK_KERNEL_TIME" in task_metrics and isinstance(task_metrics["TASK_KERNEL_TIME"], dict):
                    mean_time = task_metrics["TASK_KERNEL_TIME"].get("mean", 0)
                sorted_tasks.append((task_name, task_metrics, mean_time))

            sorted_tasks.sort(key=lambda x: x[2], reverse=True)

            for task_name, task_metrics, _ in sorted_tasks:
                method = task_metrics.get("METHOD", "Unknown")
                device = task_metrics.get("DEVICE", "Unknown")

                print(f"\n{task_name} - {method}")
                print(f"Device: {device}")
                print(f"{'Metric':<30} {'Mean':<15} {'GeoMean':<15} {'Median':<15} {'CV(%)':<10} {'Min':<15} {'Max':<15}")
                print(f"{'-'*100}")

                for metric, stats in task_metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        try:
                            if "TIME" in metric or "time" in metric.lower():
                                mean = format_time_unit(stats["mean"])
                                geomean = format_time_unit(stats["geomean"])
                                median = format_time_unit(stats["median"])
                                min_val = format_time_unit(stats["min"])
                                max_val = format_time_unit(stats["max"])
                            else:
                                mean = format_number(stats["mean"])
                                geomean = format_number(stats["geomean"])
                                median = format_number(stats["median"])
                                min_val = format_number(stats["min"])
                                max_val = format_number(stats["max"])

                            print(f"{metric:<30} {mean:<15} {geomean:<15} {median:<15} {stats['cv']:.2f}% {min_val:<15} {max_val:<15}")
                        except Exception as e:
                            print(f"{metric:<30} Error formatting statistics: {e}")
        except Exception as e:
            print(f"Error generating task metrics: {e}")

def export_to_csv(results, output_file):
    """
    Export results to a CSV file.
    """
    import csv

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Task Graph', 'Task', 'Metric', 'Count', 'Mean', 'GeoMean', 'Median', 'CV(%)', 'Min', 'Max'])

        for graph_type, graph_results in results.items():
            # Write overall metrics
            for metric, stats in graph_results["overall"].items():
                writer.writerow([
                    graph_type,
                    'OVERALL',
                    metric,
                    stats["count"],
                    stats["mean"],
                    stats["geomean"],
                    stats["median"],
                    stats["cv"],
                    stats["min"],
                    stats["max"]
                ])

            # Write task metrics
            for task_name, task_metrics in graph_results["tasks"].items():
                for metric, stats in task_metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        writer.writerow([
                            graph_type,
                            task_name,
                            metric,
                            stats["count"],
                            stats["mean"],
                            stats["geomean"],
                            stats["median"],
                            stats["cv"],
                            stats["min"],
                            stats["max"]
                        ])

def main():
    if len(sys.argv) < 2:
        print("Usage: python performance_analyzer.py <log_file> [output_csv]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing performance data from: {file_path}")

    try:
        # Parse the JSON data
        data_objects = parse_json_file(file_path)
        print(f"Successfully parsed {len(data_objects)} JSON objects")

        # Extract and organize metrics
        metrics_by_type = extract_metrics(data_objects)
        print(f"Found {len(metrics_by_type)} different task graph types")

        # Calculate statistics
        results = analyze_metrics(metrics_by_type)

        # Print the report
        print_report(results)

        # Export to CSV if specified
        if output_csv:
            export_to_csv(results, output_csv)
            print(f"\nResults exported to: {output_csv}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
