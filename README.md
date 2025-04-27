<h1 align="left">
  <img src="docs/pulse-logo.png" alt="TornadoVMPulse Logo" width="64" style="vertical-align: middle; margin-right: 10px;">
  TornadoVMPulse
</h1>

## Overview

The **TornadoVMPulse** is an interactive web application for visualizing and analyzing profiling data from [TornadoVM](https://tornadovm.readthedocs.io/en/latest/profiler.html). It supports both raw JSON/log files and pre-processed CSV files, providing deep insights into kernel execution, data transfer, memory, and power usage.

## Features
- Upload and analyze TornadoVM profiler logs (JSON, log, or CSV)
- Automatic conversion of raw logs to CSV using `tornadovm_profiler_log_json_analyzer.py`
- Interactive visualizations: time distribution, performance dashboard, memory and power usage, copy-in/out analysis, and more
- Sidebar controls for metric selection and time unit
- Downloadable generated CSVs
- Modern, user-friendly UI with a custom icon

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run src/app.py
   ```

3. **Upload a profiling log file:**
   - Drag and drop a raw JSON/log or CSV file into the uploader.
   - If a raw file is uploaded, it will be converted to CSV and available for download.

4. **Explore the dashboard:**
   - Use the sidebar to select which metrics to display and the time unit.
   - Review the visualizations and download the generated CSV if needed.

## Profiler Icon

The dashboard uses a custom icon:

<img src="profiler-visual.png" alt="TornadoVMPulse Logo" width="96">

- Place `profiler-visual.png` in the project root or `src/` directory.
- You can use this icon for branding, documentation, or as a favicon in the Streamlit app.

## References
- [TornadoVM Profiler Documentation](https://tornadovm.readthedocs.io/en/latest/profiler.html)
- [Project Repository](https://github.com/mikepapadim/TornadoVMPulse)

## License
MIT License 