# Segment LIFO Visualization Guide

Comprehensive plotting tools for analyzing battery segment behavior and detecting LIFO violations.

---

## Quick Start

### Command Line Usage

**Basic usage** (stacked area chart):
```bash
python py_script/visualization/plot_segment_lifo.py path/to/solution_timeseries.csv
```

**Individual line plots**:
```bash
python py_script/visualization/plot_segment_lifo.py path/to/solution_timeseries.csv --style individual
```

**Custom output path**:
```bash
python py_script/visualization/plot_segment_lifo.py \
    path/to/solution_timeseries.csv \
    -o my_custom_plot.html \
    --style stacked \
    --width 1600 \
    --height 1400
```

**Disable violation detection** (faster for large files):
```bash
python py_script/visualization/plot_segment_lifo.py \
    path/to/solution_timeseries.csv \
    --no-violations
```

---

## Python API Usage

### 1. Single Run Analysis

```python
from py_script.visualization.plot_segment_lifo import plot_segment_lifo_analysis

# Create comprehensive LIFO analysis
fig = plot_segment_lifo_analysis(
    csv_path="path/to/solution_timeseries.csv",
    output_path="my_plot.html",  # Optional, None = auto-generate
    show_violations=True,         # Highlight parallel charging/discharging
    plot_style='stacked',         # 'stacked' or 'individual'
    epsilon=0.1,                  # Tolerance for violation detection (kWh)
    width=1400,                   # Plot width (px)
    height=1200                   # Plot height (px)
)
```

### 2. Detect Violations Only

```python
from py_script.visualization.plot_segment_lifo import detect_parallel_operations
import pandas as pd

# Load data
df = pd.read_csv("path/to/solution_timeseries.csv")

# Detect violations
parallel_charging, parallel_discharging = detect_parallel_operations(
    df,
    epsilon=0.1  # Tolerance in kWh
)

# Check results
print(f"Parallel charging events: {len(parallel_charging)}")
print(f"Parallel discharging events: {len(parallel_discharging)}")

# Examine violations
for event in parallel_charging:
    print(f"Hour {event['hour']:.2f}: Segments {event['segments']}, "
          f"Deltas {event['deltas']}, SOC {event['soc']:.2f} kWh")
```

### 3. Compare Multiple Runs

```python
from py_script.visualization.plot_segment_lifo import plot_segment_comparison

# Compare two optimization runs
fig = plot_segment_comparison(
    csv_paths=[
        "run1_sequential_on/solution_timeseries.csv",
        "run2_sequential_off/solution_timeseries.csv"
    ],
    labels=[
        "Sequential Activation ON",
        "Sequential Activation OFF"
    ],
    output_path="comparison.html",
    width=1400,
    height=800
)
```

---

## Visualization Panels

The main plot (`plot_segment_lifo_analysis`) generates 4 panels:

### Panel 1: Segment Energy Over Time
- **Stacked mode**: Area chart showing total SOC built from individual segments
- **Individual mode**: Line traces for each segment with capacity reference line
- **Purpose**: Visualize how energy is distributed across segments

### Panel 2: Segment Energy Deltas
- Shows instantaneous energy changes (Δ) for each segment
- **Green markers**: Charging (positive Δ)
- **Red markers**: Discharging (negative Δ)
- **Red X markers**: Parallel charging violation events
- **Purpose**: Detect parallel operations (multiple segments changing simultaneously)

### Panel 3: Total SOC Trajectory
- Black line with fill showing total battery state of charge
- Capacity reference line at 4,472 kWh
- **Purpose**: Overall energy trajectory context

### Panel 4: Power Flows
- **Green**: Charging power (positive)
- **Red**: Discharging power (shown as negative for clarity)
- **Purpose**: Understand charging/discharging patterns

---

## Interpreting Results

### ✓ **Correct LIFO Behavior** (Sequential Activation ON)

**Panel 1 (Segment Energy):**
- Segments fill sequentially from bottom to top
- Segment 1 always reaches 447.2 kWh before segment 2 receives any energy
- During discharge, top segments empty completely before lower segments discharge

**Panel 2 (Deltas):**
- Only **one segment** shows non-zero delta at any time step
- No red X markers (no parallel charging violations)
- Clean vertical alignment of markers (one segment at a time)

**Example good behavior:**
```
Hour 1.00: Segment 1: +447.2 kWh  (only segment 1 charging)
Hour 1.25: Segment 2: +200.0 kWh  (only segment 2 charging)
Hour 1.50: Segment 2: +247.2 kWh  (only segment 2 charging)
```

---

### ✗ **LIFO Violation** (Sequential Activation OFF)

**Panel 1 (Segment Energy):**
- Segments may appear to fill correctly (fullness constraint still enforced)
- No obvious visual violation in stacked view

**Panel 2 (Deltas):**
- **Multiple segments** show non-zero deltas at the same time step
- **Red X markers** indicate parallel charging events
- Horizontal alignment of markers (simultaneous changes)

**Example violation:**
```
Hour 1.25: Segment 1: +447.2 kWh  }  Both charging
           Segment 2: +83.9 kWh   }  at the same time!
```

**Why this is wrong:**
- Physically impossible in a "stacked tank" model
- Distorts cyclic degradation calculations
- Violates Xu et al. 2017 LIFO assumption

---

## Common Use Cases

### 1. Validate Optimization Results

```python
from py_script.visualization.plot_segment_lifo import detect_parallel_operations
import pandas as pd

df = pd.read_csv("solution_timeseries.csv")
parallel_ch, parallel_dis = detect_parallel_operations(df)

if len(parallel_ch) == 0 and len(parallel_dis) == 0:
    print("✓ PASS: LIFO constraint properly enforced")
else:
    print(f"✗ FAIL: {len(parallel_ch) + len(parallel_dis)} violations detected")
    print("Check that REQUIRE_SEQUENTIAL_SEGMENT_ACTIVATION = True")
```

### 2. Debug Slow Solve Times

```python
# Create plots for both configurations
plot_segment_lifo_analysis(
    "sequential_on/solution_timeseries.csv",
    output_path="sequential_on_analysis.html"
)

plot_segment_lifo_analysis(
    "sequential_off/solution_timeseries.csv",
    output_path="sequential_off_analysis.html"
)

# Compare segment behavior and solve times
```

### 3. Generate Report Figures

```python
# High-resolution plot for publications
fig = plot_segment_lifo_analysis(
    csv_path="solution_timeseries.csv",
    plot_style='individual',  # Clearer for publications
    width=2000,
    height=1600
)

# Export as static image (requires kaleido)
fig.write_image("segment_analysis.png", scale=2)
```

### 4. Automated Validation in CI/CD

```python
def validate_lifo_compliance(csv_path, max_violations=0):
    """Check if optimization results comply with LIFO constraints."""
    df = pd.read_csv(csv_path)
    parallel_ch, parallel_dis = detect_parallel_operations(df)

    total_violations = len(parallel_ch) + len(parallel_dis)

    if total_violations > max_violations:
        raise AssertionError(
            f"LIFO validation failed: {total_violations} violations detected. "
            f"Set REQUIRE_SEQUENTIAL_SEGMENT_ACTIVATION = True"
        )

    return True

# In test suite
validate_lifo_compliance("validation_results/latest/solution_timeseries.csv")
```

---

## Command-Line Options

```bash
python py_script/visualization/plot_segment_lifo.py -h
```

**Required:**
- `csv_path`: Path to solution_timeseries.csv file

**Optional:**
- `-o, --output PATH`: Output HTML path (default: auto-generate)
- `--style {stacked,individual}`: Plot style (default: stacked)
- `--no-violations`: Disable violation detection
- `--epsilon FLOAT`: Tolerance for violation detection in kWh (default: 0.1)
- `--width INT`: Plot width in pixels (default: 1400)
- `--height INT`: Plot height in pixels (default: 1200)

---

## Troubleshooting

### Issue: No violations shown but plot looks wrong

**Solution**: Adjust `epsilon` parameter. Default is 0.1 kWh - may need tuning for your data:
```python
# More sensitive (detects smaller changes)
plot_segment_lifo_analysis(csv_path, epsilon=0.01)

# Less sensitive (ignores numerical noise)
plot_segment_lifo_analysis(csv_path, epsilon=1.0)
```

### Issue: Plot too large/slow to render

**Solution**: Reduce plot size or disable violation highlighting:
```python
plot_segment_lifo_analysis(
    csv_path,
    width=1000,
    height=800,
    show_violations=False  # Faster rendering
)
```

### Issue: Want static image instead of HTML

**Solution**: Install kaleido and export:
```bash
pip install kaleido
```

```python
fig = plot_segment_lifo_analysis(csv_path)
fig.write_image("plot.png", width=1600, height=1200, scale=2)
```

---

## Example Output

### Files Generated
```
validation_results/
└── optimizer_validation/
    └── 20251114_223822_notebook_test_modeliii_ch_24h_alpha1.0/
        ├── solution_timeseries.csv
        ├── performance_summary.json
        ├── segment_lifo_analysis.html          ← Stacked view
        ├── segment_lifo_individual.html        ← Individual lines
        └── SEGMENT_VIOLATION_REPORT.md         ← Text report
```

### Interpretation
- **Open HTML files** in browser for interactive plots
- **Hover over data** to see details
- **Zoom/pan** to examine specific time periods
- **Panel 2** is most important for violation detection

---

## Integration with Existing Workflow

Add to your optimization notebook after solving:

```python
# At the end of optimization notebook
from py_script.visualization.plot_segment_lifo import plot_segment_lifo_analysis

# Create visualization
solution_csv = f"{validation_dir}/solution_timeseries.csv"
plot_segment_lifo_analysis(
    csv_path=solution_csv,
    output_path=f"{validation_dir}/segment_lifo_analysis.html",
    show_violations=True,
    plot_style='stacked'
)

print(f"LIFO analysis plot saved to {validation_dir}")
```

---

## References

1. **Project Documentation**: `doc/whole_project_description.md`
2. **LIFO Bug Analysis**: `LIFO_SEGMENT_BUG_ANALYSIS.md`
3. **Optimizer Implementation**: `py_script/core/optimizer.py`
4. **Xu et al. 2017**: IEEE Trans. on Smart Grid, 9(2), 1131-1140

---

**Author**: Claude Code
**Version**: 1.0
**Date**: 2025-11-14
