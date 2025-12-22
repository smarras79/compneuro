# Complete Firing Rate Analysis Toolkit
## Full Working Implementation

This package provides a complete, ready-to-use toolkit for analyzing condition-specific firing rates in hippocampus neural data, replicating and extending MATLAB analysis workflows.

## 📁 Files Included

### Core Files (Required)
1. **`firing_rate_analyzer.py`** - The analysis module
2. **`example_analysis_1.py`** - Main analysis script

### Dependencies
- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## 🚀 Quick Start

### Installation

```bash
# Install dependencies (if needed)
pip install numpy scipy matplotlib
```

### Basic Usage

```bash
# Run analysis on your .mat file
python example_analysis_1.py /path/to/your_data.mat

# Specify custom output directory
python example_analysis_1.py /path/to/your_data.mat --output my_results
```

### Expected Data Format

Your `.mat` file should contain:
- `neur_tensor_stim1on`: Neural data (n_neurons × n_timebins × n_trials)
- `cond_matrix`: Condition matrix (n_trials × n_conditions)
- `stim1on`: Structure with `edges` field (time bin edges)

## 📊 What the Analysis Does

The script runs 7 comprehensive steps:

### STEP 1: Load Data
- Loads neural tensor and condition matrix from .mat file
- Extracts time information
- Displays data dimensions

### STEP 2: Behavioral Analysis
- Plots TA (true temporal distance) vs TP (produced temporal distance)
- Compares two attention conditions
- Calculates correlation, linear fit, and error statistics

### STEP 3: MATLAB Replication
Exactly replicates `sujay_example_zoom.m`:
- Neuron 1, conditions: col9=1, col2=1, col3=4
- Neuron 1, conditions: col9=1, col2=1, col3=5
- Neuron 3, conditions: col9=1, col2=1, col3=2
- 300-sample moving average smoothing

### STEP 4: Custom Condition Comparison
- Compares different condition values for same neuron
- Allows flexible condition specification

### STEP 5: Population Analysis
- Analyzes first 5 neurons with same condition
- Shows population-level firing patterns

### STEP 6: Attention Modulation
- Compares two attention conditions
- Shows how attention affects neural responses

### STEP 7: Summary Report
- Generates text summary with all statistics
- Lists all generated files

## 📈 Output Files

After running, you'll get:

```
results/
└── your_session_name/
    ├── behavioral_ta_vs_tp.png
    ├── firing_rates_matlab_replication.png
    ├── firing_rates_condition_comparison.png
    ├── firing_rates_population.png
    ├── firing_rates_attention_modulation.png
    └── analysis_summary.txt
```

## 🔧 Customization Guide

### Using the Module Programmatically

```python
from firing_rate_analyzer import FiringRateAnalyzer
import numpy as np
from scipy.io import loadmat

# Load your data
data = loadmat('your_file.mat')
neural_tensor = data['neur_tensor_stim1on']  # (neurons, timebins, trials)
cond_matrix = data['cond_matrix']  # (trials, conditions)
time_edges = data['stim1on']['edges'][0, 0].flatten()

# Example 1: Replicate MATLAB analysis
results = FiringRateAnalyzer.replicate_matlab_example(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    window_size=300
)

# Plot results
fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Trace 1', 'Trace 2', 'Trace 3'],
    add_vertical_lines=[0]  # Mark stimulus onset
)
fig.savefig('my_results.png', dpi=300)
```

### Example 2: Custom Conditions

```python
# Define your own conditions
conditions_list = [
    {9: 1, 3: 4},           # Column 9 == 1 AND column 3 == 4
    {9: 1, 3: 5},           # Column 9 == 1 AND column 3 == 5
    {9: 1, 3: [4, 5]},      # Column 9 == 1 AND column 3 in [4, 5]
]

# Specify which neurons to analyze
neuron_indices = [0, 0, 2]  # First two are neuron 0, last is neuron 2

# Run analysis
results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=conditions_list,
    neuron_indices=neuron_indices,
    window_size=300,
    average=True  # Average across trials
)

# Access results
print(f"Trials per condition: {results['n_trials']}")
time_vector = results['time_vector']
smoothed_rates = results['firing_rates_smooth']
```

### Example 3: Compare Same Neuron Across Conditions

```python
# Compare different task conditions for best neuron
best_neuron = 5  # Your best time cell

conditions_to_compare = [
    {2: 1, 3: 2},  # Condition A
    {2: 1, 3: 4},  # Condition B
    {2: 1, 3: 5},  # Condition C
]

results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=conditions_to_compare,
    neuron_indices=[best_neuron] * 3,  # Same neuron for all
    window_size=200,  # Smaller window = more temporal detail
    average=True
)

fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Condition A', 'Condition B', 'Condition C'],
    colors=['blue', 'orange', 'green'],
    title=f'Neuron {best_neuron+1}: Task Modulation'
)
```

### Example 4: Extract Individual Trials (No Averaging)

```python
# Get all individual trial traces
conditions = {9: 1, 2: 1, 3: 4}

results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=[conditions],
    neuron_indices=[0],
    window_size=100,
    average=False  # Return all trials, no averaging
)

# Now you have (n_timebins, n_trials) array
all_trials = results['firing_rates_smooth'][0]
print(f"Shape: {all_trials.shape}")

# Can compute trial-to-trial variability
variability = np.std(all_trials, axis=1)
mean_response = np.mean(all_trials, axis=1)
cv = variability / (mean_response + 1e-10)  # Coefficient of variation
```

## 🔍 Understanding Conditions

### Condition Dictionary Format

```python
conditions = {
    column_index: value_or_values
}
```

**Examples:**

```python
# Single value
{9: 1}  # Column 9 must equal 1

# Multiple values (OR logic within column)
{3: [4, 5, 6]}  # Column 3 must be 4 OR 5 OR 6

# Multiple columns (AND logic between columns)
{9: 1, 2: 1, 3: 4}  # col9=1 AND col2=1 AND col3=4

# Combined
{9: 1, 3: [4, 5]}  # col9=1 AND col3 in [4,5]
```

### MATLAB to Python Column Index Conversion

**Important:** MATLAB uses 1-based indexing, Python uses 0-based!

| MATLAB | Python | Typical Meaning |
|--------|--------|-----------------|
| column 1 | column 0 | TA (true temporal distance) |
| column 2 | column 1 | TP (produced temporal distance) |
| column 3 | column 2 | Success/failure |
| column 4 | column 3 | Some task parameter |
| column 10 | column 9 | Attention condition 1 |
| column 12 | column 11 | Attention condition 2 |

**Example:** If MATLAB code uses `cond_matrix(:,10)`, use `cond_matrix[:, 9]` in Python.

## 🎨 Plotting Options

### Basic Plot

```python
fig = FiringRateAnalyzer.plot_condition_specific_firing(results)
```

### Customized Plot

```python
fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Custom Label 1', 'Custom Label 2'],
    colors=['#FF6B6B', '#4ECDC4'],  # Hex colors
    title='My Custom Title',
    xlabel='Time from stimulus (s)',
    ylabel='Firing Rate (spikes/s)',
    figsize=(14, 7),
    add_vertical_lines=[0, 1.0, 2.0]  # Mark key events
)

# Save with high resolution
fig.savefig('my_plot.png', dpi=600, bbox_inches='tight')
fig.savefig('my_plot.pdf')  # Vector format
```

## 🛠️ API Reference

### FiringRateAnalyzer.filter_trials_multi_condition()

Filter trials based on multiple condition matrix columns.

**Parameters:**
- `cond_matrix`: np.ndarray, shape (n_trials, n_conditions)
- `conditions`: dict, {column_index: value(s)}

**Returns:**
- Boolean mask, shape (n_trials,)

### FiringRateAnalyzer.extract_firing_rates()

Extract firing rates for specific neuron and trials.

**Parameters:**
- `neural_tensor`: np.ndarray, shape (n_neurons, n_timebins, n_trials)
- `neuron_idx`: int, 0-indexed
- `trial_indices`: np.ndarray, boolean mask or integer indices

**Returns:**
- Firing rates, shape (n_timebins, n_selected_trials)

### FiringRateAnalyzer.smooth_firing_rates()

Apply moving average smoothing.

**Parameters:**
- `firing_rates`: np.ndarray, shape (n_timebins,) or (n_timebins, n_trials)
- `window_size`: int, smoothing window (default: 300)
- `axis`: int, axis to smooth along (default: 0)

**Returns:**
- Smoothed rates (length reduced by window_size-1)

### FiringRateAnalyzer.analyze_condition_specific_firing()

Main analysis function.

**Parameters:**
- `neural_tensor`: np.ndarray, (n_neurons, n_timebins, n_trials)
- `cond_matrix`: np.ndarray, (n_trials, n_conditions)
- `time_edges`: np.ndarray, (n_timebins,)
- `conditions_list`: list of condition dicts
- `neuron_indices`: list of int
- `window_size`: int (default: 300)
- `average`: bool (default: True)

**Returns:**
- dict with keys: 'firing_rates_raw', 'firing_rates_smooth', 'time_vector', 'n_trials', 'trial_masks'

### FiringRateAnalyzer.replicate_matlab_example()

Exact MATLAB replication.

**Parameters:**
- `neural_tensor`: np.ndarray
- `cond_matrix`: np.ndarray
- `time_edges`: np.ndarray
- `window_size`: int (default: 300)

**Returns:**
- Same as analyze_condition_specific_firing()

## ❓ Troubleshooting

### Problem: "No trials found for conditions"
**Solution:** 
- Check your condition matrix has the expected columns
- Print unique values: `np.unique(cond_matrix[:, column_idx])`
- Try relaxing conditions (fewer constraints)

### Problem: Index errors
**Solution:**
- Remember Python is 0-indexed (MATLAB is 1-indexed)
- Subtract 1 from MATLAB column numbers
- Check: `neuron_idx < neural_tensor.shape[0]`

### Problem: Smoothing produces strange results
**Solution:**
- Check window_size isn't larger than data length
- Verify firing rates aren't all zeros
- Try smaller window (50-100 samples)

### Problem: File not found
**Solution:**
```python
from pathlib import Path
mat_file = Path('data/session.mat')
if not mat_file.exists():
    print(f"File not found: {mat_file.absolute()}")
```

## 📝 Example Complete Script

Here's a minimal working example:

```python
#!/usr/bin/env python3
from firing_rate_analyzer import FiringRateAnalyzer
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load data
data = loadmat('your_data.mat')
neural_tensor = data['neur_tensor_stim1on']
cond_matrix = data['cond_matrix']
time_edges = data['stim1on']['edges'][0, 0].flatten()

# Define analysis
conditions_list = [{9: 1, 2: 1, 3: 4}, {9: 1, 2: 1, 3: 5}]
neuron_indices = [0, 0]

# Run analysis
results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor, cond_matrix, time_edges,
    conditions_list, neuron_indices, window_size=300
)

# Plot
fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Condition A', 'Condition B'],
    add_vertical_lines=[0]
)
plt.savefig('results.png', dpi=300)
plt.show()
```

## 🎯 Key Features

✅ **Exact MATLAB replication** - matches sujay_example_zoom.m output  
✅ **Flexible condition filtering** - complex AND/OR logic  
✅ **Multiple smoothing options** - adjustable window sizes  
✅ **Publication-ready plots** - customizable styling  
✅ **Comprehensive error handling** - informative messages  
✅ **Well-tested** - all functions validated  
✅ **Fast execution** - optimized NumPy operations  
✅ **Easy integration** - works with existing pipelines  

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your data format matches expectations
3. Try the minimal example above
4. Check Python and library versions

## 📄 License

MIT License - Free to use and modify

---

**Version:** 1.0  
**Last Updated:** December 2024  
**Tested with:** Python 3.8+, NumPy 1.20+, SciPy 1.7+, Matplotlib 3.3+
