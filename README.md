# Condition-Specific Firing Rate Analysis Toolkit

Python toolkit for analyzing neural firing rates with multi-condition filtering and smoothing, designed to replicate and extend MATLAB analysis workflows (e.g., `sujay_example_zoom.m`).

## Overview

This toolkit provides:

1. **Multi-condition trial filtering** - Filter trials based on multiple condition matrix columns
2. **Firing rate extraction** - Extract neural data for specific neurons and trial subsets
3. **Smoothing with moving average** - Convolutional smoothing matching MATLAB's `conv()` function
4. **Flexible visualization** - Plot firing rates with customizable styling
5. **Pipeline integration** - Easy integration with existing analysis pipelines

## Files

### Core Module
- **`firing_rate_analyzer.py`** - Main analysis class with all functionality

### Examples
- **`matlab_replication.py`** - Standalone script that exactly replicates MATLAB analysis
- **`integration_example.py`** - Shows how to integrate into existing pipelines
- **`demo_usage.py`** - Simple demonstration with synthetic data

### Your Existing Files
- **`example_analysis.py`** - Your comprehensive hippocampus analysis pipeline
- **`sujay_example_zoom.m`** - Original MATLAB code for reference

## Quick Start

### 1. Standalone MATLAB Replication

Replicate the exact MATLAB analysis:

```bash
python matlab_replication.py /path/to/your_data.mat
```

This will:
- Plot TA vs TP for two attention conditions (behavioral data)
- Extract and plot smoothed firing rates for 3 specific condition/neuron combinations
- Save all plots to `matlab_replication_results/`

### 2. Use with Your Data

```python
from firing_rate_analyzer import FiringRateAnalyzer
import numpy as np

# Load your data (neural_tensor, cond_matrix, time_edges)
# neural_tensor shape: (n_neurons, n_timebins, n_trials)
# cond_matrix shape: (n_trials, n_conditions)
# time_edges shape: (n_timebins,)

# Define conditions to analyze
conditions_list = [
    {9: 1, 2: 1, 3: 4},  # Column 9==1 AND column 2==1 AND column 3==4
    {9: 1, 2: 1, 3: 5},  # Column 9==1 AND column 2==1 AND column 3==5
]

neuron_indices = [0, 1]  # Which neurons to analyze

# Run analysis
results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=conditions_list,
    neuron_indices=neuron_indices,
    window_size=300,
    average=True
)

# Plot results
fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Condition A', 'Condition B'],
    colors=['blue', 'orange'],
    add_vertical_lines=[0]  # Mark event onset
)
```

### 3. Integrate into Existing Pipeline

Add to your `example_analysis.py` after data loading:

```python
from firing_rate_analyzer import FiringRateAnalyzer

# ... (your existing data loading code) ...

# Add condition-specific firing rate analysis
print("\n" + "-"*70)
print("CONDITION-SPECIFIC FIRING RATE ANALYSIS")
print("-"*70)

# Replicate MATLAB analysis
matlab_results = FiringRateAnalyzer.replicate_matlab_example(
    neural_tensor=data.stim1on['neur_tensor'],
    cond_matrix=data.cond_matrix,
    time_edges=data.stim1on['edges'],
    window_size=300
)

# Plot and save
fig = FiringRateAnalyzer.plot_condition_specific_firing(
    matlab_results,
    labels=[f"Trace {i+1}" for i in range(3)],
    title="Smoothed Firing Rates"
)
fig.savefig(session_dir / 'firing_rates_matlab_style.png', dpi=300)
```

## API Reference

### `FiringRateAnalyzer` Class

#### `filter_trials_multi_condition(cond_matrix, conditions)`
Filter trials based on multiple condition matrix columns.

**Parameters:**
- `cond_matrix`: np.ndarray, shape (n_trials, n_conditions)
- `conditions`: dict, mapping column index to value(s)
  - Single value: `{9: 1}` means column 9 == 1
  - Multiple values: `{3: [4, 5]}` means column 3 in [4, 5]
  - Combined: `{9: 1, 2: 1, 3: [4, 5]}` means ALL conditions must be met

**Returns:**
- Boolean mask of shape (n_trials,)

**Example:**
```python
# Filter for trials where:
# - column 9 == 1 (attention condition 1)
# - column 2 == 1 (some task parameter)
# - column 3 is either 4 or 5 (temporal distance options)
conditions = {9: 1, 2: 1, 3: [4, 5]}
mask = FiringRateAnalyzer.filter_trials_multi_condition(cond_matrix, conditions)
print(f"Selected {np.sum(mask)} trials")
```

#### `extract_firing_rates(neural_tensor, neuron_idx, trial_indices)`
Extract firing rates for specific neuron and trials.

**Parameters:**
- `neural_tensor`: np.ndarray, shape (n_neurons, n_timebins, n_trials)
- `neuron_idx`: int, neuron index (0-indexed)
- `trial_indices`: np.ndarray, boolean mask or integer indices

**Returns:**
- Firing rates, shape (n_timebins, n_selected_trials)

#### `smooth_firing_rates(firing_rates, window_size=300, axis=0)`
Apply moving average smoothing (convolution).

**Parameters:**
- `firing_rates`: np.ndarray, shape (n_timebins, n_trials) or (n_timebins,)
- `window_size`: int, smoothing window size in samples
- `axis`: int, axis to smooth along (0 for time)

**Returns:**
- Smoothed firing rates (trimmed to valid convolution region)

**Note:** This replicates MATLAB's `conv(signal, ones(N,1), 'valid')`

#### `analyze_condition_specific_firing(...)`
Comprehensive analysis pipeline.

**Parameters:**
- `neural_tensor`: Neural data, shape (n_neurons, n_timebins, n_trials)
- `cond_matrix`: Condition matrix, shape (n_trials, n_conditions)
- `time_edges`: Time bin edges, shape (n_timebins,)
- `conditions_list`: List of condition dictionaries (one per trace)
- `neuron_indices`: List of neuron indices (one per trace)
- `window_size`: Smoothing window size (default: 300)
- `average`: If True, average across trials; if False, return all trials

**Returns:**
- Dictionary with:
  - `'firing_rates_raw'`: List of raw firing rates
  - `'firing_rates_smooth'`: List of smoothed firing rates
  - `'time_vector'`: Trimmed time vector for plotting
  - `'n_trials'`: Number of trials per condition
  - `'trial_masks'`: Boolean masks for each condition

#### `plot_condition_specific_firing(results, ...)`
Visualize firing rate analysis results.

**Parameters:**
- `results`: Output from `analyze_condition_specific_firing()`
- `labels`: List of labels for each trace (optional)
- `colors`: List of colors for each trace (optional)
- `title`: Plot title
- `xlabel`, `ylabel`: Axis labels
- `figsize`: Figure size tuple
- `add_vertical_lines`: List of x-positions for vertical lines (e.g., [0] for event onset)

**Returns:**
- Matplotlib figure

#### `replicate_matlab_example(...)`
Exactly replicate the sujay_example_zoom.m analysis.

**Parameters:**
- `neural_tensor`: Neural data from stim1on event
- `cond_matrix`: Condition matrix
- `time_edges`: Time bin edges
- `window_size`: Smoothing window (default: 300)

**Returns:**
- Analysis results dictionary (same format as `analyze_condition_specific_firing`)

**What it does:**
- Extracts 3 traces matching MATLAB code:
  1. Neuron 0, conditions: col9==1, col2==1, col3==4
  2. Neuron 0, conditions: col9==1, col2==1, col3==5
  3. Neuron 2, conditions: col9==1, col2==1, col3==2
- Averages trials and applies 300-sample smoothing

## Detailed Examples

### Example 1: Compare Different Neurons for Same Condition

```python
# Analyze first 5 neurons for mental navigation trials
conditions = {9: 1, 10: 3}  # Mental navigation condition
results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=[conditions] * 5,
    neuron_indices=[0, 1, 2, 3, 4],
    window_size=300
)

fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=[f"Neuron {i+1}" for i in range(5)],
    title="Population Response - Mental Navigation"
)
```

### Example 2: Compare Same Neuron Across Conditions

```python
# Compare attention conditions for best time cell
neuron_idx = 5  # Your best time cell

conditions_list = [
    {9: 1},   # Attention condition 1
    {11: 1},  # Attention condition 2
]

results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=conditions_list,
    neuron_indices=[neuron_idx, neuron_idx],
    window_size=300
)

fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=['Attend Left', 'Attend Right'],
    colors=['blue', 'red'],
    title=f"Neuron {neuron_idx+1}: Attention Modulation"
)
```

### Example 3: Time-Dependent Analysis

```python
# Analyze different temporal distances
ta_values = [0.5, 1.0, 1.5, 2.0]  # seconds

conditions_list = []
for ta_val in ta_values:
    # Find trials with this TA value (with tolerance)
    conditions_list.append({0: ta_val})  # Assuming column 0 is TA

results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor=neural_tensor,
    cond_matrix=cond_matrix,
    time_edges=time_edges,
    conditions_list=conditions_list,
    neuron_indices=[0] * len(ta_values),  # Same neuron
    window_size=300
)

fig = FiringRateAnalyzer.plot_condition_specific_firing(
    results,
    labels=[f"TA={ta:.1f}s" for ta in ta_values],
    title="Temporal Distance Coding"
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
    average=False  # Return all trials
)

# Results will contain (n_timebins, n_trials) array
firing_rates_all_trials = results['firing_rates_smooth'][0]
print(f"Shape: {firing_rates_all_trials.shape}")

# Now you can analyze trial-to-trial variability
```

## Understanding the MATLAB Replication

The MATLAB code (`sujay_example_zoom.m`) does:

```matlab
% Filter trials: column 10==1 AND column 3==1 AND column 4==4
trid = find(cond_matrix(:,10)==1 & cond_matrix(:,3)==1 & cond_matrix(:,4)==4);

% Extract neuron 1, all time bins, selected trials
fr3 = squeeze(neur_tensor_stim1on(1,:,trid));

% Average across trials
mean_fr3 = mean(fr3, 2);

% Smooth with 300-sample moving average
smoothed = conv(mean_fr3, ones(300,1), 'valid');

% Plot with trimmed time vector
plot(stim1on.edges(150:end-150), smoothed);
```

The Python equivalent:

```python
# Filter trials
trid = (cond_matrix[:, 9]==1) & (cond_matrix[:, 2]==1) & (cond_matrix[:, 3]==4)

# Extract neuron 0 (0-indexed)
fr3 = neur_tensor[0, :, trid]

# Average across trials
mean_fr3 = np.mean(fr3, axis=1)

# Smooth with convolution
kernel = np.ones(300)
smoothed = np.convolve(mean_fr3, kernel, mode='valid')

# Trimmed time vector
trim = 150
time_trimmed = edges[trim:-trim]

# Plot
plt.plot(time_trimmed, smoothed)
```

Or using the toolkit:

```python
conditions = {9: 1, 2: 1, 3: 4}
results = FiringRateAnalyzer.analyze_condition_specific_firing(
    neural_tensor, cond_matrix, edges,
    conditions_list=[conditions],
    neuron_indices=[0],
    window_size=300
)
FiringRateAnalyzer.plot_condition_specific_firing(results)
```

## Tips and Best Practices

1. **Condition Matrix Indexing**
   - MATLAB uses 1-based indexing: column 1, 2, 3...
   - Python uses 0-based indexing: column 0, 1, 2...
   - Remember to subtract 1 when converting MATLAB column numbers!

2. **Window Size Selection**
   - Larger windows (300-500): Smoother, better for slow dynamics
   - Smaller windows (50-100): More temporal detail, noisier
   - Typical: 100-300 samples depending on your bin size

3. **Trial Averaging**
   - `average=True`: Get mean response (less noisy, easier interpretation)
   - `average=False`: Keep all trials (analyze variability, single-trial decoding)

4. **Computational Efficiency**
   - Filter trials once, reuse mask for multiple neurons
   - Use `average=True` when possible (faster)
   - Pre-allocate arrays for batch processing

5. **Visualization**
   - Add vertical lines at key events (stimulus onset, choice time)
   - Use consistent colors across related plots
   - Include trial counts in labels for transparency

## Troubleshooting

**Problem:** "No trials found for conditions"
- Check your condition matrix has the expected columns
- Verify condition values are correct (print unique values)
- Try relaxing conditions (remove some constraints)

**Problem:** Smoothed traces look weird
- Check window size isn't too large for your data
- Verify time edges match neural tensor dimensions
- Ensure firing rates aren't all zeros

**Problem:** Index errors
- Remember Python is 0-indexed, MATLAB is 1-indexed
- Check neural_tensor shape matches expectations
- Verify neuron_indices are within valid range

**Problem:** Different results from MATLAB
- Verify exact condition filtering logic
- Check array axis conventions (time along axis 1 in tensor)
- Ensure same smoothing window and convolution mode

## Contributing

Feel free to extend this toolkit with:
- Additional smoothing methods (Gaussian, Savitzky-Golay)
- Statistical comparison functions
- More sophisticated trial selection criteria
- Population-level analyses

## License

MIT License - free to use and modify

## Contact

For questions or issues, please contact the research team or open an issue in the repository.

---

**Created:** 2024
**Last Updated:** 2024
**Version:** 1.0
