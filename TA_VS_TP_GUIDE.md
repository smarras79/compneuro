# TA vs TP Plotting Feature

## Overview

This feature analyzes and visualizes the relationship between **True Temporal Distance (TA)** and **Produced Temporal Distance (TP)** in your hippocampal neural data.

## What is TA vs TP?

- **TA (True Temporal Distance)**: The actual time interval between landmarks (Column 1 of cond_matrix)
- **TP (Produced Temporal Distance)**: The time interval produced by the subject (Column 2 of cond_matrix)

This analysis shows how accurately the subject estimates temporal intervals during mental navigation.

---

## Usage

### Option 1: Integrated in Main Pipeline

The TA vs TP analysis is **automatically included** when you run the main analysis:

```bash
python example_analysis.py data.mat
```

This generates:
- `ta_vs_tp_benchmark.png` - Main benchmark plot
- `ta_vs_tp_detailed.png` - Detailed analysis with error distributions

### Option 2: Standalone Script

For **focused TA vs TP analysis only**:

```bash
python plot_ta_vs_tp.py data.mat
```

**With options:**
```bash
# Custom output directory
python plot_ta_vs_tp.py data.mat --output results/

# Color points by start landmark
python plot_ta_vs_tp.py data.mat --color-by curr

# Color points by target landmark
python plot_ta_vs_tp.py data.mat --color-by target

# Color points by success
python plot_ta_vs_tp.py data.mat --color-by succ
```

---

## Data Requirements

Your MATLAB file must contain:

```matlab
cond_matrix with at least 2 columns:
  Column 1 (index 0): TA - True temporal distance (seconds)
  Column 2 (index 1): TP - Produced temporal distance (seconds)

Optional:
  Column 10 (index 9): Trial validity mask (1 = valid, 0 = invalid)
```

### Example MATLAB code to extract:
```matlab
% In MATLAB
trid = find(cond_matrix(:,10) == 1);  % Find valid trials
ta_att1 = cond_matrix(trid, 1);       % True temporal distance
tp_att1 = cond_matrix(trid, 2);       % Produced temporal distance
```

### Python equivalent:
```python
# In Python
trial_mask = data.cond_matrix[:, 9] == 1  # Column 10 (0-indexed = 9)
ta = data.cond_matrix[trial_mask, 0]       # Column 1 (0-indexed = 0)
tp = data.cond_matrix[trial_mask, 1]       # Column 2 (0-indexed = 1)
```

---

## Output Files

### 1. Benchmark Plot (`ta_vs_tp_benchmark.png`)

A comprehensive 2×2 plot showing:

**Top-left (Main plot):**
- Scatter plot: TA vs TP
- Red dashed line: Unity (perfect performance)
- Green line: Linear regression fit
- Statistics: correlation, MAE, RMSE

**Bottom-left (Marginal):**
- Distribution of TA values

**Top-right (Marginal):**
- Distribution of TP values

**Bottom-right (Error):**
- Distribution of errors (TP - TA)
- Shows over/underestimation

### 2. Detailed Analysis (`ta_vs_tp_detailed.png`)

A 2×3 comprehensive analysis showing:

1. **Main scatter plot** (optionally colored by condition)
2. **Error vs TA**: Shows if errors depend on interval length
3. **Relative error histogram**: Percentage errors
4. **Absolute error histogram**: Absolute error magnitudes
5. **Statistics table**: All performance metrics

### 3. Statistics File (`ta_vs_tp_statistics.txt`)

Text file containing:
- Data summary
- Performance metrics (correlation, slope, R²)
- Error metrics (mean, std, MAE, RMSE)
- Error distribution (over/underestimation percentages)

---

## Interpreting the Results

### Perfect Performance
- Points fall on the **unity line** (red dashed)
- Correlation ≈ 1.0
- Slope ≈ 1.0, Intercept ≈ 0.0
- Mean error ≈ 0

### Typical Patterns

**1. Good Performance**
```
Correlation: 0.85-0.95
MAE: < 0.3 seconds
Points cluster near unity line
```

**2. Systematic Underestimation**
```
Slope < 1.0
Points below unity line
Subject produces shorter intervals than actual
```

**3. Systematic Overestimation**
```
Slope > 1.0
Points above unity line
Subject produces longer intervals than actual
```

**4. Scalar Property**
```
Error increases with TA
Error vs TA plot shows positive slope
Typical in temporal cognition
```

---

## API Usage

### In Python Scripts

```python
from hippocampus_data_loader import HippocampusDataLoader
from neural_visualization import NeuralVisualizer, save_figure

# Load data
data = HippocampusDataLoader.load_mat_file('data.mat')

# Create basic plot
fig = NeuralVisualizer.plot_ta_vs_tp(data)
save_figure(fig, 'ta_vs_tp_benchmark.png')

# Create detailed plot with coloring
fig = NeuralVisualizer.plot_ta_vs_tp_detailed(
    data, 
    trial_mask=None,  # Uses column 10 == 1 by default
    color_by='curr'   # Color by start landmark
)
save_figure(fig, 'ta_vs_tp_detailed.png')

# Use custom trial mask
my_mask = data.filter_trials(succ=1, trial_type=3)
fig = NeuralVisualizer.plot_ta_vs_tp(data, trial_mask=my_mask)
```

---

## Function Reference

### `plot_ta_vs_tp(neural_data, trial_mask=None, figsize=(10, 8))`

Create a comprehensive TA vs TP benchmark plot.

**Parameters:**
- `neural_data`: HippocampusDataLoader instance
- `trial_mask`: Optional boolean array to filter trials
  - If `None`, uses column 10 == 1 from cond_matrix
- `figsize`: Figure size tuple

**Returns:**
- Matplotlib figure with 2×2 subplot layout

**Example:**
```python
fig = NeuralVisualizer.plot_ta_vs_tp(data)
```

---

### `plot_ta_vs_tp_detailed(neural_data, trial_mask=None, color_by=None, figsize=(14, 10))`

Create a detailed TA vs TP analysis with additional visualizations.

**Parameters:**
- `neural_data`: HippocampusDataLoader instance
- `trial_mask`: Optional boolean array to filter trials
- `color_by`: Optional condition name to color points by
  - Options: `'curr'`, `'target'`, `'succ'`, `'trial_type'`, etc.
- `figsize`: Figure size tuple

**Returns:**
- Matplotlib figure with 2×3 subplot layout

**Example:**
```python
fig = NeuralVisualizer.plot_ta_vs_tp_detailed(
    data,
    color_by='curr'  # Color by start landmark
)
```

---

## Statistics Explained

### Correlation (r)
- Range: -1 to 1
- Measures linear relationship between TA and TP
- High values (>0.8) indicate good temporal estimation

### Slope
- Unity (1.0) indicates perfect scaling
- <1.0: Compression (underestimation)
- >1.0: Expansion (overestimation)

### Mean Absolute Error (MAE)
- Average absolute difference between TA and TP
- Units: seconds
- Lower is better

### Root Mean Square Error (RMSE)
- Square root of mean squared errors
- Penalizes large errors more than MAE
- Units: seconds

### Relative Error
- (TP - TA) / TA × 100
- Units: percentage
- Shows proportional accuracy

---

## Examples

### Example 1: Basic Analysis
```python
from hippocampus_data_loader import HippocampusDataLoader
from neural_visualization import NeuralVisualizer

data = HippocampusDataLoader.load_mat_file('session1.mat')
fig = NeuralVisualizer.plot_ta_vs_tp(data)
```

### Example 2: Color by Condition
```python
# Color points by start landmark
fig = NeuralVisualizer.plot_ta_vs_tp_detailed(
    data,
    color_by='curr'
)
```

### Example 3: Custom Trial Selection
```python
# Analyze only successful trials
success_trials = data.filter_trials(succ=1)
fig = NeuralVisualizer.plot_ta_vs_tp(
    data,
    trial_mask=success_trials
)
```

### Example 4: Multiple Conditions
```python
# Analyze each trial type separately
for trial_type in [1, 2, 3]:
    mask = data.filter_trials(trial_type=trial_type)
    fig = NeuralVisualizer.plot_ta_vs_tp(data, trial_mask=mask)
    save_figure(fig, f'ta_vs_tp_type{trial_type}.png')
```

---

## Troubleshooting

### Issue: "cond_matrix must have at least 2 columns"
**Solution:** Verify your data has TA and TP in columns 1 and 2

### Issue: "Column 10 not found, using all trials"
**Solution:** This is a warning, not an error. The code will use all trials instead of filtering.

### Issue: Points don't cluster near unity line
**Solution:** This may be normal! Check:
- Subject performance during the task
- Task difficulty
- Temporal ranges being tested

### Issue: Very low correlation (<0.3)
**Possible causes:**
- Subject not performing task correctly
- Wrong columns selected for TA/TP
- Data quality issues

---

## Tips for Best Results

1. **Always check your data first:**
   ```python
   print(data.summary())
   print(f"TA range: {data.cond_matrix[:, 0].min():.2f} - {data.cond_matrix[:, 0].max():.2f}")
   print(f"TP range: {data.cond_matrix[:, 1].min():.2f} - {data.cond_matrix[:, 1].max():.2f}")
   ```

2. **Use appropriate trial masks:**
   - First attempts only: More reliable
   - Successful trials only: Better performance estimates
   - All trials: Complete picture

3. **Compare across sessions:**
   - Track performance over time
   - Look for learning effects
   - Identify outlier sessions

4. **Use coloring strategically:**
   - `color_by='curr'`: See if starting position affects performance
   - `color_by='target'`: See if target affects performance
   - `color_by='succ'`: Distinguish successful vs. failed trials

---

## Version History

**v2.1+** (Current)
- Added `plot_ta_vs_tp()` function
- Added `plot_ta_vs_tp_detailed()` function
- Integrated into main analysis pipeline
- Created standalone `plot_ta_vs_tp.py` script
- Automatic detection of column 10 mask

---

## Citation

If you use the TA vs TP analysis in your research, please cite the main pipeline and mention the temporal production analysis feature.

---

**Questions?** Check the main README.md or open an issue on GitHub.
