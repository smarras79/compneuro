# Hippocampus Neural Data Analysis Pipeline

A comprehensive Python toolkit for analyzing neural activity from hippocampal recordings during mental navigation tasks in primates. This pipeline uses advanced AI and machine learning techniques to decode behavioral states, identify time cells, analyze population dynamics, and detect sequential activity patterns.

## Overview

This analysis suite is designed for neural data from experiments where animals mentally navigate between landmarks using learned sequences. The toolkit provides:

- **Data Loading**: Robust `.mat` file reader with automatic data validation
- **Temporal Coding Analysis**: Time cell identification and temporal tuning characterization
- **Machine Learning Decoding**: Multiple ML algorithms (Bayesian, SVM, Random Forest, Neural Networks) to decode landmark pairs and temporal distances
- **Dimensionality Reduction**: PCA, ICA, t-SNE, and UMAP for neural manifold analysis
- **Sequence Detection**: Algorithms to identify sequential neural activity patterns
- **Publication-Quality Visualizations**: Comprehensive plotting tools for all analyses

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
cd /path/to/compneuro
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For enhanced dimensionality reduction:
```bash
pip install umap-learn
```

## Data Structure

The pipeline expects `.mat` files with the following structure:

### Required Fields

- **`neur_tensor_trialon`**: Neural firing rates (neurons × time × trials)
  - 1ms time bins
  - Time range: -500ms to +9500ms relative to start landmark onset
  - Units: spikes/second (Hz)

- **`cond_matrix`**: Condition matrix (trials × condition_labels)
  - Each row represents one trial
  - Columns contain experimental parameters

### Optional Fields

- **`lfp_tensor_trialon`**: Local field potential data (channels × time × trials)
  - 1kHz sampling rate
  - Same time range as neural data

### Condition Labels

The `cond_matrix` typically contains these columns (in order):

| Column | Label | Description |
|--------|-------|-------------|
| 1 | `ta` | True temporal distance (seconds) |
| 2 | `tp` | Produced temporal distance by animal (seconds) |
| 3 | `curr` | Start landmark (1-6) |
| 4 | `target` | Target landmark (1-6) |
| 5 | `trial_type` | 1=visible, 2=sequence occluded, 3=fully occluded |
| 6 | `seqq` | Sequence identity (1,2=normal speed, 3=1.5× slower) |
| 7 | `succ` | Success flag (1/0) |
| 8 | `validtrials_mm` | Valid trials via mixture model |
| 9 | `attempt` | Trial attempt number |
| 10-12 | - | Additional experimental parameters |

## Quick Start

### Basic Usage

Analyze a single `.mat` file:

```bash
python example_analysis.py data/session1.mat
```

This will:
1. Load the neural data
2. Filter for mental navigation trials
3. Perform all analyses
4. Generate visualizations
5. Save results to `results/session1/`

### Analyze Multiple Files

Process all `.mat` files in a directory:

```bash
python example_analysis.py data/ --pattern "*.mat" --output results/
```

### Custom Output Directory

```bash
python example_analysis.py data/session1.mat --output my_results/
```

## Modules

### 1. Data Loader (`hippocampus_data_loader.py`)

Load and preprocess neural data from `.mat` files.

```python
from hippocampus_data_loader import HippocampusDataLoader

# Load data
data = HippocampusDataLoader.load_mat_file('session1.mat')

# View summary
print(data.summary())

# Filter trials
mental_nav_trials = data.get_mental_navigation_trials()
success_trials = data.filter_trials(succ=1, trial_type=3)

# Extract neural activity
activity = data.get_neural_activity(
    trial_mask=mental_nav_trials,
    time_window=(0, 3000)  # ms
)
```

### 2. AI-Powered Analysis (`neural_analysis_ai.py`)

#### Temporal Coding Analysis

Identify time cells and analyze temporal tuning:

```python
from neural_analysis_ai import TemporalCodingAnalyzer

analyzer = TemporalCodingAnalyzer()

# Compute temporal tuning curves
tuning = analyzer.compute_temporal_tuning(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(0, 3000),
    smooth_sigma=50.0
)

# Identify time cells
time_cells = analyzer.identify_time_cells(
    tuning,
    tmi_threshold=0.3,
    info_threshold=0.1
)

print(f"Found {np.sum(time_cells)} time cells")
```

#### Population Decoding

Decode behavioral variables using machine learning:

```python
from neural_analysis_ai import PopulationDecoder

# Decode landmark pairs
decoder = PopulationDecoder(method='random_forest')
results = decoder.decode_landmark_pairs(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(500, 2000),
    cv_folds=5
)

print(f"Decoding accuracy: {results['accuracy']:.3f}")
print(f"Chance level: {results['chance_level']:.3f}")

# Decode temporal distance (regression)
decoder = PopulationDecoder(method='bayesian')
temporal_results = decoder.decode_temporal_distance(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(500, 2000)
)

print(f"Correlation: {temporal_results['correlation']:.3f}")
```

**Available decoding methods:**
- `'bayesian'`: Gaussian Naive Bayes
- `'svm'`: Support Vector Machine with RBF kernel
- `'random_forest'`: Random Forest (100 trees)
- `'logistic'`: Logistic Regression
- `'mlp'`: Multi-Layer Perceptron Neural Network

#### Neural Manifold Analysis

Analyze population dynamics in low-dimensional space:

```python
from neural_analysis_ai import NeuralManifoldAnalyzer

# PCA analysis
analyzer = NeuralManifoldAnalyzer(method='pca', n_components=10)
manifold = analyzer.fit_transform(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(0, 3000)
)

# Check variance explained
var_exp = manifold['variance_explained']
print(f"First 3 PCs explain {np.sum(var_exp[:3])*100:.1f}% variance")

# Compute trajectory similarity
curr = data.get_condition('curr')[mental_nav_trials]
similarity = analyzer.compute_trajectory_similarity(
    manifold['trajectories'],
    curr
)
```

**Available methods:**
- `'pca'`: Principal Component Analysis
- `'ica'`: Independent Component Analysis
- `'tsne'`: t-Distributed Stochastic Neighbor Embedding
- `'umap'`: Uniform Manifold Approximation and Projection (requires umap-learn)
- `'nmf'`: Non-Negative Matrix Factorization

#### Sequence Analysis

Detect sequential neural activity patterns:

```python
from neural_analysis_ai import SequenceAnalyzer

analyzer = SequenceAnalyzer()
sequences = analyzer.detect_sequences(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(0, 3000),
    min_neurons=5
)

print(f"Sequence score: {sequences['sequence_score']:.3f}")
```

### 3. Visualization (`neural_visualization.py`)

Generate publication-quality figures:

```python
from neural_visualization import NeuralVisualizer, save_figure

# Raster plot
fig = NeuralVisualizer.plot_raster(
    activity=data.neur_tensor,
    time_vector=data.time_vector,
    trial_idx=0
)
save_figure(fig, 'raster_plot.png', dpi=300)

# Temporal tuning curves
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=12)
save_figure(fig, 'tuning_curves.png')

# Population heatmap
fig = NeuralVisualizer.plot_population_heatmap(
    activity=mean_activity,
    time_vector=time_vec,
    sort_by='peak'
)

# Decoding results
fig = NeuralVisualizer.plot_decoding_results(decoding_results)

# 3D neural trajectories
fig = NeuralVisualizer.plot_3d_trajectory(
    manifold_results,
    condition_labels=landmark_pairs
)

# Time cell summary
fig = NeuralVisualizer.plot_time_cells_summary(tuning, time_cells)

# Sequence analysis
fig = NeuralVisualizer.plot_sequence_analysis(sequences, time_vec)
```

## Output Files

The analysis pipeline generates the following outputs in the results directory:

### Figures (PNG format, 300 DPI)

- `temporal_tuning_curves.png`: Tuning curves of top temporally modulated neurons
- `population_heatmap.png`: Population activity sorted by peak time
- `time_cells_summary.png`: Time cell identification summary statistics
- `decoding_random_forest.png`: Decoding performance and confusion matrix
- `pca_variance.png`: PCA variance explained
- `neural_trajectories_3d.png`: 3D neural state space trajectories
- `sequence_analysis.png`: Sequential activity patterns

### Text Files

- `analysis_summary.txt`: Summary statistics and key findings

## Analysis Pipeline

The complete analysis pipeline (`example_analysis.py`) performs:

1. **Data Loading**: Load `.mat` file and validate structure
2. **Trial Filtering**: Extract mental navigation trials (fully occluded, normal speed, first attempts)
3. **Temporal Coding**:
   - Compute temporal tuning curves
   - Identify time cells
   - Analyze temporal information content
4. **Population Decoding**:
   - Decode landmark pairs using multiple ML methods
   - Decode temporal distances (regression)
5. **Manifold Analysis**:
   - PCA dimensionality reduction
   - 3D trajectory visualization
   - Trajectory similarity analysis
6. **Sequence Detection**:
   - Identify sequential activation patterns
   - Compute sequence scores
7. **Visualization**: Generate all plots
8. **Summary Report**: Save key findings

## Advanced Usage

### Custom Trial Filtering

```python
# Filter by multiple conditions
custom_trials = data.filter_trials(
    trial_type=3,          # Fully occluded
    seqq='<3',            # Normal speed
    succ=1,               # Successful trials only
    curr=1,               # Specific start landmark
    target='!=1'          # Target not equal to start
)

# Complex filtering with custom logic
ta = data.get_condition('ta')
tp = data.get_condition('tp')
accurate_trials = np.abs(ta - tp) < 0.5  # Within 500ms error
combined_mask = custom_trials & accurate_trials
```

### Time Window Analysis

Analyze different epochs of the task:

```python
epochs = {
    'fixation': (-500, 0),
    'start_cue': (0, 400),
    'target_cue': (400, 800),
    'navigation': (800, 3800),
    'feedback': (3800, 5000)
}

for epoch_name, (t_start, t_end) in epochs.items():
    activity = data.get_neural_activity(
        trial_mask=mental_nav_trials,
        time_window=(t_start, t_end)
    )
    # Analyze epoch-specific activity
```

### Batch Processing

Process multiple sessions:

```python
from pathlib import Path
import pandas as pd

data_dir = Path('data/')
results_list = []

for mat_file in data_dir.glob('*.mat'):
    data = HippocampusDataLoader.load_mat_file(mat_file)
    mn_trials = data.get_mental_navigation_trials()

    # Analyze
    analyzer = TemporalCodingAnalyzer()
    tuning = analyzer.compute_temporal_tuning(data, mn_trials)
    time_cells = analyzer.identify_time_cells(tuning)

    # Store results
    results_list.append({
        'session': mat_file.stem,
        'n_neurons': data.n_neurons,
        'n_trials': np.sum(mn_trials),
        'n_time_cells': np.sum(time_cells),
        'percent_time_cells': np.sum(time_cells) / data.n_neurons * 100
    })

# Create summary dataframe
df = pd.DataFrame(results_list)
print(df)
df.to_csv('session_summary.csv', index=False)
```

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No files matching '*.mat'`
- **Solution**: Check that your data files have the `.mat` extension and are in the specified directory

**Issue**: `ValueError: Missing required fields`
- **Solution**: Ensure your `.mat` file contains `neur_tensor_trialon` and `cond_matrix` fields

**Issue**: `Warning: Some classes have fewer samples than CV folds`
- **Solution**: This is normal for rare landmark pairs. The code automatically adjusts the number of cross-validation folds

**Issue**: Import error for UMAP
- **Solution**: Install UMAP: `pip install umap-learn`

**Issue**: Figures not displaying
- **Solution**: If running on a remote server, figures are saved to disk automatically

### Performance Tips

- For large datasets (>200 neurons, >1000 trials), consider:
  - Reducing `smooth_sigma` for faster temporal tuning computation
  - Using fewer cross-validation folds for decoding
  - Reducing `n_components` for dimensionality reduction
  - Analyzing subsets of neurons for initial exploration

## Citation

If you use this analysis pipeline in your research, please cite:

```bibtex
@software{hippocampus_analysis_pipeline,
  title = {AI-Powered Hippocampus Neural Data Analysis Pipeline},
  author = {Computational Neuroscience Analysis Team},
  year = {2025},
  version = {1.0}
}
```

## License

This code is provided for research purposes. Please contact the authors for commercial use.

## Support

For questions, issues, or feature requests:
- Open an issue on the GitHub repository
- Contact: [your email]

## Acknowledgments

This pipeline was developed for analyzing hippocampal recordings during mental navigation tasks, building on methods from computational neuroscience and machine learning.

## Version History

- **v1.0** (2025-11-15): Initial release
  - Data loading and preprocessing
  - Temporal coding analysis
  - ML-based decoding (5 methods)
  - Dimensionality reduction (5 methods)
  - Sequence detection
  - Comprehensive visualization suite
