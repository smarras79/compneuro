# Hippocampus Neural Data Analysis Pipeline

A comprehensive Python toolkit for analyzing neural activity from hippocampal recordings during mental navigation tasks in primates. This pipeline uses advanced AI and machine learning techniques to decode behavioral states, identify time cells, analyze population dynamics, and detect sequential activity patterns.

## 🎯 Overview

This analysis suite is designed for neural data from experiments where animals mentally navigate between landmarks using learned sequences. The toolkit provides:

* **✅ MATLAB v7.3 Support**: Robust `.mat` file reader with automatic format detection (old MATLAB and HDF5/v7.3)
* **⏱️ Temporal Coding Analysis**: Time cell identification and temporal tuning characterization
* **🧠 Machine Learning Decoding**: Multiple ML algorithms to decode landmark pairs and temporal distances
* **📊 Dimensionality Reduction**: PCA, ICA, t-SNE, and UMAP for neural manifold analysis
* **🔄 Sequence Detection**: Algorithms to identify sequential neural activity patterns
* **📈 Publication-Quality Visualizations**: Comprehensive plotting tools for all analyses

## 📦 Installation

### Prerequisites

* Python 3.7 or higher
* pip package manager

### Quick Setup

1. **Clone or download this repository**:
   ```bash
   cd /path/to/compneuro
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test installation**:
   ```bash
   python test_installation.py
   ```

   You should see:
   ```
   ✓ All required dependencies are installed!
   ✓ All custom modules loaded successfully!
   ✓ INSTALLATION SUCCESSFUL!
   ```

## 🚀 Quick Start

### Analyze a Single File

```bash
python example_analysis.py data/amadeus01172020_a_neur_tensor_joyon.mat
```

This will:
1. ✅ Load the neural data (automatically detects MATLAB v7.3 format)
2. 📊 Filter for mental navigation trials
3. 🔬 Perform all analyses
4. 📈 Generate visualizations
5. 💾 Save results to `results/amadeus01172020_a_neur_tensor_joyon/`

### Analyze Multiple Files

Process all `.mat` files in a directory:

```bash
python example_analysis.py data/ --pattern "*.mat" --output results/
```

### Custom Output Directory

```bash
python example_analysis.py data/session1.mat --output my_results/
```

## 📁 Data Structure

The pipeline expects `.mat` files with the following structure:

### Required Fields

* **`neur_tensor_trialon`**: Neural firing rates
  + Shape: `(neurons × time × trials)`
  + Time bins: 1ms
  + Time range: -500ms to +9500ms relative to start landmark onset
  + Units: spikes/second (Hz)

* **`cond_matrix`**: Condition matrix
  + Shape: `(trials × condition_labels)`
  + Each row represents one trial
  + Columns contain experimental parameters

### Optional Fields

* **`lfp_tensor_trialon`**: Local field potential data
  + Shape: `(channels × time × trials)`
  + Sampling rate: 1kHz

### Condition Labels (cond_matrix columns)

| Column | Label | Description |
|--------|-------|-------------|
| 0 | `ta` | True temporal distance (seconds) |
| 1 | `tp` | Produced temporal distance by animal (seconds) |
| 2 | `curr` | Start landmark (1-6) |
| 3 | `target` | Target landmark (1-6) |
| 4 | `trial_type` | 1=visible, 2=sequence occluded, 3=fully occluded |
| 5 | `seqq` | Sequence identity (1,2=normal, 3=slower) |
| 6 | `succ` | Success flag (1/0) |
| 7 | `validtrials_mm` | Valid trials via mixture model |
| 8 | `attempt` | Trial attempt number |

## 📚 Modules

### 1. Data Loader (`hippocampus_data_loader.py`)

Load and preprocess neural data from `.mat` files with **automatic MATLAB v7.3 detection**.

```python
from hippocampus_data_loader import HippocampusDataLoader

# Load data (works with both old MATLAB and v7.3 formats!)
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

**Key Features:**
- ✅ Automatic detection of MATLAB format (old vs v7.3)
- ✅ Handles HDF5-based MATLAB v7.3 files
- ✅ Proper array transposition for HDF5 format
- ✅ Comprehensive data validation

### 2. AI-Powered Analysis (`neural_analysis_ai.py`)

#### Temporal Coding Analysis

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
```

#### Population Decoding

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

print(f"Accuracy: {results['accuracy']:.3f}")
```

**Available methods:** `'bayesian'`, `'svm'`, `'random_forest'`, `'logistic'`, `'mlp'`

#### Neural Manifold Analysis

```python
from neural_analysis_ai import NeuralManifoldAnalyzer

analyzer = NeuralManifoldAnalyzer(method='pca', n_components=10)
manifold = analyzer.fit_transform(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(0, 3000)
)
```

**Available methods:** `'pca'`, `'ica'`, `'tsne'`, `'umap'`, `'nmf'`

#### Sequence Analysis

```python
from neural_analysis_ai import SequenceAnalyzer

analyzer = SequenceAnalyzer()
sequences = analyzer.detect_sequences(
    neural_data=data,
    trial_mask=mental_nav_trials,
    time_window=(0, 3000)
)
```

### 3. Visualization (`neural_visualization.py`)

Generate publication-quality figures:

```python
from neural_visualization import NeuralVisualizer, save_figure

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
```

## 📊 Output Files

The analysis pipeline generates:

### Figures (PNG, 300 DPI)
- `temporal_tuning_curves.png` - Tuning curves of top neurons
- `population_heatmap.png` - Population activity sorted by peak time
- `time_cells_summary.png` - Time cell identification statistics
- `decoding_random_forest.png` - Decoding performance
- `pca_variance.png` - PCA variance explained
- `neural_trajectories_3d.png` - 3D neural state space
- `sequence_analysis.png` - Sequential activity patterns

### Text Files
- `analysis_summary.txt` - Summary statistics and key findings

## 🔧 Troubleshooting

### Common Issues

**Issue:** `Error loading data: Please use HDF reader for matlab v7.3 files`
- **Solution:** This is now automatically handled! The new loader detects v7.3 format and uses h5py. Make sure you've installed all requirements: `pip install -r requirements.txt`

**Issue:** `ModuleNotFoundError: No module named 'h5py'`
- **Solution:** Install h5py: `pip install h5py`

**Issue:** `FileNotFoundError: No files matching '*.mat'`
- **Solution:** Check that your data files are in the specified directory

**Issue:** `ValueError: Missing required fields`
- **Solution:** Ensure your `.mat` file contains `neur_tensor_trialon` and `cond_matrix`

**Issue:** `Warning: Some classes have fewer samples than CV folds`
- **Solution:** This is normal for rare landmark pairs. The code automatically adjusts CV folds.

## 🎓 Complete Analysis Pipeline

The `example_analysis.py` script performs:

1. ✅ **Data Loading** - Load and validate .mat file (any MATLAB version)
2. 🎯 **Trial Filtering** - Extract mental navigation trials
3. ⏱️ **Temporal Coding** - Compute tuning curves and identify time cells
4. 🧠 **Population Decoding** - Decode landmark pairs and temporal distances
5. 📊 **Manifold Analysis** - PCA and 3D trajectory visualization
6. 🔄 **Sequence Detection** - Identify sequential activation patterns
7. 📈 **Visualization** - Generate all plots
8. 📝 **Summary Report** - Save key findings

## 📖 Advanced Usage

### Custom Trial Filtering

```python
# Multiple conditions
custom_trials = data.filter_trials(
    trial_type=3,      # Fully occluded
    seqq='<3',        # Normal speed
    succ=1,           # Successful only
    curr=1,           # Specific start landmark
    target='!=1'      # Target not equal to start
)
```

### Batch Processing

```python
from pathlib import Path
import pandas as pd

results_list = []

for mat_file in Path('data/').glob('*.mat'):
    data = HippocampusDataLoader.load_mat_file(mat_file)
    # ... analyze ...
    results_list.append({
        'session': mat_file.stem,
        'n_neurons': data.n_neurons,
        'n_time_cells': np.sum(time_cells)
    })

df = pd.DataFrame(results_list)
df.to_csv('session_summary.csv')
```

## 📄 Citation

If you use this pipeline in your research, please cite:

```
@software{hippocampus_analysis_pipeline,
  title = {AI-Powered Hippocampus Neural Data Analysis Pipeline},
  author = {Computational Neuroscience Analysis Team},
  year = {2025},
  version = {2.0}
}
```

## 📜 License

This code is provided for research purposes. Please contact the authors for commercial use.

## 🤝 Support

For questions, issues, or feature requests:
- Open an issue on the GitHub repository
- Contact the development team

## 🔄 Version History

* **v2.0** (2025-11-15): MATLAB v7.3 support
  + Automatic detection of MATLAB format
  + HDF5/h5py integration for v7.3 files
  + Improved error handling and validation
  
* **v1.0** (2025-11-14): Initial release
  + Data loading and preprocessing
  + Temporal coding analysis
  + ML-based decoding (5 methods)
  + Dimensionality reduction (5 methods)
  + Sequence detection
  + Comprehensive visualization suite

## ✨ What's New in v2.0

The major improvement is **automatic MATLAB v7.3 support**:

- ✅ Works with both old MATLAB formats and v7.3 (HDF5)
- ✅ Automatic format detection - no user intervention needed
- ✅ Proper handling of HDF5 array transposition
- ✅ Better error messages and validation

No changes needed to your analysis code - it just works!

---

**Ready to analyze your data?**

```bash
# Test installation
python test_installation.py

# Run analysis
python example_analysis.py your_data.mat
```
