# 🚀 QUICKSTART GUIDE

## Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script:
```bash
bash setup.sh
```

### Step 2: Test Installation

```bash
python test_installation.py
```

You should see:
```
✓ All required dependencies are installed!
✓ All custom modules loaded successfully!
✓ INSTALLATION SUCCESSFUL!
```

### Step 3: Run Analysis

```bash
python example_analysis.py path/to/your/data.mat
```

Replace `path/to/your/data.mat` with your actual data file.

## 📂 Example

If your data is at `data/amadeus01172020_a_neur_tensor_joyon.mat`:

```bash
python example_analysis.py data/amadeus01172020_a_neur_tensor_joyon.mat
```

Results will be saved to:
```
results/amadeus01172020_a_neur_tensor_joyon/
├── temporal_tuning_curves.png
├── population_heatmap.png
├── time_cells_summary.png
├── decoding_random_forest.png
├── decoding_bayesian.png
├── decoding_svm.png
├── pca_variance.png
├── neural_trajectories_3d.png
├── sequence_analysis.png
└── analysis_summary.txt
```

## 🔍 What Gets Analyzed?

1. **Temporal Coding**
   - Identifies time cells
   - Computes temporal tuning curves
   - Measures temporal information

2. **Population Decoding**
   - Decodes landmark pairs using multiple ML methods
   - Decodes temporal distances (regression)
   - Cross-validated performance metrics

3. **Neural Manifold**
   - PCA dimensionality reduction
   - 3D trajectory visualization
   - Within/between condition comparisons

4. **Sequence Detection**
   - Sequential activation patterns
   - Sequence scores and time lags

## 💡 Common Commands

### Analyze single file with custom output:
```bash
python example_analysis.py data.mat --output my_results/
```

### Analyze all .mat files in a directory:
```bash
python example_analysis.py data/ --pattern "*.mat"
```

### Get help:
```bash
python example_analysis.py --help
```

## ⚠️ Troubleshooting

### MATLAB v7.3 files (HDF5)
✅ **Automatically handled!** The loader detects the format and uses the correct reader.

If you see an error about v7.3 files:
```bash
pip install h5py
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Import errors
Make sure you're in the correct directory:
```bash
cd /path/to/compneuro
python test_installation.py
```

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [FIX_GUIDE.md](FIX_GUIDE.md) if you have MATLAB v7.3 files
- Explore the example scripts to customize your analysis

## 🎯 File Overview

| File | Purpose |
|------|---------|
| `hippocampus_data_loader.py` | Load .mat files (auto-detects v7.3) |
| `neural_analysis_ai.py` | All analysis algorithms |
| `neural_visualization.py` | Plotting functions |
| `example_analysis.py` | **Main script - run this!** |
| `test_installation.py` | Test your setup |
| `requirements.txt` | Python dependencies |

## ✅ Quick Validation

After running your analysis, you should have:
- ✅ 9 PNG figures (300 DPI, publication quality)
- ✅ 1 text summary file
- ✅ All files in `results/<session_name>/`

**That's it! You're ready to analyze hippocampal neural data!** 🧠
