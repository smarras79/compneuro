# 🧠 Neural Analysis Toolkit - Complete Package

## What You Have

A complete, professional-grade neural analysis toolkit for detecting Sharp-Wave Ripples (SWRs), analyzing neural-behavioral coupling, and generating publication-quality visualizations. Features automatic behavioral event extraction, time alignment correction, and comprehensive statistical analysis.

---

## File Structure

### Main Analysis Script

**`main.jl`** - Complete automated pipeline (~584 lines)
- Loads data from .mat files
- Extracts specific experimental conditions
- Applies configurable signal filtering
- Auto-detects behavioral events
- Handles time alignment automatically
- Shifts events to SWR coordinate system
- Runs comprehensive SWR analysis
- Generates publication-quality plots
- Saves complete results

**This is your entry point!** Everything else is called from here.

---

### 🔬 Core Analysis Modules

1. **`integrated_analysis_pipeline.jl`** - Main analysis engine (~600 lines)
   - SWR detection (150-250 Hz band)
   - Frequency band analysis (delta through high-gamma)
   - Event-triggered analysis
   - Statistical enrichment calculations
   - ML-based clustering
   - Automatic visualization generation
   - Result saving and reporting

2. **`swr_detection.jl`** - Classical ripple detection (~400 lines)
   - Bandpass filtering (ripple band)
   - Envelope detection
   - Threshold-based event detection
   - Duration and amplitude filtering
   - Peak finding and feature extraction
   - Event-triggered analysis around behavioral events

3. **`neural_spectral_analysis.jl`** - Frequency analysis (~350 lines)
   - Power spectral density (Welch method)
   - Multi-taper spectrograms
   - Wavelet time-frequency analysis
   - Band power calculations (delta, theta, alpha, beta, gamma, ripple)
   - Relative power normalization

4. **`ml_pattern_detection.jl`** - Machine learning (~450 lines)
   - Statistical feature extraction
   - K-means clustering of events
   - Anomaly detection (isolation forest)
   - KNN classification
   - PCA dimensionality reduction
   - Pattern recognition in neural events

---

### 🎨 Visualization & Filtering

5. **`enhanced_visualization.jl`** - Publication-quality plots (~168 lines)
   - High-resolution figures (600 DPI)
   - Signal + behavioral event markers
   - Spectrograms with event overlays
   - Event-triggered averages
   - Multi-panel summary figures
   - Colorblind-friendly palettes

6. **`auxiliary_functions.jl`** - Signal filtering suite
   - 7 filter types:
     - `:none` - Raw signal
     - `:moving_average` - General smoothing
     - `:gaussian` - Shape-preserving
     - `:savitzky_golay` - Peak-preserving
     - `:butterworth` - Frequency cutoff
     - `:median` - Spike removal
     - `:exponential` - Adaptive smoothing
   - Automatic length adjustment
   - Parameter optimization

---

### Event Extraction

7. **`behavioral_event_extraction.jl`** - Automatic event detection (~300 lines)
   - Auto-detects motion/position columns
   - Extracts motion onset/offset times
   - Position-change based detection
   - Velocity threshold method
   - Fallback to trial-based events
   - Time alignment with neural data

---

### 📚 Documentation

8. **`README.md`** - This file
    - Package structure
    - File descriptions
    - Quick reference
    
9. **`GETTING_STARTED.md`** ⭐ - START HERE!
   - Step-by-step walkthrough
   - Configuration guide
   - Complete workflow explanation
   - Troubleshooting tips
   
10. **`NEURAL_ANALYSIS_GUIDE.md`** - In-depth guide
    - Scientific background
    - Algorithm details
    - Parameter tuning
    - Advanced usage

11. **`DATA_LOADING_GUIDE.md`** - Data extraction
    - MAT file structure
    - Condition extraction
    - Trial selection
    - Data preprocessing

12. **`FILTERING_GUIDE.md`** - Signal filtering
    - Filter comparison
    - When to use each filter
    - Parameter selection
    - Examples

13. **`TROUBLESHOOTING.md`** - Common issues
    - Error messages
    - Solutions
    - Debugging tips

14. **`QUICK_REFERENCE.md`** - One-page cheat sheet
    - Key functions
    - Essential parameters
    - Quick examples

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Configure

Edit the top of `main.jl`:

```julia
# ========== CONFIGURATION ==========
fs = 1000.0              # Sampling frequency (Hz)
window_size = 300        # Filter window size
ineuron = 1              # Which neuron to analyze
selected_filter = :moving_average  # Filter type
```

### Step 2: Run

```bash
julia main.jl
```

### Step 3: View Results

Check `./neural_analysis_output/` for:
- 📊 Plots of neural activity
- 📈 SWR detection results
- 🎯 Event-triggered analysis
- 📋 Complete text summary

**That's it!** The script handles everything automatically.

---

## 📊 What Gets Analyzed

### 1. Sharp-Wave Ripple Detection
- Bandpass filter (150-250 Hz)
- Envelope detection
- Threshold crossing (3σ)
- Duration filtering (30-200 ms)
- Feature extraction (amplitude, duration, peak time)

**Output:** List of detected SWR events with properties

### 2. Frequency Band Analysis
- Power spectral density
- Band power calculations:
  - Delta: 0.5-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-30 Hz
  - Low Gamma: 30-80 Hz
  - High Gamma: 80-150 Hz
  - Ripple: 150-250 Hz

**Output:** Relative power in each band, spectrograms

### 3. Behavioral Event Extraction
- Auto-detects motion from behavioral data
- Identifies onset and offset times
- Filters events to match neural signal bounds
- Shifts to SWR detection coordinate system

**Output:** Dictionary of event times by type

### 4. Event-Triggered Analysis
- Finds SWRs within ±250ms of behavioral events
- Calculates occurrence rates
- Computes enrichment factors
- Statistical significance testing

**Output:** SWR counts per event type, enrichment ratios

### 5. Machine Learning
- K-means clustering of SWR events
- Feature-based classification
- Anomaly detection
- Pattern recognition

**Output:** Event clusters, classifications, anomaly scores

---

## 🔧 Customization

### Change SWR Detection Threshold

```julia
config = Dict(
    "swr_threshold_sd" => 2.5  # Lower = more SWRs (more sensitive)
)
```

### Try Different Filter

```julia
selected_filter = :gaussian
filter_params = Dict(:sigma => 100.0)
```

### Analyze Different Neuron

```julia
ineuron = 2  # Or 3, 4, ...
```

### Adjust Event Window

```julia
config = Dict(
    "event_window_ms" => 1000.0  # ±500ms instead of ±250ms
)
```

---

## 📈 Complete Workflow

```
┌─────────────────────────┐
│   CONFIGURATION         │
│  (fs, filters, neuron)  │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   LOAD DATA             │
│  (.mat file → arrays)   │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   EXTRACT CONDITIONS    │
│  (specific trials)      │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   SIGNAL FILTERING      │
│  (smooth, denoise)      │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   BEHAVIORAL EVENTS     │
│  (auto-detect motion)   │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   TIME ALIGNMENT        │
│  (filter & shift)       │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   SWR DETECTION         │
│  (150-250 Hz events)    │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   FREQUENCY ANALYSIS    │
│  (band powers, PSD)     │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   EVENT COUPLING        │
│  (SWRs near behavior)   │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   ML CLUSTERING         │
│  (pattern recognition)  │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   VISUALIZATION         │
│  (publication plots)    │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│   SAVE RESULTS          │
│  (plots + summary)      │
└─────────────────────────┘
```

---

## Key Features

### ✅ Automatic Time Alignment
- Handles coordinate system mismatches
- Filters events to signal bounds
- Shifts to SWR detection coordinates
- **No manual intervention needed!**

### ✅ Flexible Filtering
- 7 filter types
- Easy parameter adjustment
- Automatic length matching
- Compare filters visually

### ✅ Auto Event Detection
- Detects motion from behavioral data
- Falls back to trial onsets if needed
- Validates event quality
- Reports extraction success

### ✅ Comprehensive Analysis
- Classical signal processing
- Modern ML techniques
- Statistical testing
- Enrichment calculations

---

## Output Files

After running `main.jl`, check `./neural_analysis_output/`:

```
neural_analysis_output/
├── neural_plot_1_moving_average.png      # Smoothed neural data
├── scatter_1_moving_average.png          # Behavioral scatter
├── psd.png                                # Power spectrum
├── spectrogram.png                        # Time-frequency
├── swr_events.png                         # Example SWRs
├── event_triggered_swr.png                # SWRs + behavior
├── ml_clustering.png                      # Event clusters
├── frequency_bands.png                    # Band comparison
├── signal_with_events.png                 # Signal + markers
└── analysis_summary.txt                   # Full report
```

---

## 🔍 Understanding Time Coordinates

**Critical Concept:** Three time coordinate systems

1. **Original Recording** (`edges`):
   ```
   Range: -2.0s to +2.0s (event-aligned)
   Length: 3998 samples
   ```

2. **Filtered Signal** (`time_bins`):
   ```
   Range: -1.7s to +1.7s (trimmed by filtering)
   Length: 1700 samples
   ```

3. **SWR Detection** (internal):
   ```
   Range: 0.0s to 3.4s (always starts at 0)
   Length: 1700 samples
   ```

**The script handles all conversions automatically!**

Event at -1.5s → shifted to 0.2s for SWR detection ✓

---

## Scientific Background

### Sharp-Wave Ripples (SWRs)
- High-frequency oscillations (150-250 Hz)
- Brief duration (30-200 ms)
- Associated with memory consolidation
- Occur during quiet rest and sleep
- Coupled with behavioral states

### Analysis Methods
1. **Bandpass Filtering**: Isolate ripple band
2. **Envelope Detection**: Find oscillation peaks
3. **Threshold Crossing**: Detect significant events
4. **Feature Extraction**: Characterize each event
5. **Event Coupling**: Relate to behavior
6. **Statistical Testing**: Assess significance

### Machine Learning
- **Clustering**: Find event subtypes
- **Classification**: Predict event categories
- **Anomaly Detection**: Identify unusual patterns
- **PCA**: Reduce dimensionality

---

## 🛠Dependencies

### Required Packages
```julia
using MAT          # Load .mat files
using Statistics   # Mean, std, etc.
using DSP          # Signal processing
using Plots        # Visualization
using Printf       # Formatted output
```

### Install Missing Packages
```julia
using Pkg
Pkg.add(["MAT", "Statistics", "DSP", "Plots"])
```

---

## 📖 Documentation Roadmap

**New User:**
1. Read `GETTING_STARTED.md` ⭐
2. Run `main.jl`
3. Check output files
4. Adjust parameters

**Understanding Data:**
1. Read `DATA_LOADING_GUIDE.md`
2. Learn MAT file structure
3. Extract your conditions
4. Customize analysis

**Improving Signal:**
1. Read `FILTERING_GUIDE.md`
2. Try different filters
3. Optimize parameters
4. Compare results

**Deep Dive:**
1. Read `NEURAL_ANALYSIS_GUIDE.md`
2. Understand algorithms
3. Tune detection parameters
4. Interpret results

**Troubleshooting:**
1. Check `TROUBLESHOOTING.md`
2. Look up error messages
3. Follow solutions
4. Report issues

---

## 🎯 Common Use Cases

### Use Case 1: Basic SWR Detection
```julia
# Just run the script!
julia main.jl
```

### Use Case 2: Compare Filters
```julia
# Run with different filters
selected_filter = :moving_average
# ... run ...

selected_filter = :gaussian
# ... run ...

# Compare output plots
```

### Use Case 3: Analyze Multiple Neurons
```julia
# Run for each neuron
for neuron in 1:3
    global ineuron = neuron
    # ... run analysis ...
end
```

### Use Case 4: Parameter Sweep
```julia
# Test different thresholds
for threshold in [2.0, 2.5, 3.0, 3.5]
    config = Dict("swr_threshold_sd" => threshold)
    # ... run with config ...
end
```

---

## 🔧 Advanced Customization

### Modify SWR Detection Algorithm

Edit `swr_detection.jl`:
```julia
function detect_swr_events(signal, fs; 
                           ripple_band=(150.0, 250.0),
                           threshold_sd=3.0,
                           min_duration=30.0,
                           max_duration=200.0)
    # Your modifications here
end
```

### Add Custom Frequency Bands

Edit `neural_spectral_analysis.jl`:
```julia
bands = Dict(
    "custom_band" => (100.0, 120.0),  # Add your band
    # ...
)
```

### Custom Event Extraction

Edit `behavioral_event_extraction.jl`:
```julia
function extract_custom_events(data, ...)
    # Your extraction logic
end
```

---

## 📊 Result Interpretation

### SWR Count
- **Low (<10)**: Increase sensitivity (lower threshold)
- **Normal (10-100)**: Good detection
- **High (>100)**: Decrease sensitivity or check signal quality

### Event Enrichment
- **<1.0**: SWRs less common during behavior (suppression)
- **~1.0**: No coupling
- **>1.0**: SWRs more common during behavior (activation)
- **>2.0**: Strong coupling

### Frequency Bands
- **High Delta/Theta**: Sleep-like state
- **High Alpha**: Resting state
- **High Beta/Gamma**: Active processing
- **High Ripple**: Strong SWR activity

---

## 🐛 Troubleshooting Quick Guide

| Error | Solution |
|-------|----------|
| KeyError | Check MAT file keys: `println(keys(data))` |
| DimensionMismatch | Verify tensor shape: `println(size(neur_tensor))` |
| No SWRs detected | Lower threshold: `swr_threshold_sd = 2.0` |
| Time alignment warnings | Normal! Script handles automatically |
| First-run viz error | Normal! Precompilation fix handles it |
| LoadError | Check file paths are correct |

**See `TROUBLESHOOTING.md` for complete solutions**

---

## 📚 Additional Resources

- **Example Analysis:** `example_complete_analysis.jl`
- **Filter Comparison:** `filter_comparison_demo.jl`
- **Length Diagnostic:** `test_filter_lengths.jl`
- **Filter Examples:** `FILTER_EXAMPLES.md`
- **Quick Reference:** `QUICK_REFERENCE.md`

---

## 🎉 Summary

This toolkit provides:
- ✅ Complete automated pipeline
- ✅ Publication-quality output
- ✅ Flexible customization
- ✅ Comprehensive documentation
- ✅ Robust error handling
- ✅ Professional visualization

**Everything you need for neural analysis!**

---

## 🚀 Getting Started Command

```bash
# Clone/download the toolkit
cd neural-analysis-toolkit

# Install dependencies (first time only)
julia -e 'using Pkg; Pkg.add(["MAT", "Statistics", "DSP", "Plots"])'

# Run analysis
julia main.jl

# Check results
ls ./neural_analysis_output/

# Success! 🎉
```

---

**Version:** 2.0  
**Last Updated:** December 2024  
**Maintainer:** Computational Neuroscience Lab

For questions, issues, or feature requests, please refer to the documentation or contact the development team.

**Happy Analyzing!** 🧠✨
