# Neural Analysis Toolkit - Complete User Guide

## 🎯 Overview

This toolkit provides comprehensive AI/ML-enhanced analysis for neural time series data, specifically designed for detecting Sharp-Wave Ripples (SWRs) and analyzing patterns in hippocampal recordings.

### Key Features

✅ **6 Signal Filtering Options**
- Moving average, Gaussian, Savitzky-Golay, Butterworth, Median, Exponential

✅ **Spectral Analysis**
- Power spectral density (PSD)
- Spectrograms
- Wavelet transforms
- Frequency band analysis (delta, theta, alpha, beta, gamma, ripple)

✅ **Sharp-Wave Ripple Detection**
- Classical threshold-based detection
- Bandpass filtering in ripple band (150-250 Hz)
- Envelope detection and event validation
- Feature extraction for each event

✅ **Event-Triggered Analysis**
- Detect SWRs around behavioral events
- Analyze neural activity at motion onset/offset
- Time-locked analysis with custom windows

✅ **Machine Learning Pattern Detection**
- Feature extraction (12+ statistical features)
- K-means clustering for event classification
- Anomaly detection (Isolation Forest-inspired)
- PCA dimensionality reduction for visualization

✅ **Comprehensive Visualizations**
- Automated plot generation
- Publication-quality figures
- Interactive analysis reports

---

## 📦 Modules

The toolkit consists of 4 main modules:

### 1. `neural_spectral_analysis.jl`
Frequency-domain analysis tools

**Functions:**
- `compute_psd()` - Power spectral density
- `compute_spectrogram()` - Time-frequency analysis
- `compute_wavelet_transform()` - Wavelet analysis
- `analyze_frequency_bands()` - Band power analysis
- `identify_dominant_frequencies()` - Peak detection

### 2. `swr_detection.jl`
Sharp-Wave Ripple detection algorithms

**Functions:**
- `detect_swr_classical()` - Threshold-based SWR detection
- `detect_swr_at_events()` - Event-triggered SWR analysis
- `extract_event_features()` - Feature extraction
- `bandpass_filter()` - Butterworth bandpass filtering
- `compute_envelope()` - Hilbert envelope
- `visualize_swr_events()` - Event visualization

### 3. `ml_pattern_detection.jl`
Machine learning pattern recognition

**Functions:**
- `extract_statistical_features()` - Extract 12+ features
- `sliding_window_features()` - Windowed feature extraction
- `kmeans_clustering()` - K-means algorithm
- `detect_anomalies_isolation()` - Anomaly detection
- `detect_anomalies_statistical()` - Z-score based detection
- `classify_events_knn()` - K-nearest neighbors classifier
- `pca_reduction()` - Dimensionality reduction
- `analyze_event_patterns()` - Comprehensive pattern analysis

### 4. `integrated_analysis_pipeline.jl`
Unified analysis workflow

**Main Functions:**
- `analyze_neural_data_comprehensive()` - Full pipeline
- `quick_swr_analysis()` - Fast SWR detection
- `save_analysis_results()` - Export results
- `default_analysis_config()` - Configuration template

---

## 🚀 Quick Start

### Basic Usage

```julia
# Load the integrated pipeline (loads all modules)
include("integrated_analysis_pipeline.jl")

# Load your neural data
signal = your_neural_data  # Vector{Float64}
fs = 1000.0  # Sampling frequency in Hz

# Run quick SWR analysis
results = quick_swr_analysis(signal, fs; plot_results=true)

# View results
println("Detected $(results["n_events"]) SWR events")
```

### Complete Analysis Workflow

```julia
# 1. Define behavioral events (optional)
behavioral_events = Dict(
    "motion_onset" => [1.2, 3.5, 7.8],  # times in seconds
    "motion_offset" => [2.1, 4.3, 8.5]
)

# 2. Configure analysis
config = Dict(
    "ripple_band" => (150.0, 250.0),
    "swr_threshold_sd" => 3.0,
    "swr_min_duration" => 30.0,
    "swr_max_duration" => 200.0,
    "event_window_ms" => 500.0,
    "n_clusters" => 3
)

# 3. Run comprehensive analysis
results = analyze_neural_data_comprehensive(
    signal,
    behavioral_events;
    fs=fs,
    config=config
)

# 4. Save results
save_analysis_results(results, "./output_directory")
```

### With Signal Filtering

```julia
# Load your data
signal_raw = load_neural_data()

# Apply filtering first
include("sujay_example_zoom_likeMatlab_enhanced.jl")

signal_filtered = apply_neural_filter(
    signal_raw,
    :gaussian,  # Filter type
    300;        # Window size
    sigma=50    # Filter parameter
)

# Then analyze
results = analyze_neural_data_comprehensive(signal_filtered, nothing; fs=1000.0)
```

---

## 📊 Understanding the Results

### SWR Detection Results

```julia
results["swr_detection"] = Dict(
    "events" => [
        (start_sample=100, end_sample=150, peak_sample=125, 
         peak_amplitude=5.2, duration_ms=50.0),
        ...
    ],
    "envelope" => [...],           # Amplitude envelope
    "filtered_signal" => [...],    # Ripple-band signal
    "threshold" => 3.5,            # Detection threshold
    "n_events" => 42               # Total events
)
```

### Event-Triggered Results

```julia
results["event_triggered_motion_onset"] = Dict(
    "event_1" => Dict(
        "event_time" => 1.2,
        "event_sample" => 1200,
        "window" => (950, 1450),
        "swr_events" => [...],     # SWRs near this event
        "n_swr" => 3
    ),
    ...
)
```

### ML Pattern Analysis

```julia
results["ml_analysis"] = Dict(
    "n_events" => 42,
    "feature_matrix" => [...],     # 42 × 12 matrix
    "feature_names" => [...],
    "cluster_labels" => [...],     # Cluster for each event
    "n_clusters" => 3,
    "anomaly_scores" => [...],
    "is_anomaly" => [...],         # Boolean anomaly flags
    "pca_transformed" => [...],    # 2D PCA coordinates
    "explained_variance" => [0.45, 0.23]
)
```

---

## 🔧 Configuration Options

### SWR Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ripple_band` | (150, 250) Hz | Frequency range for ripples |
| `swr_threshold_sd` | 3.0 | Threshold in standard deviations |
| `swr_min_duration` | 30 ms | Minimum ripple duration |
| `swr_max_duration` | 200 ms | Maximum ripple duration |
| `merge_threshold_ms` | 50 ms | Merge events closer than this |

### Spectral Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `psd_window` | 512 | Window size for PSD |
| `spec_window` | 256 | Window for spectrogram |
| `compute_spectrogram` | true | Generate spectrogram |

### ML Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 3 | Number of k-means clusters |
| `contamination` | 0.1 | Expected anomaly proportion |

---

## 📈 Visualization Guide

### Generated Plots

1. **Power Spectral Density (PSD)**
   - Shows frequency content
   - Identifies dominant frequencies
   - Useful for quality check

2. **Spectrogram**
   - Time-frequency representation
   - Shows how frequencies change over time
   - Reveals transient events

3. **SWR Events**
   - Individual ripple examples
   - Shows raw signal with event markers
   - Verifies detection quality

4. **ML Clustering**
   - PCA visualization of event types
   - Shows event groupings
   - Identifies anomalies

5. **Frequency Band Power**
   - Bar chart of power distribution
   - Delta, theta, alpha, beta, gamma, ripple
   - Useful for state classification

---

## 🧪 Example Applications

### 1. Basic SWR Detection

```julia
include("integrated_analysis_pipeline.jl")

# Your data
signal = load_lfp_data()
fs = 1000.0

# Detect
results = quick_swr_analysis(signal, fs)

# Extract event times
swr_times = [e.peak_sample / fs for e in results["events"]]
println("SWRs at: ", swr_times)
```

### 2. Compare Different Filters

```julia
# Test multiple filters on same data
filters = [:moving_average, :gaussian, :butterworth]

for ftype in filters
    signal_filtered = apply_neural_filter(signal_raw, ftype, 300)
    results = quick_swr_analysis(signal_filtered, fs; plot_results=false)
    println("$ftype: $(results["n_events"]) SWRs")
end
```

### 3. Analyze Motion-Related SWRs

```julia
# Extract motion times from behavioral data
motion_onsets = extract_motion_onsets(behavioral_data)
motion_offsets = extract_motion_offsets(behavioral_data)

events = Dict(
    "motion_onset" => motion_onsets,
    "motion_offset" => motion_offsets
)

# Analyze
results = analyze_neural_data_comprehensive(signal, events; fs=fs)

# Check if SWRs occur more at onset or offset
onset_swr = results["event_triggered_motion_onset"]
offset_swr = results["event_triggered_motion_offset"]

println("SWRs at onset: $(sum([e["n_swr"] for (k,e) in onset_swr if k != "all_swr" && k != "total_events"]))")
println("SWRs at offset: $(sum([e["n_swr"] for (k,e) in offset_swr if k != "all_swr" && k != "total_events"]))")
```

### 4. Classify SWR Types

```julia
# Run ML analysis
results = analyze_neural_data_comprehensive(signal, nothing; fs=fs)

ml = results["ml_analysis"]

# Find "typical" SWRs (largest cluster)
cluster_counts = [sum(ml["cluster_labels"] .== i) for i in 1:ml["n_clusters"]]
typical_cluster = argmax(cluster_counts)

typical_swr_indices = findall(ml["cluster_labels"] .== typical_cluster)
anomalous_swr_indices = findall(ml["is_anomaly"])

println("Typical SWRs: $(length(typical_swr_indices))")
println("Anomalous SWRs: $(length(anomalous_swr_indices))")
```

### 5. Frequency Band Analysis by State

```julia
# Analyze different behavioral states
states = Dict(
    "rest" => signal_rest,
    "active" => signal_active
)

for (state_name, signal_data) in states
    results = analyze_neural_data_comprehensive(signal_data, nothing; fs=fs)
    bands = results["frequency_bands"]
    
    println("\n$state_name:")
    for (band, info) in bands
        if band != "total_power" && band != "freq" && band != "power_spectrum"
            println("  $band: $(round(info["relative_power"]*100, digits=1))%")
        end
    end
end
```

---

## 🎓 Scientific Background

### Sharp-Wave Ripples

**What are they?**
- High-frequency oscillations (150-250 Hz)
- Brief duration (50-150 ms)
- Occur in hippocampus during rest/sleep
- Associated with memory consolidation

**Detection approach:**
1. Bandpass filter (150-250 Hz)
2. Compute amplitude envelope (Hilbert)
3. Threshold at mean + 3-8 SD
4. Validate duration and frequency

**Interpretation:**
- More SWRs → more memory replay
- Timing relative to behavior is important
- Ripple frequency relates to information content

### Frequency Bands

| Band | Frequency | Associated With |
|------|-----------|----------------|
| Delta | 0.5-4 Hz | Deep sleep |
| Theta | 4-8 Hz | Navigation, REM |
| Alpha | 8-13 Hz | Relaxed wakefulness |
| Beta | 13-30 Hz | Active thinking |
| Low Gamma | 30-80 Hz | Attention, perception |
| High Gamma | 80-150 Hz | Cognitive processing |
| Ripple | 150-250 Hz | Memory replay |

---

## 🐛 Troubleshooting

### No SWRs Detected

**Problem:** `results["n_events"] = 0`

**Solutions:**
1. Lower threshold: `swr_threshold_sd = 2.5`
2. Widen frequency band: `ripple_band = (100, 300)`
3. Check signal quality: Plot raw signal
4. Adjust duration limits: `swr_min_duration = 20`

### Too Many False Positives

**Problem:** Detected events don't look like ripples

**Solutions:**
1. Raise threshold: `swr_threshold_sd = 4.0`
2. Narrow frequency band: `ripple_band = (150, 200)`
3. Stricter duration: `swr_min_duration = 40`
4. Use ML to filter: Check `is_anomaly` flags

### Length Mismatch Errors

**Problem:** `BoundsError` during plotting

**Solution:** Already fixed in latest version!
- Time vectors automatically adjusted
- All filters handle edge effects consistently

### Butterworth Filter Fails

**Problem:** `MethodError` with Lowpass

**Solution:** Already fixed!
- Frequencies normalized internally
- Ensure `cutoff_freq < fs/2`

### Out of Memory

**Problem:** Large datasets crash

**Solutions:**
1. Process in chunks:
   ```julia
   for i in 1:n_chunks
       segment = signal[start:end]
       results_i = quick_swr_analysis(segment, fs)
   end
   ```

2. Reduce spectrogram resolution:
   ```julia
   config["spec_window"] = 128  # Instead of 256
   config["compute_spectrogram"] = false  # Skip if not needed
   ```

---

## 📚 References

### SWR Detection Methods
- Buzsáki et al. (1992) - High-frequency network oscillation in the hippocampus
- Cheng & Frank (2008) - New experiences enhance coordinated neural activity

### ML in Neuroscience
- Stevenson et al. (2008) - Inferring functional connections
- Glaser et al. (2020) - Machine learning for neural data analysis

### Signal Processing
- Oppenheim & Schafer - Discrete-Time Signal Processing
- Mallat - A Wavelet Tour of Signal Processing

---

## 🔄 Updates and Extensions

### Coming Soon
- Deep learning SWR detector (CNN/LSTM)
- Phase-amplitude coupling analysis
- Granger causality for connectivity
- Real-time detection capabilities
- Multi-channel analysis

### Want to Contribute?
The code is modular! Add your own:
- New filter types in `apply_neural_filter()`
- Detection algorithms in `swr_detection.jl`
- ML methods in `ml_pattern_detection.jl`
- Analysis pipelines in `integrated_analysis_pipeline.jl`

---

## 💡 Best Practices

### 1. Always Visualize
```julia
# Check raw signal quality
plot(signal[1:10000])

# Verify detected events
visualize_swr_events(signal, results["events"], fs)
```

### 2. Start Conservative
```julia
# Begin with strict parameters
swr_threshold_sd = 4.0
swr_min_duration = 50.0

# Then relax if needed
```

### 3. Validate with ML
```julia
# Use anomaly detection to find outliers
ml = results["ml_analysis"]
good_events = results["swr_detection"]["events"][.!ml["is_anomaly"]]
```

### 4. Document Your Parameters
```julia
# Save configuration with results
config_used = Dict(
    "filter" => :gaussian,
    "window" => 300,
    "ripple_band" => (150, 250),
    "threshold" => 3.0
)
```

### 5. Compare Methods
```julia
# Try different thresholds
for thresh in [2.5, 3.0, 3.5, 4.0]
    results = detect_swr_classical(signal, fs; threshold_sd=thresh)
    println("Threshold $thresh: $(results["n_events"]) events")
end
```

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the example scripts
3. Examine module documentation (docstrings)
4. Try the diagnostic tools (`test_filter_lengths.jl`)

---

## 🎉 Summary

This toolkit provides everything you need for:
- ✅ Professional neural signal filtering
- ✅ Accurate SWR detection
- ✅ Comprehensive spectral analysis
- ✅ Advanced ML pattern recognition
- ✅ Event-triggered analysis
- ✅ Beautiful visualizations

**One command runs it all:**
```julia
results = analyze_neural_data_comprehensive(signal, events; fs=1000.0)
```

Happy analyzing! 🧠✨
