# Getting Started - Neural Analysis Toolkit

## 🎯 Quick Start Guide

This guide walks you through the **actual** analysis pipeline as implemented in `main.jl`.

---

## What You'll Learn

1. How to configure the analysis
2. How to load and extract your data
3. How to apply signal filtering
4. How to extract behavioral events
5. How to run comprehensive SWR analysis
6. How to interpret and save results

---

## Step 1: Configure Your Analysis

At the top of `main.jl`, set your parameters:

```julia
# ========== CONFIGURATION ==========
fs = 1000.0           # Sampling frequency (Hz) - MUST MATCH YOUR SYSTEM
window_size = 300     # Filter window size
ineuron = 1           # Which neuron to analyze (1, 2, 3, ...)

# Select filter type
selected_filter = :moving_average  # Options: :none, :moving_average, 
                                   #          :gaussian, :savitzky_golay,
                                   #          :butterworth, :median, :exponential
```

**Key Configuration:**
- `fs`: Your recording system's sampling rate
- `window_size`: Larger = smoother signal, smaller = more detail
- `ineuron`: Which neuron in your multi-neuron recording
- `selected_filter`: How to smooth the signal (or `:none` for raw data)

---

## Step 2: Load Your Data

The script loads a `.mat` file with specific structure:

```julia
# Load the .mat file
data = matread("./data/amadeus01172020_a_neur_tensor_stim1on.mat")
```

**What's in the file:**
- `neur_tensor_stim1on`: 3D array (neurons × time × trials)
- `cond_matrix`: Behavioral conditions for each trial
- `edges` or `stim1on`: Time base for recordings
- `cond_label`: Labels for condition matrix columns

**Data Structure:**
```
neur_tensor_stim1on: (3 neurons, 3998 time points, 120 trials)
cond_matrix: (120 trials, 12 condition columns)
edges: (3998,) time points from -2.0s to +2.0s
```

---

## Step 3: Extract Neural Data

The script extracts specific experimental conditions:

```julia
# Extract condition 4: specific experimental parameters
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[ineuron, :, trid]

# Compute mean across trials
fr3_mean = vec(mean(fr3, dims=2))
```

**What's happening:**
1. Find trials matching specific conditions
2. Extract neural data for those trials
3. Average across trials to get mean firing rate

---

## Step 4: Apply Signal Filtering

Multiple filter options are available:

```julia
# Apply the selected filter
fr3_smooth = apply_neural_filter(fr3_mean, selected_filter, window_size; 
                                 filter_params...)
```

**Available Filters:**

| Filter | Best For | Parameters |
|--------|----------|------------|
| `:none` | Raw signal | None |
| `:moving_average` | General smoothing | window_size |
| `:gaussian` | Smooth, preserves shape | window_size, sigma |
| `:savitzky_golay` | Preserves peaks | window_size, poly_order |
| `:butterworth` | Remove high frequency | cutoff_freq, filter_order |
| `:median` | Remove spikes | window_size |
| `:exponential` | Adaptive smoothing | alpha |

**Important:** Filtering trims the signal edges! The script automatically adjusts time bins to match.

---

## Step 5: Extract Behavioral Events

The script automatically extracts when behavior occurred:

```julia
# Auto-detect motion column
suggested_column = auto_detect_motion_column(cond_matrix)

# Extract motion events
behavioral_events = extract_motion_events(
    cond_matrix, 
    edges;
    position_column=suggested_column,
    threshold_quantile=0.75,  # Top 25% of changes
    motion_duration=0.5,       # 500ms per event
    fs=fs,
    method=:position_change
)
```

**What You Get:**
```julia
behavioral_events = Dict(
    "motion_onset" => [times when motion started],
    "motion_offset" => [times when motion stopped],
    "n_events" => total count
)
```

**Fallback:** If no motion detected, uses trial onset times instead.

---

## Step 6: Time Alignment (Automatic)

The script checks and fixes time alignment issues:

```julia
# Check if events are outside filtered signal bounds
if n_events_outside > 0
    # Filter events to match signal
    # Remove events outside the filtered range
end

# Shift events to match SWR detection coordinate system
time_offset = time_bins[1]  # e.g., -1.7s
behavioral_events_shifted = Dict{String, Any}()

for (event_type, times) in behavioral_events
    shifted_times = times .- time_offset  # Convert to 0-based
    behavioral_events_shifted[event_type] = shifted_times
end
```

**Why This Matters:**
- Original recording: -2.0s to +2.0s (event-aligned)
- Filtered signal: -1.7s to +1.7s (trimmed by filtering)
- SWR detection: 0.0s to 3.4s (internal time axis)

Events must be shifted to match the SWR detection coordinate system!

---

## Step 7: Run Comprehensive Analysis

```julia
results = analyze_neural_data_comprehensive(
    signal_filtered,
    behavioral_events;
    fs=fs,
    config=Dict(
        "ripple_band" => (150.0, 250.0),      # SWR frequency range
        "swr_threshold_sd" => 3.0,             # Detection threshold
        "swr_min_duration" => 30.0,            # Minimum 30ms
        "swr_max_duration" => 200.0,           # Maximum 200ms
        "event_window_ms" => 500.0,            # ±250ms around events
        "n_clusters" => 3                      # ML clustering
    )
)
```

**What Gets Analyzed:**
1. ✅ **SWR Detection**: Finds sharp-wave ripples in 150-250 Hz band
2. ✅ **Frequency Analysis**: Power in delta, theta, alpha, beta, gamma bands
3. ✅ **Event-Triggered Analysis**: SWRs near behavioral events
4. ✅ **Enrichment**: Are SWRs more common during behavior?
5. ✅ **ML Clustering**: Groups similar SWR events

---

## Step 8: View Results

The script displays comprehensive results:

```julia
# Console output shows:
📊 NEURAL EVENTS DETECTED (Sharp-Wave Ripples):
  Total SWRs: 42
  Duration: 65.3 ± 15.2 ms
  Amplitude: 4.5 ± 1.2

🎯 BEHAVIORAL EVENTS EXTRACTED:
  motion_onset: 459 events
  motion_offset: 468 events

🔗 EVENT-TRIGGERED ANALYSIS:
  SWRs within ±250ms of motion_onset: 125
  Rate: 0.27 SWRs per event
  Enrichment: 2.3x baseline

📈 FREQUENCY BAND POWER:
  delta (0.5-4 Hz): 18.7%
  theta (4-8 Hz): 20.2%
  alpha (8-13 Hz): 15.3%
  beta (13-30 Hz): 22.1%
  low_gamma (30-80 Hz): 12.2%
  high_gamma (80-150 Hz): 8.4%
  ripple (150-250 Hz): 3.1%
```

---

## Output Files

Results are saved to `./neural_analysis_output/`:

```
neural_analysis_output/
├── neural_plot_1_moving_average.png       # Smoothed firing rates
├── scatter_1_moving_average.png           # Behavioral scatter plots
├── psd.png                                 # Power spectral density
├── spectrogram.png                         # Time-frequency analysis
├── swr_events.png                          # Example SWR events
├── event_triggered_swr.png                 # SWRs around behavior
├── ml_clustering.png                       # Event clusters
├── frequency_bands.png                     # Band power comparison
├── signal_with_events.png                  # Signal + event markers
└── analysis_summary.txt                    # Complete text report
```

---

## Customizing the Analysis

### Change Detection Threshold

```julia
config = Dict(
    "swr_threshold_sd" => 2.5  # Lower = more sensitive (more SWRs)
                                # Higher = more selective (fewer SWRs)
)
```

### Change Filter Type

```julia
selected_filter = :gaussian
filter_params = Dict(:sigma => 100.0)  # Smoother
```

### Analyze Different Neuron

```julia
ineuron = 2  # Or 3, 4, ... depending on your data
```

### Extract Different Conditions

```julia
# Modify the condition extraction
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 4] .== 5))  # Different condition
```

---

## Complete Workflow Summary

```
1. CONFIGURE
   ├─ Set fs, window_size, ineuron
   └─ Choose filter type

2. LOAD DATA
   ├─ Read .mat file
   └─ Extract experimental conditions

3. PROCESS SIGNAL
   ├─ Average across trials
   ├─ Apply filtering
   └─ Adjust time bins

4. EXTRACT EVENTS
   ├─ Auto-detect behavioral events
   ├─ Filter events to signal bounds
   └─ Shift to SWR time base

5. ANALYZE
   ├─ Detect SWRs
   ├─ Analyze frequencies
   ├─ Event-triggered analysis
   └─ ML clustering

6. RESULTS
   ├─ Display summary
   ├─ Save plots
   └─ Write report
```

---

## Troubleshooting

### Error: "LoadError: KeyError"
```julia
# Check available keys in your data
println(keys(data))

# Adjust variable names accordingly
neur_tensor = data["your_actual_key"]
```

### Error: "DimensionMismatch"
```julia
# Check tensor dimensions
println(size(neur_tensor_stim1on))

# Should be: (neurons, time, trials)
```

### No SWRs Detected
```julia
# Lower the threshold
config = Dict("swr_threshold_sd" => 2.0)

# Or check if signal is in correct units (should be firing rate in Hz)
```

### Time Alignment Warnings
```julia
# This is normal! The script automatically handles it
# Events outside filtered signal bounds are removed
# Then shifted to match SWR detection coordinate system
```

### First-Run Visualization Error
```julia
# Also normal! The precompilation fix handles it
# If error persists on second run, check enhanced_visualization.jl
```

---

## Advanced: Understanding Time Coordinates

**Three Time Coordinate Systems:**

1. **Original Recording** (`edges`):
   - Range: -2.0s to +2.0s
   - Length: 3998 samples
   - Centered on stimulus onset

2. **Filtered Signal** (`time_bins`):
   - Range: -1.7s to +1.7s  
   - Length: 1700 samples
   - Trimmed by filtering

3. **SWR Detection Internal**:
   - Range: 0.0s to 3.4s
   - Length: 1700 samples
   - Always starts at 0

**The script handles all conversions automatically!**

---

## Next Steps

1. ✅ Run `main.jl` with default settings
2. 📊 Check `./neural_analysis_output/` for results
3. 🔧 Adjust configuration parameters
4. 📖 Read full documentation for deeper understanding
5. 🧪 Try different filter types and thresholds

---

## Quick Reference

**Essential Files:**
- `main.jl` - Main analysis script (this document)
- `integrated_analysis_pipeline.jl` - Analysis functions
- `behavioral_event_extraction.jl` - Event detection
- `enhanced_visualization.jl` - Publication-quality plots
- `auxiliary_functions.jl` - Filtering functions

**Key Parameters:**
- `fs = 1000.0` - Sampling frequency
- `selected_filter = :moving_average` - Filter type
- `swr_threshold_sd = 3.0` - Detection threshold
- `event_window_ms = 500.0` - Time window around events

**Run the script:**
```bash
julia main.jl
```

**That's it!** The script is fully automated and handles all the complexity internally. 🎉

---

## Summary: What main.jl Does

```julia
# Pseudocode version
configure_parameters()
load_data()
extract_conditions()
apply_filtering()
extract_behavioral_events()
check_and_fix_time_alignment()
shift_events_to_swr_coordinates()
run_comprehensive_analysis()
display_and_save_results()
```

**All automatic. All integrated. All publication-ready.** ✨
