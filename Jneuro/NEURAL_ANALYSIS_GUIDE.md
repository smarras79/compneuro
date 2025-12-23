# Sharp-Wave Ripple (SWR) Detection and Analysis Pipeline
## Comprehensive Guide for Parameter Tuning and Usage

---

## Overview

This Julia pipeline performs **hippocampal neural data analysis** with a focus on:
1. **Sharp-Wave Ripple (SWR) Detection**: Identifying high-frequency oscillations (150-250 Hz) characteristic of memory consolidation
2. **Behavioral Event Extraction**: Detecting motion/position changes from experimental recordings
3. **Neural-Behavioral Coupling**: Quantifying temporal relationships between SWRs and behavior
4. **Machine Learning Pattern Analysis**: Clustering SWRs into distinct types based on their features

---

## Pipeline Architecture

```
Raw Neural Data (.mat file)
    ↓
Signal Preprocessing & Filtering
    ↓
┌─────────────────────┬────────────────────────┐
│   SWR Detection     │  Behavioral Events     │
│   (Neural Signal)   │  (Position/Motion)     │
└─────────────────────┴────────────────────────┘
    ↓
Event-Triggered Analysis
    ↓
Statistical & ML Analysis
    ↓
Visualization & Reports
```

---

## Section-by-Section Code Walkthrough

### 1. Configuration Setup (Lines 1-21)

```julia
fs = 500.0           # Sampling frequency (Hz)
window_size = 300    # Filter window size
```

**What it does:**
- Defines fundamental parameters for signal processing
- Sets up the temporal resolution of your analysis

**Key Parameters:**
- `fs` (500 Hz): Should match your recording system's sampling rate
- `window_size` (300 samples = 600 ms at 500 Hz): Controls smoothing granularity

**When to adjust:**
- ✅ **Use fs = 500 Hz** if your system records at 500 samples/second
- ✅ **Use fs = 1000 Hz** or **fs = 1500 Hz** for higher sampling rate recordings
- ⚠️ Incorrect `fs` will cause time misalignment in all downstream analyses

---

### 2. Data Loading (Lines 23-27)

```julia
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
```

**What it does:**
- Loads preprocessed neural activity data from MATLAB format
- Expected structure: `neur_tensor_stim1on` with dimensions [neurons × time × trials]

**Requirements:**
- File must contain: `neur_tensor_stim1on`, `cond_matrix`, `cond_label`, `stim1on`
- Alternative file noted: `amadeus01172020_a_neur_tensor_joyon.mat`

---

### 3. Filter Selection (Lines 29-53)

```julia
selected_filter = :moving_average
```

**What it does:**
- Applies temporal smoothing to reduce noise while preserving neural dynamics

**Available Filters:**
| Filter Type | Best For | Parameters |
|------------|----------|------------|
| `:moving_average` | General smoothing, interpretable | `window_size` |
| `:gaussian` | Smooth without sharp edges | `sigma` |
| `:savitzky_golay` | Preserving peaks/troughs | `poly_order` |
| `:butterworth` | Frequency-specific cutoff | `cutoff_freq`, `filter_order` |
| `:median` | Removing outlier spikes | `window_size` |
| `:exponential` | Real-time applications | `alpha` |

**Tuning Recommendations:**
```julia
# Conservative (more smoothing)
window_size = 500      # ~1 second smoothing at 500 Hz

# Moderate (balanced)
window_size = 300      # ~600 ms (default)

# Aggressive (minimal smoothing)
window_size = 100      # ~200 ms
```

⚠️ **Caution:** Over-smoothing (large `window_size`) can obscure rapid SWR events!

---

### 4. Neural Data Extraction (Lines 64-81)

```julia
# Condition 4: column 10==1 & column 3==1 & column 4==4
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]
```

**What it does:**
- Selects trials matching specific experimental conditions
- Extracts firing rates for the first neuron across selected trials

**Condition Logic:**
- **Column 10**: Likely indicates valid/included trials
- **Column 3**: Could represent stimulus type or task phase
- **Column 4**: Values 4 and 5 represent different conditions being compared

**To customize:** Check `cond_label` output to understand what each column represents in your experiment.

---

### 5. Behavioral Data Visualization (Lines 83-95)

```julia
neural_plots_scatter(ta_att1, tp_att1, "1")
```

**What it does:**
- Visualizes behavioral variables (e.g., target angle vs. target position)
- Creates separate plots for different attention conditions

**Purpose:** Quality control to ensure behavioral data is sensible before neural analysis.

---

### 6. Signal Filtering Application (Lines 98-134)

```julia
fr3_smooth = apply_neural_filter(fr3_mean, selected_filter, window_size; filter_params...)
```

**What it does:**
- Computes trial-averaged firing rates
- Applies selected filter to reduce noise
- Handles edge effects by trimming filtered signals

**Critical Detail:**
```julia
additional_trim = (length(fr3_mean) - length(fr3_smooth)) ÷ 2
time_start = 150 + additional_trim
```
This ensures time bins align perfectly with filtered signals (prevents plotting errors).

---

### 7. Behavioral Event Extraction (Lines 136-181)

```julia
behavioral_events = extract_motion_events(
    cond_matrix, 
    edges;
    position_column=suggested_column,
    threshold_quantile=0.75,
    motion_duration=0.5,
    fs=fs,
    method=:position_change
)
```

**What it does:**
- **Auto-detects** which column in `cond_matrix` contains position/motion data
- Identifies times when the animal moved (large position changes)
- Falls back to trial onset times if motion detection fails

**Key Parameters for Tuning:**

| Parameter | Default | Effect | Tuning Guide |
|-----------|---------|--------|--------------|
| `threshold_quantile` | 0.75 | Sensitivity of motion detection | **0.90** = Only large movements<br>**0.75** = Moderate (default)<br>**0.50** = Detect subtle movements |
| `motion_duration` | 0.5 s | Assumed duration of each motion event | Adjust based on your task (e.g., 0.3 s for rapid movements) |
| `position_column` | Auto | Which behavioral column to analyze | Manually set if auto-detection fails (e.g., `position_column=2`) |

**Fallback Behavior:**
```julia
# If no motion detected:
behavioral_events = extract_trial_events(cond_matrix, edges; fs=fs)
```
Uses trial onset times instead (useful for event-related designs).

---

### 8. Comprehensive SWR Analysis (Lines 189-218)

```julia
results = analyze_neural_data_comprehensive(
    signal_filtered,
    behavioral_events;
    fs=fs,
    config=Dict(
        "ripple_band" => (150.0, 250.0),
        "swr_threshold_sd" => 3.0,
        "swr_min_duration" => 30.0,
        "swr_max_duration" => 200.0,
        "event_window_ms" => 500.0,
        "n_clusters" => 3
    )
)
```

**What it does:**
This is the **core analysis engine** with four main components:

#### A. **SWR Detection Algorithm**
1. Bandpass filter signal to ripple frequencies (150-250 Hz)
2. Compute power envelope (Hilbert transform)
3. Detect periods where power exceeds threshold
4. Filter by duration criteria

#### B. **Event-Triggered Analysis**
- For each behavioral event, counts SWRs within ±250 ms window
- Calculates enrichment: Are SWRs more common near behavior?

#### C. **ML Pattern Clustering**
- Extracts features from each SWR (duration, amplitude, frequency content)
- Uses K-means to identify SWR subtypes

#### D. **Frequency Band Analysis**
- Computes power spectrum
- Quantifies energy in delta, theta, alpha, beta, gamma bands

---

## Critical Parameters for SWR Detection

### 🎯 Primary Tuning Parameters

#### 1. **Ripple Frequency Band**
```julia
"ripple_band" => (150.0, 250.0)  # Hz
```

**Biological Background:**
- **Rat hippocampus**: 150-250 Hz (default)
- **Mouse hippocampus**: 150-250 Hz
- **Human hippocampus**: 80-140 Hz (different!)

**Tuning Guide:**
```julia
# For mouse/rat in vivo
"ripple_band" => (150.0, 250.0)   # Standard

# For human recordings
"ripple_band" => (80.0, 140.0)    # Slower ripples

# Narrow band (more specific)
"ripple_band" => (180.0, 220.0)   # Center of rat SWR range
```

---

#### 2. **Detection Threshold**
```julia
"swr_threshold_sd" => 3.0  # Standard deviations
```

**Effect on Detection:**
| Threshold | Sensitivity | Specificity | Use When... |
|-----------|-------------|-------------|-------------|
| **2.0 SD** | High (more SWRs) | Lower (more false positives) | Exploratory analysis, want to capture all possible events |
| **3.0 SD** | Moderate (balanced) | Good (default) | Standard analysis |
| **4.0 SD** | Low (fewer SWRs) | High (very strict) | Want only the clearest, strongest SWRs |

**Recommended Tuning Strategy:**
1. Start with **3.0 SD** (default)
2. Inspect example SWR events in output plots
3. If too many noise artifacts → **increase to 3.5 or 4.0**
4. If missing obvious SWRs → **decrease to 2.5**

---

#### 3. **Duration Constraints**
```julia
"swr_min_duration" => 30.0   # milliseconds
"swr_max_duration" => 200.0  # milliseconds
```

**Biological Constraints:**
- **Typical SWR**: 50-100 ms duration
- **Range**: 30-200 ms in literature

**Tuning Logic:**
```julia
# Very strict (high confidence)
"swr_min_duration" => 40.0
"swr_max_duration" => 150.0

# Permissive (capture brief events)
"swr_min_duration" => 20.0
"swr_max_duration" => 300.0
```

⚠️ **Common Issue:** If `swr_min_duration` is too large, you'll miss brief but valid SWRs!

---

#### 4. **Event Window for Behavioral Coupling**
```julia
"event_window_ms" => 500.0  # ±250 ms around behavior
```

**Interpretation:**
- **500 ms window** = Look 250 ms before and 250 ms after each behavioral event
- Used to calculate SWR enrichment near behavior

**Tuning:**
```julia
# Tight coupling (immediate relationship)
"event_window_ms" => 200.0   # ±100 ms

# Standard
"event_window_ms" => 500.0   # ±250 ms (default)

# Broad coupling (delayed effects)
"event_window_ms" => 1000.0  # ±500 ms
```

---

### 🔍 Secondary Parameters

#### 5. **ML Clustering**
```julia
"n_clusters" => 3
```

**What it determines:**
- Number of distinct SWR types to identify
- Based on features like duration, amplitude, spectral content

**Tuning:**
```julia
"n_clusters" => 2   # Simple categorization (e.g., weak vs strong)
"n_clusters" => 3   # Default (e.g., weak, medium, strong)
"n_clusters" => 4   # Fine-grained types
```

**How to choose:** Look at `ml_clustering.png` output. If clusters overlap heavily, reduce `n_clusters`.

---

## Results Interpretation Guide

### Section 1: Neural Events Detected

```
📊 NEURAL EVENTS DETECTED (Sharp-Wave Ripples):
  Total SWRs: 127
  Duration: 68.3 ± 24.1 ms
  Amplitude: 4.2 ± 1.3
```

**What this tells you:**
- **Total SWRs**: Number of detected events across entire recording
- **Duration**: Average length (should be 30-200 ms range)
- **Amplitude**: Peak power (in SD units above baseline)

**Quality Checks:**
✅ **Good SWR detection:**
- Duration: 40-150 ms (within biological range)
- Amplitude: 3-6 SD (clear above noise)
- Total: 50-500 SWRs per session (depends on recording length)

❌ **Problematic Detection:**
- Duration < 20 ms → Likely noise spikes, increase `swr_min_duration`
- Duration > 250 ms → Likely artifacts, decrease `swr_max_duration`
- Amplitude < 2 SD → Threshold too low, increase `swr_threshold_sd`
- Total SWRs > 1000 → Over-sensitive, increase threshold

---

### Section 2: Behavioral Events

```
🎯 BEHAVIORAL EVENTS EXTRACTED:
  motion_onset: 89 events
  motion_offset: 89 events
```

**Validation:**
- Number should match your experimental design
- Example: 90 trials with motion → expect ~90 motion_onset events

**If extraction failed:**
```
⚠️  No motion events detected from automatic extraction.
    Trying fallback: using trial-based events...
```

**Solutions:**
1. Manually specify `position_column` (check which column has position data)
2. Lower `threshold_quantile` to 0.5 (more sensitive)
3. Use trial onsets instead (disable motion detection)

---

### Section 3: Neural-Behavioral Enrichment

```
🔬 NEURAL-BEHAVIORAL COMPARISON:
  Near motion_onset:
    Total SWRs within ±250ms: 45
    Mean SWRs per event: 0.51
    Enrichment: 2.3x
    ⭐ ENRICHED - SWRs occur MORE near this behavior
```

**Enrichment Factor Interpretation:**

| Enrichment | Meaning | Biological Interpretation |
|------------|---------|---------------------------|
| **> 2.0x** | Strong enrichment | SWRs are temporally locked to behavior (e.g., memory encoding) |
| **1.5-2.0x** | Moderate enrichment | Weak coupling |
| **0.8-1.2x** | Random | No relationship (SWRs independent of behavior) |
| **< 0.7x** | Depletion | SWRs suppressed during behavior (e.g., active exploration inhibits replay) |

**Example Scenarios:**
- **Enrichment = 2.3x** → SWRs occur more than twice as often near motion compared to baseline
- **Enrichment = 0.5x** → SWRs are suppressed during motion (consistent with literature: SWRs occur during rest, not movement)

---

### Section 4: ML Pattern Analysis

```
🤖 MACHINE LEARNING PATTERN ANALYSIS:
  SWR event types found: 3
    Type 1: 45 events (35.4%)
    Type 2: 63 events (49.6%)
    Type 3: 19 events (15.0%)
  Anomalous SWRs: 7
```

**Interpretation:**
- **Type 1-3**: Distinct SWR subtypes based on features
- **Anomalous**: Outlier events (unusual duration, amplitude, or spectral content)

**Typical Patterns:**
- **Type 1**: Brief, weak SWRs
- **Type 2**: Standard SWRs (most common)
- **Type 3**: Long, high-amplitude SWRs

Check `ml_clustering.png` to see feature distributions for each type.

---

## Troubleshooting Guide

### Problem: No SWRs Detected

**Symptoms:**
```
Total SWRs: 0
```

**Solutions:**
1. **Lower detection threshold:**
   ```julia
   "swr_threshold_sd" => 2.0  # From 3.0
   ```

2. **Broaden frequency band:**
   ```julia
   "ripple_band" => (100.0, 300.0)  # From (150, 250)
   ```

3. **Relax duration constraints:**
   ```julia
   "swr_min_duration" => 20.0
   "swr_max_duration" => 300.0
   ```

4. **Check signal quality:**
   - Inspect `psd.png`: Is there power in the 150-250 Hz range?
   - If not, your recording may not contain ripples (check electrode placement)

---

### Problem: Too Many False Positives

**Symptoms:**
```
Total SWRs: 5000+
Duration: 15.2 ± 8.3 ms  (too brief)
```

**Solutions:**
1. **Increase detection threshold:**
   ```julia
   "swr_threshold_sd" => 4.0  # From 3.0
   ```

2. **Increase minimum duration:**
   ```julia
   "swr_min_duration" => 40.0  # From 30.0
   ```

3. **Narrow frequency band:**
   ```julia
   "ripple_band" => (180.0, 220.0)  # More specific
   ```

---

### Problem: Behavioral Event Extraction Fails

**Symptoms:**
```
⚠️  No motion events detected
```

**Solutions:**
1. **Manually set position column:**
   ```julia
   position_column = 2  # Instead of auto-detect
   ```

2. **Lower sensitivity:**
   ```julia
   threshold_quantile = 0.50  # From 0.75
   ```

3. **Use trial onsets instead:**
   ```julia
   # Comment out motion extraction, use:
   behavioral_events = extract_trial_events(cond_matrix, edges; fs=fs)
   ```

---

## Output Files Explained

After running, the pipeline generates:

| File | Content | Use For |
|------|---------|---------|
| `psd.png` | Power spectral density | Check if ripple frequencies (150-250 Hz) are present |
| `spectrogram.png` | Time-frequency plot | Visualize when high-frequency activity occurs |
| `swr_events.png` | Example detected SWRs | Quality control: Do these look like real ripples? |
| `ml_clustering.png` | SWR type distributions | Understand heterogeneity in your SWR population |
| `frequency_bands.png` | Power across bands | See relative contributions of delta, theta, gamma, etc. |
| `analysis_summary.txt` | Numerical results | Copy statistics for papers/presentations |

---

## Recommended Parameter Sets

### Conservative (High Specificity)
**Use when:** You want only the clearest, highest-confidence SWRs

```julia
config = Dict(
    "ripple_band" => (170.0, 230.0),    # Narrow band
    "swr_threshold_sd" => 4.0,           # Strict threshold
    "swr_min_duration" => 40.0,          # Exclude brief noise
    "swr_max_duration" => 150.0,         # Exclude long artifacts
    "event_window_ms" => 200.0           # Tight temporal coupling
)
```

---

### Balanced (Default)
**Use when:** Standard exploratory analysis

```julia
config = Dict(
    "ripple_band" => (150.0, 250.0),
    "swr_threshold_sd" => 3.0,
    "swr_min_duration" => 30.0,
    "swr_max_duration" => 200.0,
    "event_window_ms" => 500.0
)
```

---

### Sensitive (High Recall)
**Use when:** You want to capture all possible SWRs (e.g., pilot studies)

```julia
config = Dict(
    "ripple_band" => (120.0, 280.0),    # Broad band
    "swr_threshold_sd" => 2.5,           # Permissive threshold
    "swr_min_duration" => 20.0,          # Allow brief events
    "swr_max_duration" => 250.0,
    "event_window_ms" => 1000.0          # Broad temporal window
)
```

---

## Quick Start Checklist

- [ ] Set correct `fs` to match your recording system
- [ ] Load appropriate `.mat` file
- [ ] Choose filter type (start with `:moving_average`)
- [ ] Verify `cond_matrix` column meanings via `cond_label`
- [ ] Set `ripple_band` based on species (150-250 Hz for rodents)
- [ ] Start with default `swr_threshold_sd = 3.0`
- [ ] Check behavioral event extraction output
- [ ] Inspect generated plots (`swr_events.png`) for quality
- [ ] Adjust parameters if needed based on results
- [ ] Interpret enrichment factors for neural-behavioral coupling

---

## References & Biological Context

**Sharp-Wave Ripples:**
- **Frequency**: 150-250 Hz in rodents, 80-140 Hz in humans
- **Duration**: 30-150 ms typical
- **Function**: Memory consolidation, spatial replay
- **Occurrence**: Predominantly during rest/sleep, suppressed during movement

**Key Literature:**
- Buzsáki (2015) *Hippocampus*: "Hippocampal sharp wave-ripple: A cognitive biomarker"
- Wilson & McNaughton (1994): Reactivation of ensemble activity during sleep
- O'Neill et al. (2010): Place-selective firing and SWR generation

---

## Contact & Support

For questions about:
- **Parameter tuning**: Start with default values, then adjust based on output quality
- **Biological interpretation**: Consult literature on hippocampal physiology
- **Code issues**: Check that all dependencies are loaded (`myplots.jl`, `integrated_analysis_pipeline.jl`, etc.)

**Validation Strategy:**
1. Visual inspection of detected events
2. Comparison to hand-labeled ground truth (if available)
3. Consistency with known biology (duration, frequency, behavioral coupling)

---

*Document Version: 1.0*  
*Last Updated: December 2025*  
*Pipeline Version: Comprehensive Neural Analysis v2.0*
