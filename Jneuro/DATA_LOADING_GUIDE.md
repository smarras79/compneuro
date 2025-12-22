# Data Loading Guide - From .mat Files to Neural Analysis

## 🎯 Problem Solved

In documentation, you see examples like:
```julia
signal = your_neural_data  # ❓ Where does this come from??
```

This guide shows **exactly** where `your_neural_data` comes from and how to extract it.

---

## 📁 Your Data Structure

### File: `amadeus01172020_a_neur_tensor_stim1on.mat`

```julia
using MAT

# Load the file
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")

# What's inside?
println(keys(data))
# Output: ["cond_label", "cond_matrix", "neur_tensor_stim1on", "stim1on"]
```

### Data Contents

```julia
# 1. neur_tensor_stim1on: 3D array of neural activity
#    Dimensions: (neurons × time × trials)
#    Example: (3, 3998, 120) = 3 neurons, 3998 time bins, 120 trials
neur_tensor_stim1on = data["neur_tensor_stim1on"]

# 2. cond_matrix: Trial conditions/parameters
#    Dimensions: (trials × parameters)
#    Each row is a trial, columns are behavioral parameters
cond_matrix = data["cond_matrix"]

# 3. cond_label: Labels for condition matrix columns
#    Tells you what each column in cond_matrix means
cond_label = data["cond_label"]

# 4. stim1on: Time information (edges, bins, etc.)
stim1on = data["stim1on"]
```

---

## 📊 Extracting Neural Signal for Analysis

### Method 1: Single Neuron, Averaged Across Trials

```julia
using MAT
using Statistics

# Load data
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
neur_tensor = data["neur_tensor_stim1on"]

# Extract neuron 1, all time points, all trials
neuron_1_data = neur_tensor[1, :, :]  # (time × trials)

# Average across trials to get single time series
your_neural_data = vec(mean(neuron_1_data, dims=2))

# Now you have: Vector{Float64} with 3998 samples
println("Signal length: $(length(your_neural_data))")
# Output: Signal length: 3998
```

### Method 2: Specific Neuron, Specific Condition

```julia
using MAT
using Statistics

# Load data
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
cond_matrix = data["cond_matrix"]
neur_tensor = data["neur_tensor_stim1on"]

# Select trials based on condition
# Example: column 10 == 1, column 3 == 1, column 4 == 4
trial_indices = findall(
    (cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 4)
)

# Extract neuron 1 data for these trials only
neuron_data = neur_tensor[1, :, trial_indices]

# Average across selected trials
your_neural_data = vec(mean(neuron_data, dims=2))

println("Signal from $(length(trial_indices)) trials")
println("Signal length: $(length(your_neural_data))")
```

### Method 3: All Neurons Concatenated

```julia
using MAT
using Statistics

data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
neur_tensor = data["neur_tensor_stim1on"]

n_neurons = size(neur_tensor, 1)
n_time = size(neur_tensor, 2)

# Concatenate all neurons
all_neurons = []
for neuron_idx in 1:n_neurons
    neuron_data = neur_tensor[neuron_idx, :, :]
    signal = vec(mean(neuron_data, dims=2))
    push!(all_neurons, signal)
end

# Use first neuron for analysis (or concatenate them)
your_neural_data = all_neurons[1]
```

---

## 🎯 Complete Working Examples

### Example 1: Quick Analysis on Neuron 1

```julia
using MAT
using Statistics

# ===== LOAD DATA =====
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
neur_tensor = data["neur_tensor_stim1on"]

# ===== EXTRACT NEURAL SIGNAL =====
neuron_1_data = neur_tensor[1, :, :]
your_neural_data = vec(mean(neuron_1_data, dims=2))

# ===== RUN ANALYSIS =====
include("integrated_analysis_pipeline.jl")

fs = 1000.0  # Sampling frequency (Hz)
results = quick_swr_analysis(your_neural_data, fs; plot_results=true)

println("Detected $(results["n_events"]) SWRs")
```

### Example 2: Condition-Specific Analysis

```julia
using MAT
using Statistics

# ===== LOAD DATA =====
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
cond_matrix = data["cond_matrix"]
neur_tensor = data["neur_tensor_stim1on"]

# ===== SELECT TRIALS =====
# Find trials where animal attended to stimulus 1 (column 10 == 1)
trial_mask = cond_matrix[:, 10] .== 1
trial_indices = findall(trial_mask)

# ===== EXTRACT NEURAL SIGNAL =====
neuron_data = neur_tensor[1, :, trial_indices]
your_neural_data = vec(mean(neuron_data, dims=2))

# ===== RUN ANALYSIS =====
include("integrated_analysis_pipeline.jl")

fs = 1000.0
results = analyze_neural_data_comprehensive(
    your_neural_data,
    nothing;  # No behavioral events yet
    fs=fs
)

save_analysis_results(results, "./condition_specific_results")
```

### Example 3: With Behavioral Event Extraction

```julia
using MAT
using Statistics

# ===== LOAD DATA =====
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
cond_matrix = data["cond_matrix"]
neur_tensor = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

# ===== SELECT TRIALS =====
trial_indices = findall(cond_matrix[:, 10] .== 1)

# ===== EXTRACT NEURAL SIGNAL =====
neuron_data = neur_tensor[1, :, trial_indices]
your_neural_data = vec(mean(neuron_data, dims=2))

# ===== EXTRACT BEHAVIORAL EVENTS =====
# Example: Detect when target angle changes significantly
target_angles = cond_matrix[trial_indices, 1]  # Column 1 = target angle
angle_changes = abs.(diff(target_angles))

# Find trials with large changes
motion_threshold = quantile(angle_changes, 0.75)
motion_trial_indices = findall(angle_changes .> motion_threshold)

# Convert to time (assuming uniform trial distribution)
samples_per_trial = length(your_neural_data) ÷ length(trial_indices)
fs = 1000.0

motion_onset_samples = motion_trial_indices .* samples_per_trial
motion_onset_samples = motion_onset_samples[motion_onset_samples .<= length(your_neural_data)]
motion_onset_times = motion_onset_samples ./ fs

# Create events dictionary
behavioral_events = Dict(
    "motion_onset" => motion_onset_times
)

# ===== RUN ANALYSIS =====
include("integrated_analysis_pipeline.jl")

results = analyze_neural_data_comprehensive(
    your_neural_data,
    behavioral_events;
    fs=fs
)

save_analysis_results(results, "./with_behavioral_events")
```

---

## 🔍 Understanding Your Data Dimensions

### Neural Tensor Structure

```julia
data = matread("your_file.mat")
neur_tensor = data["neur_tensor_stim1on"]

# Check dimensions
println("Tensor shape: $(size(neur_tensor))")
# Output: (3, 3998, 120)
#         ↑   ↑     ↑
#         │   │     └─ Number of trials
#         │   └─────── Number of time bins
#         └─────────── Number of neurons

n_neurons, n_time, n_trials = size(neur_tensor)
```

### Accessing Data

```julia
# Single neuron, single trial
trial_1_neuron_1 = neur_tensor[1, :, 1]  # Vector of length n_time

# Single neuron, all trials
all_trials_neuron_1 = neur_tensor[1, :, :]  # Matrix (time × trials)

# All neurons, single trial
trial_1_all_neurons = neur_tensor[:, :, 1]  # Matrix (neurons × time)

# Single timepoint, all neurons, all trials
timepoint_100 = neur_tensor[:, 100, :]  # Matrix (neurons × trials)
```

---

## 📋 Standard Workflow Template

Use this template for your own analyses:

```julia
# ==========================================
# TEMPLATE: Neural Data Analysis Workflow
# ==========================================

using MAT
using Statistics

# ===== 1. LOAD DATA =====
data = matread("path/to/your/file.mat")
neur_tensor = data["neur_tensor_stim1on"]
cond_matrix = data["cond_matrix"]

# ===== 2. SELECT DATA =====
# Option A: Use all trials
trial_indices = 1:size(neur_tensor, 3)

# Option B: Select specific condition
# trial_indices = findall(cond_matrix[:, COLUMN] .== VALUE)

# ===== 3. SELECT NEURON(S) =====
# Option A: Single neuron
neuron_idx = 1

# Option B: Multiple neurons (analyze separately)
# neuron_idx = [1, 2, 3]

# ===== 4. EXTRACT SIGNAL =====
neuron_data = neur_tensor[neuron_idx, :, trial_indices]
your_neural_data = vec(mean(neuron_data, dims=2))

# ===== 5. OPTIONAL: FILTER SIGNAL =====
include("sujay_example_zoom_likeMatlab_enhanced.jl")

signal_filtered = apply_neural_filter(
    your_neural_data,
    :gaussian,  # or :moving_average, :butterworth, etc.
    300;        # window size
    sigma=50.0
)

# ===== 6. RUN ANALYSIS =====
include("integrated_analysis_pipeline.jl")

fs = 1000.0  # Adjust based on your sampling rate

results = analyze_neural_data_comprehensive(
    signal_filtered,  # or your_neural_data if not filtered
    nothing,          # or behavioral_events dictionary
    fs=fs
)

# ===== 7. SAVE RESULTS =====
save_analysis_results(results, "./my_analysis_results")

# ===== 8. EXAMINE RESULTS =====
println("Detected $(results["swr_detection"]["n_events"]) SWRs")

# Access specific results
swr_events = results["swr_detection"]["events"]
ml_clusters = results["ml_analysis"]["cluster_labels"]
frequency_bands = results["frequency_bands"]
```

---

## 🎨 Visual Guide

```
.mat File
    ↓
matread()
    ↓
data = Dict(
    "neur_tensor_stim1on" → [neurons × time × trials]
    "cond_matrix" → [trials × conditions]
    "cond_label" → [condition names]
    "stim1on" → [time info]
)
    ↓
Select neuron & trials
    ↓
neur_tensor[neuron_idx, :, trial_indices]
    ↓                    ↓
    [time × trials]      Average across trials
                         ↓
                    your_neural_data = vec(mean(..., dims=2))
                         ↓
                    Vector{Float64} with n_time samples
                         ↓
                    Ready for analysis!
```

---

## ❓ Common Questions

### Q: How do I know my sampling frequency?

**A:** Check your data acquisition system documentation. Common values:
- 1000 Hz (1 kHz) - Standard for LFP
- 2000 Hz (2 kHz) - High-resolution LFP
- 30000 Hz (30 kHz) - Spike recordings

Or estimate from time bins:
```julia
edges = stim1on["edges"]
total_time = edges[end] - edges[1]  # seconds
n_samples = length(edges)
fs = n_samples / total_time
println("Estimated fs: $(fs) Hz")
```

### Q: Which neuron should I analyze?

**A:** Depends on your research question:
- **Single neuron**: Focus on most active neuron
- **Population average**: Average across all neurons
- **Each separately**: Analyze all, compare results

Find most active neuron:
```julia
# Calculate mean firing rate for each neuron
activity_levels = [mean(neur_tensor[i, :, :]) for i in 1:size(neur_tensor, 1)]
most_active = argmax(activity_levels)
println("Most active neuron: $most_active")
```

### Q: How do I select trials by condition?

**A:** Use cond_matrix:
```julia
# View condition labels
println(data["cond_label"])

# Select trials where column X equals Y
trials = findall(cond_matrix[:, X] .== Y)

# Multiple conditions (AND)
trials = findall(
    (cond_matrix[:, X] .== Y) .& 
    (cond_matrix[:, Z] .== W)
)

# Multiple conditions (OR)
trials = findall(
    (cond_matrix[:, X] .== Y) .| 
    (cond_matrix[:, Z] .== W)
)
```

### Q: What if my data is in a different format?

**A:** The key requirement is: `Vector{Float64}`

```julia
# If you have a matrix
matrix_data = rand(1000, 10)  # (time × trials)
your_neural_data = vec(mean(matrix_data, dims=2))

# If you have CSV
using CSV, DataFrames
df = CSV.read("data.csv", DataFrame)
your_neural_data = df.neural_signal  # Assuming column name

# If you have numpy array (from Python)
# Save in Python: np.save("signal.npy", array)
# Load in Julia:
using NPZ
your_neural_data = npzread("signal.npy")
```

---

## ✅ Quick Checklist

Before running analysis, ensure:

- [ ] Data loaded: `data = matread("file.mat")`
- [ ] Neural signal extracted: `your_neural_data = vec(mean(...))`
- [ ] Signal is Vector{Float64}: `typeof(your_neural_data)`
- [ ] Signal has reasonable length: `length(your_neural_data) > 1000`
- [ ] No NaN or Inf values: `any(isnan.(your_neural_data))`
- [ ] Sampling frequency known: `fs = 1000.0`
- [ ] Analysis modules loaded: `include("integrated_analysis_pipeline.jl")`

Now you're ready to analyze! 🎉

---

## 📝 Summary

### The Key Line

```julia
# This is "your_neural_data"!
your_neural_data = vec(mean(neur_tensor[neuron_idx, :, trial_indices], dims=2))
```

It comes from:
1. Loading .mat file
2. Selecting neuron(s)
3. Selecting trial(s)
4. Averaging across trials
5. Converting to 1D vector

### Now All Examples Make Sense!

When documentation says:
```julia
signal = your_neural_data
results = quick_swr_analysis(signal, 1000.0)
```

You know it means:
```julia
# Extract the signal (as shown above)
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))

# Then analyze it
results = quick_swr_analysis(your_neural_data, 1000.0)
```

---

**No more confusion!** 🎯
