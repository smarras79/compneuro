# Getting Started - Neural Analysis Toolkit

## 🎯 Everything You Need in One Place

This is your 5-minute quick-start guide. For details, see the full documentation.

---

## Step 1: Load Your Data (THE MISSING PIECE!)

```julia
using MAT
using Statistics

# Load the .mat file
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")

# Extract the neural tensor
neur_tensor = data["neur_tensor_stim1on"]
# This is a 3D array: (neurons × time × trials)
# Example shape: (3, 3998, 120)

# Extract signal for neuron 1, averaged across all trials
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))

# Check what you have
println("Signal type: ", typeof(your_neural_data))     # Vector{Float64}
println("Signal length: ", length(your_neural_data))   # 3998
```

**This is what "your_neural_data" means in all the examples!**

---

## Step 2: Run Quick Analysis

```julia
# Load the analysis pipeline
include("integrated_analysis_pipeline.jl")

# Set sampling frequency (Hz)
fs = 1000.0

# Run quick SWR detection
results = quick_swr_analysis(your_neural_data, fs; plot_results=true)

# See what you found
println("Detected $(results["n_events"]) Sharp-Wave Ripples")
```

**That's it! You just detected SWRs.** 🎉

---

## Step 3: Run Complete Analysis

```julia
# For comprehensive analysis with all features:
results = analyze_neural_data_comprehensive(
    your_neural_data,
    nothing;  # No behavioral events yet
    fs=fs
)

# Save all results
save_analysis_results(results, "./my_results")
```

**Check ./my_results/ for plots and summary!**

---

## Optional: Add Signal Filtering

```julia
# Before analysis, filter the signal
signal_filtered = apply_neural_filter(
    your_neural_data,
    :gaussian,  # Filter type
    300;        # Window size
    sigma=50.0  # Smoothness
)

# Then analyze the filtered signal
results = quick_swr_analysis(signal_filtered, fs)
```

---

## Optional: Add Behavioral Events

```julia
# Define when events happened (times in seconds)
behavioral_events = Dict(
    "motion_onset" => [1.2, 3.5, 7.8, 10.2],
    "motion_offset" => [2.1, 4.3, 8.5, 11.1]
)

# Run analysis with events
results = analyze_neural_data_comprehensive(
    your_neural_data,
    behavioral_events;
    fs=fs
)

# Now you can see SWRs at specific behavioral times!
```

---

## Complete Working Script

Copy and paste this entire script:

```julia
# ==========================================
# COMPLETE NEURAL ANALYSIS SCRIPT
# ==========================================

using MAT
using Statistics

println("=" ^70)
println("NEURAL ANALYSIS SCRIPT")
println("=" ^70)

# ===== 1. LOAD DATA =====
println("\n[1/4] Loading data...")

data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
neur_tensor = data["neur_tensor_stim1on"]

# Extract signal
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))

println("  Signal length: $(length(your_neural_data)) samples")
println("  ✓ Data loaded")

# ===== 2. LOAD ANALYSIS TOOLS =====
println("\n[2/4] Loading analysis pipeline...")

include("integrated_analysis_pipeline.jl")

println("  ✓ Pipeline loaded")

# ===== 3. RUN ANALYSIS =====
println("\n[3/4] Running analysis...")

fs = 1000.0  # Sampling frequency in Hz

results = analyze_neural_data_comprehensive(
    your_neural_data,
    nothing;  # No behavioral events
    fs=fs
)

println("  ✓ Analysis complete")

# ===== 4. DISPLAY AND SAVE RESULTS =====
println("\n[4/4] Results:")

# SWR count
n_swr = results["swr_detection"]["n_events"]
println("  Detected $n_swr Sharp-Wave Ripples")

# Frequency bands
if haskey(results, "frequency_bands")
    println("\n  Frequency Band Power:")
    bands = results["frequency_bands"]
    for band in sort(collect(keys(bands)))
        if band != "total_power" && band != "freq" && band != "power_spectrum"
            info = bands[band]
            rel_power = get(info, "relative_power", 0.0) * 100
            println("    $(rpad(band, 12)): $(round(rel_power, digits=1))%")
        end
    end
end

# Save results
println("\n  Saving results...")
save_analysis_results(results, "./neural_analysis_output")

println("\n" * "=" ^70)
println("✓ COMPLETE!")
println("Check './neural_analysis_output/' for plots and summary")
println("=" ^70)
```

**Save this as `my_analysis.jl` and run with: `julia my_analysis.jl`**

---

## What You Get

After running, you'll have:

### Console Output
```
======================================================================
NEURAL ANALYSIS SCRIPT
======================================================================

[1/4] Loading data...
  Signal length: 3998 samples
  ✓ Data loaded

[2/4] Loading analysis pipeline...
  ✓ Pipeline loaded

[3/4] Running analysis...
  ✓ Analysis complete

[4/4] Results:
  Detected 42 Sharp-Wave Ripples

  Frequency Band Power:
    alpha       : 15.3%
    beta        : 22.1%
    delta       : 18.7%
    high_gamma  : 8.4%
    low_gamma   : 12.2%
    ripple      : 3.1%
    theta       : 20.2%

  Saving results...

======================================================================
✓ COMPLETE!
Check './neural_analysis_output/' for plots and summary
======================================================================
```

### Files Created
```
neural_analysis_output/
├── psd.png                    # Power spectrum
├── spectrogram.png            # Time-frequency
├── swr_events.png             # Example ripples
├── ml_clustering.png          # Event clusters
├── frequency_bands.png        # Band powers
└── analysis_summary.txt       # Text report
```

---

## Troubleshooting

### Error: "cannot find file"
```julia
# Check your path
pwd()  # Shows current directory

# Adjust path to data
data = matread("path/to/your/data.mat")
```

### Error: "KeyError: neur_tensor_stim1on"
```julia
# Check what keys are available
println(keys(data))

# Use the correct key name
neur_tensor = data["your_actual_key_name"]
```

### Error: "DimensionMismatch"
```julia
# Check tensor dimensions
println(size(neur_tensor))  # Should be (neurons, time, trials)

# Make sure you're averaging the right dimension
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))
#                                                          ↑
#                                                     dim=2 averages across trials
```

### No SWRs detected
```julia
# Lower the threshold
config = Dict("swr_threshold_sd" => 2.5)  # Instead of default 3.0

results = analyze_neural_data_comprehensive(
    your_neural_data, nothing; 
    fs=fs, 
    config=config
)
```

---

## Next Steps

1. ✅ You've run basic analysis
2. 📖 Read `DATA_LOADING_GUIDE.md` for advanced data extraction
3. 🎨 Read `FILTERING_GUIDE.md` to improve signal quality
4. 🧠 Read `NEURAL_ANALYSIS_GUIDE.md` for in-depth understanding
5. 🔧 Customize parameters for your specific research question

---

## File Guide

**Start Here:**
- `GETTING_STARTED.md` ← You are here!
- `DATA_LOADING_GUIDE.md` - How to extract your data
- `QUICK_REFERENCE.md` - One-page cheat sheet

**For Analysis:**
- `integrated_analysis_pipeline.jl` - Main analysis code
- `example_complete_analysis.jl` - Complete working example

**For Deep Dive:**
- `NEURAL_ANALYSIS_GUIDE.md` - Complete documentation
- `README_PACKAGE_OVERVIEW.md` - Package overview

**For Problems:**
- `TROUBLESHOOTING.md` - Common issues
- `BUG_FIXES_V1.1.md` - Recent fixes

---

## Summary

### The Key Points

1. **Your data comes from**: `vec(mean(neur_tensor[1, :, :], dims=2))`
2. **Quick analysis**: `quick_swr_analysis(your_neural_data, 1000.0)`
3. **Complete analysis**: `analyze_neural_data_comprehensive(your_neural_data, nothing; fs=1000.0)`
4. **Results are saved**: Check the output directory for plots

### The Three Lines You Need

```julia
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))
include("integrated_analysis_pipeline.jl")
results = quick_swr_analysis(your_neural_data, 1000.0)
```

**That's literally it!** Everything else is optional enhancement. 🎉

---

**Ready?** Copy the complete script above and run it! 🚀
