# Neural Signal Filtering - Usage Guide

## Overview

The enhanced neural signal processing code now supports **6 different filter types** while keeping your original moving average implementation as the default. You can easily switch between filters to find the best one for your neural data analysis.

## Quick Start

### 1. Using the Enhanced Main Script

In `sujay_example_zoom_likeMatlab_enhanced.jl`, simply change the filter selection:

```julia
# Line ~230: Choose your filter
selected_filter = :moving_average  # Change this line!
```

Available options:
- `:moving_average` (default - your original filter)
- `:gaussian`
- `:savitzky_golay`
- `:butterworth`
- `:median`
- `:exponential`

### 2. Running the Comparison Tool

To see all filters side-by-side:

```bash
julia filter_comparison_demo.jl
```

This generates:
- `filter_comparison.png` - Individual subplots for each filter
- `filter_overlay.png` - All filters overlaid for easy comparison

## Filter Types Explained

### 1. Moving Average (`:moving_average`)
**Your original filter - unchanged!**

- **How it works**: Simple box filter, averages over window
- **Best for**: General smoothing, easy interpretation
- **Pros**: Fast, simple, preserves amplitude
- **Cons**: Poor frequency response, can blur sharp features

```julia
selected_filter = :moving_average
window_size = 300
```

### 2. Gaussian (`:gaussian`)
**Smooth bell-shaped kernel**

- **How it works**: Weighted average with Gaussian kernel
- **Best for**: Smooth noise reduction while preserving shape
- **Pros**: Better frequency response, natural smoothing
- **Cons**: Still blurs sharp transitions

```julia
selected_filter = :gaussian
window_size = 300
filter_params[:sigma] = 50  # Control smoothness (lower = smoother)
```

### 3. Savitzky-Golay (`:savitzky_golay`)
**Polynomial fitting in local windows**

- **How it works**: Fits polynomial to local data
- **Best for**: Preserving peaks and sharp features
- **Pros**: Maintains signal features, good for derivatives
- **Cons**: Can amplify noise if poly_order is too high

```julia
selected_filter = :savitzky_golay
window_size = 301  # Should be odd
filter_params[:poly_order] = 3  # Polynomial degree (2-5 typical)
```

### 4. Butterworth (`:butterworth`)
**Classic frequency-domain lowpass filter**

- **How it works**: Removes frequencies above cutoff
- **Best for**: Removing specific frequency components
- **Pros**: Sharp frequency cutoff, zero-phase (using filtfilt)
- **Cons**: Can ring near sharp transitions

```julia
selected_filter = :butterworth
filter_params[:cutoff_freq] = 0.05  # Hz (absolute frequency)
filter_params[:filter_order] = 4     # Higher = sharper cutoff
filter_params[:fs] = 1.0             # Sampling frequency (Hz)

# Note: cutoff_freq is normalized internally to Nyquist frequency (fs/2)
# So cutoff_freq should be < fs/2
# Example: If fs=1.0, valid cutoff range is 0.001 to 0.49
```

### 5. Median (`:median`)
**Nonlinear edge-preserving filter**

- **How it works**: Takes median value in window
- **Best for**: Removing spikes and outliers
- **Pros**: Excellent at removing impulse noise, preserves edges
- **Cons**: Can create "stairstep" artifacts, computationally slower

```julia
selected_filter = :median
window_size = 301  # Should be odd
```

### 6. Exponential Moving Average (`:exponential`)
**Weighted average with exponential decay**

- **How it works**: Recent data weighted more heavily
- **Best for**: Tracking trends, adaptive smoothing
- **Pros**: Responsive to recent changes, causal filter
- **Cons**: Asymmetric response (lags rising edges)

```julia
selected_filter = :exponential
filter_params[:alpha] = 0.05  # Smoothing (0.01-0.2 typical, lower = smoother)
```

## Advanced Usage

### Customizing Filter Parameters

The main script includes a `filter_params` dictionary where you can adjust filter-specific parameters:

```julia
filter_params = Dict(
    :cutoff_freq => 0.05,      # Butterworth cutoff (Hz)
    :fs => 1.0,                # Sampling frequency
    :filter_order => 4,        # Butterworth order
    :poly_order => 3,          # Savitzky-Golay polynomial order
    :alpha => 0.05,            # Exponential smoothing factor
    :sigma => window_size/6    # Gaussian std dev
)
```

### Using the Filter Function Directly

You can also use the `apply_neural_filter()` function directly in your own code:

```julia
# Basic usage
filtered_data = apply_neural_filter(raw_data, :gaussian, 300)

# With custom parameters
filtered_data = apply_neural_filter(
    raw_data, 
    :butterworth, 
    300;
    cutoff_freq=0.08,
    filter_order=6
)
```

## Choosing the Right Filter

### Decision Guide

| **Your Goal** | **Recommended Filter** | **Why?** |
|---------------|----------------------|----------|
| Match original analysis | `:moving_average` | Same as before |
| Smooth noise, preserve shape | `:gaussian` | Best general-purpose upgrade |
| Keep peaks sharp | `:savitzky_golay` | Preserves features |
| Remove specific frequencies | `:butterworth` | Frequency control |
| Remove spike artifacts | `:median` | Robust to outliers |
| Track changing trends | `:exponential` | Adaptive |

### Parameter Tuning Tips

1. **Window Size**
   - Larger window = more smoothing
   - Typical range: 100-500 samples
   - Current default: 300 (same as original)

2. **Butterworth Cutoff**
   - Lower cutoff = more smoothing
   - Start with: 0.05-0.1 Hz
   - Adjust based on data inspection

3. **Savitzky-Golay Polynomial**
   - Low order (2-3): smoother
   - High order (4-5): preserves more detail
   - Too high: can amplify noise

4. **Exponential Alpha**
   - Lower (0.01-0.05): smoother, slower response
   - Higher (0.1-0.2): more responsive, less smooth

## Example Workflow

```julia
# 1. Run comparison to see all filters
# julia filter_comparison_demo.jl

# 2. Based on plots, choose a filter (e.g., Gaussian looks best)

# 3. Edit main script
selected_filter = :gaussian
window_size = 300
filter_params[:sigma] = 45

# 4. Run main analysis
# julia sujay_example_zoom_likeMatlab_enhanced.jl

# 5. Fine-tune parameters if needed
```

## Performance Notes

- **Fastest**: Moving average, Exponential
- **Medium**: Gaussian, Median
- **Slowest**: Butterworth (requires filtfilt), Savitzky-Golay
- All filters are fast enough for typical neural datasets

## Troubleshooting

### "Window size must be odd" error
```julia
# For Savitzky-Golay and Median, use odd window sizes
window_size = 301  # Not 300
```

### Signal looks over-smoothed
```julia
# Reduce window size or adjust parameters
window_size = 150  # Instead of 300
# OR
filter_params[:sigma] = 30  # Smaller sigma for Gaussian
```

### Signal still noisy
```julia
# Increase smoothing
window_size = 500
# OR for Butterworth
filter_params[:cutoff_freq] = 0.03  # Lower cutoff
```

### Butterworth creates artifacts
```julia
# Reduce filter order or increase cutoff
filter_params[:filter_order] = 2
filter_params[:cutoff_freq] = 0.1
```

## Files Generated

1. **sujay_example_zoom_likeMatlab_enhanced.jl**
   - Enhanced main script with filter selection
   - Keeps all original functionality
   - Simply change `selected_filter` variable

2. **filter_comparison_demo.jl**
   - Standalone comparison tool
   - Generates visualization plots
   - Computes filter statistics

## Questions?

The code includes extensive documentation. To see all filter options:

```julia
print_filter_options()  # Run this in the script
```

## Migration from Original Code

**Nothing breaks!** The default `:moving_average` filter produces identical results to your original code. You can:

1. Use the enhanced script as-is (same results)
2. Experiment with other filters by changing one line
3. Keep the original script if you prefer

The enhanced version is a strict superset of functionality - it only adds options without changing the default behavior.
