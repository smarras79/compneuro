# Quick Filter Switching Examples

This file shows example configurations for different use cases.
Copy-paste these into your `sujay_example_zoom_likeMatlab_enhanced.jl` file
at the "FILTER SELECTION" section (around line 240).

## Example 1: General Smoothing (Recommended Upgrade from Moving Average)

```julia
selected_filter = :gaussian
window_size = 300
filter_params[:sigma] = 50  # Adjust for more/less smoothing
```

Why: Better frequency response than moving average, similar smoothness.

---

## Example 2: Preserve Peak Shapes (For Spike Detection)

```julia
selected_filter = :savitzky_golay
window_size = 301  # Must be odd
filter_params[:poly_order] = 3  # 2-3 for smoothing, 4-5 for features
```

Why: Best at preserving sharp features while reducing noise.

---

## Example 3: Remove Outlier Spikes

```julia
selected_filter = :median
window_size = 201  # Smaller window for spike detection
```

Why: Excellent at removing transient artifacts without affecting baseline.

---

## Example 4: Frequency-Based Filtering

```julia
selected_filter = :butterworth
window_size = 300
filter_params[:cutoff_freq] = 0.05  # Must be < fs/2 (here fs=1.0, so < 0.5)
filter_params[:filter_order] = 4
filter_params[:fs] = 1.0  # Sampling frequency

# Note: cutoff_freq is normalized internally to Nyquist frequency
# Valid range: slightly above 0 to slightly below fs/2
# Example ranges:
#   - If fs=1.0: use cutoff_freq from 0.001 to 0.49
#   - If fs=100.0: use cutoff_freq from 0.1 to 49.9
```

Why: When you want to remove specific frequency components.

---

## Example 5: Minimal Smoothing (Light Touch)

```julia
selected_filter = :gaussian
window_size = 100  # Much smaller window
filter_params[:sigma] = 20
```

Why: When you want to preserve most detail but reduce high-frequency noise.

---

## Example 6: Aggressive Smoothing (Strong Denoising)

```julia
selected_filter = :gaussian
window_size = 500  # Large window
filter_params[:sigma] = 100
```

Why: When data is very noisy and you want to see general trends.

---

## Example 7: Adaptive/Tracking (For Changing Baselines)

```julia
selected_filter = :exponential
window_size = 300
filter_params[:alpha] = 0.1  # 0.01-0.05 smooth, 0.1-0.2 responsive
```

Why: Good for tracking slowly changing trends or baselines.

---

## Example 8: Original Behavior (No Change)

```julia
selected_filter = :moving_average
window_size = 300
```

Why: Keep exactly the same behavior as before the enhancement.

---

## How to Choose

### Quick Decision Tree:

**Q: Do you want to keep original behavior?**
→ YES: Use `:moving_average`

**Q: Is your data very noisy?**
→ YES: Try `:gaussian` with large window (400-500)

**Q: Do you need to detect/preserve peaks?**
→ YES: Use `:savitzky_golay` with poly_order=3

**Q: Do you have outlier spikes?**
→ YES: Use `:median` with smaller window (100-200)

**Q: Do you know the noise frequency?**
→ YES: Use `:butterworth` and set cutoff_freq appropriately

**Q: Need to track changing baseline?**
→ YES: Use `:exponential` with alpha=0.05-0.1

**Q: Want general improvement?**
→ Use `:gaussian` with window_size=300, sigma=50

---

## Testing Your Choice

After selecting a filter:

1. **Run the main script**:
   ```bash
   julia sujay_example_zoom_likeMatlab_enhanced.jl
   ```

2. **Check the output**:
   - Look at the plot
   - Check if features are preserved
   - Verify noise is reduced

3. **Compare all options** (if unsure):
   ```bash
   julia filter_comparison_demo.jl
   ```
   This shows all filters side-by-side

4. **Fine-tune**:
   - Adjust window_size (bigger = smoother)
   - Adjust filter-specific parameters
   - Re-run until satisfied

---

## Pro Tips

### Tip 1: Start Conservative
Begin with the default parameters, then increase smoothing if needed:
```julia
# Start here
window_size = 300
# If too noisy, try
window_size = 400
# Still noisy?
window_size = 500
```

### Tip 2: Compare Original vs Filtered
Keep the original signal in a separate variable:
```julia
fr3_original = copy(fr3_mean)
fr3_smooth = apply_neural_filter(fr3_mean, selected_filter, window_size; filter_params...)
# Now you can plot both to compare
```

### Tip 3: Try Multiple Filters
Create plots with 2-3 different filters to compare:
```julia
fr3_gaussian = apply_neural_filter(fr3_mean, :gaussian, 300)
fr3_savgol = apply_neural_filter(fr3_mean, :savitzky_golay, 301)
# Plot both and compare
```

### Tip 4: Document Your Choice
Add a comment explaining why you chose specific parameters:
```julia
# Using Gaussian with sigma=50 because:
# - Preserves peak shapes better than moving average
# - Removes high-frequency noise from electrode drift
# - Window=300 matches temporal resolution of behavioral data
selected_filter = :gaussian
filter_params[:sigma] = 50
```

### Tip 5: Batch Processing
If analyzing multiple sessions, create a config dict:
```julia
# Configuration for all analyses
analysis_config = Dict(
    :filter => :gaussian,
    :window_size => 300,
    :sigma => 50
)

# Use it
selected_filter = analysis_config[:filter]
window_size = analysis_config[:window_size]
```

---

## Common Parameter Ranges

### Window Size
- **Very small** (50-100): Minimal smoothing, preserves detail
- **Small** (100-200): Light smoothing, good for clean data  
- **Medium** (200-400): Standard smoothing, most common
- **Large** (400-600): Heavy smoothing, for very noisy data
- **Very large** (>600): Extreme smoothing, only for baseline trends

### Gaussian Sigma
- **Tight** (window_size/10): Subtle smoothing
- **Standard** (window_size/6): Balanced (default)
- **Loose** (window_size/4): Aggressive smoothing

### Butterworth Cutoff Frequency
**IMPORTANT**: Cutoff frequency must be less than fs/2 (Nyquist frequency)
- **Very low** (0.001-0.03 × fs/2): Heavy filtering, slow oscillations only
- **Low** (0.03-0.08 × fs/2): Standard filtering
- **Medium** (0.08-0.15 × fs/2): Light filtering, preserves faster changes  
- **High** (0.15-0.4 × fs/2): Minimal filtering

Example: If fs=1.0 (default), then fs/2=0.5, so:
- Very low: 0.0005-0.015 Hz
- Low: 0.015-0.04 Hz  
- Medium: 0.04-0.075 Hz
- High: 0.075-0.2 Hz

### Savitzky-Golay Polynomial Order
- **2**: Maximum smoothing
- **3**: Good balance (default)
- **4**: More feature preservation
- **5**: Minimal smoothing

### Exponential Alpha
- **Very smooth** (0.01-0.03): Slow baseline tracking
- **Smooth** (0.03-0.08): Standard tracking
- **Responsive** (0.08-0.15): Fast tracking, less smooth
- **Very responsive** (0.15-0.3): Minimal smoothing

---

## Save Your Configuration

Once you find settings you like, document them:

```julia
# === OPTIMAL FILTER CONFIGURATION ===
# Determined on: 2024-XX-XX
# Dataset: Amadeus hippocampal recordings
# Reasoning: Gaussian filter preserves transient responses
#            while removing high-frequency electrode noise
# =====================================

selected_filter = :gaussian
window_size = 350
filter_params = Dict(
    :sigma => 60,
    # Other params stay at defaults
)
```

This makes it easy to:
- Reproduce your analysis
- Explain your methods
- Apply same settings to new data
