# Filter Implementation - Troubleshooting Guide

## Common Issues Quick Reference

### Issue 1: Butterworth Filter MethodError

**Error Message**:
```
MethodError: no method matching Lowpass(::Float64; fs::Float64)
This method may not support any kwargs.
```

**What Happened**: Different versions of DSP.jl have different APIs for creating Butterworth filters.

**Fixed In Current Version**: The code now normalizes cutoff frequency to Nyquist frequency internally, which is compatible with all DSP.jl versions.

**What You Need to Know**:
- `cutoff_freq` parameter is in Hz (absolute frequency)
- `fs` parameter is your sampling frequency in Hz  
- Internally, cutoff is normalized: `normalized_cutoff = cutoff_freq / (fs/2)`
- **Important**: `cutoff_freq` must be less than `fs/2`

**Examples**:
```julia
# If fs = 1.0 (default), use cutoff_freq < 0.5
filter_params[:cutoff_freq] = 0.05  # ✓ Valid (0.05 < 0.5)
filter_params[:cutoff_freq] = 0.3   # ✓ Valid (0.3 < 0.5)
filter_params[:cutoff_freq] = 0.6   # ✗ Invalid (0.6 > 0.5)

# If you know your actual sampling rate, e.g., fs = 100 Hz
filter_params[:fs] = 100.0
filter_params[:cutoff_freq] = 10.0  # ✓ Valid (10 < 50)
```

---

### Issue 2: BoundsError with Different Filters

#### What Was Happening

When switching from `:moving_average` to other filter types, you might have seen:
```
BoundsError: attempt to access 3399-element Vector{Float64} at index [1:3699]
```

### Why It Happened

Different filters handle edge effects differently:

1. **Convolution-based filters** (moving average, gaussian):
   - Input: N samples
   - Output: N - (window_size - 1) samples
   - Example: 4000 input → 3701 output (with window=300)

2. **Other filters** (butterworth, median, etc.):
   - Apply filtering differently
   - May have different edge trimming behavior
   - Need consistent trimming to match convolution filters

The original code assumed a fixed time vector length, but filtered signals have different lengths depending on the filter type.

### The Fix

The enhanced scripts now:

1. **Calculate actual filtered signal length**
   ```julia
   fr3_smooth = apply_neural_filter(fr3_mean, selected_filter, window_size; filter_params...)
   # Length is now: length(fr3_mean) - (window_size - 1)
   ```

2. **Adjust time vector to match**
   ```julia
   additional_trim = (length(fr3_mean) - length(fr3_smooth)) ÷ 2
   time_start = 150 + additional_trim
   time_end = length(edges) - 150 - additional_trim
   time_bins = edges[time_start:time_end]
   ```

3. **Verify before plotting**
   ```julia
   if length(time_bins) != length(fr3_smooth)
       # Auto-adjust to minimum length
       min_len = min(length(time_bins), length(fr3_smooth))
       time_bins = time_bins[1:min_len]
       fr3_smooth = fr3_smooth[1:min_len]
   end
   ```

### Debug Output

Both scripts now print diagnostic information:

```
Input signal lengths:
  fr3_mean: 4000
  fr4_mean: 4000
  edges: 4000

Using filter: gaussian
Window size: 300

Filtered signal lengths:
  fr3_smooth: 3701
  fr4_smooth: 3701
  time_bins: 3701
```

If lengths don't match, you'll see:
```
WARNING: Length mismatch detected, adjusting...
  Adjusted to length: 3701
```

## Understanding Filter Outputs

### Edge Handling Summary

| Filter Type | Edge Trimming | Output Length |
|-------------|---------------|---------------|
| moving_average | m-1 total (m = window_size) | N - (m-1) |
| gaussian | m-1 total | N - (m-1) |
| savitzky_golay | win-1 total | N - (win-1) |
| butterworth | Manual trim to match | N - (window÷2)*2 |
| median | Manual trim to match | N - (window÷2)*2 |
| exponential | Manual trim to match | N - (window÷2)*2 |

All filters are now adjusted to have consistent output lengths.

### Why "Valid" Mode?

The filters use "valid" mode convolution, which:
- ✓ Only returns samples where the kernel fully overlaps the signal
- ✓ Avoids edge artifacts from zero-padding
- ✓ Ensures reliable smoothing throughout the result
- ✗ Reduces output length by (window_size - 1)

This is appropriate for neural data where edge effects could introduce artifacts.

## Common Issues and Solutions

### Issue 1: "Window size must be odd"

**For**: Savitzky-Golay and Median filters

**Solution**:
```julia
window_size = 301  # Use odd number (not 300)
```

Or let the code auto-adjust (already implemented).

### Issue 2: Signal still too noisy

**Try**:
1. Increase window size:
   ```julia
   window_size = 500  # More smoothing
   ```

2. Adjust filter-specific parameters:
   ```julia
   # For Gaussian
   filter_params[:sigma] = window_size / 4  # More smoothing
   
   # For Butterworth
   filter_params[:cutoff_freq] = 0.03  # Lower cutoff
   
   # For Exponential
   filter_params[:alpha] = 0.02  # Lower alpha (smoother)
   ```

### Issue 3: Signal over-smoothed (peaks lost)

**Try**:
1. Decrease window size:
   ```julia
   window_size = 150  # Less smoothing
   ```

2. Use feature-preserving filters:
   ```julia
   selected_filter = :savitzky_golay  # Preserves peaks
   ```

3. Adjust parameters for less smoothing:
   ```julia
   # For Gaussian
   filter_params[:sigma] = window_size / 8  # Less smoothing
   
   # For Butterworth
   filter_params[:cutoff_freq] = 0.1  # Higher cutoff
   ```

### Issue 4: Butterworth creates oscillations

This is "ringing" - Butterworth response to sharp transitions.

**Solutions**:
1. Lower filter order:
   ```julia
   filter_params[:filter_order] = 2  # Gentler response
   ```

2. Increase cutoff frequency:
   ```julia
   filter_params[:cutoff_freq] = 0.1  # Less aggressive
   ```

3. Try Gaussian instead (no ringing):
   ```julia
   selected_filter = :gaussian
   ```

**Note on Butterworth frequencies**: The cutoff frequency is automatically normalized to the Nyquist frequency (fs/2). Make sure `cutoff_freq < fs/2`. For example, if `fs=1.0`, use `cutoff_freq` between 0.001 and 0.49.

### Issue 5: Different filters give very different results

This is **expected**! Each filter has different characteristics:

- **Moving average**: Uniform smoothing, preserves amplitude
- **Gaussian**: Smooth gradual transitions
- **Savitzky-Golay**: Preserves peaks, sharper features
- **Butterworth**: Frequency-based, can create ringing
- **Median**: Removes outliers, creates steps
- **Exponential**: Tracks trends, asymmetric

**What to do**:
1. Run `filter_comparison_demo.jl` to see all filters
2. Choose based on what features you want to preserve
3. Fine-tune parameters for your specific data

## Verification Steps

### Step 1: Check Signal Lengths
```julia
# Run test script
julia test_filter_lengths.jl
```

Should show:
```
✓ Time vector and filtered signal have matching lengths!
✓ Can safely plot(time_matched, filtered_valid)
```

### Step 2: Compare Filters Visually
```julia
# Run comparison
julia filter_comparison_demo.jl
```

Generates plots showing all filters side-by-side.

### Step 3: Verify Your Choice
```julia
# In main script, check output
julia sujay_example_zoom_likeMatlab_enhanced.jl
```

Look for:
```
Using filter: gaussian
Window size: 300

Input signal lengths:
  fr3_mean: 4000
  ...

Filtered signal lengths:
  fr3_smooth: 3701
  time_bins: 3701

✓ Neural data processed with gaussian filter
```

No warnings = everything is working!

## Performance Notes

### Filter Speed (Relative)

1. **Fastest**: 
   - Moving average (simple convolution)
   - Exponential (single pass)

2. **Medium**:
   - Gaussian (convolution with computed kernel)
   - Median (local sorting)

3. **Slower**:
   - Butterworth (filtfilt = two passes)
   - Savitzky-Golay (polynomial fitting)

For typical neural datasets (thousands of samples), all are fast enough (< 1 second).

### Memory Usage

All filters have similar memory footprint:
- Input array + output array + kernel
- Peak memory: ~3x input size
- Not a concern for typical datasets

## Advanced: Creating Custom Filters

Want to add your own filter? Follow this pattern:

```julia
elseif filter_type == :my_custom_filter
    # Your filtering logic here
    filtered = my_filter_function(data, window_size)
    
    # IMPORTANT: Trim to match other filters
    trim = window_size ÷ 2
    return filtered[(trim+1):(end-trim)]
```

Key points:
1. Return same-length output as other filters
2. Trim edges consistently
3. Handle edge cases (window size too large, etc.)

## Getting Help

If you encounter issues:

1. **Check lengths**: Run `test_filter_lengths.jl`
2. **Compare filters**: Run `filter_comparison_demo.jl`
3. **Check debug output**: Look at printed signal lengths
4. **Try default**: Switch to `:moving_average` to verify data loads correctly

Most issues are solved by ensuring time vector and signal lengths match!

## Summary

✓ **The fix**: Automatically adjusts time vectors to match filtered signals
✓ **Works for all filters**: Consistent length handling
✓ **Debug output**: Shows what's happening at each step
✓ **Graceful fallback**: Auto-adjusts if mismatch detected

You can now switch between any filter type without worrying about length mismatches!
