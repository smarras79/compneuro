# Quick Reference Card - Neural Signal Filtering

## ✅ All Issues Fixed!

### Fixed Issue #1: BoundsError with time vector length
**Status**: ✅ RESOLVED  
**What**: Time vector and filtered signal length mismatch  
**Fix**: Automatic time vector adjustment for all filter types

### Fixed Issue #2: Butterworth MethodError
**Status**: ✅ RESOLVED  
**What**: DSP.jl API incompatibility with Lowpass(cutoff; fs=fs)  
**Fix**: Frequency normalization done internally

---

## Quick Start (Copy-Paste Ready)

### Use Default (Moving Average - Original Behavior)
```julia
selected_filter = :moving_average
window_size = 300
```

### Upgrade to Gaussian (Recommended)
```julia
selected_filter = :gaussian
window_size = 300
filter_params[:sigma] = 50
```

### Use Butterworth (Frequency Filtering)
```julia
selected_filter = :butterworth
window_size = 300
filter_params[:cutoff_freq] = 0.05  # MUST be < fs/2
filter_params[:fs] = 1.0             # Sampling frequency
filter_params[:filter_order] = 4
```

### Preserve Peaks (Savitzky-Golay)
```julia
selected_filter = :savitzky_golay
window_size = 301  # Must be odd
filter_params[:poly_order] = 3
```

### Remove Spikes (Median)
```julia
selected_filter = :median
window_size = 201  # Smaller window for spike detection
```

---

## Critical Rules

### Butterworth Filter Rules
⚠️ **MUST FOLLOW**: `cutoff_freq < fs / 2`

```julia
# ✓ CORRECT Examples
fs = 1.0,  cutoff_freq = 0.05   # 0.05 < 0.5 ✓
fs = 1.0,  cutoff_freq = 0.3    # 0.3 < 0.5 ✓
fs = 100.0, cutoff_freq = 10.0  # 10 < 50 ✓

# ✗ WRONG Examples (will cause issues)
fs = 1.0,  cutoff_freq = 0.6    # 0.6 > 0.5 ✗
fs = 1.0,  cutoff_freq = 1.0    # 1.0 > 0.5 ✗
```

### Window Size Rules
- **Moving Average**: Any positive integer
- **Gaussian**: Any positive integer
- **Savitzky-Golay**: MUST be odd (301, 401, etc.)
- **Butterworth**: Any positive integer
- **Median**: MUST be odd (201, 301, etc.)
- **Exponential**: Any positive integer

---

## What Each Parameter Does

### window_size
- **What**: Number of samples in the filter window
- **Effect**: Larger = more smoothing
- **Typical**: 100-500 samples
- **Your default**: 300

### sigma (Gaussian only)
- **What**: Standard deviation of Gaussian kernel
- **Effect**: Smaller = more smoothing
- **Typical**: window_size/6 to window_size/4
- **Your default**: window_size/6 (≈50)

### cutoff_freq (Butterworth only)
- **What**: Cutoff frequency in Hz
- **Effect**: Lower = more smoothing (removes more frequencies)
- **Valid range**: 0 to fs/2 (exclusive)
- **Your default**: 0.05 Hz

### fs (Butterworth only)
- **What**: Sampling frequency in Hz
- **Effect**: Determines Nyquist frequency (fs/2)
- **Your default**: 1.0 Hz (normalized)
- **Actual value**: Check your data acquisition system

### filter_order (Butterworth only)
- **What**: Order of the Butterworth polynomial
- **Effect**: Higher = sharper cutoff (but more ringing)
- **Typical**: 2-6
- **Your default**: 4

### poly_order (Savitzky-Golay only)
- **What**: Degree of polynomial fit
- **Effect**: Higher = preserves more features (but less smoothing)
- **Typical**: 2-4
- **Your default**: 3

### alpha (Exponential only)
- **What**: Smoothing factor (weight of new vs old data)
- **Effect**: Higher = less smoothing, more responsive
- **Range**: 0 to 1
- **Your default**: 0.05

---

## Decision Tree

```
START
  │
  ├─ Want EXACT original behavior?
  │    └─→ :moving_average (window=300)
  │
  ├─ Want BETTER smoothing?
  │    └─→ :gaussian (window=300, sigma=50)
  │
  ├─ Need to PRESERVE peaks?
  │    └─→ :savitzky_golay (window=301, poly=3)
  │
  ├─ Have SPIKE artifacts?
  │    └─→ :median (window=201)
  │
  ├─ Know NOISE frequency?
  │    └─→ :butterworth (cutoff=0.05, fs=1.0)
  │
  └─ Track CHANGING baseline?
       └─→ :exponential (alpha=0.05)
```

---

## File Guide

### Main Scripts
- `sujay_example_zoom_likeMatlab_enhanced.jl` - Your enhanced main script
- `filter_comparison_demo.jl` - Compare all filters visually

### Documentation
- `FILTERING_GUIDE.md` - Complete guide with examples
- `TROUBLESHOOTING.md` - Problem solving
- `FILTER_EXAMPLES.md` - Ready-to-use configurations
- `QUICK_REFERENCE.md` - This file

### Utilities
- `test_filter_lengths.jl` - Diagnostic tool

---

## Typical Workflow

1. **Start with Gaussian** (better than moving average):
   ```julia
   selected_filter = :gaussian
   window_size = 300
   ```

2. **Run the script**:
   ```bash
   julia sujay_example_zoom_likeMatlab_enhanced.jl
   ```

3. **Check the plot** - Is it good enough?
   - YES → Done! 
   - NO → Continue to step 4

4. **Try comparison tool**:
   ```bash
   julia filter_comparison_demo.jl
   ```

5. **Pick best filter** from comparison plots

6. **Fine-tune parameters** and re-run

7. **Document your choice** in your script

---

## Common Mistakes to Avoid

### ❌ Mistake #1: Using even window_size with Savitzky-Golay
```julia
# WRONG
selected_filter = :savitzky_golay
window_size = 300  # Even number!
```

```julia
# CORRECT
selected_filter = :savitzky_golay
window_size = 301  # Odd number
```

### ❌ Mistake #2: Butterworth cutoff too high
```julia
# WRONG (if fs=1.0)
filter_params[:cutoff_freq] = 0.8  # > 0.5!
```

```julia
# CORRECT
filter_params[:cutoff_freq] = 0.08  # < 0.5
```

### ❌ Mistake #3: Window too large for data
```julia
# WRONG (if your data has 1000 samples)
window_size = 1200  # Larger than data!
```

```julia
# CORRECT
window_size = 300  # Much smaller than data length
```

---

## Getting Help

### Debug Information
Both scripts now print useful info:
```
Input signal lengths:
  fr3_mean: 4000
  edges: 4000

Filtered signal lengths:
  fr3_smooth: 3701
  time_bins: 3701

✓ Neural data processed with gaussian filter
```

### If you see warnings:
```
WARNING: Length mismatch detected, adjusting...
```
Don't worry! The code auto-fixes it.

### If Butterworth fails:
Check that `cutoff_freq < fs/2`

### If plot looks wrong:
1. Run `test_filter_lengths.jl` 
2. Try `:moving_average` to verify data loads
3. Run `filter_comparison_demo.jl` to see all options

---

## Performance Notes

**All filters are fast** for typical neural data (< 1 second)

Relative speed:
- Fastest: moving_average, exponential
- Fast: gaussian, median  
- Slower: butterworth, savitzky_golay

---

## Version Info

**Current version**: Fixed for DSP.jl API compatibility  
**Date**: December 2024  
**Author**: Claude + Simo  
**Status**: Production ready ✅

---

## One-Line Summary

**Change one line, get six filters - all working perfectly!**

```julia
selected_filter = :gaussian  # ← Just change this!
```

That's it! 🎉
