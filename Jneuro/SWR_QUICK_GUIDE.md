# Quick Reference: SWR Detection in Multi-Neuron Analysis

## New Files

✅ `multi_neuron_swr.jl` - SWR detection engine  
✅ `multi_neuron_swr_plots.jl` - SWR visualization  
✅ `main_enhanced.jl` - Updated with SWR integration  

## What You Get

### Console Output
- Total SWRs detected across all neurons
- Per-neuron statistics (count, rate, duration, frequency)
- Activity rankings (which neurons have most SWRs)
- Co-rippling analysis (which neurons ripple together)
- Synchrony scores (coordination measure)

### Plots (5 files)
1. **swr_comparison.png** - SWR statistics by neuron
2. **co_ripple_matrix.png** - Co-rippling probability heatmap
3. **swr_raster.png** - Temporal distribution of SWRs
4. **swr_properties.png** - Duration/amplitude/frequency distributions
5. **swr_synchrony.png** - Synchrony scores

## Quick Start

### Run with SWR Detection
```bash
julia main_enhanced.jl
```

SWR analysis runs automatically after comparative metrics!

### Configure Detection

Edit `main_enhanced.jl` around line 862:

```julia
# Enable/disable
ENABLE_SWR_DETECTION = true  # or false

# Adjust sensitivity (line 870)
threshold_sd=3.0    # Lower = more SWRs detected
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ripple_band` | (150, 250) Hz | Frequency range for ripples |
| `threshold_sd` | 3.0 | Detection threshold (SD above baseline) |
| `min_duration_ms` | 30.0 | Minimum SWR duration |
| `max_duration_ms` | 200.0 | Maximum SWR duration |
| `co_ripple_window_ms` | 50.0 | Time window for co-rippling |

## Typical Output

```
📊 SWR Summary by Neuron:
  Total SWRs detected: 523

  Neuron  1:  45 SWRs  |  0.15 Hz  |  68.3 ms  |  182.4 Hz peak  |  Sync=0.342
  Neuron  5:  52 SWRs  |  0.17 Hz  |  65.1 ms  |  189.3 Hz peak  |  Sync=0.456

🔗 Co-Rippling Analysis:
  Found 8 neuron pairs with frequent co-rippling (>20%):
    Neurons  5 ↔ 13: 38.2% co-ripple probability
```

## Interpreting Results

### SWR Rate
- **High (>0.2 Hz)**: Very active, central to ripple generation
- **Normal (0.1-0.2 Hz)**: Typical hippocampal activity
- **Low (<0.1 Hz)**: Less involved in ripple events

### Co-Rippling
- **>30%**: Strong functional coupling
- **10-30%**: Moderate coordination
- **<10%**: Independent activity

### Synchrony Score
- **High (>0.4)**: Hub neuron, coordinates with many others
- **Medium (0.2-0.4)**: Moderate network integration
- **Low (<0.2)**: Peripheral or independent

## Quick Adjustments

### More Sensitive (finds more SWRs)
```julia
threshold_sd=2.5
min_duration_ms=25.0
```

### More Specific (only clear SWRs)
```julia
threshold_sd=3.5
min_duration_ms=40.0
```

### Broader Frequency Range
```julia
ripple_band=(120.0, 300.0)
```

## Output Location

All results saved to:
```
./neural_analysis_output/multi_neuron_analysis/swr_analysis/
```

## Troubleshooting

**No SWRs detected?**
- Lower `threshold_sd` to 2.0
- Check sampling rate (`fs` should be ≥1000 Hz)

**Too many SWRs?**
- Raise `threshold_sd` to 3.5
- Tighten duration range

**Crashes?**
- Check for NaN values in data
- Verify `fs` is defined correctly

---

**That's it!** SWR detection now runs automatically as part of your multi-neuron analysis pipeline. 🎉
