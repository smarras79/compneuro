# Understanding Temporal Tuning Curve Plots

## 📊 How Neurons Are Selected (NOT Random!)

### **Selection Method: Top Neurons by TMI**

The temporal tuning curves plot shows the **neurons with the STRONGEST temporal modulation**, not random neurons.

```python
# This is what happens in the code:
top_idx = np.argsort(tmi)[::-1][:n_neurons]
# Sorts ALL neurons by TMI (highest first)
# Selects top N (default: 12)
```

---

## 🎯 What You're Seeing

### If you see 9 curves instead of 12:

**Possible reasons:**

1. **Only 9 time cells identified**
   - TMI threshold (default: 0.3)
   - Temporal info threshold (default: 0.1 bits)
   - Only 9 neurons passed both criteria

2. **Limited by data quality**
   - Insufficient trials for reliable tuning
   - Only 9 neurons show clear temporal modulation

3. **Display limitation**
   - Code selects top 12 by TMI
   - But may filter to only "significant" ones

---

## 🔍 What Gets Plotted

### The Plot Shows:
- **Top 12 neurons** (by default) ranked by TMI
- Each panel shows one neuron's temporal tuning curve
- Title shows: `Neuron ID (TMI=X.XX)`
- **These are the "best" temporally-tuned neurons in your data**

### TMI (Temporal Modulation Index):
```
TMI = std(firing_rate_over_time) / mean(firing_rate)

High TMI (>0.5): Strong temporal tuning
Medium TMI (0.3-0.5): Moderate tuning  
Low TMI (<0.3): Weak/no tuning
```

### Neurons are selected by:
1. Calculate TMI for ALL neurons
2. Sort neurons: highest TMI → lowest TMI
3. Take top 12 (or n_neurons parameter)
4. Plot these 12 in order

---

## 📈 New Features (v2.4+)

### **Now you get MORE information:**

#### **1. Enhanced Titles**
Each subplot now shows:
- Neuron index
- TMI value
- Temporal information (bits)

```
Neuron 42
TMI=0.85, Info=0.42bits
```

#### **2. Overall Title**
```
Top 12 Neurons by TMI
(out of 150 total neurons)
```
This tells you how many neurons you have total!

#### **3. Additional Plot: All Time Cells**

If you have ≤24 time cells, you now get an **additional plot**:
- `all_time_cells_tuning.png`
- Shows **every** time cell (not just top 12)
- Sorted by peak time
- All curves in red

If you have >24 time cells:
- Too many to plot individually
- See `population_heatmap.png` instead

---

## 🎨 Customization Options

### **1. Change Number of Neurons**

```python
# Plot top 20 neurons
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=20)

# Plot top 6 neurons (2 rows)
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=6)

# Plot top 30 neurons (10 rows!)
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=30)
```

### **2. Sort by Different Criteria**

```python
# Sort by TMI (default)
fig = NeuralVisualizer.plot_temporal_tuning(tuning, sort_by='tmi')

# Sort by temporal information
fig = NeuralVisualizer.plot_temporal_tuning(tuning, sort_by='info')

# Sort by peak time
fig = NeuralVisualizer.plot_temporal_tuning(tuning, sort_by='peak_time')
```

### **3. Set Minimum TMI Threshold**

```python
# Only plot neurons with TMI >= 0.5 (very strict)
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=12, min_tmi=0.5)

# Only plot neurons with TMI >= 0.2 (more lenient)
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=12, min_tmi=0.2)
```

---

## 💡 Understanding Your Results

### **Scenario 1: You see exactly 12 curves**
✅ You have at least 12 neurons with measurable temporal tuning  
✅ These are the top 12 by TMI  
✅ Everything is normal!

### **Scenario 2: You see 9 curves**
⚠️ You have exactly 9 neurons meeting the criteria  

**Why?**
- Only 9 neurons classified as "time cells"
- TMI threshold eliminated others
- Check your data quality

**What to do:**
```python
# Check how many time cells you have
n_time_cells = np.sum(time_cells)
print(f"Time cells: {n_time_cells}")

# Check TMI distribution
print(f"TMI range: {np.min(tmi)} to {np.max(tmi)}")
print(f"Neurons with TMI>0.3: {np.sum(tmi > 0.3)}")

# Plot more neurons if available
fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=20, min_tmi=0.1)
```

### **Scenario 3: You see fewer than expected**
⚠️ Limited temporal tuning in your data

**Possible causes:**
- Insufficient trials (need ~50+ for reliable tuning)
- Wrong time window (neurons tuned outside 0-3000ms)
- Wrong alignment (using stim1on vs trialon)
- Low firing rates overall
- Task doesn't engage temporal coding

**What to do:**
1. Check number of trials: `data.n_trials`
2. Check time window coverage
3. Try different alignment (stim1on vs trialon)
4. Lower thresholds temporarily to explore

---

## 📊 Interpreting the Plots

### **Good Temporal Tuning:**
```
Clear peak in firing rate
TMI > 0.5
Smooth curve (not noisy)
Peak occurs within trial window
```

### **Moderate Tuning:**
```
Visible peak but broader
TMI 0.3-0.5
Some variability
```

### **Weak Tuning:**
```
Flat or very broad
TMI < 0.3
Might not be a true time cell
```

---

## 🔬 Advanced Analysis

### **Plot Specific Neurons**

If you want to plot specific neurons (not just top by TMI):

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Manually select neurons
my_neurons = [5, 12, 23, 45, 67, 89]  # Your neurons of interest

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, neuron_idx in enumerate(my_neurons):
    ax = axes[i]
    ax.plot(tuning['time_vector'], tuning['tuning_curves'][neuron_idx, :], 'b-', linewidth=2)
    ax.fill_between(tuning['time_vector'], 0, tuning['tuning_curves'][neuron_idx, :], alpha=0.3)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Neuron {neuron_idx} (TMI={tuning["tmi"][neuron_idx]:.2f})')
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('my_selected_neurons.png', dpi=300)
```

### **Plot All Neurons (as grid)**

For ALL neurons (if you don't have too many):

```python
n_neurons = tuning['n_neurons']

if n_neurons <= 100:  # Reasonable limit
    n_cols = 10
    n_rows = int(np.ceil(n_neurons / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2))
    axes = axes.flatten()
    
    for i in range(n_neurons):
        ax = axes[i]
        ax.plot(tuning['time_vector'], tuning['tuning_curves'][i, :], linewidth=0.5)
        ax.set_title(f'N{i}', fontsize=6)
        ax.axis('off')
    
    for i in range(n_neurons, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('all_neurons_grid.png', dpi=300)
```

---

## ✅ Summary

### What the temporal tuning plot shows:
✅ **Top neurons** ranked by temporal modulation strength (TMI)  
✅ **NOT random** - these are your best time cells  
✅ Default: top 12, but you can customize  
✅ Each curve shows how one neuron's firing varies over time  

### If you see fewer curves than expected:
1. Check `n_time_cells` in your analysis
2. Look at TMI distribution
3. Consider lowering thresholds
4. Check you have enough trials
5. Verify correct time alignment

### New in v2.4+:
✅ Shows both TMI and temporal information  
✅ Overall title with neuron count  
✅ Additional plot for all time cells (if ≤24)  
✅ Better customization options  

---

**The neurons shown are your dataset's BEST temporal encoders!** 🧠⏱️
