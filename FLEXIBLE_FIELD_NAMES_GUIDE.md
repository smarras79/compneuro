# Handling Different MATLAB File Naming Conventions

## The Problem

Your MATLAB files may use different field names depending on:
- Different experimental protocols (stim1on, stim2on, trialon, etc.)
- Different preprocessing pipelines
- Different lab conventions
- Different versions of analysis code

For example, you encountered:
- ❌ Expected: `neur_tensor_trialon`
- ✅ Your file has: `neur_tensor_stim1on`

## The Solution (v2.3)

The data loader now **automatically searches** for neural tensor fields using multiple naming patterns!

### Supported Neural Tensor Names

The loader tries these field names in order:
1. `neur_tensor_trialon` (original expected name)
2. `neur_tensor_stim1on` ← **YOUR FILE**
3. `neur_tensor_stim2on`
4. `neur_tensor_stimon`
5. `neur_tensor`
6. `neural_tensor`
7. `spikes`
8. `firing_rates`

### Supported LFP Tensor Names

Similarly for LFP data:
1. `lfp_tensor_trialon`
2. `lfp_tensor_stim1on` ← **YOUR FILE (if present)**
3. `lfp_tensor_stim2on`
4. `lfp_tensor_stimon`
5. `lfp_tensor`
6. `lfp`

---

## Usage

### No Changes Needed!

Just load your file normally:

```python
from hippocampus_data_loader import HippocampusDataLoader

# Works with ANY of the supported field names
data = HippocampusDataLoader.load_mat_file('amadeus01172020_a_neur_tensor_stim1on.mat')
```

You'll see:
```
✓ Found neural data field: 'neur_tensor_stim1on'
✓ Loaded amadeus01172020_a_neur_tensor_stim1on.mat using h5py (MATLAB v7.3)
✓ Data validation passed
```

---

## Inspecting Unknown Files

If you have a file with unknown structure, use the inspector:

```bash
python inspect_mat_file.py your_file.mat
```

This shows:
- All available fields
- Field shapes and data types
- Whether the file can be loaded
- Suggestions for loading

### Example Output:

```
Inspecting: amadeus01172020_a_neur_tensor_stim1on.mat
======================================================================
Format: MATLAB v7.3 (HDF5)

Available fields:
  cond_matrix                    shape=(500, 12)  dtype=float64
  neur_tensor_stim1on           shape=(150, 9999, 500)  dtype=float64
  stim1on/aft                    shape=(1,)  dtype=float64
  stim1on/bef                    shape=(1,)  dtype=float64
  tensor_structure               shape=(1,)  dtype=object
  cond_label                     shape=(12,)  dtype=object
======================================================================

LOADING SUGGESTIONS
======================================================================

Attempting to load file...
✓ File loaded successfully!

Data shape: (150, 9999, 500) (neurons × time × trials)
Condition matrix shape: (500, 12)
```

---

## Multiple Files with Different Names

You can mix files with different naming conventions:

```python
# These all work!
data1 = HippocampusDataLoader.load_mat_file('session_trialon.mat')
# ✓ Found neural data field: 'neur_tensor_trialon'

data2 = HippocampusDataLoader.load_mat_file('session_stim1on.mat')
# ✓ Found neural data field: 'neur_tensor_stim1on'

data3 = HippocampusDataLoader.load_mat_file('session_stim2on.mat')
# ✓ Found neural data field: 'neur_tensor_stim2on'
```

---

## Adding Custom Field Names

If your file uses a completely different name, you can add it to the loader:

### Method 1: Edit the loader (permanent)

In `hippocampus_data_loader.py`, add your field name to the list:

```python
def _get_neural_tensor(self) -> np.ndarray:
    # List of possible field names for neural data
    possible_names = [
        'neur_tensor_trialon',
        'neur_tensor_stim1on',
        # ... existing names ...
        'your_custom_field_name',  # ← Add here
    ]
```

### Method 2: Direct access (temporary)

For one-off files:

```python
from hippocampus_data_loader import HippocampusDataLoader

# Load file normally (will fail to find tensor)
# But you can access raw data dictionary
loader = HippocampusDataLoader.__new__(HippocampusDataLoader)
loader.filepath = Path('your_file.mat')
loader.data_dict = HippocampusDataLoader._load_mat_file('your_file.mat')

# Manually set the tensor
loader.neur_tensor = loader.data_dict['your_custom_name']
loader.cond_matrix = loader.data_dict['cond_matrix']

# Continue normally
loader._validate_data()
# etc.
```

---

## Understanding Different Event Alignments

Different field names often represent different time alignments:

### `neur_tensor_trialon`
- Aligned to trial onset
- Time 0 = trial starts
- Use for: Overall trial analysis

### `neur_tensor_stim1on`
- Aligned to stimulus 1 onset
- Time 0 = first stimulus appears
- Use for: Analyzing response to first landmark

### `neur_tensor_stim2on`
- Aligned to stimulus 2 onset
- Time 0 = second stimulus appears
- Use for: Analyzing response to second landmark

### `neur_tensor_stimon`
- Aligned to generic stimulus onset
- Time 0 = stimulus appears
- Use for: General stimulus response

---

## Best Practices

### 1. Always Inspect Unknown Files First

```bash
python inspect_mat_file.py unknown_file.mat
```

### 2. Check What Was Loaded

```python
data = HippocampusDataLoader.load_mat_file('file.mat')
print(data.summary())
```

### 3. Verify Time Alignment

Different field names may have different time vectors:

```python
print(f"Time range: {data.time_vector[0]} to {data.time_vector[-1]} ms")
print(f"Zero point represents: [check field name for alignment]")
```

### 4. Document Your Files

Keep track of what each field name means in your specific dataset:

```python
# Good practice: Add comments
data_stim1 = HippocampusDataLoader.load_mat_file('session_stim1on.mat')
# Note: This is aligned to first landmark onset
# Time 0 = landmark 1 appears
# Use for analyzing immediate landmark responses
```

---

## Troubleshooting

### Issue: "Could not find neural tensor field"

**Solution:** Inspect the file to see what fields are available:
```bash
python inspect_mat_file.py your_file.mat
```

Then either:
1. Add your field name to the loader's search list
2. Rename your MATLAB variable before saving
3. Use the direct access method

### Issue: Different time ranges

Different alignments may have different time vectors:
- `trialon`: -500 to 9499 ms
- `stim1on`: -200 to 1999 ms
- `stim2on`: -200 to 1999 ms

**Solution:** Always check `data.time_vector` and adjust your analysis windows accordingly.

### Issue: HDF5 references (#refs#)

You may see fields like `#refs#/a`, `#refs#/b` in the inspector.

**Solution:** These are internal HDF5 references, not actual data. Ignore them. The loader automatically filters them out.

---

## Examples

### Example 1: Load stim1on file

```python
data = HippocampusDataLoader.load_mat_file('amadeus01172020_a_neur_tensor_stim1on.mat')
# ✓ Found neural data field: 'neur_tensor_stim1on'
print(data.summary())
```

### Example 2: Batch process mixed files

```python
from pathlib import Path

for mat_file in Path('data/').glob('*.mat'):
    try:
        data = HippocampusDataLoader.load_mat_file(mat_file)
        print(f"✓ Loaded {mat_file.name}")
        print(f"  Shape: {data.neur_tensor.shape}")
    except Exception as e:
        print(f"✗ Failed to load {mat_file.name}: {e}")
```

### Example 3: Compare different alignments

```python
# Load different alignments
data_trial = HippocampusDataLoader.load_mat_file('session_trialon.mat')
data_stim1 = HippocampusDataLoader.load_mat_file('session_stim1on.mat')
data_stim2 = HippocampusDataLoader.load_mat_file('session_stim2on.mat')

print(f"Trial onset time range: {data_trial.time_vector[0]} to {data_trial.time_vector[-1]}")
print(f"Stim1 onset time range: {data_stim1.time_vector[0]} to {data_stim1.time_vector[-1]}")
print(f"Stim2 onset time range: {data_stim2.time_vector[0]} to {data_stim2.time_vector[-1]}")
```

---

## Summary

✅ **No code changes needed** - The loader automatically finds your neural data  
✅ **Supports multiple naming conventions** - trialon, stim1on, stim2on, etc.  
✅ **Easy inspection** - Use `inspect_mat_file.py` to see file structure  
✅ **Backwards compatible** - Old code still works  
✅ **Extensible** - Easy to add new field names  

Your `neur_tensor_stim1on` files now work perfectly! 🎉

---

## Version History

**v2.3** (Current)
- ✅ Flexible field name detection
- ✅ Support for stim1on, stim2on, stimon alignments
- ✅ Added inspect_mat_file.py utility
- ✅ Better error messages with available fields

**v2.2**
- TA vs TP plotting

**v2.1**
- Bug fix: Dynamic time vector

**v2.0**
- MATLAB v7.3 support

---

For questions or issues, refer to the main README.md or open an issue.
