# Troubleshooting Guide - HDF5 Edges Extraction

## Issue
The error `ValueError: x and y must have same first dimension, but have shapes (0,) and (253,)` indicates that `time_axis` is empty while `smooth_fr3` has 253 points.

This means the `edges` array is not being extracted correctly from the HDF5 file structure.

## Diagnostic Steps

### Step 1: Inspect the HDF5 structure
Run the diagnostic script first:

```bash
python inspect_h5.py
```

This will show you:
- How `stim1on` is stored in the HDF5 file
- Whether it's a Dataset (with references) or a Group
- The structure of `edges` within `stim1on`
- Sample values from the edges array

### Step 2: Common HDF5 Structure Patterns

**Pattern 1: Reference to a struct**
```
stim1on → Dataset → Reference → Group → edges
```
In this case, `stim1on[0,0]` contains a reference that needs to be dereferenced.

**Pattern 2: Direct group**
```
stim1on → Group → edges
```
In this case, `edges` can be accessed directly.

### Step 3: Based on the diagnostic output

The updated script handles both patterns:

```python
if isinstance(stim1on_ref, h5py.Dataset):
    # Pattern 1: It's a reference
    ref = stim1on_ref[0, 0]
    stim1on_struct = h5file[ref]
    edges = np.array(stim1on_struct['edges']).flatten()
else:
    # Pattern 2: Direct access
    edges = np.array(stim1on_ref['edges']).flatten()
```

## Expected Debug Output

When you run the updated script, you should see:

```
Loading ./data/amadeus01172020_a_neur_tensor_stim1on.mat using h5py (v7.3 format)
Processing HDF5 format data...
cond_label: HDF5 object references (not loaded)
cond_matrix shape: (2254, 12)
neur_tensor_stim1on shape: (37, 3998, 2254)
edges shape: (3998,)  # <-- This should NOT be (0,)
edges range: [-1.500, 2.497]  # <-- Should show actual time values

Debug info:
mean_fr3 shape: (3998,)
mean_fr4 shape: (3998,)
smooth_fr3 shape: (3699,)  # = 3998 - 300 + 1
smooth_fr4 shape: (3699,)
time_axis shape: (3699,)  # <-- Should match smooth_fr3
edges length: 3998
```

## Manual Fix (if needed)

If the automatic extraction still fails, you can manually check the structure:

```python
import h5py
import numpy as np

with h5py.File('./data/amadeus01172020_a_neur_tensor_stim1on.mat', 'r') as f:
    # Check what stim1on is
    print(f['stim1on'])
    
    # If it's a Dataset with references:
    ref = f['stim1on'][0, 0]
    edges = np.array(f[ref]['edges']).flatten()
    
    # Or if it's a Group:
    edges = np.array(f['stim1on']['edges']).flatten()
```

## Key Points

1. HDF5 v7.3 stores MATLAB structs as:
   - **Groups** (direct access to fields)
   - **Datasets with object references** (need dereferencing)

2. Arrays are **transposed** in HDF5 (column-major vs row-major)

3. The `flatten()` is essential to convert from 2D (3998, 1) to 1D (3998,)

## Next Steps

1. Run `inspect_h5.py` first to understand your file structure
2. Share the output so I can adjust the extraction code if needed
3. The updated script includes better error handling and debug output
