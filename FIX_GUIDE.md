# Fix for MATLAB v7.3 Loading Error

## Problem
Your code is trying to load a MATLAB v7.3 file using `scipy.io.loadmat`, which doesn't support this format. MATLAB v7.3 files use HDF5 format and require the `h5py` library.

## Error Message
```
Error loading data/amadeus01172020_a_neur_tensor_joyon.mat: 
Please use HDF reader for matlab v7.3 files, e.g. h5py
```

## Solution Overview

The fix involves two main changes to `hippocampus_data_loader.py`:

1. Add `h5py` import
2. Modify the `_load_mat_file` method to automatically detect and handle both formats
3. Add a new `_load_hdf5_mat` method for v7.3 files

## Quick Fix Steps

### Option 1: Replace the entire file (RECOMMENDED)

1. **Install h5py** (if not already installed):
   ```bash
   pip install h5py
   ```

2. **Replace the file**:
   ```bash
   cd /path/to/compneuro
   cp hippocampus_data_loader.py hippocampus_data_loader.py.backup
   cp hippocampus_data_loader_fixed.py hippocampus_data_loader.py
   ```

3. **Test**:
   ```bash
   python example_analysis.py data/amadeus01172020_a_neur_tensor_joyon.mat
   ```

### Option 2: Manual patch

If you prefer to modify your existing file manually:

1. **Install h5py**:
   ```bash
   pip install h5py
   ```

2. **Add import** at the top of `hippocampus_data_loader.py`:
   ```python
   import h5py
   ```

3. **Replace the `_load_mat_file` method** with:
   ```python
   @staticmethod
   def _load_mat_file(filepath: str) -> Dict[str, Any]:
       """
       Load MATLAB file with automatic detection of format (old vs v7.3).
       """
       filepath = Path(filepath)
       if not filepath.exists():
           raise FileNotFoundError(f"File not found: {filepath}")
       
       # Try loading with scipy first (for older MATLAB formats)
       try:
           data = loadmat(str(filepath), squeeze_me=False, struct_as_record=False)
           print(f"Loaded {filepath.name} using scipy.io.loadmat (MATLAB < v7.3)")
           return data
       except NotImplementedError:
           # File is in HDF5/v7.3 format, use h5py
           print(f"Loading {filepath.name} using h5py (MATLAB v7.3)")
           return HippocampusDataLoader._load_hdf5_mat(filepath)
       except Exception as e:
           raise RuntimeError(f"Error loading {filepath}: {str(e)}")
   ```

4. **Add this new method** to the class:
   ```python
   @staticmethod
   def _load_hdf5_mat(filepath: Path) -> Dict[str, Any]:
       """
       Load MATLAB v7.3 (HDF5) file.
       """
       data_dict = {}
       
       with h5py.File(filepath, 'r') as f:
           # Recursively load all datasets
           def load_dataset(name, obj):
               if isinstance(obj, h5py.Dataset):
                   data = obj[()]
                   
                   # Handle MATLAB's column-major vs numpy's row-major
                   if data.ndim > 1:
                       data = data.T
                   
                   # Skip object references
                   if obj.dtype == np.dtype('O'):
                       return
                   
                   data_dict[name] = data
           
           # Visit all items in the file
           f.visititems(load_dataset)
           
           # Also get top-level datasets directly
           for key in f.keys():
               if key not in data_dict and isinstance(f[key], h5py.Dataset):
                   data = f[key][()]
                   if data.ndim > 1:
                       data = data.T
                   data_dict[key] = data
       
       return data_dict
   ```

## What Changed?

### Before (Original Code)
```python
@staticmethod
def _load_mat_file(filepath: str) -> Dict[str, Any]:
    """Load MATLAB file."""
    try:
        return loadmat(str(filepath), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}: {str(e)}")
```

This would fail on v7.3 files with the error you saw.

### After (Fixed Code)
The new code:
1. **First tries** `scipy.io.loadmat` (for old MATLAB formats)
2. **If that fails** with `NotImplementedError`, it switches to `h5py` (for v7.3)
3. **Handles the transpose** needed for HDF5 files (MATLAB is column-major, NumPy is row-major)
4. **Prints which method** was used (helpful for debugging)

## Testing the Fix

After applying the fix, test with:

```bash
cd /path/to/compneuro
python -c "
from hippocampus_data_loader import HippocampusDataLoader
data = HippocampusDataLoader.load_mat_file('data/amadeus01172020_a_neur_tensor_joyon.mat')
print(data.summary())
"
```

You should see:
```
Loading amadeus01172020_a_neur_tensor_joyon.mat using h5py (MATLAB v7.3)
Data Summary for amadeus01172020_a_neur_tensor_joyon.mat
============================================================
Neural tensor shape: (X, Y, Z) (neurons × time × trials)
...
```

## Troubleshooting

### If you get "ModuleNotFoundError: No module named 'h5py'":
```bash
pip install h5py
# or
pip install --break-system-packages h5py  # if on certain Linux systems
```

### If the data loads but has wrong dimensions:
The transpose handling in `_load_hdf5_mat` should fix this, but if you still have issues:
- Check that `neur_tensor_trialon` has shape (neurons, time, trials)
- Check that `cond_matrix` has shape (trials, conditions)

### If you get "Unable to open file":
- Verify the file path is correct
- Ensure the .mat file is not corrupted
- Try opening it in MATLAB to verify it's valid

## Why This Happens

MATLAB changed its default save format:
- **Before MATLAB R2006b**: Used MAT-File Level 5 format (readable by scipy)
- **After MATLAB R2006b**: Can use `-v7.3` flag, which uses HDF5 format (requires h5py)

Your file was saved with `-v7.3`, hence the error.

## Additional Notes

The fixed loader now handles:
- ✅ Old MATLAB formats (< v7.3)
- ✅ New MATLAB formats (v7.3 with HDF5)
- ✅ Automatic format detection
- ✅ Proper array transposition for HDF5
- ✅ Informative console output

## Files Provided

1. **hippocampus_data_loader_fixed.py** - Complete replacement file
2. **fix_matlab_loader.py** - Patch helper script
3. **FIX_GUIDE.md** - This document

Choose the option that works best for you!
