# Data Inspection Feature - Documentation

## Overview

The data inspection feature allows you to **review the contents and structure** of your MAT file before running the full analysis pipeline. This helps you:

1. **Verify** the file loaded correctly
2. **Understand** the data structure
3. **Check** for expected variables
4. **Inspect** dimensions and data ranges
5. **Pause** before committing to a long analysis

---

## What You See

When you run `main_enhanced.jl` with inspection enabled, you'll see:

### 1. Summary Table

```
======================================================================
MAT FILE CONTENTS INSPECTION
======================================================================

📊 Found 4 variable(s) in MAT file:

┌──────────────────────────────┬────────────────────┬───────────────┐
│ Variable Name                │ Type               │ Dimensions    │
├──────────────────────────────┼────────────────────┼───────────────┤
│ cond_label                   │ Array{String,2}    │ 1×10          │
│ cond_matrix                  │ Array{Float64,2}   │ 3998×10       │
│ neur_tensor_stim1on          │ Array{Float64,3}   │ 37×3998×50    │
│ stim1on                      │ Array{Float64,1}   │ 50            │
└──────────────────────────────┴────────────────────┴───────────────┘
```

### 2. Detailed Information

```
======================================================================
DETAILED VARIABLE INFORMATION
======================================================================

[1] Variable: "cond_label"
  ├─ Type: Matrix{String}
  ├─ Dimensions: (1, 10)
  ├─ Element type: String
  ├─ Total elements: 10
  └─ Values: ["col1" "col2" "col3" "col4" "col5" "col6" "col7" "col8" "col9" "col10"]

[2] Variable: "cond_matrix"
  ├─ Type: Matrix{Float64}
  ├─ Dimensions: (3998, 10)
  ├─ Element type: Float64
  ├─ Total elements: 39980
  ├─ Range: [0.0, 145.0]
  ├─ Mean: 23.4567
  ├─ Std: 12.3456
  └─ Sample corner: [1.0 2.0; 3.0 4.0]

[3] Variable: "neur_tensor_stim1on"
  ├─ Type: Array{Float64,3}
  ├─ Dimensions: (37, 3998, 50)
  ├─ Element type: Float64
  ├─ Total elements: 7396300
  ├─ Range: [0.0, 89.234]
  ├─ Mean: 8.4523
  ├─ Std: 5.2341
  └─ Sample corner: [12.3 14.5; 9.8 11.2]

[4] Variable: "stim1on"
  ├─ Type: Vector{Float64}
  ├─ Dimensions: (50,)
  ├─ Element type: Float64
  ├─ Total elements: 50
  ├─ Range: [1.0, 50.0]
  ├─ Mean: 25.5
  ├─ Std: 14.5774
  └─ Values: [1.0, 2.0, 3.0, 4.0, 5.0, ..., 50.0]
```

### 3. Validation

```
🔍 Validating expected variables...
  ✓ Found: cond_label
  ✓ Found: cond_matrix
  ✓ Found: neur_tensor_stim1on
  ✓ Found: stim1on

✓ All expected variables are present!
```

### 4. Pause for Confirmation

```
──────────────────────────────────────────────────────────────────────
⏸️  PAUSED: Review the data structure above
──────────────────────────────────────────────────────────────────────

Continue with analysis? (press Enter to continue, or Ctrl+C to abort):
```

---

## Configuration

### Enable/Disable Inspection

In `main_enhanced.jl`, find this line (around line 38):

```julia
ENABLE_DATA_INSPECTION = true  # Set to false to skip inspection and pause
```

**Enabled (default):**
```julia
ENABLE_DATA_INSPECTION = true
```
- Shows full inspection report
- Validates expected variables
- Pauses for user confirmation

**Disabled (for automated runs):**
```julia
ENABLE_DATA_INSPECTION = false
```
- Skips inspection entirely
- No pause
- Proceeds directly to analysis

---

## Use Cases

### 1. First-Time Analysis

**Always enable inspection** when:
- Running analysis on a new dataset
- Unsure about data structure
- Want to verify file loaded correctly

### 2. Production/Automated Runs

**Disable inspection** when:
- Running batch processing
- Data structure is known and stable
- Running in automated scripts
- Don't want manual intervention

### 3. Debugging

**Enable inspection** to:
- Check for missing variables
- Verify data dimensions match expectations
- Inspect data ranges for anomalies
- Find NaN or Inf values

---

## Understanding the Output

### Variable Types

Common types you'll see:

- **Array{Float64,1}** - 1D array (vector)
- **Array{Float64,2}** - 2D array (matrix)
- **Array{Float64,3}** - 3D array (tensor)
- **Array{String,2}** - 2D string array
- **Float64** - Single number (scalar)

### Dimensions

For the neural data tensor `neur_tensor_stim1on`:
```
Dimensions: (37, 3998, 50)
         │    │     │
         │    │     └── 50 trials
         │    └──────── 3998 time bins
         └───────────── 37 neurons
```

### Statistics

For numeric arrays, you'll see:
- **Range**: [min, max] values
- **Mean**: Average value
- **Std**: Standard deviation
- **Special values**: Count of NaN or Inf (if present)

### Sample Values

- **Small arrays** (<10 elements): Shows all values
- **Medium arrays** (10-100 elements): Shows first few values
- **Large arrays** (>100 elements): Shows corner samples

---

## Interactive Usage

### Continue Analysis

After reviewing the data:
1. Read the information displayed
2. Verify dimensions match expectations
3. Check that required variables are present
4. Press **Enter** to continue

### Abort Analysis

If something looks wrong:
1. Press **Ctrl+C** to abort
2. Fix the data file or configuration
3. Run again

---

## Advanced Features

### Quick Inspect Single Variable

If you want to inspect just one variable:

```julia
# After loading data
using .DataInspector
quick_inspect(data, "neur_tensor_stim1on")
```

### Custom Validation

To check for different variables:

```julia
# Modify the expected variables list
expected_variables = ["var1", "var2", "var3"]
validate_expected_variables(data, expected_variables)
```

### Programmatic Inspection

Use inspection in your own code:

```julia
using .DataInspector

# Load data
data = matread("myfile.mat")

# Inspect
inspect_mat_file(data)

# Validate
if validate_expected_variables(data, ["required_var"])
    # Proceed with analysis
else
    error("Missing required variables!")
end
```

---

## Troubleshooting

### Issue: Inspection takes too long

**Cause:** Very large arrays (>100 million elements)

**Solution:**
```julia
ENABLE_DATA_INSPECTION = false  # Disable for very large files
```

### Issue: Can't see all information

**Cause:** Console buffer too small

**Solution:**
- Scroll up in your terminal
- Redirect output to file:
  ```bash
  julia main_enhanced.jl > output.log 2>&1
  ```

### Issue: Expected variables not found

**Cause:** Wrong file or file structure changed

**Check:**
1. Verify file path is correct
2. Check variable names in MATLAB/Octave
3. Ensure file was saved correctly

**Fix:**
- Update expected variables list
- Use correct file
- Re-save MAT file with proper structure

### Issue: Special values (NaN/Inf) detected

**Cause:** Data preprocessing issue or missing values

**Action:**
- Review preprocessing steps
- Check for division by zero
- Verify data quality before analysis

---

## Example Session

```
Configuration:
  Sampling frequency: 1000.0 Hz
  Window size: 300
  Neuron 1
  Data inspection: enabled

======================================================================
MAT FILE CONTENTS INSPECTION
======================================================================

📊 Found 4 variable(s) in MAT file:

[Variable table displayed...]

======================================================================
DETAILED VARIABLE INFORMATION
======================================================================

[Detailed info for each variable...]

🔍 Validating expected variables...
  ✓ Found: cond_label
  ✓ Found: cond_matrix
  ✓ Found: neur_tensor_stim1on
  ✓ Found: stim1on

✓ All expected variables are present!

──────────────────────────────────────────────────────────────────────
⏸️  PAUSED: Review the data structure above
──────────────────────────────────────────────────────────────────────

Continue with analysis? (press Enter to continue, or Ctrl+C to abort): [Enter]

✓ Continuing with analysis...

Using filter: moving_average
Window size: 300

[Analysis continues...]
```

---

## Benefits

✅ **Verify data integrity** before long analysis  
✅ **Catch errors early** (missing variables, wrong dimensions)  
✅ **Understand data structure** without opening MATLAB  
✅ **Document data format** for reproducibility  
✅ **Prevent wasted computation** on incorrect data  
✅ **Interactive control** over when to proceed  

---

## Files

- **data_inspector.jl** - Inspection utility module
- **main_enhanced.jl** - Integrated inspection (lines 38, 52-66)

---

## Summary

The data inspection feature provides:

1. **Automatic inspection** of all variables in MAT file
2. **Detailed information** about dimensions, types, and statistics
3. **Validation** of expected variables
4. **Pause point** for user review and confirmation
5. **Configuration option** to enable/disable

**Default behavior:** Enabled (shows inspection and pauses)  
**For automation:** Disable with `ENABLE_DATA_INSPECTION = false`

This feature helps ensure you're working with the correct data before investing time in analysis!
