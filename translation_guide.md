# MATLAB to Python Translation Guide
## Neural Data Analysis Script

### Key Translation Points

#### 1. **Array Indexing**
- **MATLAB**: 1-indexed, `cond_matrix(:,10)` means column 10
- **Python**: 0-indexed, `cond_matrix[:, 9]` means column 10

#### 2. **Data Loading**
```matlab
% MATLAB
load('./data/amadeus01172020_a_neur_tensor_stim1on.mat')
```

```python
# Python - handles both old and v7.3 formats
data = load_mat_file('./data/amadeus01172020_a_neur_tensor_stim1on.mat')
```

**Important**: The function automatically detects and handles:
- Old MATLAB format (.mat < v7.3) using `scipy.io.loadmat`
- New MATLAB format (.mat v7.3 HDF5) using `h5py`
- Automatic transposition for HDF5 files (they store in column-major, Python uses row-major)

#### 3. **Finding Elements**
```matlab
% MATLAB
trid = find(cond_matrix(:,10)==1);
```

```python
# Python
trid = np.where(cond_matrix[:, 9] == 1)[0]
```

**Note**: `np.where()` returns a tuple, so `[0]` extracts the array of indices.

#### 4. **Logical Operations**
```matlab
% MATLAB
trid = find(cond_matrix(:,10)==1 & cond_matrix(:,3)==1 & cond_matrix(:,4)==4);
```

```python
# Python - element-wise logical AND
trid = np.where((cond_matrix[:, 9] == 1) & 
                (cond_matrix[:, 2] == 1) & 
                (cond_matrix[:, 3] == 4))[0]
```

**Important**: Use `&` for element-wise AND, not `and`. Parentheses are required!

#### 5. **Array Squeezing**
```matlab
% MATLAB
fr3 = squeeze(neur_tensor_stim1on(1,:,trid));
```

```python
# Python
fr3 = np.squeeze(neur_tensor_stim1on[0, :, trid])
```

#### 6. **Convolution**
```matlab
% MATLAB - 'valid' mode returns only fully overlapping part
conv(mean(fr3,2), ones(300,1), 'valid')
```

```python
# Python
mean_fr3 = np.mean(fr3, axis=1)  # Mean across trials (columns)
smooth_fr3 = np.convolve(mean_fr3, np.ones(300), mode='valid')
```

**Key difference**: 
- MATLAB `mean(fr3, 2)` means mean along dimension 2 (columns)
- Python `np.mean(fr3, axis=1)` means mean along axis 1 (also columns in 2D)

#### 7. **Array Slicing**
```matlab
% MATLAB
stim1on.edges(150:end-150)
```

```python
# Python
edges[149:-150]  # 149 because Python is 0-indexed, -150 excludes last 150
```

### Plotting Differences

#### MATLAB
```matlab
subplot(2,2,1);
plot(ta_att1, tp_att1, 'ok');
```

#### Python
```python
plt.subplot(2, 2, 1)
plt.plot(ta_att1, tp_att1, 'ok', markerfacecolor='black')
```

### Data Structure Handling

The script handles MATLAB structures gracefully:

```python
# For old format .mat files, stim1on might be a structured array
if hasattr(stim1on, 'dtype') and stim1on.dtype.names:
    edges = stim1on['edges'][0, 0]
# For v7.3 HDF5 files, it's already extracted
else:
    edges = stim1on['edges']
```

### Common Issues and Solutions

1. **Dimension mismatch**: MATLAB is column-major, Python/NumPy is row-major
   - Solution: Transpose when needed, or specify axis correctly

2. **Mean operation**: Ensure correct axis
   - `mean(fr3, 2)` in MATLAB → `np.mean(fr3, axis=1)` in Python

3. **Indexing errors**: Remember Python is 0-indexed
   - Column 10 in MATLAB = index 9 in Python

4. **Logical operations**: Use element-wise operators
   - Use `&` not `and`, `|` not `or`
   - Always use parentheses around conditions

### Required Python Packages

```bash
pip install numpy scipy matplotlib h5py
```

### Running the Script

```bash
python sujay_example_zoom.py
```

Make sure your data file exists at:
```
./data/amadeus01172020_a_neur_tensor_stim1on.mat
```

### Output

The script produces:
1. **Figure 1**: Behavioral data (2 subplots)
   - True vs produced temporal distance for two attention conditions
2. **Figure 2**: Neural firing rates (smoothed)
   - Two conditions plotted with 300-sample moving average
3. **Console output**: Data shapes and unique temporal distances
