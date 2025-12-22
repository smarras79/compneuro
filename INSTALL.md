# Quick Setup Guide

## Installation

Your .mat file is in MATLAB v7.3 format (HDF5), so you need to install h5py:

```bash
pip install h5py
```

Or install all dependencies at once:

```bash
pip install numpy scipy matplotlib h5py
```

## Verify Installation

```bash
python -c "import numpy, scipy, matplotlib, h5py; print('All dependencies installed!')"
```

## Run the Analysis

```bash
python example_analysis_1.py data/amadeus01172020_a_neur_tensor_stim1on.mat
```

## If You Get Import Errors

If you're using a virtual environment:

```bash
# Create virtual environment
python -m venv myenv

# Activate it
# On macOS/Linux:
source myenv/bin/activate
# On Windows:
myenv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib h5py

# Run analysis
python example_analysis_1.py your_data.mat
```

## What's Been Fixed

The updated `example_analysis_1.py` now automatically:
1. Tries to load with scipy (for older .mat files)
2. Falls back to h5py if the file is MATLAB v7.3 format
3. Handles the HDF5 data structure correctly
4. Transposes arrays as needed (MATLAB stores column-major, Python uses row-major)

The code works with both old and new MATLAB formats automatically!
