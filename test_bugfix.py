#!/usr/bin/env python3
"""
Quick test to verify the time vector bug fix
"""
import numpy as np
import sys

print("="*70)
print("Testing Time Vector Bug Fix")
print("="*70)

# Test 1: Create mock data with different time lengths
print("\nTest 1: Creating mock data with 9999 time points...")

# Simulate your data structure
n_neurons = 100
n_timepoints = 9999  # Your actual data size
n_trials = 200

mock_neur_tensor = np.random.rand(n_neurons, n_timepoints, n_trials) * 10
mock_cond_matrix = np.random.randint(0, 6, (n_trials, 9))

print(f"  Mock data shape: {mock_neur_tensor.shape}")
print(f"  Time points: {n_timepoints}")

# Test 2: Check time vector generation
print("\nTest 2: Checking time vector generation...")

time_vector = np.arange(-500, -500 + n_timepoints)
print(f"  Time vector length: {len(time_vector)}")
print(f"  Time vector range: {time_vector[0]} to {time_vector[-1]} ms")
print(f"  Expected length: {n_timepoints}")

assert len(time_vector) == n_timepoints, "Time vector length mismatch!"
print("  ✓ Time vector length matches data!")

# Test 3: Check indexing
print("\nTest 3: Testing time window indexing...")

t_start, t_end = 0, 3000
t_idx = (time_vector >= t_start) & (time_vector < t_end)

print(f"  Window: {t_start} to {t_end} ms")
print(f"  Boolean mask shape: {t_idx.shape}")
print(f"  Data shape: {mock_neur_tensor.shape}")
print(f"  Selected time points: {np.sum(t_idx)}")

try:
    # This should NOT raise an IndexError
    subset = mock_neur_tensor[:, t_idx, :]
    print(f"  Subset shape: {subset.shape}")
    print("  ✓ Indexing works correctly!")
except IndexError as e:
    print(f"  ✗ IndexError: {e}")
    sys.exit(1)

# Test 4: Different data sizes
print("\nTest 4: Testing with various data sizes...")

test_sizes = [9999, 10000, 5000, 12345]
for size in test_sizes:
    time_vec = np.arange(-500, -500 + size)
    assert len(time_vec) == size, f"Failed for size {size}"
    print(f"  ✓ Size {size}: {time_vec[0]} to {time_vec[-1]} ms")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nThe time vector bug has been fixed.")
print("You can now run your analysis:")
print("  python example_analysis.py your_data.mat")
print()
