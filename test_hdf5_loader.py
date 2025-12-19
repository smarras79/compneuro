"""
Test script for verifying HDF5 MATLAB file loading.

This script helps diagnose and test loading of MATLAB v7.3 files.

Usage:
    python test_hdf5_loader.py <path_to_mat_file>
"""

import sys
from pathlib import Path
import h5py
import numpy as np

from hippocampus_data_loader import HippocampusDataLoader


def inspect_hdf5_structure(filepath: Path):
    """
    Inspect the structure of an HDF5-based MATLAB file.

    Args:
        filepath: Path to .mat file
    """
    print("\n" + "="*70)
    print("  HDF5 FILE STRUCTURE INSPECTION")
    print("="*70)

    with h5py.File(filepath, 'r') as f:
        print(f"\nFile: {filepath.name}")
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  - {key}")

        print("\nDetailed structure:")

        def print_structure(name, obj, indent=0):
            """Recursively print HDF5 structure."""
            prefix = "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{prefix}{name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{prefix}{name}: Group")

        f.visititems(print_structure)

        # Check for required fields
        print("\n" + "-"*70)
        print("Required field check:")
        required = ['neur_tensor_trialon', 'cond_matrix']
        for field in required:
            if field in f:
                shape = f[field].shape
                dtype = f[field].dtype
                print(f"  ✓ {field:25s}: shape={shape}, dtype={dtype}")
            else:
                print(f"  ✗ {field:25s}: MISSING")

        # Check optional fields
        print("\nOptional field check:")
        optional = ['lfp_tensor_trialon']
        for field in optional:
            if field in f:
                shape = f[field].shape
                dtype = f[field].dtype
                print(f"  ✓ {field:25s}: shape={shape}, dtype={dtype}")
            else:
                print(f"  - {field:25s}: not present")


def test_load_file(filepath: Path):
    """
    Test loading the file with the enhanced loader.

    Args:
        filepath: Path to .mat file
    """
    print("\n" + "="*70)
    print("  TESTING DATA LOADER")
    print("="*70)

    try:
        # Load the file
        neural_data = HippocampusDataLoader.load_mat_file(filepath)

        # Print summary
        print("\n" + neural_data.summary())

        # Test basic operations
        print("\n" + "-"*70)
        print("Testing basic operations:")

        # Get mental navigation trials
        mn_trials = neural_data.get_mental_navigation_trials()
        print(f"\n✓ Mental navigation trials: {np.sum(mn_trials)}/{neural_data.n_trials}")

        # Get neural activity
        activity = neural_data.get_neural_activity(mn_trials, (0, 3000))
        print(f"✓ Neural activity shape: {activity.shape}")
        print(f"  (neurons × time × trials)")

        # Check data ranges
        print(f"✓ Neural activity range: [{np.min(activity):.3f}, {np.max(activity):.3f}]")
        print(f"✓ Mean firing rate: {np.mean(activity):.3f} Hz")

        print("\n" + "="*70)
        print("  SUCCESS: File loaded and validated!")
        print("="*70)

        return neural_data

    except Exception as e:
        print("\n" + "="*70)
        print("  ERROR LOADING FILE")
        print("="*70)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_hdf5_loader.py <path_to_mat_file>")
        print("\nExample:")
        print("  python test_hdf5_loader.py data/amadeus01172020_a_neur_tensor_stim1on.mat")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Step 1: Inspect file structure
    try:
        inspect_hdf5_structure(filepath)
    except Exception as e:
        print(f"\nError inspecting file: {str(e)}")
        print("This may not be an HDF5 file.")

    # Step 2: Test loading
    neural_data = test_load_file(filepath)

    if neural_data is not None:
        print("\n✓ All tests passed! The file can be used with example_analysis_1.py")
        print(f"\nNext steps:")
        print(f"  python example_analysis_1.py {filepath} --output results")


if __name__ == "__main__":
    main()
