#!/usr/bin/env python3
"""
MATLAB File Inspector

Utility script to inspect the contents of MATLAB files without loading all data.
Useful for understanding file structure before analysis.

Usage:
    python inspect_mat_file.py data.mat
    python inspect_mat_file.py data/*.mat
"""

import sys
from pathlib import Path
import argparse

from hippocampus_data_loader import HippocampusDataLoader


def inspect_file(filepath: str):
    """
    Inspect a single MATLAB file.
    
    Args:
        filepath: Path to .mat file
    """
    try:
        HippocampusDataLoader.inspect_mat_file(filepath)
        
        # Try to provide loading suggestions
        print("\n" + "="*70)
        print("LOADING SUGGESTIONS")
        print("="*70)
        
        # Check if file can be loaded
        try:
            print("\nAttempting to load file...")
            data = HippocampusDataLoader.load_mat_file(filepath)
            print(f"✓ File loaded successfully!")
            print(f"\nData shape: {data.neur_tensor.shape} (neurons × time × trials)")
            print(f"Condition matrix shape: {data.cond_matrix.shape}")
            
            # Show first few rows of condition matrix
            print(f"\nFirst 3 rows of condition matrix:")
            for i in range(min(3, data.cond_matrix.shape[0])):
                print(f"  Trial {i+1}: {data.cond_matrix[i, :min(5, data.cond_matrix.shape[1])]}")
            
        except Exception as e:
            print(f"✗ Could not load file automatically: {e}")
            print("\nTo load this file, you may need to:")
            print("  1. Check the field names match expected patterns")
            print("  2. Ensure data has correct dimensions (neurons × time × trials)")
            print("  3. Add your specific field name to the loader")
        
    except Exception as e:
        print(f"\n✗ Error inspecting file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Inspect MATLAB files to see their structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect single file
  python inspect_mat_file.py data.mat
  
  # Inspect multiple files
  python inspect_mat_file.py data/*.mat
  
  # Inspect with pattern
  python inspect_mat_file.py "data/*stim1on.mat"
        """
    )
    
    parser.add_argument('files', type=str, nargs='+',
                       help='Path(s) to .mat file(s)')
    
    args = parser.parse_args()
    
    # Process each file
    files = []
    for pattern in args.files:
        matched = list(Path('.').glob(pattern))
        if matched:
            files.extend(matched)
        elif Path(pattern).exists():
            files.append(Path(pattern))
    
    if not files:
        print(f"Error: No files found matching: {args.files}")
        sys.exit(1)
    
    print(f"\nFound {len(files)} file(s) to inspect\n")
    
    for filepath in files:
        inspect_file(str(filepath))
        if len(files) > 1:
            print("\n" + "="*70)
            print()


if __name__ == '__main__':
    main()
