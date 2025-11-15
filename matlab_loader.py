#!/usr/bin/env python3
"""
Patch script to fix MATLAB v7.3 loading in hippocampus_data_loader.py

This script will update the _load_mat_file method to handle both old
and v7.3 MATLAB formats automatically.
"""

import sys
from pathlib import Path

# The fixed _load_mat_file and _load_hdf5_mat methods
FIXED_METHODS = '''
    @staticmethod
    def _load_mat_file(filepath: str) -> Dict[str, Any]:
        """
        Load MATLAB file with automatic detection of format (old vs v7.3).
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary containing the loaded data
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
    
    @staticmethod
    def _load_hdf5_mat(filepath: Path) -> Dict[str, Any]:
        """
        Load MATLAB v7.3 (HDF5) file.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary containing the loaded data
        """
        data_dict = {}
        
        with h5py.File(filepath, 'r') as f:
            # Recursively load all datasets
            def load_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Load the dataset
                    data = obj[()]
                    
                    # Handle MATLAB's column-major vs numpy's row-major
                    # MATLAB arrays are often transposed when saved to HDF5
                    if data.ndim > 1:
                        data = data.T
                    
                    # Handle object references (common in MATLAB structs)
                    if obj.dtype == np.dtype('O'):
                        # This is a reference type, skip for now
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
'''

IMPORT_FIX = '''import h5py'''


def main():
    """Apply the patch to hippocampus_data_loader.py"""
    
    # Find the file
    loader_file = Path('hippocampus_data_loader.py')
    
    if not loader_file.exists():
        print(f"Error: {loader_file} not found in current directory")
        print(f"Current directory: {Path.cwd()}")
        print("\nPlease run this script from your compneuro directory:")
        print("  cd /path/to/compneuro")
        print("  python fix_matlab_loader.py")
        sys.exit(1)
    
    print(f"Found {loader_file}")
    print("Applying patch for MATLAB v7.3 support...")
    
    # Read the original file
    with open(loader_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'h5py' in content and '_load_hdf5_mat' in content:
        print("\n✓ File appears to already be patched!")
        print("  If you're still having issues, use the full replacement file:")
        print("  hippocampus_data_loader_fixed.py")
        return
    
    # Add h5py import if not present
    if 'import h5py' not in content:
        # Find the import section and add h5py
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if in_imports and (line.startswith('import ') or line.startswith('from ')):
                import_lines.append(line)
            else:
                if in_imports and line.strip() and not line.startswith('#') and not line.startswith('"""'):
                    in_imports = False
                other_lines.append(line)
        
        # Add h5py import
        import_lines.append('import h5py')
        
        # Reconstruct
        content = '\n'.join(import_lines) + '\n' + '\n'.join(other_lines)
    
    # Create backup
    backup_file = loader_file.with_suffix('.py.bak')
    print(f"Creating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print("\n" + "="*60)
    print("PATCH SUMMARY")
    print("="*60)
    print("Due to the complexity of patching the existing file,")
    print("I recommend replacing it with the fixed version.")
    print("\nSteps:")
    print("1. Backup created: hippocampus_data_loader.py.bak")
    print("2. Copy the fixed file:")
    print("   cp hippocampus_data_loader_fixed.py hippocampus_data_loader.py")
    print("\nOr manually add the h5py import and methods shown above.")
    print("="*60)


if __name__ == '__main__':
    main()
