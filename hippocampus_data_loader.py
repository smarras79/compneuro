"""
Hippocampus Neural Data Loader
Supports both old MATLAB formats and v7.3 (HDF5) formats
"""
import numpy as np
import h5py
from scipy.io import loadmat
from pathlib import Path
from typing import Optional, Dict, Any, Tuple


class HippocampusDataLoader:
    """
    Loader for hippocampus neural data with support for both old and v7.3 MATLAB formats.
    
    Expected data structure:
    - neur_tensor_trialon: (neurons × time × trials) neural firing rates in Hz
    - cond_matrix: (trials × conditions) experimental conditions
    - lfp_tensor_trialon: (channels × time × trials) optional LFP data
    """
    
    def __init__(self, filepath: str):
        """Initialize with data from .mat file."""
        self.filepath = Path(filepath)
        self.data_dict = self._load_mat_file(filepath)
        
        # Extract main fields - try multiple possible names for neural tensor
        self.neur_tensor = self._get_neural_tensor()
        self.cond_matrix = self._get_array('cond_matrix')
        
        # Optional LFP data - try multiple possible names
        self.lfp_tensor = self._get_lfp_tensor()
        
        # Validate data
        self._validate_data()
        
        # Store dimensions
        self.n_neurons, self.n_timepoints, self.n_trials = self.neur_tensor.shape
        
        # Create time vector based on actual data size
        # Assuming 1ms bins, starting from -500ms
        self.time_vector = np.arange(-500, -500 + self.n_timepoints)
        
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
            print(f"✓ Loaded {filepath.name} using scipy.io.loadmat (MATLAB < v7.3)")
            return data
        except NotImplementedError:
            # File is in HDF5/v7.3 format, use h5py
            print(f"✓ Loading {filepath.name} using h5py (MATLAB v7.3)")
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
    
    def _get_array(self, field_name: str, required: bool = True) -> Optional[np.ndarray]:
        """
        Get array from loaded data dictionary.
        
        Args:
            field_name: Name of the field to extract
            required: Whether this field is required
            
        Returns:
            Numpy array or None if not required and not found
        """
        if field_name not in self.data_dict:
            if required:
                available_fields = [k for k in self.data_dict.keys() 
                                  if not k.startswith('__') and not k.startswith('#refs#')]
                raise ValueError(f"Required field '{field_name}' not found in data file.\n"
                               f"Available fields: {available_fields}")
            return None
        
        data = self.data_dict[field_name]
        
        # Ensure it's a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        return data
    
    def _get_neural_tensor(self) -> np.ndarray:
        """
        Find and extract neural tensor from data, trying multiple possible field names.
        
        Returns:
            Neural tensor array (neurons × time × trials)
        """
        # List of possible field names for neural data (in order of preference)
        possible_names = [
            'neur_tensor_trialon',
            'neur_tensor_stim1on',
            'neur_tensor_stim2on',
            'neur_tensor_stimon',
            'neur_tensor',
            'neural_tensor',
            'spikes',
            'firing_rates'
        ]
        
        # Try each possible name
        for name in possible_names:
            if name in self.data_dict:
                print(f"✓ Found neural data field: '{name}'")
                data = self.data_dict[name]
                
                # Ensure it's a numpy array
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                
                return data
        
        # If none found, raise error with helpful message
        available_fields = [k for k in self.data_dict.keys() 
                          if not k.startswith('__') and not k.startswith('#refs#')]
        
        raise ValueError(
            f"Could not find neural tensor field in data file.\n"
            f"Tried: {possible_names}\n"
            f"Available fields: {available_fields}\n\n"
            f"Please ensure your .mat file contains one of the expected field names,\n"
            f"or modify the code to include your specific field name."
        )
    
    def _get_lfp_tensor(self) -> Optional[np.ndarray]:
        """
        Find and extract LFP tensor if available.
        
        Returns:
            LFP tensor array or None if not found
        """
        # List of possible field names for LFP data
        possible_names = [
            'lfp_tensor_trialon',
            'lfp_tensor_stim1on',
            'lfp_tensor_stim2on',
            'lfp_tensor_stimon',
            'lfp_tensor',
            'lfp'
        ]
        
        # Try each possible name
        for name in possible_names:
            if name in self.data_dict:
                print(f"✓ Found LFP data field: '{name}'")
                data = self.data_dict[name]
                
                # Ensure it's a numpy array
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                
                return data
        
        # LFP is optional, so return None if not found
        return None
    
    def _validate_data(self):
        """Validate the loaded data has correct dimensions."""
        # Check neural tensor
        if self.neur_tensor.ndim != 3:
            raise ValueError(f"neur_tensor_trialon should be 3D (neurons × time × trials), "
                           f"got shape {self.neur_tensor.shape}")
        
        # Check condition matrix
        if self.cond_matrix.ndim != 2:
            raise ValueError(f"cond_matrix should be 2D (trials × conditions), "
                           f"got shape {self.cond_matrix.shape}")
        
        # Check trial counts match
        n_trials_neural = self.neur_tensor.shape[2]
        n_trials_cond = self.cond_matrix.shape[0]
        
        if n_trials_neural != n_trials_cond:
            raise ValueError(f"Trial count mismatch: neural data has {n_trials_neural} trials, "
                           f"condition matrix has {n_trials_cond} trials")
        
        print(f"✓ Data validation passed")
    
    @classmethod
    def load_mat_file(cls, filepath: str) -> 'HippocampusDataLoader':
        """
        Class method to load data from .mat file.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            HippocampusDataLoader instance
        """
        return cls(filepath)
    
    @staticmethod
    def inspect_mat_file(filepath: str) -> Dict[str, Any]:
        """
        Inspect a MATLAB file and show available fields without loading full data.
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"\nInspecting: {filepath.name}")
        print("=" * 70)
        
        # Try loading with scipy first
        try:
            import scipy.io
            mat = scipy.io.whosmat(str(filepath))
            print(f"Format: MATLAB < v7.3 (readable by scipy)")
            print(f"\nAvailable fields:")
            for name, shape, dtype in mat:
                if not name.startswith('__'):
                    print(f"  {name:<30} shape={shape}  dtype={dtype}")
            
        except (NotImplementedError, ValueError):
            # HDF5 format
            print(f"Format: MATLAB v7.3 (HDF5)")
            print(f"\nAvailable fields:")
            
            with h5py.File(filepath, 'r') as f:
                def print_field(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if not name.startswith('#refs#'):
                            print(f"  {name:<30} shape={obj.shape}  dtype={obj.dtype}")
                
                f.visititems(print_field)
                
                # Also print top-level
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and not key.startswith('#refs#'):
                        print(f"  {key:<30} shape={f[key].shape}  dtype={f[key].dtype}")
        
        print("=" * 70)
        
        return {
            'filepath': filepath,
            'exists': True
        }
    
    def summary(self) -> str:
        """Generate a summary of the loaded data."""
        summary_lines = [
            f"\nData Summary: {self.filepath.name}",
            "=" * 70,
            f"Neural tensor shape: {self.neur_tensor.shape} (neurons × time × trials)",
            f"  - {self.n_neurons} neurons",
            f"  - {self.n_timepoints} time points (1ms bins)",
            f"  - {self.n_trials} trials",
            f"Condition matrix shape: {self.cond_matrix.shape} (trials × conditions)",
            f"Time vector: {self.time_vector[0]} to {self.time_vector[-1]} ms ({len(self.time_vector)} points)",
            f"Mean firing rate: {np.mean(self.neur_tensor):.2f} Hz",
            f"Max firing rate: {np.max(self.neur_tensor):.2f} Hz",
        ]
        
        if self.lfp_tensor is not None:
            summary_lines.append(f"LFP tensor shape: {self.lfp_tensor.shape}")
        
        summary_lines.append("=" * 70)
        
        return "\n".join(summary_lines)
    
    def get_condition(self, condition_name: str) -> np.ndarray:
        """
        Extract a specific condition from the condition matrix.
        
        Args:
            condition_name: Name of the condition (e.g., 'ta', 'tp', 'curr', 'target')
            
        Returns:
            Array of condition values for all trials
        """
        condition_map = {
            'ta': 0,      # True temporal distance
            'tp': 1,      # Produced temporal distance
            'curr': 2,    # Start landmark
            'target': 3,  # Target landmark
            'trial_type': 4,  # Trial type
            'seqq': 5,    # Sequence identity
            'succ': 6,    # Success flag
            'validtrials_mm': 7,  # Valid trials (mixture model)
            'attempt': 8  # Attempt number
        }
        
        if condition_name not in condition_map:
            raise ValueError(f"Unknown condition '{condition_name}'. "
                           f"Available: {list(condition_map.keys())}")
        
        col_idx = condition_map[condition_name]
        
        if col_idx >= self.cond_matrix.shape[1]:
            raise ValueError(f"Condition '{condition_name}' (column {col_idx}) not found. "
                           f"Condition matrix has {self.cond_matrix.shape[1]} columns")
        
        return self.cond_matrix[:, col_idx]
    
    def filter_trials(self, **conditions) -> np.ndarray:
        """
        Filter trials based on conditions.
        
        Args:
            **conditions: Condition name and value pairs
                         Can use operators: '!=', '<', '>', '<=', '>='
                         Example: trial_type=3, succ=1, seqq='<3'
        
        Returns:
            Boolean mask for trials matching all conditions
        """
        mask = np.ones(self.n_trials, dtype=bool)
        
        for cond_name, cond_value in conditions.items():
            cond_array = self.get_condition(cond_name)
            
            # Handle string operators
            if isinstance(cond_value, str):
                if cond_value.startswith('!='):
                    value = float(cond_value[2:])
                    mask &= (cond_array != value)
                elif cond_value.startswith('<='):
                    value = float(cond_value[2:])
                    mask &= (cond_array <= value)
                elif cond_value.startswith('>='):
                    value = float(cond_value[2:])
                    mask &= (cond_array >= value)
                elif cond_value.startswith('<'):
                    value = float(cond_value[1:])
                    mask &= (cond_array < value)
                elif cond_value.startswith('>'):
                    value = float(cond_value[1:])
                    mask &= (cond_array > value)
                else:
                    raise ValueError(f"Invalid operator in condition value: {cond_value}")
            else:
                # Direct equality
                mask &= (cond_array == cond_value)
        
        return mask
    
    def get_mental_navigation_trials(self) -> np.ndarray:
        """
        Get trials for mental navigation analysis (fully occluded, normal speed, first attempts).
        
        Returns:
            Boolean mask for mental navigation trials
        """
        return self.filter_trials(trial_type=3, seqq='<3', attempt=1)
    
    def get_neural_activity(self, 
                           trial_mask: Optional[np.ndarray] = None,
                           time_window: Optional[Tuple[int, int]] = None,
                           neurons: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract neural activity for specific trials, time window, and neurons.
        
        Args:
            trial_mask: Boolean mask for trials (default: all trials)
            time_window: Tuple of (start_ms, end_ms) (default: all time)
            neurons: Array of neuron indices (default: all neurons)
        
        Returns:
            Neural activity array (neurons × time × trials)
        """
        activity = self.neur_tensor.copy()
        
        # Filter neurons
        if neurons is not None:
            activity = activity[neurons, :, :]
        
        # Filter time
        if time_window is not None:
            t_start, t_end = time_window
            t_idx = (self.time_vector >= t_start) & (self.time_vector < t_end)
            activity = activity[:, t_idx, :]
        
        # Filter trials
        if trial_mask is not None:
            activity = activity[:, :, trial_mask]
        
        return activity


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hippocampus_data_loader.py <path_to_mat_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"\n{'='*70}")
    print(f"Testing Hippocampus Data Loader")
    print(f"{'='*70}")
    
    try:
        # Load data
        data = HippocampusDataLoader.load_mat_file(filepath)
        
        # Print summary
        print(data.summary())
        
        # Try filtering
        print("\n" + "=" * 70)
        print("Trial Filtering Examples")
        print("=" * 70)
        
        mn_trials = data.get_mental_navigation_trials()
        print(f"Mental navigation trials: {np.sum(mn_trials)} / {data.n_trials}")
        
        # Try extracting activity
        print("\nExtracting neural activity for mental navigation trials...")
        activity = data.get_neural_activity(
            trial_mask=mn_trials,
            time_window=(0, 3000)
        )
        print(f"Extracted activity shape: {activity.shape}")
        print(f"Mean firing rate: {np.mean(activity):.2f} Hz")
        
        print("\n" + "=" * 70)
        print("✓ Data loader test SUCCESSFUL!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
