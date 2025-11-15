"""
Fixed Hippocampus Data Loader with HDF5/v7.3 MATLAB file support
"""
import numpy as np
import h5py
from scipy.io import loadmat
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

class HippocampusDataLoader:
    """
    Loader for hippocampus neural data with support for both old and v7.3 MATLAB formats.
    """
    
    def __init__(self, filepath: str):
        """Initialize with data from .mat file."""
        self.filepath = Path(filepath)
        self.data_dict = self._load_mat_file(filepath)
        
        # Extract main fields
        self.neur_tensor = self._get_array('neur_tensor_trialon')
        self.cond_matrix = self._get_array('cond_matrix')
        
        # Optional LFP data
        self.lfp_tensor = self._get_array('lfp_tensor_trialon', required=False)
        
        # Validate data
        self._validate_data()
        
        # Create time vector (assuming 1ms bins, -500 to 9500ms)
        self.time_vector = np.arange(-500, 9500)
        
        # Store dimensions
        self.n_neurons, self.n_timepoints, self.n_trials = self.neur_tensor.shape
        
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
                raise ValueError(f"Required field '{field_name}' not found in data file. "
                               f"Available fields: {list(self.data_dict.keys())}")
            return None
        
        data = self.data_dict[field_name]
        
        # Ensure it's a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        return data
    
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
    
    def summary(self) -> str:
        """Generate a summary of the loaded data."""
        summary_lines = [
            f"Data Summary for {self.filepath.name}",
            "=" * 60,
            f"Neural tensor shape: {self.neur_tensor.shape} (neurons × time × trials)",
            f"  - {self.n_neurons} neurons",
            f"  - {self.n_timepoints} time points",
            f"  - {self.n_trials} trials",
            f"Condition matrix shape: {self.cond_matrix.shape} (trials × conditions)",
            f"Time vector: {self.time_vector[0]} to {self.time_vector[-1]} ms",
        ]
        
        if self.lfp_tensor is not None:
            summary_lines.append(f"LFP tensor shape: {self.lfp_tensor.shape}")
        
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


def test_loader():
    """Test the data loader with both MATLAB formats."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hippocampus_data_loader_fixed.py <path_to_mat_file>")
        return
    
    filepath = sys.argv[1]
    
    print(f"\nLoading: {filepath}")
    print("-" * 60)
    
    try:
        # Load data
        data = HippocampusDataLoader.load_mat_file(filepath)
        
        # Print summary
        print(data.summary())
        
        # Try filtering
        print("\n" + "=" * 60)
        print("Testing trial filtering...")
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
        
        print("\n✓ Data loader test successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loader()
