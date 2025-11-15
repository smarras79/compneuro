"""
Hippocampus Neural Data Loader
===============================
Module for loading and preprocessing neural activity data from hippocampus recordings
during mental navigation tasks in monkeys.

Author: Computational Neuroscience Analysis Pipeline
Date: 2025-11-15
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class NeuralData:
    """
    Container for neural recording data from hippocampus.

    Attributes:
        neur_tensor: Neural firing rates (neurons × time × trials)
                    Time bins: 1ms resolution, -500ms to +9500ms relative to start landmark
        lfp_tensor: LFP data at 1kHz sampling (channels × time × trials)
        cond_matrix: Condition matrix (trials × condition_labels)
        condition_labels: Names of condition variables
        time_vector: Optional time vector in milliseconds. If None, defaults to -500 to +9500ms
        n_neurons: Number of recorded neurons
        n_trials: Number of trials
        n_timepoints: Number of time points
    """
    neur_tensor: np.ndarray
    lfp_tensor: np.ndarray
    cond_matrix: np.ndarray
    condition_labels: List[str]
    filename: str = ""
    time_vector: Optional[np.ndarray] = None

    # Derived properties
    n_neurons: int = field(init=False)
    n_trials: int = field(init=False)
    n_timepoints: int = field(init=False)

    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.n_neurons = self.neur_tensor.shape[0]
        self.n_timepoints = self.neur_tensor.shape[1]
        self.n_trials = self.neur_tensor.shape[2]

        # Set time vector if not provided
        if self.time_vector is None:
            # Default: -500ms to +9500ms in 1ms bins (standard hippocampus recording)
            if self.n_timepoints == 10001:
                self.time_vector = np.arange(-500, 9501, 1)
            else:
                # For synthetic/test data, create a simple time vector
                self.time_vector = np.arange(self.n_timepoints)

        # Validate dimensions
        assert len(self.time_vector) == self.n_timepoints, \
            f"Time vector length {len(self.time_vector)} doesn't match timepoints {self.n_timepoints}"
        assert self.cond_matrix.shape[0] == self.n_trials, \
            f"Condition matrix trials {self.cond_matrix.shape[0]} doesn't match neural data {self.n_trials}"

    def get_condition(self, label: str) -> np.ndarray:
        """
        Get condition values for a specific label.

        Args:
            label: Condition label name (e.g., 'ta', 'trial_type', 'seqq')

        Returns:
            Array of condition values across trials
        """
        if label not in self.condition_labels:
            raise ValueError(f"Label '{label}' not found. Available: {self.condition_labels}")

        idx = self.condition_labels.index(label)
        return self.cond_matrix[:, idx]

    def filter_trials(self, **conditions) -> np.ndarray:
        """
        Filter trials based on condition criteria.

        Args:
            **conditions: Keyword arguments specifying condition filters
                         e.g., trial_type=3, seqq='<3', succ=1

        Returns:
            Boolean mask of trials matching all conditions
        """
        mask = np.ones(self.n_trials, dtype=bool)

        for label, criterion in conditions.items():
            cond_values = self.get_condition(label)

            if isinstance(criterion, str):
                # Handle string comparisons like '<3', '>=2'
                if criterion.startswith('<='):
                    mask &= (cond_values <= float(criterion[2:]))
                elif criterion.startswith('>='):
                    mask &= (cond_values >= float(criterion[2:]))
                elif criterion.startswith('<'):
                    mask &= (cond_values < float(criterion[1:]))
                elif criterion.startswith('>'):
                    mask &= (cond_values > float(criterion[1:]))
                elif criterion.startswith('!='):
                    mask &= (cond_values != float(criterion[2:]))
                else:
                    mask &= (cond_values == float(criterion))
            else:
                # Direct equality
                mask &= (cond_values == criterion)

        return mask

    def get_mental_navigation_trials(self) -> np.ndarray:
        """
        Extract trials for pure mental navigation analysis.

        Filters for:
        - Fully occluded trials (trial_type==3)
        - Normal speed sequences (seqq<3)
        - First attempts only (attempt==1)

        Returns:
            Boolean mask of mental navigation trials
        """
        return self.filter_trials(trial_type=3, seqq='<3', attempt=1)

    def get_neural_activity(self, trial_mask: Optional[np.ndarray] = None,
                           time_window: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract neural activity for specified trials and time window.

        Args:
            trial_mask: Boolean mask for trial selection (default: all trials)
            time_window: (start_ms, end_ms) relative to start landmark onset

        Returns:
            Neural tensor subset (neurons × time × selected_trials)
        """
        data = self.neur_tensor.copy()

        if time_window is not None:
            start_ms, end_ms = time_window
            time_idx = (self.time_vector >= start_ms) & (self.time_vector <= end_ms)
            data = data[:, time_idx, :]

        if trial_mask is not None:
            data = data[:, :, trial_mask]

        return data

    def summary(self) -> str:
        """Generate summary statistics of the dataset."""
        summary_lines = [
            f"Neural Data Summary - {self.filename}",
            "=" * 60,
            f"Neurons: {self.n_neurons}",
            f"Trials: {self.n_trials}",
            f"Time points: {self.n_timepoints} (1ms bins from {self.time_vector[0]}ms to {self.time_vector[-1]}ms)",
            f"LFP channels: {self.lfp_tensor.shape[0] if self.lfp_tensor is not None else 0}",
            "",
            "Trial Type Distribution:",
            f"  Visible (type=1): {np.sum(self.get_condition('trial_type') == 1)} trials",
            f"  Sequence occluded (type=2): {np.sum(self.get_condition('trial_type') == 2)} trials",
            f"  Fully occluded (type=3): {np.sum(self.get_condition('trial_type') == 3)} trials",
            "",
            f"Mental navigation trials: {np.sum(self.get_mental_navigation_trials())} trials",
            f"Success rate: {np.mean(self.get_condition('succ')) * 100:.1f}%",
            "",
            f"Condition labels: {', '.join(self.condition_labels)}"
        ]
        return "\n".join(summary_lines)


class HippocampusDataLoader:
    """
    Loader for hippocampus neural recording data from .mat files.

    Expected .mat file structure:
    - neur_tensor_trialon: (neurons × time × trials) neural firing rates
    - lfp_tensor_trialon: (channels × time × trials) LFP data at 1kHz
    - cond_matrix: (trials × conditions) condition matrix

    Condition labels (12 columns):
    1. ta - True temporal distance (seconds)
    2. tp - Produced temporal distance by animal (seconds)
    3. curr - Start landmark (1-6)
    4. target - Target landmark (1-6)
    5. trial_type - 1=visible, 2=sequence occluded, 3=fully occluded
    6. seqq - Sequence identity (1,2=normal, 3=1.5× slower)
    7. succ - Success (1/0)
    8. validtrials_mm - Valid trials via mixture model
    9-12. Additional experimental parameters
    """

    # Default condition labels based on experimental design
    DEFAULT_CONDITION_LABELS = [
        'ta',              # True temporal distance
        'tp',              # Produced temporal distance
        'curr',            # Start landmark
        'target',          # Target landmark
        'trial_type',      # Visual feedback condition
        'seqq',            # Sequence identity/speed
        'succ',            # Success flag
        'validtrials_mm',  # Valid trials (mixture model)
        'attempt',         # Trial attempt number
        'reaction_time',   # Reaction time (ms)
        'nav_duration',    # Navigation duration (ms)
        'trial_id'         # Trial identifier
    ]

    @staticmethod
    def load_mat_file(filepath: Union[str, Path],
                     condition_labels: Optional[List[str]] = None) -> NeuralData:
        """
        Load neural data from a .mat file.

        Args:
            filepath: Path to .mat file
            condition_labels: Optional list of condition label names
                            (defaults to DEFAULT_CONDITION_LABELS)

        Returns:
            NeuralData object containing all experimental data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required fields are missing from .mat file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"Loading data from: {filepath.name}")

        try:
            # Load .mat file
            mat_data = sio.loadmat(str(filepath))

            # Extract main data arrays
            required_fields = ['neur_tensor_trialon', 'cond_matrix']
            missing_fields = [f for f in required_fields if f not in mat_data]

            if missing_fields:
                available_fields = [k for k in mat_data.keys() if not k.startswith('__')]
                raise ValueError(
                    f"Missing required fields: {missing_fields}\n"
                    f"Available fields: {available_fields}"
                )

            neur_tensor = mat_data['neur_tensor_trialon']
            cond_matrix = mat_data['cond_matrix']

            # LFP data is optional
            if 'lfp_tensor_trialon' in mat_data:
                lfp_tensor = mat_data['lfp_tensor_trialon']
            else:
                warnings.warn("LFP data not found in .mat file")
                # Create placeholder with correct dimensions
                lfp_tensor = np.zeros((0, neur_tensor.shape[1], neur_tensor.shape[2]))

            # Use provided labels or defaults
            if condition_labels is None:
                condition_labels = HippocampusDataLoader.DEFAULT_CONDITION_LABELS[:cond_matrix.shape[1]]

            # Ensure we have the right number of labels
            if len(condition_labels) != cond_matrix.shape[1]:
                warnings.warn(
                    f"Number of condition labels ({len(condition_labels)}) doesn't match "
                    f"condition matrix columns ({cond_matrix.shape[1]}). Using defaults."
                )
                condition_labels = HippocampusDataLoader.DEFAULT_CONDITION_LABELS[:cond_matrix.shape[1]]

            print(f"✓ Loaded {neur_tensor.shape[0]} neurons, "
                  f"{neur_tensor.shape[2]} trials, "
                  f"{neur_tensor.shape[1]} timepoints")

            # Create NeuralData object
            neural_data = NeuralData(
                neur_tensor=neur_tensor,
                lfp_tensor=lfp_tensor,
                cond_matrix=cond_matrix,
                condition_labels=condition_labels,
                filename=filepath.name
            )

            return neural_data

        except Exception as e:
            raise RuntimeError(f"Error loading {filepath}: {str(e)}")

    @staticmethod
    def load_multiple_files(directory: Union[str, Path],
                           pattern: str = "*.mat",
                           condition_labels: Optional[List[str]] = None) -> List[NeuralData]:
        """
        Load multiple .mat files from a directory.

        Args:
            directory: Directory containing .mat files
            pattern: Glob pattern for file matching (default: "*.mat")
            condition_labels: Optional list of condition label names

        Returns:
            List of NeuralData objects, one per file
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        mat_files = sorted(directory.glob(pattern))

        if not mat_files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")

        print(f"Found {len(mat_files)} .mat files")

        datasets = []
        for mat_file in mat_files:
            try:
                data = HippocampusDataLoader.load_mat_file(mat_file, condition_labels)
                datasets.append(data)
            except Exception as e:
                warnings.warn(f"Failed to load {mat_file.name}: {str(e)}")
                continue

        print(f"\nSuccessfully loaded {len(datasets)}/{len(mat_files)} files")
        return datasets


def main():
    """Example usage of the data loader."""
    import sys

    # Example: Load a single file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        data = HippocampusDataLoader.load_mat_file(filepath)
        print("\n" + data.summary())

        # Example: Extract mental navigation trials
        mn_mask = data.get_mental_navigation_trials()
        print(f"\n Mental navigation trials: {np.sum(mn_mask)}/{data.n_trials}")

        # Example: Get neural activity during navigation period (0-3000ms)
        nav_activity = data.get_neural_activity(
            trial_mask=mn_mask,
            time_window=(0, 3000)
        )
        print(f"Navigation activity shape: {nav_activity.shape}")

    else:
        print("Usage: python hippocampus_data_loader.py <path_to_mat_file>")
        print("\nOr use in Python:")
        print("  from hippocampus_data_loader import HippocampusDataLoader")
        print("  data = HippocampusDataLoader.load_mat_file('data.mat')")
        print("  print(data.summary())")


if __name__ == "__main__":
    main()
