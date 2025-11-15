"""
AI-Powered Neural Analysis for Hippocampus Data
================================================
Advanced analysis tools using machine learning and AI techniques for analyzing
neural activity during mental navigation tasks.

Includes:
- Temporal coding analysis
- Population decoding (Bayesian, SVM, Neural Networks)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Trajectory analysis in neural state space
- Sequence detection and replay analysis
- Time cell identification

Author: Computational Neuroscience Analysis Pipeline
Date: 2025-11-15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.neural_network import MLPClassifier
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False

from hippocampus_data_loader import NeuralData


class TemporalCodingAnalyzer:
    """
    Analyze temporal coding properties of neurons during mental navigation.

    Methods for identifying time cells, sequence cells, and temporal tuning.
    """

    @staticmethod
    def compute_temporal_tuning(neural_data: NeuralData,
                               trial_mask: np.ndarray,
                               time_window: Tuple[int, int] = (0, 3000),
                               smooth_sigma: float = 50.0) -> Dict:
        """
        Compute temporal tuning curves for each neuron.

        Args:
            neural_data: NeuralData object
            trial_mask: Boolean mask for trial selection
            time_window: Time window for analysis (ms)
            smooth_sigma: Gaussian smoothing kernel width (ms)

        Returns:
            Dictionary containing tuning curves and statistics
        """
        # Get neural activity
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        n_neurons, n_time, n_trials = activity.shape

        # Compute mean firing rate across trials
        mean_fr = np.mean(activity, axis=2)  # (neurons × time)

        # Smooth tuning curves
        if smooth_sigma > 0:
            mean_fr_smooth = gaussian_filter1d(mean_fr, sigma=smooth_sigma, axis=1)
        else:
            mean_fr_smooth = mean_fr

        # Compute temporal modulation index
        # TMI = (max_fr - min_fr) / (max_fr + min_fr)
        max_fr = np.max(mean_fr_smooth, axis=1)
        min_fr = np.min(mean_fr_smooth, axis=1)
        tmi = (max_fr - min_fr) / (max_fr + min_fr + 1e-10)

        # Peak time (time of maximum firing)
        time_vec = neural_data.time_vector[
            (neural_data.time_vector >= time_window[0]) &
            (neural_data.time_vector <= time_window[1])
        ]
        peak_times = time_vec[np.argmax(mean_fr_smooth, axis=1)]

        # Temporal information content (bits)
        temporal_info = TemporalCodingAnalyzer._compute_temporal_information(activity)

        return {
            'tuning_curves': mean_fr_smooth,
            'tuning_curves_raw': mean_fr,
            'time_vector': time_vec,
            'temporal_modulation_index': tmi,
            'peak_times': peak_times,
            'temporal_information': temporal_info,
            'max_firing_rate': max_fr,
            'min_firing_rate': min_fr
        }

    @staticmethod
    def _compute_temporal_information(activity: np.ndarray,
                                     n_bins: int = 20) -> np.ndarray:
        """
        Compute temporal information content for each neuron.

        Information = Σ p(t) * (λ(t)/λ_mean) * log2(λ(t)/λ_mean)

        Args:
            activity: Neural activity (neurons × time × trials)
            n_bins: Number of temporal bins

        Returns:
            Temporal information in bits for each neuron
        """
        n_neurons, n_time, n_trials = activity.shape

        # Bin the time dimension
        time_bins = np.array_split(np.arange(n_time), n_bins)
        temporal_info = np.zeros(n_neurons)

        for neuron in range(n_neurons):
            fr_binned = np.array([
                np.mean(activity[neuron, bin_idx, :])
                for bin_idx in time_bins
            ])

            mean_fr = np.mean(fr_binned)
            if mean_fr < 0.1:  # Skip silent neurons
                continue

            # Compute information
            p_t = 1.0 / n_bins  # Uniform time distribution
            info = 0
            for fr in fr_binned:
                if fr > 0:
                    info += p_t * (fr / mean_fr) * np.log2(fr / mean_fr)

            temporal_info[neuron] = max(0, info)

        return temporal_info

    @staticmethod
    def identify_time_cells(tuning_results: Dict,
                           tmi_threshold: float = 0.3,
                           info_threshold: float = 0.1) -> np.ndarray:
        """
        Identify time cells based on temporal tuning properties.

        Args:
            tuning_results: Output from compute_temporal_tuning
            tmi_threshold: Minimum temporal modulation index
            info_threshold: Minimum temporal information (bits)

        Returns:
            Boolean array indicating time cells
        """
        tmi = tuning_results['temporal_modulation_index']
        temp_info = tuning_results['temporal_information']

        is_time_cell = (tmi >= tmi_threshold) & (temp_info >= info_threshold)
        return is_time_cell


class PopulationDecoder:
    """
    Decode behavioral/cognitive variables from population neural activity
    using machine learning approaches.
    """

    def __init__(self, method: str = 'bayesian'):
        """
        Initialize decoder.

        Args:
            method: Decoding method - 'bayesian', 'svm', 'random_forest', 'logistic', 'mlp'
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()

    def decode_landmark_pairs(self,
                             neural_data: NeuralData,
                             trial_mask: np.ndarray,
                             time_window: Tuple[int, int] = (500, 2000),
                             cv_folds: int = 5) -> Dict:
        """
        Decode which landmark pair is being mentally navigated.

        Args:
            neural_data: NeuralData object
            trial_mask: Trials to include
            time_window: Time window for decoding (ms)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with decoding accuracy and confusion matrix
        """
        # Get neural activity
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        n_neurons, n_time, n_trials = activity.shape

        # Average over time window to get population vector per trial
        X = np.mean(activity, axis=1).T  # (trials × neurons)

        # Create labels: landmark pairs (curr, target)
        curr = neural_data.get_condition('curr')[trial_mask].astype(int)
        target = neural_data.get_condition('target')[trial_mask].astype(int)
        y = curr * 10 + target  # Encode as single integer

        # Check if we have enough samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        min_samples = np.min(counts)

        if min_samples < cv_folds:
            warnings.warn(f"Some classes have fewer samples ({min_samples}) than CV folds ({cv_folds}). "
                         f"Reducing folds to {min_samples}")
            cv_folds = max(2, min_samples)

        # Train decoder
        if self.method == 'bayesian':
            accuracy, cm = self._decode_naive_bayes(X, y, cv_folds)
        elif self.method == 'svm':
            accuracy, cm = self._decode_svm(X, y, cv_folds)
        elif self.method == 'random_forest':
            accuracy, cm = self._decode_random_forest(X, y, cv_folds)
        elif self.method == 'logistic':
            accuracy, cm = self._decode_logistic(X, y, cv_folds)
        elif self.method == 'mlp' and NN_AVAILABLE:
            accuracy, cm = self._decode_mlp(X, y, cv_folds)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'landmark_pairs': unique_classes,
            'n_pairs': len(unique_classes),
            'chance_level': 1.0 / len(unique_classes)
        }

    def decode_temporal_distance(self,
                                neural_data: NeuralData,
                                trial_mask: np.ndarray,
                                time_window: Tuple[int, int] = (500, 2000)) -> Dict:
        """
        Decode temporal distance from neural activity (regression).

        Args:
            neural_data: NeuralData object
            trial_mask: Trials to include
            time_window: Time window for decoding (ms)

        Returns:
            Dictionary with correlation between actual and predicted distances
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict

        # Get neural activity
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        X = np.mean(activity, axis=1).T  # (trials × neurons)

        # Get true temporal distances
        y = neural_data.get_condition('ta')[trial_mask]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Ridge regression with cross-validation
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X_scaled, y, cv=5)

        # Compute metrics
        correlation = np.corrcoef(y, y_pred)[0, 1]
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        return {
            'correlation': correlation,
            'rmse': rmse,
            'true_distances': y,
            'predicted_distances': y_pred
        }

    def _decode_naive_bayes(self, X, y, cv_folds):
        """Naive Bayes decoder."""
        from sklearn.naive_bayes import GaussianNB

        X_scaled = self.scaler.fit_transform(X)
        model = GaussianNB()

        # Cross-validation
        scores = cross_val_score(model, X_scaled, y, cv=cv_folds)

        # Fit full model for confusion matrix
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)

        return np.mean(scores), cm

    def _decode_svm(self, X, y, cv_folds):
        """SVM decoder."""
        X_scaled = self.scaler.fit_transform(X)
        model = SVC(kernel='rbf', C=1.0, gamma='scale')

        scores = cross_val_score(model, X_scaled, y, cv=cv_folds)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)

        return np.mean(scores), cm

    def _decode_random_forest(self, X, y, cv_folds):
        """Random Forest decoder."""
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        scores = cross_val_score(model, X, y, cv=cv_folds)
        model.fit(X, y)
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        return np.mean(scores), cm

    def _decode_logistic(self, X, y, cv_folds):
        """Logistic regression decoder."""
        X_scaled = self.scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)

        scores = cross_val_score(model, X_scaled, y, cv=cv_folds)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)

        return np.mean(scores), cm

    def _decode_mlp(self, X, y, cv_folds):
        """Multi-layer perceptron decoder."""
        X_scaled = self.scaler.fit_transform(X)
        model = MLPClassifier(hidden_layer_sizes=(100, 50),
                             max_iter=500,
                             random_state=42)

        scores = cross_val_score(model, X_scaled, y, cv=cv_folds)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)

        return np.mean(scores), cm


class NeuralManifoldAnalyzer:
    """
    Analyze neural population dynamics in low-dimensional manifolds.

    Uses dimensionality reduction to visualize and analyze neural trajectories
    during mental navigation.
    """

    def __init__(self, method: str = 'pca', n_components: int = 3):
        """
        Initialize manifold analyzer.

        Args:
            method: 'pca', 'ica', 'tsne', 'umap', 'nmf'
            n_components: Number of components/dimensions
        """
        self.method = method
        self.n_components = n_components
        self.model = None
        self.scaler = StandardScaler()

    def fit_transform(self, neural_data: NeuralData,
                     trial_mask: np.ndarray,
                     time_window: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Fit dimensionality reduction and transform neural activity.

        Args:
            neural_data: NeuralData object
            trial_mask: Trials to include
            time_window: Optional time window (ms)

        Returns:
            Dictionary with low-dimensional representation and model
        """
        # Get neural activity
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        n_neurons, n_time, n_trials = activity.shape

        # Reshape: (time × trials, neurons)
        X = activity.transpose(1, 2, 0).reshape(-1, n_neurons)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Apply dimensionality reduction
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components)
            X_reduced = self.model.fit_transform(X_scaled)
            var_explained = self.model.explained_variance_ratio_

        elif self.method == 'ica':
            self.model = FastICA(n_components=self.n_components, random_state=42)
            X_reduced = self.model.fit_transform(X_scaled)
            var_explained = None

        elif self.method == 'tsne':
            perplexity = min(30, (X_scaled.shape[0] - 1) // 3)
            self.model = TSNE(n_components=self.n_components,
                            perplexity=perplexity,
                            random_state=42)
            X_reduced = self.model.fit_transform(X_scaled)
            var_explained = None

        elif self.method == 'umap' and UMAP_AVAILABLE:
            self.model = umap.UMAP(n_components=self.n_components,
                                  random_state=42)
            X_reduced = self.model.fit_transform(X_scaled)
            var_explained = None

        elif self.method == 'nmf':
            # NMF requires non-negative data
            X_nonneg = X - X.min() + 1e-10
            self.model = NMF(n_components=self.n_components, random_state=42)
            X_reduced = self.model.fit_transform(X_nonneg)
            var_explained = None

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Reshape back to (time, trials, components)
        trajectories = X_reduced.reshape(n_time, n_trials, self.n_components)

        return {
            'trajectories': trajectories,
            'X_reduced': X_reduced,
            'variance_explained': var_explained,
            'model': self.model,
            'method': self.method
        }

    def compute_trajectory_similarity(self, trajectories: np.ndarray,
                                     condition_labels: np.ndarray) -> Dict:
        """
        Compute similarity between neural trajectories for different conditions.

        Args:
            trajectories: Neural trajectories (time × trials × components)
            condition_labels: Condition label for each trial

        Returns:
            Dictionary with similarity metrics
        """
        n_time, n_trials, n_comp = trajectories.shape
        unique_conditions = np.unique(condition_labels)

        # Compute mean trajectory for each condition
        mean_trajectories = {}
        for cond in unique_conditions:
            cond_mask = condition_labels == cond
            mean_trajectories[cond] = np.mean(trajectories[:, cond_mask, :], axis=1)

        # Compute pairwise trajectory distances
        n_cond = len(unique_conditions)
        distance_matrix = np.zeros((n_cond, n_cond))

        for i, cond1 in enumerate(unique_conditions):
            for j, cond2 in enumerate(unique_conditions):
                traj1 = mean_trajectories[cond1]
                traj2 = mean_trajectories[cond2]

                # Euclidean distance averaged over time
                dist = np.mean(np.sqrt(np.sum((traj1 - traj2) ** 2, axis=1)))
                distance_matrix[i, j] = dist

        return {
            'mean_trajectories': mean_trajectories,
            'distance_matrix': distance_matrix,
            'conditions': unique_conditions
        }


class SequenceAnalyzer:
    """
    Detect and analyze sequential activity patterns and replay events.
    """

    @staticmethod
    def detect_sequences(neural_data: NeuralData,
                        trial_mask: np.ndarray,
                        time_window: Tuple[int, int] = (0, 3000),
                        min_neurons: int = 5) -> Dict:
        """
        Detect sequential activation patterns across neurons.

        Args:
            neural_data: NeuralData object
            trial_mask: Trials to analyze
            time_window: Time window for analysis (ms)
            min_neurons: Minimum neurons participating in sequence

        Returns:
            Dictionary with sequence detection results
        """
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        n_neurons, n_time, n_trials = activity.shape

        # Average across trials
        mean_activity = np.mean(activity, axis=2)

        # Find peak time for each neuron
        peak_times = np.argmax(mean_activity, axis=1)

        # Sort neurons by peak time
        sorted_idx = np.argsort(peak_times)
        sorted_activity = mean_activity[sorted_idx, :]

        # Compute sequence score (how sequential is the activation?)
        sequence_score = SequenceAnalyzer._compute_sequence_score(sorted_activity)

        return {
            'sorted_activity': sorted_activity,
            'neuron_order': sorted_idx,
            'peak_times': peak_times[sorted_idx],
            'sequence_score': sequence_score
        }

    @staticmethod
    def _compute_sequence_score(activity_matrix: np.ndarray) -> float:
        """
        Compute how sequential the activity is (0=random, 1=perfect sequence).

        Uses correlation between neuron index and peak time.
        """
        n_neurons = activity_matrix.shape[0]
        peak_times = np.argmax(activity_matrix, axis=1)

        # Correlation between neuron index and peak time
        neuron_idx = np.arange(n_neurons)
        correlation, _ = stats.spearmanr(neuron_idx, peak_times)

        return max(0, correlation)  # Return 0 if negative correlation


def print_analysis_summary(results: Dict, title: str = "Analysis Results"):
    """Pretty print analysis results."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            if value.size < 10:
                print(f"{key}: {value}")
            else:
                print(f"{key}: array shape {value.shape}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print("AI-Powered Neural Analysis Module")
    print("==================================")
    print("\nAvailable analysis tools:")
    print("  - TemporalCodingAnalyzer: Time cell identification, temporal tuning")
    print("  - PopulationDecoder: ML-based decoding (Bayesian, SVM, Random Forest, etc.)")
    print("  - NeuralManifoldAnalyzer: Dimensionality reduction (PCA, t-SNE, UMAP)")
    print("  - SequenceAnalyzer: Sequential activity and replay detection")
    print("\nImport and use with your data:")
    print("  from neural_analysis_ai import TemporalCodingAnalyzer, PopulationDecoder")
