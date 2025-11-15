"""
Neural Analysis AI Module
Temporal coding, population decoding, manifold analysis, and sequence detection
"""
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, r2_score
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class TemporalCodingAnalyzer:
    """Analyze temporal coding in neural populations."""
    
    def __init__(self):
        pass
    
    def compute_temporal_tuning(self, 
                                neural_data,
                                trial_mask: np.ndarray,
                                time_window: Tuple[int, int] = (0, 3000),
                                smooth_sigma: float = 50.0) -> Dict[str, Any]:
        """
        Compute temporal tuning curves for all neurons.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials to include
            time_window: Time window in ms
            smooth_sigma: Gaussian smoothing sigma in ms
            
        Returns:
            Dictionary with tuning curves and statistics
        """
        print(f"\nComputing temporal tuning curves...")
        print(f"  Time window: {time_window[0]} to {time_window[1]} ms")
        print(f"  Smoothing sigma: {smooth_sigma} ms")
        
        # Extract neural activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        # Get time vector for this window
        t_idx = (neural_data.time_vector >= time_window[0]) & \
                (neural_data.time_vector < time_window[1])
        time_vec = neural_data.time_vector[t_idx]
        
        n_neurons, n_time, n_trials = activity.shape
        
        # Compute mean activity across trials
        mean_activity = np.mean(activity, axis=2)
        
        # Smooth
        smooth_activity = np.zeros_like(mean_activity)
        for i in range(n_neurons):
            smooth_activity[i, :] = gaussian_filter1d(mean_activity[i, :], smooth_sigma)
        
        # Compute temporal modulation index (TMI)
        tmi = np.std(smooth_activity, axis=1) / (np.mean(smooth_activity, axis=1) + 1e-10)
        
        # Compute temporal information (bits)
        temporal_info = np.zeros(n_neurons)
        for i in range(n_neurons):
            # Bin firing rates
            fr = smooth_activity[i, :]
            mean_fr = np.mean(fr)
            if mean_fr > 0:
                # Compute mutual information
                p_t = np.ones(len(fr)) / len(fr)  # Uniform time distribution
                info = np.sum(p_t * (fr / mean_fr) * np.log2((fr + 1e-10) / (mean_fr + 1e-10)))
                temporal_info[i] = max(0, info)
        
        # Find peak time for each neuron
        peak_times = np.argmax(smooth_activity, axis=1)
        peak_times_ms = time_vec[peak_times]
        
        print(f"✓ Computed tuning for {n_neurons} neurons across {n_trials} trials")
        print(f"  Mean TMI: {np.mean(tmi):.3f}")
        print(f"  Mean temporal info: {np.mean(temporal_info):.3f} bits")
        
        return {
            'tuning_curves': smooth_activity,
            'mean_activity': mean_activity,
            'time_vector': time_vec,
            'tmi': tmi,
            'temporal_info': temporal_info,
            'peak_times': peak_times,
            'peak_times_ms': peak_times_ms,
            'n_neurons': n_neurons,
            'n_trials': n_trials
        }
    
    def identify_time_cells(self,
                           tuning_results: Dict[str, Any],
                           tmi_threshold: float = 0.3,
                           info_threshold: float = 0.1) -> np.ndarray:
        """
        Identify time cells based on temporal modulation.
        
        Args:
            tuning_results: Output from compute_temporal_tuning
            tmi_threshold: Minimum TMI to be considered a time cell
            info_threshold: Minimum temporal information (bits)
            
        Returns:
            Boolean array indicating time cells
        """
        tmi = tuning_results['tmi']
        temporal_info = tuning_results['temporal_info']
        
        time_cells = (tmi > tmi_threshold) & (temporal_info > info_threshold)
        
        n_time_cells = np.sum(time_cells)
        pct = 100 * n_time_cells / len(time_cells)
        
        print(f"\n✓ Identified {n_time_cells} time cells ({pct:.1f}%)")
        print(f"  TMI threshold: {tmi_threshold}")
        print(f"  Info threshold: {info_threshold} bits")
        
        return time_cells


class PopulationDecoder:
    """Decode behavioral variables from population activity."""
    
    def __init__(self, method: str = 'bayesian'):
        """
        Initialize decoder.
        
        Args:
            method: 'bayesian', 'svm', 'random_forest', 'logistic', 'mlp'
        """
        self.method = method
        self.model = self._create_model()
        
    def _create_model(self):
        """Create sklearn model based on method."""
        if self.method == 'bayesian':
            return GaussianNB()
        elif self.method == 'svm':
            return SVC(kernel='rbf', C=1.0, gamma='scale')
        elif self.method == 'random_forest':
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.method == 'logistic':
            return LogisticRegression(max_iter=1000, C=1.0)
        elif self.method == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def decode_landmark_pairs(self,
                              neural_data,
                              trial_mask: np.ndarray,
                              time_window: Tuple[int, int] = (500, 2000),
                              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Decode landmark pairs from neural activity.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials
            time_window: Time window for decoding
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with decoding results
        """
        print(f"\nDecoding landmark pairs using {self.method}...")
        
        # Extract activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        # Average over time to get trial-level features
        X = np.mean(activity, axis=1).T  # (trials, neurons)
        
        # Get labels
        curr = neural_data.get_condition('curr')[trial_mask]
        target = neural_data.get_condition('target')[trial_mask]
        labels = curr * 10 + target  # Encode as single number
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check for sufficient samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = np.min(counts)
        
        if min_samples < cv_folds:
            cv_folds = max(2, min_samples)
            print(f"  Adjusting CV folds to {cv_folds} (min class has {min_samples} samples)")
        
        # Cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_scaled, labels, cv=kfold, scoring='accuracy')
        
        # Fit final model
        self.model.fit(X_scaled, labels)
        predictions = self.model.predict(X_scaled)
        
        # Confusion matrix
        conf_mat = confusion_matrix(labels, predictions)
        
        # Compute chance level
        chance_level = 1.0 / len(unique_labels)
        
        print(f"✓ Decoding accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"  Chance level: {chance_level:.3f}")
        print(f"  Number of landmark pairs: {len(unique_labels)}")
        
        return {
            'accuracy': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,
            'confusion_matrix': conf_mat,
            'labels': labels,
            'predictions': predictions,
            'unique_labels': unique_labels,
            'chance_level': chance_level,
            'method': self.method
        }
    
    def decode_temporal_distance(self,
                                 neural_data,
                                 trial_mask: np.ndarray,
                                 time_window: Tuple[int, int] = (500, 2000),
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """
        Decode temporal distance (regression).
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials
            time_window: Time window for decoding
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with regression results
        """
        print(f"\nDecoding temporal distance (regression)...")
        
        # Extract activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        # Average over time
        X = np.mean(activity, axis=1).T  # (trials, neurons)
        
        # Get temporal distance labels
        y = neural_data.get_condition('ta')[trial_mask]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Ridge regression
        model = Ridge(alpha=1.0)
        
        # Cross-validation (R^2 score)
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
        
        # Fit final model
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        
        # Compute correlation
        correlation = np.corrcoef(y, predictions)[0, 1]
        
        print(f"✓ R² score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"  Correlation: {correlation:.3f}")
        
        return {
            'r2': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,
            'correlation': correlation,
            'true_values': y,
            'predictions': predictions
        }


class NeuralManifoldAnalyzer:
    """Analyze neural population dynamics in low-dimensional space."""
    
    def __init__(self, method: str = 'pca', n_components: int = 10):
        """
        Initialize manifold analyzer.
        
        Args:
            method: 'pca', 'ica', 'tsne', 'umap', 'nmf'
            n_components: Number of components
        """
        self.method = method
        self.n_components = n_components
        self.model = self._create_model()
        
    def _create_model(self):
        """Create dimensionality reduction model."""
        if self.method == 'pca':
            return PCA(n_components=self.n_components)
        elif self.method == 'ica':
            return FastICA(n_components=self.n_components, random_state=42, max_iter=500)
        elif self.method == 'tsne':
            return TSNE(n_components=min(3, self.n_components), random_state=42)
        elif self.method == 'umap':
            try:
                from umap import UMAP
                return UMAP(n_components=min(3, self.n_components), random_state=42)
            except ImportError:
                print("Warning: UMAP not installed, falling back to PCA")
                return PCA(n_components=self.n_components)
        elif self.method == 'nmf':
            return NMF(n_components=self.n_components, random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_transform(self,
                     neural_data,
                     trial_mask: np.ndarray,
                     time_window: Tuple[int, int] = (0, 3000)) -> Dict[str, Any]:
        """
        Project neural activity into low-dimensional manifold.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials
            time_window: Time window
            
        Returns:
            Dictionary with manifold results
        """
        print(f"\nComputing neural manifold using {self.method}...")
        
        # Extract activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        n_neurons, n_time, n_trials = activity.shape
        
        # Reshape to (samples, features) where samples = time points × trials
        X = activity.transpose(2, 1, 0).reshape(-1, n_neurons)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit and transform
        X_reduced = self.model.fit_transform(X_scaled)
        
        # Reshape back to (trials, time, components)
        trajectories = X_reduced.reshape(n_trials, n_time, -1)
        
        # Get variance explained if applicable
        variance_explained = None
        if hasattr(self.model, 'explained_variance_ratio_'):
            variance_explained = self.model.explained_variance_ratio_
            print(f"✓ Variance explained by first 3 PCs: "
                  f"{100*np.sum(variance_explained[:3]):.1f}%")
        
        print(f"✓ Projected to {X_reduced.shape[1]} dimensions")
        
        return {
            'trajectories': trajectories,
            'reduced_data': X_reduced,
            'variance_explained': variance_explained,
            'method': self.method,
            'n_components': X_reduced.shape[1]
        }
    
    def compute_trajectory_similarity(self,
                                     trajectories: np.ndarray,
                                     labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute similarity between neural trajectories.
        
        Args:
            trajectories: (trials, time, components) array
            labels: Condition labels for each trial
            
        Returns:
            Dictionary with similarity metrics
        """
        print("\nComputing trajectory similarity...")
        
        n_trials, n_time, n_components = trajectories.shape
        
        # Compute pairwise distances
        distances = np.zeros((n_trials, n_trials))
        
        for i in range(n_trials):
            for j in range(i+1, n_trials):
                # Euclidean distance between trajectories
                dist = np.sqrt(np.sum((trajectories[i] - trajectories[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Compute within vs between condition distances
        unique_labels = np.unique(labels)
        within_dists = []
        between_dists = []
        
        for i in range(n_trials):
            for j in range(i+1, n_trials):
                if labels[i] == labels[j]:
                    within_dists.append(distances[i, j])
                else:
                    between_dists.append(distances[i, j])
        
        within_mean = np.mean(within_dists) if within_dists else 0
        between_mean = np.mean(between_dists) if between_dists else 0
        
        print(f"✓ Mean within-condition distance: {within_mean:.2f}")
        print(f"  Mean between-condition distance: {between_mean:.2f}")
        
        return {
            'distance_matrix': distances,
            'within_distance': within_mean,
            'between_distance': between_mean,
            'separability': between_mean / (within_mean + 1e-10)
        }


class SequenceAnalyzer:
    """Detect sequential neural activity patterns."""
    
    def __init__(self):
        pass
    
    def detect_sequences(self,
                        neural_data,
                        trial_mask: np.ndarray,
                        time_window: Tuple[int, int] = (0, 3000),
                        min_neurons: int = 5) -> Dict[str, Any]:
        """
        Detect sequential activation patterns.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials
            time_window: Time window
            min_neurons: Minimum neurons to consider a sequence
            
        Returns:
            Dictionary with sequence detection results
        """
        print(f"\nDetecting sequential activation patterns...")
        
        # Extract activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        # Average across trials
        mean_activity = np.mean(activity, axis=2)
        
        # Find peak time for each neuron
        peak_times = np.argmax(mean_activity, axis=1)
        
        # Sort neurons by peak time
        sorted_idx = np.argsort(peak_times)
        
        # Compute sequence score (correlation between neuron index and peak time)
        sequence_score = np.corrcoef(np.arange(len(peak_times)), peak_times)[0, 1]
        
        # Compute time lags between consecutive neurons
        sorted_peaks = peak_times[sorted_idx]
        time_lags = np.diff(sorted_peaks)
        
        print(f"✓ Sequence score: {sequence_score:.3f}")
        print(f"  Mean time lag: {np.mean(time_lags):.1f} ms")
        
        return {
            'sequence_score': sequence_score,
            'sorted_neurons': sorted_idx,
            'peak_times': peak_times,
            'time_lags': time_lags,
            'sorted_activity': mean_activity[sorted_idx, :]
        }


if __name__ == "__main__":
    print("Neural Analysis AI Module")
    print("Import this module to use the analysis classes:")
    print("  - TemporalCodingAnalyzer")
    print("  - PopulationDecoder")
    print("  - NeuralManifoldAnalyzer")
    print("  - SequenceAnalyzer")
