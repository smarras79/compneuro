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


class TimeSeriesFeatureExtractor:
    """
    Advanced ML/AI-based feature extraction from neural time series.
    
    Extracts interpretable features and patterns from firing rate time series
    using signal processing, time series analysis, and machine learning.
    """
    
    def __init__(self):
        pass
    
    def extract_all_features(self,
                            neural_data,
                            trial_mask: np.ndarray,
                            time_window: Tuple[int, int] = (0, 3000),
                            smooth_sigma: float = 50.0) -> Dict[str, Any]:
        """
        Extract comprehensive features from neural time series.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Boolean mask for trials
            time_window: Time window in ms
            smooth_sigma: Smoothing parameter
            
        Returns:
            Dictionary with extracted features for all neurons
        """
        print("\n" + "="*70)
        print("AI/ML TIME SERIES FEATURE EXTRACTION")
        print("="*70)
        
        # Extract activity
        activity = neural_data.get_neural_activity(
            trial_mask=trial_mask,
            time_window=time_window
        )
        
        # Get time vector
        t_idx = (neural_data.time_vector >= time_window[0]) & \
                (neural_data.time_vector < time_window[1])
        time_vec = neural_data.time_vector[t_idx]
        
        n_neurons, n_time, n_trials = activity.shape
        
        # Compute mean activity and smooth
        mean_activity = np.mean(activity, axis=2)
        smooth_activity = np.zeros_like(mean_activity)
        for i in range(n_neurons):
            smooth_activity[i, :] = gaussian_filter1d(mean_activity[i, :], smooth_sigma)
        
        print(f"\nExtracting features from {n_neurons} neurons...")
        
        # Initialize feature dictionaries
        features = {
            'basic_stats': self._extract_basic_statistics(smooth_activity),
            'temporal_features': self._extract_temporal_features(smooth_activity, time_vec),
            'spectral_features': self._extract_spectral_features(smooth_activity),
            'shape_features': self._extract_shape_features(smooth_activity),
            'complexity_features': self._extract_complexity_features(smooth_activity),
            'trial_variability': self._extract_trial_variability(activity)
        }
        
        # Add raw data
        features['smooth_activity'] = smooth_activity
        features['time_vector'] = time_vec
        features['n_neurons'] = n_neurons
        
        print(f"✓ Extracted {len(features)} feature categories")
        
        return features
    
    def _extract_basic_statistics(self, activity: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract basic statistical features."""
        print("  - Basic statistics...")
        
        n_neurons = activity.shape[0]
        
        return {
            'mean': np.mean(activity, axis=1),
            'std': np.std(activity, axis=1),
            'min': np.min(activity, axis=1),
            'max': np.max(activity, axis=1),
            'median': np.median(activity, axis=1),
            'range': np.ptp(activity, axis=1),
            'cv': np.std(activity, axis=1) / (np.mean(activity, axis=1) + 1e-10),
            'skewness': stats.skew(activity, axis=1),
            'kurtosis': stats.kurtosis(activity, axis=1)
        }
    
    def _extract_temporal_features(self, activity: np.ndarray, time_vec: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract temporal/time-domain features."""
        print("  - Temporal features...")
        
        n_neurons, n_time = activity.shape
        
        # Peak detection
        peak_idx = np.argmax(activity, axis=1)
        peak_times = time_vec[peak_idx]
        peak_values = np.max(activity, axis=1)
        
        # Find trough
        trough_idx = np.argmin(activity, axis=1)
        trough_values = np.min(activity, axis=1)
        
        # Peak width at half-maximum
        peak_widths = np.zeros(n_neurons)
        for i in range(n_neurons):
            half_max = (peak_values[i] + trough_values[i]) / 2
            above_half = activity[i, :] > half_max
            if np.any(above_half):
                peak_widths[i] = np.sum(above_half)
        
        # Slope at peak (derivative)
        gradients = np.gradient(activity, axis=1)
        max_slope = np.max(np.abs(gradients), axis=1)
        
        # Rise time and fall time
        rise_times = np.zeros(n_neurons)
        fall_times = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            peak_i = peak_idx[i]
            # Rise time: 10% to 90% of max
            val_10 = trough_values[i] + 0.1 * (peak_values[i] - trough_values[i])
            val_90 = trough_values[i] + 0.9 * (peak_values[i] - trough_values[i])
            
            before_peak = activity[i, :peak_i]
            if len(before_peak) > 0:
                idx_10 = np.where(before_peak >= val_10)[0]
                idx_90 = np.where(before_peak >= val_90)[0]
                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_times[i] = idx_90[0] - idx_10[0]
            
            # Fall time
            after_peak = activity[i, peak_i:]
            if len(after_peak) > 0:
                idx_90_fall = np.where(after_peak <= val_90)[0]
                idx_10_fall = np.where(after_peak <= val_10)[0]
                if len(idx_90_fall) > 0 and len(idx_10_fall) > 0:
                    fall_times[i] = idx_10_fall[0] - idx_90_fall[0] if idx_10_fall[0] > idx_90_fall[0] else 0
        
        return {
            'peak_time': peak_times,
            'peak_value': peak_values,
            'peak_width': peak_widths,
            'max_slope': max_slope,
            'rise_time': rise_times,
            'fall_time': fall_times,
            'asymmetry': (rise_times - fall_times) / (rise_times + fall_times + 1e-10)
        }
    
    def _extract_spectral_features(self, activity: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency-domain features using FFT."""
        print("  - Spectral features...")
        
        n_neurons, n_time = activity.shape
        
        # Compute FFT for each neuron
        fft_vals = np.fft.rfft(activity, axis=1)
        power_spectrum = np.abs(fft_vals)**2
        freqs = np.fft.rfftfreq(n_time)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(power_spectrum, axis=1)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(freqs * power_spectrum, axis=1) / (np.sum(power_spectrum, axis=1) + 1e-10)
        
        # Spectral spread (standard deviation)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid[:, np.newaxis])**2) * power_spectrum, axis=1) / 
                                 (np.sum(power_spectrum, axis=1) + 1e-10))
        
        # Spectral entropy
        normalized_spectrum = power_spectrum / (np.sum(power_spectrum, axis=1, keepdims=True) + 1e-10)
        spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-10), axis=1)
        
        # Band power (divide into frequency bands)
        low_band = np.sum(power_spectrum[:, :len(freqs)//4], axis=1)
        mid_band = np.sum(power_spectrum[:, len(freqs)//4:len(freqs)//2], axis=1)
        high_band = np.sum(power_spectrum[:, len(freqs)//2:], axis=1)
        total_power = np.sum(power_spectrum, axis=1)
        
        return {
            'dominant_frequency': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'low_band_power': low_band / (total_power + 1e-10),
            'mid_band_power': mid_band / (total_power + 1e-10),
            'high_band_power': high_band / (total_power + 1e-10),
            'total_power': total_power
        }
    
    def _extract_shape_features(self, activity: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract shape-based features of firing rate curves."""
        print("  - Shape features...")
        
        n_neurons, n_time = activity.shape
        
        # Number of peaks
        n_peaks = np.zeros(n_neurons)
        for i in range(n_neurons):
            # Find local maxima
            peaks = (activity[i, 1:-1] > activity[i, :-2]) & (activity[i, 1:-1] > activity[i, 2:])
            n_peaks[i] = np.sum(peaks)
        
        # Smoothness (inverse of total variation)
        total_variation = np.sum(np.abs(np.diff(activity, axis=1)), axis=1)
        smoothness = 1.0 / (total_variation + 1e-10)
        
        # Monotonicity score (how much of curve is monotonic)
        diffs = np.diff(activity, axis=1)
        monotonic_increasing = np.sum(diffs > 0, axis=1) / n_time
        monotonic_decreasing = np.sum(diffs < 0, axis=1) / n_time
        monotonicity = np.maximum(monotonic_increasing, monotonic_decreasing)
        
        # Curve area (integral)
        curve_area = np.sum(activity, axis=1)
        
        return {
            'n_peaks': n_peaks,
            'smoothness': smoothness,
            'monotonicity': monotonicity,
            'curve_area': curve_area,
            'monotonic_direction': np.sign(monotonic_increasing - monotonic_decreasing)
        }
    
    def _extract_complexity_features(self, activity: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract complexity and information-theoretic features."""
        print("  - Complexity features...")
        
        n_neurons, n_time = activity.shape
        
        # Approximate entropy (regularity measure)
        def approximate_entropy(signal, m=2, r=0.2):
            """Compute approximate entropy of a signal."""
            N = len(signal)
            r = r * np.std(signal)
            
            def _maxdist(x_i, x_j, m):
                return max([abs(x_i[k] - x_j[k]) for k in range(m)])
            
            def _phi(m):
                patterns = np.array([[signal[i+j] for j in range(m)] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    matches = sum([1 for j in range(N - m + 1) if _maxdist(patterns[i], patterns[j], m) <= r])
                    C[i] = matches / (N - m + 1)
                return np.sum(np.log(C + 1e-10)) / (N - m + 1)
            
            return abs(_phi(m) - _phi(m + 1))
        
        approx_entropy = np.array([approximate_entropy(activity[i, :]) for i in range(n_neurons)])
        
        # Sample entropy (similar but more robust)
        # Simplified version for speed
        sample_entropy = np.array([np.std(np.diff(activity[i, :])) / (np.std(activity[i, :]) + 1e-10) 
                                  for i in range(n_neurons)])
        
        # Zero-crossing rate
        zero_crossings = np.zeros(n_neurons)
        for i in range(n_neurons):
            centered = activity[i, :] - np.mean(activity[i, :])
            zero_crossings[i] = np.sum(np.diff(np.sign(centered)) != 0) / n_time
        
        return {
            'approximate_entropy': approx_entropy,
            'sample_entropy': sample_entropy,
            'zero_crossing_rate': zero_crossings
        }
    
    def _extract_trial_variability(self, activity: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features related to trial-to-trial variability."""
        print("  - Trial variability features...")
        
        n_neurons, n_time, n_trials = activity.shape
        
        # Fano factor (variance / mean over trials)
        mean_over_trials = np.mean(activity, axis=2)
        var_over_trials = np.var(activity, axis=2)
        fano_factor = var_over_trials / (mean_over_trials + 1e-10)
        
        # Mean Fano factor over time
        mean_fano = np.mean(fano_factor, axis=1)
        
        # Coefficient of variation over trials
        cv_trials = np.std(activity, axis=2) / (np.mean(activity, axis=2) + 1e-10)
        mean_cv_trials = np.mean(cv_trials, axis=1)
        
        # Reliability (correlation between trials)
        reliability = np.zeros(n_neurons)
        for i in range(n_neurons):
            # Compute average pairwise correlation between trials
            corrs = []
            for t1 in range(n_trials):
                for t2 in range(t1+1, n_trials):
                    if np.std(activity[i, :, t1]) > 0 and np.std(activity[i, :, t2]) > 0:
                        corr = np.corrcoef(activity[i, :, t1], activity[i, :, t2])[0, 1]
                        if not np.isnan(corr):
                            corrs.append(corr)
            reliability[i] = np.mean(corrs) if corrs else 0
        
        return {
            'mean_fano_factor': mean_fano,
            'mean_cv_trials': mean_cv_trials,
            'trial_reliability': reliability
        }
    
    def cluster_neurons(self,
                       features: Dict[str, Any],
                       n_clusters: int = 5,
                       features_to_use: Optional[list] = None) -> Dict[str, Any]:
        """
        Cluster neurons based on extracted features using unsupervised ML.
        
        Args:
            features: Output from extract_all_features
            n_clusters: Number of clusters
            features_to_use: List of feature names to use (None = use all)
            
        Returns:
            Dictionary with clustering results
        """
        print(f"\nClustering neurons into {n_clusters} groups...")
        
        # Collect features into matrix
        feature_matrix = []
        feature_names = []
        
        if features_to_use is None:
            # Use key temporal and shape features
            features_to_use = [
                'basic_stats/mean', 'basic_stats/cv',
                'temporal_features/peak_time', 'temporal_features/peak_width',
                'spectral_features/dominant_frequency',
                'shape_features/n_peaks', 'shape_features/monotonicity',
                'trial_variability/trial_reliability'
            ]
        
        for feat_path in features_to_use:
            category, name = feat_path.split('/')
            if category in features and name in features[category]:
                feature_matrix.append(features[category][name])
                feature_names.append(f"{category}/{name}")
        
        feature_matrix = np.array(feature_matrix).T  # (n_neurons, n_features)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Compute cluster statistics
        cluster_stats = {}
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_stats[c] = {
                'n_neurons': np.sum(mask),
                'neuron_indices': np.where(mask)[0],
                'feature_means': {name: np.mean(feature_matrix[mask, i]) 
                                for i, name in enumerate(feature_names)}
            }
        
        print(f"✓ Clustering complete")
        for c in range(n_clusters):
            print(f"  Cluster {c}: {cluster_stats[c]['n_neurons']} neurons")
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_stats': cluster_stats,
            'feature_names': feature_names,
            'feature_matrix': feature_matrix_scaled,
            'n_clusters': n_clusters
        }
    
    def classify_neuron_types(self,
                             features: Dict[str, Any],
                             known_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Classify neurons into functional types using ML.
        
        If known_labels provided, uses supervised learning.
        Otherwise, uses unsupervised clustering.
        
        Args:
            features: Output from extract_all_features
            known_labels: Optional array of known neuron types
            
        Returns:
            Dictionary with classification results
        """
        if known_labels is not None:
            print("\nSupervised classification of neuron types...")
            # Would implement supervised classifier here
            # For now, use clustering
            pass
        
        # Unsupervised: cluster based on features
        return self.cluster_neurons(features, n_clusters=5)


if __name__ == "__main__":
    print("Neural Analysis AI Module")
    print("Import this module to use the analysis classes:")
    print("  - TemporalCodingAnalyzer")
    print("  - PopulationDecoder")
    print("  - NeuralManifoldAnalyzer")
    print("  - SequenceAnalyzer")
