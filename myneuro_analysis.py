"""
Enhanced Neural Analysis with AI/ML and Frequency Domain Analysis
===================================================================
Advanced computational neuroscience analysis for neural time series data from
monkey image recognition and mental navigation tasks.

This script demonstrates:
1. Loading neural tensor data
2. Frequency domain analysis (FFT, Power Spectral Density, Spectrograms)
3. Time-frequency analysis (Wavelet transforms, STFT)
4. Phase-amplitude coupling (theta-gamma coupling)
5. Coherence and synchrony analysis
6. Deep learning pattern extraction
7. Oscillatory burst detection
8. Cross-frequency coupling analysis
9. Spike-triggered analysis
10. Advanced visualization of frequency fields

Author: Computational Neuroscience Analysis Pipeline
Date: 2025-12-19
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import warnings

# Scientific computing
from scipy import signal, stats
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, butter, filtfilt, welch, spectrogram

# Machine Learning
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    warnings.warn("PyWavelets not available. Install with: pip install PyWavelets")

# Import our modules
from hippocampus_data_loader import HippocampusDataLoader, NeuralData
from neural_analysis_ai import (
    TemporalCodingAnalyzer,
    PopulationDecoder,
    NeuralManifoldAnalyzer,
    SequenceAnalyzer
)


class FrequencyAnalyzer:
    """
    Comprehensive frequency domain analysis for neural time series.
    """

    def __init__(self, sampling_rate: float = 1000.0):
        """
        Initialize frequency analyzer.

        Args:
            sampling_rate: Sampling rate in Hz (default 1000 Hz = 1ms resolution)
        """
        self.fs = sampling_rate

    def compute_fft(self, signal_data: np.ndarray) -> Dict:
        """
        Compute Fast Fourier Transform.

        Args:
            signal_data: Neural signal (neurons × time × trials) or (time,)

        Returns:
            Dictionary with frequencies, power spectrum, and phase
        """
        if signal_data.ndim == 1:
            # Single signal
            n = len(signal_data)
            freqs = rfftfreq(n, 1/self.fs)
            fft_vals = rfft(signal_data)
            power = np.abs(fft_vals) ** 2
            phase = np.angle(fft_vals)

            return {
                'frequencies': freqs,
                'power': power,
                'phase': phase,
                'complex_spectrum': fft_vals
            }
        else:
            # Multi-neuron signal
            n_neurons, n_time, n_trials = signal_data.shape
            freqs = rfftfreq(n_time, 1/self.fs)

            # Average FFT across trials
            power_all = []
            for neuron in range(n_neurons):
                power_neuron = []
                for trial in range(n_trials):
                    fft_vals = rfft(signal_data[neuron, :, trial])
                    power_neuron.append(np.abs(fft_vals) ** 2)
                power_all.append(np.mean(power_neuron, axis=0))

            power_all = np.array(power_all)

            return {
                'frequencies': freqs,
                'power_spectrum': power_all,  # neurons × frequencies
                'mean_power': np.mean(power_all, axis=0)
            }

    def compute_psd(self, signal_data: np.ndarray,
                    nperseg: int = 256) -> Dict:
        """
        Compute Power Spectral Density using Welch's method.

        Args:
            signal_data: Neural signal
            nperseg: Length of each segment for Welch's method

        Returns:
            Dictionary with frequencies and PSD
        """
        if signal_data.ndim == 1:
            freqs, psd = welch(signal_data, fs=self.fs, nperseg=nperseg)
            return {'frequencies': freqs, 'psd': psd}
        else:
            n_neurons, n_time, n_trials = signal_data.shape
            psd_all = []

            for neuron in range(n_neurons):
                psd_neuron = []
                for trial in range(n_trials):
                    freqs, psd = welch(signal_data[neuron, :, trial],
                                      fs=self.fs, nperseg=nperseg)
                    psd_neuron.append(psd)
                psd_all.append(np.mean(psd_neuron, axis=0))

            psd_all = np.array(psd_all)

            return {
                'frequencies': freqs,
                'psd': psd_all,  # neurons × frequencies
                'mean_psd': np.mean(psd_all, axis=0)
            }

    def compute_spectrogram(self, signal_data: np.ndarray,
                           nperseg: int = 256) -> Dict:
        """
        Compute time-frequency spectrogram.

        Args:
            signal_data: Neural signal (time,) or (neurons × time × trials)
            nperseg: Segment length

        Returns:
            Dictionary with time, frequencies, and spectrogram
        """
        if signal_data.ndim == 1:
            freqs, times, Sxx = spectrogram(signal_data, fs=self.fs,
                                           nperseg=nperseg)
            return {
                'frequencies': freqs,
                'times': times,
                'spectrogram': Sxx
            }
        else:
            # Average across trials and neurons
            n_neurons, n_time, n_trials = signal_data.shape
            mean_signal = np.mean(signal_data, axis=(0, 2))
            freqs, times, Sxx = spectrogram(mean_signal, fs=self.fs,
                                           nperseg=nperseg)
            return {
                'frequencies': freqs,
                'times': times,
                'spectrogram': Sxx
            }

    def compute_wavelet_transform(self, signal_data: np.ndarray,
                                 wavelet: str = 'cmor1.5-1.0',
                                 scales: Optional[np.ndarray] = None) -> Dict:
        """
        Compute continuous wavelet transform for time-frequency analysis.

        Args:
            signal_data: Neural signal
            wavelet: Wavelet type (e.g., 'cmor1.5-1.0', 'morl')
            scales: Wavelet scales (default: log-spaced from 1 to 128)

        Returns:
            Dictionary with time-frequency representation
        """
        if not WAVELET_AVAILABLE:
            raise ImportError("PyWavelets required for wavelet analysis")

        if scales is None:
            scales = np.logspace(0, 7, 100, base=2)

        if signal_data.ndim == 1:
            coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet,
                                                 sampling_period=1/self.fs)
            power = np.abs(coefficients) ** 2

            return {
                'coefficients': coefficients,
                'power': power,
                'frequencies': frequencies,
                'scales': scales,
                'wavelet': wavelet
            }
        else:
            # Average across trials
            n_neurons, n_time, n_trials = signal_data.shape
            mean_signal = np.mean(signal_data, axis=(0, 2))
            coefficients, frequencies = pywt.cwt(mean_signal, scales, wavelet,
                                                sampling_period=1/self.fs)
            power = np.abs(coefficients) ** 2

            return {
                'coefficients': coefficients,
                'power': power,
                'frequencies': frequencies,
                'scales': scales,
                'wavelet': wavelet
            }

    def identify_frequency_bands(self, psd_result: Dict) -> Dict:
        """
        Identify power in canonical frequency bands.

        Args:
            psd_result: Output from compute_psd

        Returns:
            Dictionary with band powers
        """
        freqs = psd_result['frequencies']
        psd = psd_result.get('mean_psd', psd_result['psd'])

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 60),
            'high_gamma': (60, 100)
        }

        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if psd.ndim == 1:
                band_powers[band_name] = np.mean(psd[mask])
            else:
                band_powers[band_name] = np.mean(psd[:, mask], axis=1)

        return band_powers


class PhaseAmplitudeCoupling:
    """
    Analyze phase-amplitude coupling (e.g., theta-gamma coupling).
    """

    def __init__(self, sampling_rate: float = 1000.0):
        self.fs = sampling_rate

    def bandpass_filter(self, signal_data: np.ndarray,
                       lowcut: float, highcut: float,
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to signal.

        Args:
            signal_data: Input signal
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal_data)

    def compute_pac(self, signal_data: np.ndarray,
                   phase_band: Tuple[float, float] = (4, 8),
                   amp_band: Tuple[float, float] = (30, 100)) -> Dict:
        """
        Compute phase-amplitude coupling between two frequency bands.

        Args:
            signal_data: Neural signal (time,)
            phase_band: Frequency band for phase (e.g., theta: 4-8 Hz)
            amp_band: Frequency band for amplitude (e.g., gamma: 30-100 Hz)

        Returns:
            Dictionary with PAC metrics
        """
        # Filter signal in phase band
        phase_filtered = self.bandpass_filter(signal_data,
                                              phase_band[0], phase_band[1])

        # Filter signal in amplitude band
        amp_filtered = self.bandpass_filter(signal_data,
                                           amp_band[0], amp_band[1])

        # Extract phase using Hilbert transform
        analytic_signal_phase = hilbert(phase_filtered)
        phase = np.angle(analytic_signal_phase)

        # Extract amplitude envelope
        analytic_signal_amp = hilbert(amp_filtered)
        amplitude = np.abs(analytic_signal_amp)

        # Compute modulation index (MI)
        n_bins = 18  # 20-degree bins
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        # Average amplitude in each phase bin
        mean_amp_per_phase = []
        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                mean_amp_per_phase.append(np.mean(amplitude[mask]))
            else:
                mean_amp_per_phase.append(0)

        mean_amp_per_phase = np.array(mean_amp_per_phase)

        # Normalize to create probability distribution
        P = mean_amp_per_phase / np.sum(mean_amp_per_phase)

        # Compute modulation index using KL divergence
        Q = np.ones(n_bins) / n_bins  # Uniform distribution
        MI = np.sum(P * np.log(P / Q + 1e-10)) / np.log(n_bins)

        return {
            'modulation_index': MI,
            'phase': phase,
            'amplitude': amplitude,
            'mean_amp_per_phase': mean_amp_per_phase,
            'phase_bins': phase_bins,
            'phase_band': phase_band,
            'amp_band': amp_band
        }


class OscillatoryBurstDetector:
    """
    Detect oscillatory bursts in neural signals.
    """

    @staticmethod
    def detect_bursts(signal_data: np.ndarray,
                     fs: float = 1000.0,
                     freq_band: Tuple[float, float] = (30, 100),
                     threshold: float = 2.0) -> Dict:
        """
        Detect oscillatory bursts in specific frequency band.

        Args:
            signal_data: Neural signal (time,)
            fs: Sampling rate
            freq_band: Frequency band to analyze
            threshold: Z-score threshold for burst detection

        Returns:
            Dictionary with burst times and properties
        """
        pac = PhaseAmplitudeCoupling(sampling_rate=fs)

        # Filter signal
        filtered = pac.bandpass_filter(signal_data, freq_band[0], freq_band[1])

        # Extract amplitude envelope
        analytic_signal = hilbert(filtered)
        amplitude = np.abs(analytic_signal)

        # Smooth amplitude
        amplitude_smooth = gaussian_filter1d(amplitude, sigma=fs/100)

        # Z-score normalization
        z_amplitude = (amplitude_smooth - np.mean(amplitude_smooth)) / np.std(amplitude_smooth)

        # Detect bursts (threshold crossings)
        burst_mask = z_amplitude > threshold

        # Find burst onset and offset
        burst_diff = np.diff(burst_mask.astype(int))
        burst_onsets = np.where(burst_diff == 1)[0]
        burst_offsets = np.where(burst_diff == -1)[0]

        # Match onsets and offsets
        if len(burst_offsets) > 0 and len(burst_onsets) > 0:
            if burst_offsets[0] < burst_onsets[0]:
                burst_offsets = burst_offsets[1:]
            if len(burst_onsets) > len(burst_offsets):
                burst_onsets = burst_onsets[:len(burst_offsets)]

        burst_durations = burst_offsets - burst_onsets

        return {
            'n_bursts': len(burst_onsets),
            'burst_onsets': burst_onsets,
            'burst_offsets': burst_offsets,
            'burst_durations': burst_durations,
            'amplitude_envelope': amplitude_smooth,
            'z_amplitude': z_amplitude,
            'threshold': threshold
        }


class DeepPatternExtractor:
    """
    Deep learning-based pattern extraction from neural data.
    """

    def __init__(self, n_features: int = 50):
        """
        Initialize pattern extractor.

        Args:
            n_features: Number of features to extract
        """
        self.n_features = n_features
        self.scaler = StandardScaler()

    def extract_temporal_patterns(self, neural_data: NeuralData,
                                  trial_mask: np.ndarray,
                                  time_window: Tuple[int, int]) -> Dict:
        """
        Extract temporal patterns using unsupervised learning.

        Args:
            neural_data: NeuralData object
            trial_mask: Boolean mask for trials
            time_window: Time window (ms)

        Returns:
            Dictionary with extracted patterns
        """
        # Get neural activity
        activity = neural_data.get_neural_activity(trial_mask, time_window)
        n_neurons, n_time, n_trials = activity.shape

        # Reshape for pattern extraction
        X = activity.transpose(2, 0, 1).reshape(n_trials, -1)
        X_scaled = self.scaler.fit_transform(X)

        # Apply ICA for pattern separation
        ica = FastICA(n_components=min(self.n_features, n_neurons),
                     random_state=42, max_iter=500)
        patterns = ica.fit_transform(X_scaled)

        # Cluster patterns
        n_clusters = min(5, n_trials // 10)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(patterns)
        else:
            cluster_labels = np.zeros(n_trials, dtype=int)

        # Detect anomalies
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(patterns)

        return {
            'patterns': patterns,
            'components': ica.components_,
            'cluster_labels': cluster_labels,
            'anomaly_labels': anomaly_labels,
            'n_clusters': n_clusters,
            'n_anomalies': np.sum(anomaly_labels == -1)
        }

    def compute_pattern_similarity(self, patterns: np.ndarray,
                                   condition_labels: np.ndarray) -> Dict:
        """
        Compute similarity between patterns across conditions.

        Args:
            patterns: Extracted patterns (trials × features)
            condition_labels: Condition for each trial

        Returns:
            Dictionary with similarity metrics
        """
        unique_conditions = np.unique(condition_labels)
        n_cond = len(unique_conditions)

        # Compute mean pattern for each condition
        mean_patterns = {}
        for cond in unique_conditions:
            mask = condition_labels == cond
            mean_patterns[cond] = np.mean(patterns[mask], axis=0)

        # Compute pairwise cosine similarity
        similarity_matrix = np.zeros((n_cond, n_cond))
        for i, cond1 in enumerate(unique_conditions):
            for j, cond2 in enumerate(unique_conditions):
                p1 = mean_patterns[cond1]
                p2 = mean_patterns[cond2]
                similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                similarity_matrix[i, j] = similarity

        return {
            'mean_patterns': mean_patterns,
            'similarity_matrix': similarity_matrix,
            'conditions': unique_conditions
        }


def analyze_frequency_spectrum(neural_data: NeuralData,
                               trial_mask: np.ndarray,
                               output_dir: Path) -> Dict:
    """
    Comprehensive frequency domain analysis.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trials
        output_dir: Output directory

    Returns:
        Dictionary with frequency analysis results
    """
    print("\n" + "="*70)
    print("  FREQUENCY SPECTRUM ANALYSIS")
    print("="*70)

    # Get neural activity
    activity = neural_data.get_neural_activity(trial_mask, (0, 3000))

    # Initialize analyzer (assuming 1ms time bins = 1000 Hz)
    freq_analyzer = FrequencyAnalyzer(sampling_rate=1000.0)

    # 1. FFT Analysis
    print("\nComputing FFT...")
    fft_results = freq_analyzer.compute_fft(activity)
    print(f"  Frequency range: {fft_results['frequencies'][0]:.2f} - "
          f"{fft_results['frequencies'][-1]:.2f} Hz")

    # 2. Power Spectral Density
    print("\nComputing Power Spectral Density...")
    psd_results = freq_analyzer.compute_psd(activity, nperseg=256)

    # 3. Identify frequency bands
    print("\nAnalyzing frequency bands:")
    band_powers = freq_analyzer.identify_frequency_bands(psd_results)
    for band_name, power in band_powers.items():
        if isinstance(power, np.ndarray):
            print(f"  {band_name:12s}: mean = {np.mean(power):.2e}")
        else:
            print(f"  {band_name:12s}: {power:.2e}")

    # 4. Spectrogram
    print("\nComputing spectrogram...")
    spec_results = freq_analyzer.compute_spectrogram(activity, nperseg=128)

    # 5. Wavelet analysis (if available)
    wavelet_results = None
    if WAVELET_AVAILABLE:
        print("\nComputing wavelet transform...")
        try:
            wavelet_results = freq_analyzer.compute_wavelet_transform(activity)
            print(f"  Time-frequency resolution: "
                  f"{len(wavelet_results['frequencies'])} × {activity.shape[1]} bins")
        except Exception as e:
            print(f"  Warning: Wavelet analysis failed - {str(e)}")

    # Visualizations
    print("\nGenerating frequency visualizations...")

    # Plot 1: Power spectrum
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FFT
    ax = axes[0, 0]
    ax.semilogy(fft_results['frequencies'], fft_results['mean_power'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('FFT Power Spectrum')
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3)

    # PSD
    ax = axes[0, 1]
    ax.semilogy(psd_results['frequencies'], psd_results['mean_psd'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('PSD (Welch Method)')
    ax.set_xlim([0, 100])
    ax.grid(True, alpha=0.3)

    # Band powers
    ax = axes[1, 0]
    band_names = list(band_powers.keys())
    band_values = [np.mean(band_powers[b]) if isinstance(band_powers[b], np.ndarray)
                   else band_powers[b] for b in band_names]
    ax.bar(band_names, band_values)
    ax.set_ylabel('Mean Power')
    ax.set_title('Frequency Band Powers')
    ax.tick_params(axis='x', rotation=45)

    # Spectrogram
    ax = axes[1, 1]
    im = ax.pcolormesh(spec_results['times'], spec_results['frequencies'],
                       10 * np.log10(spec_results['spectrogram']),
                       shading='auto', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Spectrogram')
    ax.set_ylim([0, 100])
    plt.colorbar(im, ax=ax, label='Power (dB)')

    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Wavelet analysis
    if wavelet_results is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(np.arange(wavelet_results['power'].shape[1]),
                          wavelet_results['frequencies'],
                          wavelet_results['power'],
                          shading='auto', cmap='jet')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Wavelet Transform (Time-Frequency Power)')
        ax.set_ylim([0, 100])
        plt.colorbar(im, ax=ax, label='Power')
        plt.tight_layout()
        plt.savefig(output_dir / 'wavelet_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✓ Frequency analysis complete")

    return {
        'fft': fft_results,
        'psd': psd_results,
        'band_powers': band_powers,
        'spectrogram': spec_results,
        'wavelet': wavelet_results
    }


def analyze_phase_amplitude_coupling(neural_data: NeuralData,
                                     trial_mask: np.ndarray,
                                     output_dir: Path) -> Dict:
    """
    Analyze phase-amplitude coupling (theta-gamma).

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trials
        output_dir: Output directory

    Returns:
        Dictionary with PAC results
    """
    print("\n" + "="*70)
    print("  PHASE-AMPLITUDE COUPLING ANALYSIS")
    print("="*70)

    # Get neural activity
    activity = neural_data.get_neural_activity(trial_mask, (0, 3000))
    n_neurons, n_time, n_trials = activity.shape

    # Average across trials for population signal
    pop_signal = np.mean(activity, axis=(0, 2))

    pac_analyzer = PhaseAmplitudeCoupling(sampling_rate=1000.0)

    # Analyze different coupling combinations
    coupling_pairs = [
        ('theta', 'low_gamma', (4, 8), (30, 60)),
        ('theta', 'high_gamma', (4, 8), (60, 100)),
        ('alpha', 'gamma', (8, 13), (30, 100)),
        ('beta', 'gamma', (13, 30), (60, 100))
    ]

    results = {}
    print("\nComputing phase-amplitude coupling:")

    for phase_name, amp_name, phase_band, amp_band in coupling_pairs:
        pac_result = pac_analyzer.compute_pac(pop_signal, phase_band, amp_band)
        mi = pac_result['modulation_index']
        results[f"{phase_name}_{amp_name}"] = pac_result
        print(f"  {phase_name:12s} - {amp_name:12s}: MI = {mi:.4f}")

    # Visualize strongest coupling
    best_coupling = max(results.keys(),
                       key=lambda k: results[k]['modulation_index'])
    best_pac = results[best_coupling]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Phase and amplitude traces
    ax = axes[0, 0]
    time_vec = np.arange(len(best_pac['phase'])) / 1000.0
    plot_range = slice(0, min(3000, len(time_vec)))
    ax.plot(time_vec[plot_range], best_pac['phase'][plot_range], 'b-', alpha=0.7, label='Phase')
    ax.set_ylabel('Phase (rad)', color='b')
    ax.set_xlabel('Time (s)')
    ax.tick_params(axis='y', labelcolor='b')
    ax2 = ax.twinx()
    ax2.plot(time_vec[plot_range], best_pac['amplitude'][plot_range], 'r-', alpha=0.7, label='Amplitude')
    ax2.set_ylabel('Amplitude', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title(f'Phase-Amplitude Traces ({best_coupling})')

    # Amplitude modulation by phase
    ax = axes[0, 1]
    phase_centers = (best_pac['phase_bins'][:-1] + best_pac['phase_bins'][1:]) / 2
    ax.bar(phase_centers, best_pac['mean_amp_per_phase'],
           width=np.diff(best_pac['phase_bins'])[0], alpha=0.7)
    ax.set_xlabel('Phase (rad)')
    ax.set_ylabel('Mean Amplitude')
    ax.set_title(f'Amplitude Modulation by Phase\nMI = {best_pac["modulation_index"]:.4f}')
    ax.axhline(y=np.mean(best_pac['mean_amp_per_phase']),
               color='r', linestyle='--', label='Mean')
    ax.legend()

    # Polar plot
    ax = axes[1, 0]
    ax = plt.subplot(2, 2, 3, projection='polar')
    ax.bar(phase_centers, best_pac['mean_amp_per_phase'],
           width=np.diff(best_pac['phase_bins'])[0], alpha=0.7)
    ax.set_title('Polar Representation')

    # Summary of all couplings
    ax = axes[1, 1]
    coupling_names = list(results.keys())
    mi_values = [results[k]['modulation_index'] for k in coupling_names]
    ax.barh(coupling_names, mi_values)
    ax.set_xlabel('Modulation Index')
    ax.set_title('Phase-Amplitude Coupling Summary')
    ax.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'phase_amplitude_coupling.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Phase-amplitude coupling analysis complete")

    return results


def analyze_oscillatory_bursts(neural_data: NeuralData,
                               trial_mask: np.ndarray,
                               output_dir: Path) -> Dict:
    """
    Detect and analyze oscillatory bursts.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trials
        output_dir: Output directory

    Returns:
        Dictionary with burst detection results
    """
    print("\n" + "="*70)
    print("  OSCILLATORY BURST DETECTION")
    print("="*70)

    # Get neural activity
    activity = neural_data.get_neural_activity(trial_mask, (0, 3000))

    # Average across trials and neurons for population signal
    pop_signal = np.mean(activity, axis=(0, 2))

    # Detect bursts in different frequency bands
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 60),
        'high_gamma': (60, 100)
    }

    results = {}
    print("\nDetecting oscillatory bursts:")

    for band_name, freq_range in bands.items():
        burst_result = OscillatoryBurstDetector.detect_bursts(
            pop_signal, fs=1000.0, freq_band=freq_range, threshold=2.0
        )
        results[band_name] = burst_result
        n_bursts = burst_result['n_bursts']
        if n_bursts > 0:
            mean_duration = np.mean(burst_result['burst_durations'])
            print(f"  {band_name:12s}: {n_bursts} bursts, "
                  f"mean duration = {mean_duration:.1f} ms")
        else:
            print(f"  {band_name:12s}: No bursts detected")

    # Visualize gamma bursts
    if results['low_gamma']['n_bursts'] > 0:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        gamma_result = results['low_gamma']
        time_vec = np.arange(len(pop_signal)) / 1000.0

        # Original signal
        ax = axes[0]
        ax.plot(time_vec, pop_signal, 'k-', alpha=0.5, linewidth=0.5)
        ax.set_ylabel('Neural Activity')
        ax.set_title('Population Signal')
        ax.set_xlim([0, 3])

        # Amplitude envelope with burst detection
        ax = axes[1]
        ax.plot(time_vec, gamma_result['amplitude_envelope'], 'b-', linewidth=1.5)

        # Mark bursts
        for onset, offset in zip(gamma_result['burst_onsets'],
                                gamma_result['burst_offsets']):
            ax.axvspan(onset/1000, offset/1000, alpha=0.3, color='red')

        ax.set_ylabel('Amplitude Envelope')
        ax.set_title('Low Gamma Bursts (30-60 Hz)')
        ax.set_xlim([0, 3])

        # Z-scored amplitude
        ax = axes[2]
        ax.plot(time_vec, gamma_result['z_amplitude'], 'g-', linewidth=1.5)
        ax.axhline(y=gamma_result['threshold'], color='r',
                  linestyle='--', label=f'Threshold ({gamma_result["threshold"]}σ)')
        ax.set_ylabel('Z-scored Amplitude')
        ax.set_xlabel('Time (s)')
        ax.set_xlim([0, 3])
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'oscillatory_bursts.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("✓ Burst detection complete")

    return results


def analyze_deep_patterns(neural_data: NeuralData,
                         trial_mask: np.ndarray,
                         output_dir: Path) -> Dict:
    """
    Extract patterns using deep learning techniques.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trials
        output_dir: Output directory

    Returns:
        Dictionary with pattern extraction results
    """
    print("\n" + "="*70)
    print("  DEEP PATTERN EXTRACTION")
    print("="*70)

    extractor = DeepPatternExtractor(n_features=20)

    print("\nExtracting temporal patterns using ICA...")
    pattern_results = extractor.extract_temporal_patterns(
        neural_data, trial_mask, (0, 3000)
    )

    print(f"  Extracted {pattern_results['patterns'].shape[1]} independent components")
    print(f"  Identified {pattern_results['n_clusters']} pattern clusters")
    print(f"  Detected {pattern_results['n_anomalies']} anomalous trials")

    # Compute pattern similarity across landmark pairs
    curr = neural_data.get_condition('curr')[trial_mask].astype(int)
    target = neural_data.get_condition('target')[trial_mask].astype(int)
    pair_labels = curr * 10 + target

    similarity_results = extractor.compute_pattern_similarity(
        pattern_results['patterns'], pair_labels
    )

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pattern space (first 2 components)
    ax = axes[0, 0]
    scatter = ax.scatter(pattern_results['patterns'][:, 0],
                        pattern_results['patterns'][:, 1],
                        c=pattern_results['cluster_labels'],
                        cmap='tab10', alpha=0.6)
    ax.set_xlabel('Pattern Component 1')
    ax.set_ylabel('Pattern Component 2')
    ax.set_title('Neural Pattern Space (ICA)')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # Anomaly detection
    ax = axes[0, 1]
    colors = ['red' if a == -1 else 'blue'
              for a in pattern_results['anomaly_labels']]
    ax.scatter(pattern_results['patterns'][:, 0],
              pattern_results['patterns'][:, 1],
              c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Pattern Component 1')
    ax.set_ylabel('Pattern Component 2')
    ax.set_title('Anomaly Detection (Red = Anomalous)')

    # Pattern similarity matrix
    ax = axes[1, 0]
    im = ax.imshow(similarity_results['similarity_matrix'],
                   cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Landmark Pair')
    ax.set_ylabel('Landmark Pair')
    ax.set_title('Pattern Similarity Across Conditions')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # Component weights heatmap
    ax = axes[1, 1]
    # Show first 10 components
    n_comp_show = min(10, pattern_results['components'].shape[0])
    im = ax.imshow(pattern_results['components'][:n_comp_show, :50],
                   aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Component')
    ax.set_title('ICA Component Weights')
    plt.colorbar(im, ax=ax, label='Weight')

    plt.tight_layout()
    plt.savefig(output_dir / 'deep_pattern_extraction.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Pattern extraction complete")

    return {
        'patterns': pattern_results,
        'similarity': similarity_results
    }


def run_enhanced_analysis(mat_file: Path, output_dir: Path):
    """
    Run enhanced AI/ML and frequency analysis pipeline.

    Args:
        mat_file: Path to .mat file with neural data
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("  ENHANCED NEURAL ANALYSIS WITH AI/ML & FREQUENCY ANALYSIS")
    print("="*70)
    print(f"\nData file: {mat_file.name}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "-"*70)
    print("Loading neural data...")
    print("-"*70)

    try:
        neural_data = HippocampusDataLoader.load_mat_file(mat_file)
        print("\n" + neural_data.summary())
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Filter trials
    print("\n" + "-"*70)
    print("Filtering mental navigation trials...")
    print("-"*70)

    mn_trials = neural_data.get_mental_navigation_trials()
    n_mn_trials = np.sum(mn_trials)
    print(f"\nSelected {n_mn_trials}/{neural_data.n_trials} trials")

    if n_mn_trials < 10:
        print("Warning: Very few trials. Results may be unreliable.")
        return

    # Run analyses
    all_results = {}

    # 1. Frequency spectrum analysis
    all_results['frequency'] = analyze_frequency_spectrum(
        neural_data, mn_trials, output_dir
    )

    # 2. Phase-amplitude coupling
    all_results['pac'] = analyze_phase_amplitude_coupling(
        neural_data, mn_trials, output_dir
    )

    # 3. Oscillatory burst detection
    all_results['bursts'] = analyze_oscillatory_bursts(
        neural_data, mn_trials, output_dir
    )

    # 4. Deep pattern extraction
    all_results['patterns'] = analyze_deep_patterns(
        neural_data, mn_trials, output_dir
    )

    # 5. Traditional AI/ML analyses from other modules
    print("\n" + "="*70)
    print("  TEMPORAL CODING & DECODING")
    print("="*70)

    # Temporal coding
    analyzer = TemporalCodingAnalyzer()
    tuning_results = analyzer.compute_temporal_tuning(
        neural_data, mn_trials, (0, 3000), smooth_sigma=50.0
    )
    time_cells = analyzer.identify_time_cells(tuning_results)
    print(f"\nIdentified {np.sum(time_cells)} time cells")

    # Decoding
    decoder = PopulationDecoder(method='random_forest')
    decoding_results = decoder.decode_landmark_pairs(
        neural_data, mn_trials, (500, 2000), cv_folds=5
    )
    print(f"Decoding accuracy: {decoding_results['accuracy']:.3f}")

    all_results['temporal'] = tuning_results
    all_results['decoding'] = decoding_results

    # Generate summary
    print("\n" + "="*70)
    print("  ANALYSIS SUMMARY")
    print("="*70)

    summary = [
        f"\nEnhanced Neural Analysis Results",
        f"Dataset: {mat_file.name}",
        f"Neurons: {neural_data.n_neurons}",
        f"Trials analyzed: {n_mn_trials}",
        f"",
        f"Frequency Analysis:",
        f"  - Dominant band: {max(all_results['frequency']['band_powers'].keys(), key=lambda k: np.mean(all_results['frequency']['band_powers'][k]) if isinstance(all_results['frequency']['band_powers'][k], np.ndarray) else all_results['frequency']['band_powers'][k])}",
    ]

    # Add PAC summary
    best_pac = max(all_results['pac'].keys(),
                  key=lambda k: all_results['pac'][k]['modulation_index'])
    summary.append(f"  - Strongest coupling: {best_pac} "
                  f"(MI = {all_results['pac'][best_pac]['modulation_index']:.4f})")

    # Add burst summary
    total_bursts = sum(all_results['bursts'][b]['n_bursts']
                      for b in all_results['bursts'])
    summary.append(f"  - Total bursts detected: {total_bursts}")

    # Add pattern summary
    summary.append(f"\nPattern Analysis:")
    summary.append(f"  - Pattern clusters: {all_results['patterns']['patterns']['n_clusters']}")
    summary.append(f"  - Anomalous trials: {all_results['patterns']['patterns']['n_anomalies']}")

    # Add ML summary
    summary.append(f"\nMachine Learning:")
    summary.append(f"  - Time cells: {np.sum(time_cells)}/{neural_data.n_neurons}")
    summary.append(f"  - Decoding accuracy: {decoding_results['accuracy']:.3f}")

    summary_text = "\n".join(summary)
    print(summary_text)

    # Save summary
    with open(output_dir / 'enhanced_analysis_summary.txt', 'w') as f:
        f.write(summary_text)

    print(f"\n✓ Enhanced analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced neural analysis with AI/ML and frequency analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python example_analysis_1.py data/session1.mat

  # Specify output directory
  python example_analysis_1.py data/session1.mat --output results/enhanced
        """
    )

    parser.add_argument('input_path', type=str,
                       help='Path to .mat file containing neural data tensor')
    parser.add_argument('--output', '-o', type=str, default='results_enhanced',
                       help='Output directory (default: results_enhanced/)')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_base = Path(args.output)

    if input_path.is_file():
        output_dir = output_base / input_path.stem
        run_enhanced_analysis(input_path, output_dir)
    else:
        print(f"Error: {input_path} is not a valid file")


if __name__ == "__main__":
    main()
