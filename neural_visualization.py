"""
Neural Data Visualization Tools
================================
Visualization functions for hippocampal neural data analysis.

Author: Computational Neuroscience Analysis Pipeline
Date: 2025-11-15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


class NeuralVisualizer:
    """Visualization tools for neural data analysis."""

    @staticmethod
    def plot_raster(activity: np.ndarray,
                   time_vector: np.ndarray,
                   trial_idx: int = 0,
                   threshold: float = None,
                   title: str = "Neural Raster Plot",
                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Create raster plot of neural activity for a single trial.

        Args:
            activity: Neural activity (neurons × time × trials)
            time_vector: Time vector in ms
            trial_idx: Which trial to plot
            threshold: Spike threshold (if None, shows firing rates)
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        n_neurons = activity.shape[0]
        trial_data = activity[:, :, trial_idx]

        if threshold is not None:
            # Binary raster
            spike_times = []
            spike_neurons = []
            for neuron in range(n_neurons):
                spike_idx = np.where(trial_data[neuron, :] > threshold)[0]
                spike_times.extend(time_vector[spike_idx])
                spike_neurons.extend([neuron] * len(spike_idx))

            ax.scatter(spike_times, spike_neurons, s=1, c='black', alpha=0.5)
        else:
            # Heatmap of firing rates
            im = ax.imshow(trial_data, aspect='auto',
                          extent=[time_vector[0], time_vector[-1], n_neurons, 0],
                          cmap='hot', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron #')
        ax.set_title(f"{title} - Trial {trial_idx}")
        ax.axvline(0, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='Start cue')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_temporal_tuning(tuning_results: Dict,
                            neuron_indices: Optional[List[int]] = None,
                            n_neurons: int = 12,
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot temporal tuning curves for multiple neurons.

        Args:
            tuning_results: Output from TemporalCodingAnalyzer
            neuron_indices: Specific neurons to plot (if None, picks top by TMI)
            n_neurons: Number of neurons to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        tuning_curves = tuning_results['tuning_curves']
        time_vec = tuning_results['time_vector']
        tmi = tuning_results['temporal_modulation_index']
        peak_times = tuning_results['peak_times']

        if neuron_indices is None:
            # Select neurons with highest temporal modulation
            neuron_indices = np.argsort(tmi)[-n_neurons:][::-1]

        n_cols = 3
        n_rows = int(np.ceil(len(neuron_indices) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, neuron in enumerate(neuron_indices):
            ax = axes[idx]

            # Plot tuning curve
            ax.plot(time_vec, tuning_curves[neuron, :], 'b-', linewidth=2)
            ax.fill_between(time_vec, 0, tuning_curves[neuron, :],
                           alpha=0.3, color='blue')

            # Mark peak
            ax.axvline(peak_times[neuron], color='red', linestyle='--',
                      linewidth=1.5, alpha=0.7)

            # Mark start cue
            ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            ax.set_title(f"Neuron {neuron} | TMI={tmi[neuron]:.3f}\nPeak={peak_times[neuron]:.0f}ms",
                        fontsize=10)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(neuron_indices), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Temporal Tuning Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_population_heatmap(activity: np.ndarray,
                               time_vector: np.ndarray,
                               sort_by: str = 'peak',
                               title: str = "Population Activity",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot population activity heatmap.

        Args:
            activity: Mean neural activity (neurons × time)
            time_vector: Time vector in ms
            sort_by: 'peak' or 'rate' or 'none'
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort neurons
        if sort_by == 'peak':
            peak_times = np.argmax(activity, axis=1)
            sort_idx = np.argsort(peak_times)
        elif sort_by == 'rate':
            mean_rates = np.mean(activity, axis=1)
            sort_idx = np.argsort(mean_rates)[::-1]
        else:
            sort_idx = np.arange(activity.shape[0])

        sorted_activity = activity[sort_idx, :]

        # Plot heatmap
        im = ax.imshow(sorted_activity, aspect='auto',
                      extent=[time_vector[0], time_vector[-1],
                             activity.shape[0], 0],
                      cmap='viridis', interpolation='nearest')

        ax.axvline(0, color='white', linestyle='--', linewidth=2,
                  alpha=0.8, label='Start cue')

        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Neuron # (sorted)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_decoding_results(decoding_results: Dict,
                             figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Visualize decoding performance.

        Args:
            decoding_results: Output from PopulationDecoder
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

        # Accuracy comparison
        ax1 = fig.add_subplot(gs[0])
        accuracy = decoding_results['accuracy']
        chance = decoding_results['chance_level']

        bars = ax1.bar(['Decoder', 'Chance'], [accuracy, chance],
                      color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Decoding Performance', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.axhline(chance, color='red', linestyle='--', linewidth=1, alpha=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        # Confusion matrix
        ax2 = fig.add_subplot(gs[1])
        cm = decoding_results['confusion_matrix']

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = ax2.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted Landmark Pair', fontsize=11)
        ax2.set_ylabel('True Landmark Pair', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Proportion', fontsize=10)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_3d_trajectory(manifold_results: Dict,
                          condition_labels: np.ndarray,
                          n_conditions: int = 6,
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot 3D neural trajectories in low-dimensional space.

        Args:
            manifold_results: Output from NeuralManifoldAnalyzer
            condition_labels: Condition label for each trial
            n_conditions: Number of conditions to plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        trajectories = manifold_results['trajectories']
        n_time, n_trials, n_comp = trajectories.shape

        if n_comp < 3:
            raise ValueError("Need at least 3 components for 3D plot")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Color palette
        colors = sns.color_palette("husl", n_conditions)

        unique_conditions = np.unique(condition_labels)[:n_conditions]

        for idx, cond in enumerate(unique_conditions):
            cond_mask = condition_labels == cond
            cond_trajectories = trajectories[:, cond_mask, :]

            # Plot mean trajectory
            mean_traj = np.mean(cond_trajectories, axis=1)

            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                   color=colors[idx], linewidth=3, alpha=0.8,
                   label=f'Condition {int(cond)}')

            # Mark start and end
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], mean_traj[0, 2],
                      color=colors[idx], s=200, marker='o', alpha=0.9,
                      edgecolors='black', linewidths=2)
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1], mean_traj[-1, 2],
                      color=colors[idx], s=200, marker='s', alpha=0.9,
                      edgecolors='black', linewidths=2)

        ax.set_xlabel(f'{manifold_results["method"].upper()} 1', fontsize=12)
        ax.set_ylabel(f'{manifold_results["method"].upper()} 2', fontsize=12)
        ax.set_zlabel(f'{manifold_results["method"].upper()} 3', fontsize=12)
        ax.set_title('Neural State Space Trajectories', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_pca_variance(manifold_results: Dict,
                         figsize: Tuple[int, int] = (10, 5)) -> plt.Figure:
        """
        Plot variance explained by PCA components.

        Args:
            manifold_results: Output from NeuralManifoldAnalyzer (must use PCA)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if manifold_results['variance_explained'] is None:
            raise ValueError("Variance explained only available for PCA")

        var_exp = manifold_results['variance_explained']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Individual variance
        ax1.bar(range(1, len(var_exp) + 1), var_exp * 100,
               alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained (%)', fontsize=12)
        ax1.set_title('Variance per Component', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Cumulative variance
        cum_var = np.cumsum(var_exp) * 100
        ax2.plot(range(1, len(cum_var) + 1), cum_var, 'o-',
                linewidth=2, markersize=8, color='darkgreen')
        ax2.axhline(90, color='red', linestyle='--', linewidth=1,
                   alpha=0.7, label='90% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_sequence_analysis(sequence_results: Dict,
                              time_vector: np.ndarray,
                              figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Visualize sequential neural activity.

        Args:
            sequence_results: Output from SequenceAnalyzer
            time_vector: Time vector in ms
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        sorted_activity = sequence_results['sorted_activity']
        peak_times = sequence_results['peak_times']
        sequence_score = sequence_results['sequence_score']

        # Activity heatmap
        im = ax1.imshow(sorted_activity, aspect='auto',
                       extent=[time_vector[0], time_vector[-1],
                              sorted_activity.shape[0], 0],
                       cmap='viridis', interpolation='nearest')

        # Overlay peak times
        ax1.plot(time_vector[peak_times], np.arange(len(peak_times)),
                'r--', linewidth=2, alpha=0.7, label='Peak times')

        ax1.set_ylabel('Neuron # (sorted by peak)', fontsize=12)
        ax1.set_title(f'Sequential Activity (Sequence Score: {sequence_score:.3f})',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')

        cbar = plt.colorbar(im, ax=ax1, label='Firing Rate (Hz)')

        # Peak time distribution
        ax2.bar(range(len(peak_times)), time_vector[peak_times],
               color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Neuron # (sorted)', fontsize=12)
        ax2.set_ylabel('Peak Time (ms)', fontsize=12)
        ax2.set_title('Peak Time Sequence', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_time_cells_summary(tuning_results: Dict,
                               time_cell_mask: np.ndarray,
                               figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Summary visualization of time cell properties.

        Args:
            tuning_results: Output from TemporalCodingAnalyzer
            time_cell_mask: Boolean mask indicating time cells
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        tmi = tuning_results['temporal_modulation_index']
        temp_info = tuning_results['temporal_information']
        peak_times = tuning_results['peak_times']

        # TMI distribution
        axes[0].hist(tmi[~time_cell_mask], bins=30, alpha=0.5,
                    label='Other neurons', color='gray', edgecolor='black')
        axes[0].hist(tmi[time_cell_mask], bins=30, alpha=0.7,
                    label='Time cells', color='red', edgecolor='black')
        axes[0].set_xlabel('Temporal Modulation Index', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('TMI Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Temporal information
        axes[1].hist(temp_info[~time_cell_mask], bins=30, alpha=0.5,
                    label='Other neurons', color='gray', edgecolor='black')
        axes[1].hist(temp_info[time_cell_mask], bins=30, alpha=0.7,
                    label='Time cells', color='red', edgecolor='black')
        axes[1].set_xlabel('Temporal Information (bits)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Information Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # Peak time distribution (time cells only)
        axes[2].hist(peak_times[time_cell_mask], bins=20, alpha=0.7,
                    color='steelblue', edgecolor='black')
        axes[2].set_xlabel('Peak Time (ms)', fontsize=11)
        axes[2].set_ylabel('Count', fontsize=11)
        axes[2].set_title(f'Time Cell Coverage ({np.sum(time_cell_mask)} cells)',
                         fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Time Cell Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution (dots per inch)
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to: {filename}")


if __name__ == "__main__":
    print("Neural Visualization Module")
    print("===========================")
    print("\nAvailable visualization tools:")
    print("  - plot_raster: Raster plots of neural activity")
    print("  - plot_temporal_tuning: Temporal tuning curves")
    print("  - plot_population_heatmap: Population activity heatmaps")
    print("  - plot_decoding_results: ML decoding performance")
    print("  - plot_3d_trajectory: 3D neural state space trajectories")
    print("  - plot_sequence_analysis: Sequential activity patterns")
    print("  - plot_time_cells_summary: Time cell identification summary")
    print("\nImport and use:")
    print("  from neural_visualization import NeuralVisualizer")
