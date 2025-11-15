"""
Neural Visualization Module
Publication-quality plots for neural data analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, Dict, Any
from pathlib import Path

# Set style
sns.set_style('white')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'


class NeuralVisualizer:
    """Static methods for neural data visualization."""
    
    @staticmethod
    def plot_raster(activity: np.ndarray,
                   time_vector: np.ndarray,
                   trial_idx: int = 0,
                   threshold: float = 5.0,
                   figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot raster plot for a single trial.
        
        Args:
            activity: (neurons, time, trials) array
            time_vector: Time points in ms
            trial_idx: Which trial to plot
            threshold: Firing rate threshold for spike display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        trial_data = activity[:, :, trial_idx]
        n_neurons, n_time = trial_data.shape
        
        # Plot spikes
        for i in range(n_neurons):
            spike_times = time_vector[trial_data[i, :] > threshold]
            ax.scatter(spike_times, np.ones_like(spike_times) * i, 
                      c='black', s=1, marker='|')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron')
        ax.set_title(f'Raster Plot - Trial {trial_idx}')
        ax.set_ylim(-1, n_neurons)
        
        sns.despine()
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_temporal_tuning(tuning_results: Dict[str, Any],
                            n_neurons: int = 12,
                            figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Plot temporal tuning curves for top neurons.
        
        Args:
            tuning_results: Output from TemporalCodingAnalyzer
            n_neurons: Number of neurons to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        tuning_curves = tuning_results['tuning_curves']
        time_vec = tuning_results['time_vector']
        tmi = tuning_results['tmi']
        
        # Select top neurons by TMI
        top_idx = np.argsort(tmi)[-n_neurons:]
        
        # Create subplots
        n_rows = int(np.ceil(n_neurons / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, neuron_idx in enumerate(top_idx):
            ax = axes[i]
            ax.plot(time_vec, tuning_curves[neuron_idx, :], 'k-', linewidth=2)
            ax.fill_between(time_vec, 0, tuning_curves[neuron_idx, :], alpha=0.3)
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'Neuron {neuron_idx} (TMI={tmi[neuron_idx]:.2f})')
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)
        
        # Hide unused subplots
        for i in range(len(top_idx), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_population_heatmap(activity: np.ndarray,
                               time_vector: np.ndarray,
                               sort_by: str = 'peak',
                               figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot population activity heatmap.
        
        Args:
            activity: (neurons, time) array
            time_vector: Time points
            sort_by: 'peak' or 'none'
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if sort_by == 'peak':
            # Sort by peak time
            peak_times = np.argmax(activity, axis=1)
            sort_idx = np.argsort(peak_times)
            activity_sorted = activity[sort_idx, :]
        else:
            activity_sorted = activity
        
        # Normalize each neuron
        activity_norm = np.zeros_like(activity_sorted)
        for i in range(activity_sorted.shape[0]):
            max_val = np.max(activity_sorted[i, :])
            if max_val > 0:
                activity_norm[i, :] = activity_sorted[i, :] / max_val
        
        im = ax.imshow(activity_norm, aspect='auto', cmap='hot', 
                       extent=[time_vector[0], time_vector[-1], 0, activity.shape[0]])
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron (sorted by peak time)')
        ax.set_title('Population Activity Heatmap')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Firing Rate')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_cells_summary(tuning_results: Dict[str, Any],
                                time_cells: np.ndarray,
                                figsize: tuple = (14, 5)) -> plt.Figure:
        """
        Plot summary of time cell identification.
        
        Args:
            tuning_results: Output from TemporalCodingAnalyzer
            time_cells: Boolean array of time cells
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, figure=fig)
        
        # TMI distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(tuning_results['tmi'][~time_cells], bins=30, alpha=0.5, 
                label='Non-time cells', color='gray')
        ax1.hist(tuning_results['tmi'][time_cells], bins=30, alpha=0.5,
                label='Time cells', color='red')
        ax1.set_xlabel('Temporal Modulation Index')
        ax1.set_ylabel('Count')
        ax1.set_title('TMI Distribution')
        ax1.legend()
        sns.despine(ax=ax1)
        
        # Temporal information distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(tuning_results['temporal_info'][~time_cells], bins=30, alpha=0.5,
                label='Non-time cells', color='gray')
        ax2.hist(tuning_results['temporal_info'][time_cells], bins=30, alpha=0.5,
                label='Time cells', color='red')
        ax2.set_xlabel('Temporal Information (bits)')
        ax2.set_ylabel('Count')
        ax2.set_title('Information Distribution')
        ax2.legend()
        sns.despine(ax=ax2)
        
        # Peak time distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(tuning_results['peak_times_ms'][time_cells], bins=20, color='red', alpha=0.7)
        ax3.set_xlabel('Peak Time (ms)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Time Cell Peak Times (n={np.sum(time_cells)})')
        sns.despine(ax=ax3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_decoding_results(results: Dict[str, Any],
                             figsize: tuple = (12, 5)) -> plt.Figure:
        """
        Plot decoding results with confusion matrix.
        
        Args:
            results: Output from PopulationDecoder
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, figure=fig)
        
        # Accuracy bar plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar([0, 1], [results['accuracy'], results['chance_level']], 
               color=['steelblue', 'gray'], alpha=0.7)
        ax1.errorbar([0], [results['accuracy']], yerr=[results['std']], 
                    fmt='none', color='black', capsize=5)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Decoder', 'Chance'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Decoding Performance ({results["method"]})')
        ax1.set_ylim([0, 1])
        ax1.axhline(results['chance_level'], color='gray', linestyle='--', alpha=0.5)
        sns.despine(ax=ax1)
        
        # Confusion matrix
        ax2 = fig.add_subplot(gs[0, 1])
        conf_mat = results['confusion_matrix']
        
        # Normalize
        conf_mat_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        im = ax2.imshow(conf_mat_norm, cmap='Blues', aspect='auto')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Confusion Matrix (Normalized)')
        
        plt.colorbar(im, ax=ax2, label='Proportion')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_3d_trajectory(manifold_results: Dict[str, Any],
                          condition_labels: np.ndarray,
                          figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Plot 3D neural trajectories.
        
        Args:
            manifold_results: Output from NeuralManifoldAnalyzer
            condition_labels: Condition label for each trial
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        trajectories = manifold_results['trajectories']
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each trajectory
        unique_conds = np.unique(condition_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_conds)))
        
        for i, cond in enumerate(unique_conds):
            cond_mask = condition_labels == cond
            cond_trajectories = trajectories[cond_mask, :, :3]
            
            # Plot mean trajectory
            mean_traj = np.mean(cond_trajectories, axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                   color=colors[i], linewidth=2, label=f'Condition {int(cond)}')
            
            # Plot start point
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1], mean_traj[0, 2],
                      color=colors[i], s=100, marker='o', edgecolor='black')
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title('Neural State Space Trajectories')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_sequence_analysis(sequence_results: Dict[str, Any],
                              time_vector: np.ndarray,
                              figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot sequence analysis results.
        
        Args:
            sequence_results: Output from SequenceAnalyzer
            time_vector: Time vector
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Sorted activity heatmap
        ax1 = axes[0]
        sorted_activity = sequence_results['sorted_activity']
        
        # Normalize
        sorted_norm = np.zeros_like(sorted_activity)
        for i in range(sorted_activity.shape[0]):
            max_val = np.max(sorted_activity[i, :])
            if max_val > 0:
                sorted_norm[i, :] = sorted_activity[i, :] / max_val
        
        im = ax1.imshow(sorted_norm, aspect='auto', cmap='hot',
                       extent=[time_vector[0], time_vector[-1], 0, sorted_activity.shape[0]])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Neuron (sorted by peak)')
        ax1.set_title(f'Sequential Activity (score={sequence_results["sequence_score"]:.3f})')
        plt.colorbar(im, ax=ax1, label='Normalized FR')
        
        # Peak times
        ax2 = axes[1]
        sorted_peaks = sequence_results['peak_times'][sequence_results['sorted_neurons']]
        ax2.plot(sorted_peaks, 'o-', color='steelblue', markersize=4)
        ax2.set_xlabel('Neuron Index (sorted)')
        ax2.set_ylabel('Peak Time (ms)')
        ax2.set_title('Peak Times of Sorted Neurons')
        ax2.grid(True, alpha=0.3)
        sns.despine(ax=ax2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pca_variance(manifold_results: Dict[str, Any],
                         figsize: tuple = (10, 5)) -> plt.Figure:
        """
        Plot PCA variance explained.
        
        Args:
            manifold_results: Output from NeuralManifoldAnalyzer
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if manifold_results['variance_explained'] is None:
            print("No variance explained available (not PCA)")
            return None
        
        var_exp = manifold_results['variance_explained']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        ax1 = axes[0]
        ax1.bar(range(1, len(var_exp) + 1), var_exp * 100, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Individual Variance Explained')
        sns.despine(ax=ax1)
        
        # Cumulative variance
        ax2 = axes[1]
        cum_var = np.cumsum(var_exp) * 100
        ax2.plot(range(1, len(cum_var) + 1), cum_var, 'o-', 
                color='steelblue', markersize=6)
        ax2.axhline(90, color='red', linestyle='--', alpha=0.5, label='90%')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        sns.despine(ax=ax2)
        
        plt.tight_layout()
        return fig


def save_figure(fig: plt.Figure, 
                filepath: str, 
                dpi: int = 300,
                bbox_inches: str = 'tight'):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Output path
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"✓ Saved figure: {filepath}")
    plt.close(fig)


if __name__ == "__main__":
    print("Neural Visualization Module")
    print("Import this module to use visualization functions:")
    print("  - NeuralVisualizer (static methods)")
    print("  - save_figure()")
