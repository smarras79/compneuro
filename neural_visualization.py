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
    
    @staticmethod
    def plot_ta_vs_tp(neural_data,
                     trial_mask: Optional[np.ndarray] = None,
                     figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Plot True Temporal Distance (TA) vs Produced Temporal Distance (TP).
        
        This creates a benchmark plot showing how accurately the subject produced
        temporal intervals. Points along the diagonal indicate perfect performance.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Optional boolean mask to filter trials
                       If None, uses column 10 == 1 from cond_matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get TA and TP from condition matrix
        ta = neural_data.cond_matrix[:, 0]  # Column 1 (0-indexed)
        tp = neural_data.cond_matrix[:, 1]  # Column 2 (0-indexed)
        
        # Apply trial mask
        if trial_mask is None:
            # Default: use column 10 == 1 (9 in 0-indexed)
            if neural_data.cond_matrix.shape[1] > 9:
                trial_mask = neural_data.cond_matrix[:, 9] == 1
                print(f"Using default mask: column 10 == 1 ({np.sum(trial_mask)} trials)")
            else:
                trial_mask = np.ones(len(ta), dtype=bool)
                print(f"Column 10 not found, using all {len(ta)} trials")
        
        # Filter data
        ta_filtered = ta[trial_mask]
        tp_filtered = tp[trial_mask]
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Plot data points
        scatter = ax_main.scatter(ta_filtered, tp_filtered, 
                                 alpha=0.6, s=50, c='steelblue', 
                                 edgecolors='black', linewidth=0.5)
        
        # Add unity line (perfect performance)
        min_val = min(np.min(ta_filtered), np.min(tp_filtered))
        max_val = max(np.max(ta_filtered), np.max(tp_filtered))
        ax_main.plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, alpha=0.7, label='Unity (Perfect)')
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(ta_filtered, tp_filtered)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax_main.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.7,
                    label=f'Fit: y={slope:.2f}x+{intercept:.2f}\n$r$={r_value:.3f}, $p$<{p_value:.1e}')
        
        # Calculate error metrics
        error = tp_filtered - ta_filtered
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        
        # Labels and title
        ax_main.set_xlabel('True Temporal Distance (TA) [s]', fontsize=12)
        ax_main.set_ylabel('Produced Temporal Distance (TP) [s]', fontsize=12)
        ax_main.set_title(f'Temporal Production Performance\n'
                         f'n={len(ta_filtered)} trials, MAE={mae:.3f}s, RMSE={rmse:.3f}s',
                         fontsize=13)
        ax_main.legend(loc='upper left', frameon=True, fontsize=10)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal', adjustable='box')
        sns.despine(ax=ax_main)
        
        # Marginal histogram for TA (bottom)
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_bottom.hist(ta_filtered, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax_bottom.set_xlabel('TA [s]', fontsize=10)
        ax_bottom.set_ylabel('Count', fontsize=10)
        ax_bottom.set_title('TA Distribution', fontsize=10)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        sns.despine(ax=ax_bottom)
        
        # Marginal histogram for TP (right)
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
        ax_right.hist(tp_filtered, bins=30, orientation='horizontal', 
                     color='steelblue', alpha=0.7, edgecolor='black')
        ax_right.set_ylabel('TP [s]', fontsize=10)
        ax_right.set_xlabel('Count', fontsize=10)
        ax_right.set_title('TP Distribution', fontsize=10, rotation=270, pad=15)
        plt.setp(ax_main.get_yticklabels(), visible=False)
        sns.despine(ax=ax_right)
        
        # Error histogram (bottom right)
        ax_error = fig.add_subplot(gs[1, 1])
        ax_error.hist(error, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax_error.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax_error.set_xlabel('Error (TP-TA) [s]', fontsize=9)
        ax_error.set_ylabel('Count', fontsize=9)
        ax_error.set_title(f'Error Distribution\nMean={np.mean(error):.3f}s', fontsize=9)
        sns.despine(ax=ax_error)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_ta_vs_tp_detailed(neural_data,
                               trial_mask: Optional[np.ndarray] = None,
                               color_by: Optional[str] = None,
                               figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Detailed TA vs TP plot with additional analyses.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Optional boolean mask to filter trials
            color_by: Optional condition to color points by ('curr', 'target', 'succ', etc.)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get TA and TP
        ta = neural_data.cond_matrix[:, 0]
        tp = neural_data.cond_matrix[:, 1]
        
        # Apply trial mask
        if trial_mask is None:
            if neural_data.cond_matrix.shape[1] > 9:
                trial_mask = neural_data.cond_matrix[:, 9] == 1
            else:
                trial_mask = np.ones(len(ta), dtype=bool)
        
        ta_filtered = ta[trial_mask]
        tp_filtered = tp[trial_mask]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Main scatter plot with optional coloring
        ax1 = fig.add_subplot(gs[0, :2])
        
        if color_by is not None:
            try:
                color_data = neural_data.get_condition(color_by)[trial_mask]
                scatter = ax1.scatter(ta_filtered, tp_filtered, 
                                    c=color_data, cmap='tab10',
                                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax1, label=color_by.upper())
            except:
                print(f"Warning: Could not color by '{color_by}', using default")
                ax1.scatter(ta_filtered, tp_filtered, 
                           alpha=0.6, s=50, c='steelblue',
                           edgecolors='black', linewidth=0.5)
        else:
            ax1.scatter(ta_filtered, tp_filtered, 
                       alpha=0.6, s=50, c='steelblue',
                       edgecolors='black', linewidth=0.5)
        
        # Unity line
        min_val = min(np.min(ta_filtered), np.min(tp_filtered))
        max_val = max(np.max(ta_filtered), np.max(tp_filtered))
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, alpha=0.7, label='Unity')
        
        # Regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(ta_filtered, tp_filtered)
        line_x = np.array([min_val, max_val])
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.7,
                label=f'Fit: r={r_value:.3f}')
        
        ax1.set_xlabel('True Temporal Distance (TA) [s]')
        ax1.set_ylabel('Produced Temporal Distance (TP) [s]')
        ax1.set_title(f'TA vs TP (n={len(ta_filtered)} trials)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        sns.despine(ax=ax1)
        
        # Error vs TA
        ax2 = fig.add_subplot(gs[0, 2])
        error = tp_filtered - ta_filtered
        ax2.scatter(ta_filtered, error, alpha=0.5, s=30, c='coral')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('TA [s]')
        ax2.set_ylabel('Error (TP-TA) [s]')
        ax2.set_title('Error vs TA')
        ax2.grid(True, alpha=0.3)
        sns.despine(ax=ax2)
        
        # Relative error
        ax3 = fig.add_subplot(gs[1, 0])
        relative_error = (tp_filtered - ta_filtered) / ta_filtered * 100
        ax3.hist(relative_error, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Relative Error (%)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Relative Error\nMedian={np.median(relative_error):.1f}%')
        sns.despine(ax=ax3)
        
        # Absolute error
        ax4 = fig.add_subplot(gs[1, 1])
        abs_error = np.abs(error)
        ax4.hist(abs_error, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Absolute Error [s]')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Absolute Error\nMAE={np.mean(abs_error):.3f}s')
        sns.despine(ax=ax4)
        
        # Statistics table
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate statistics
        stats_text = f"""
        PERFORMANCE STATISTICS
        {'='*30}
        N trials:        {len(ta_filtered)}
        
        Correlation:     {r_value:.4f}
        Slope:           {slope:.4f}
        Intercept:       {intercept:.4f}
        p-value:         {p_value:.2e}
        
        Mean Error:      {np.mean(error):.4f} s
        Std Error:       {np.std(error):.4f} s
        MAE:             {np.mean(abs_error):.4f} s
        RMSE:            {np.sqrt(np.mean(error**2)):.4f} s
        
        Median Rel Err:  {np.median(relative_error):.2f}%
        
        TA Range:        [{np.min(ta_filtered):.2f}, {np.max(ta_filtered):.2f}] s
        TP Range:        [{np.min(tp_filtered):.2f}, {np.max(tp_filtered):.2f}] s
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rt_vs_ta(neural_data,
                     trial_mask: Optional[np.ndarray] = None,
                     rt_column: int = 11,
                     figsize: tuple = (10, 8)) -> plt.Figure:
        """
        Plot Reaction Time (RT) vs True Temporal Distance (TA).
        
        This analyzes how response time varies with the temporal interval.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Optional boolean mask to filter trials
                       If None, uses column 10 == 1 from cond_matrix
            rt_column: Column index for reaction time (default: 11, which is column 12)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get TA and RT from condition matrix
        ta = neural_data.cond_matrix[:, 0]  # Column 1 (0-indexed)
        
        # Check if RT column exists
        if neural_data.cond_matrix.shape[1] <= rt_column:
            raise ValueError(
                f"RT column {rt_column} (0-indexed) not found in condition matrix.\n"
                f"Condition matrix has {neural_data.cond_matrix.shape[1]} columns.\n"
                f"Please specify the correct column index using rt_column parameter."
            )
        
        rt = neural_data.cond_matrix[:, rt_column]
        
        # Apply trial mask
        if trial_mask is None:
            # Default: use column 10 == 1 (9 in 0-indexed)
            if neural_data.cond_matrix.shape[1] > 9:
                trial_mask = neural_data.cond_matrix[:, 9] == 1
                print(f"Using default mask: column 10 == 1 ({np.sum(trial_mask)} trials)")
            else:
                trial_mask = np.ones(len(ta), dtype=bool)
                print(f"Column 10 not found, using all {len(ta)} trials")
        
        # Filter data and remove invalid RT values (NaN, negative, or extremely large)
        ta_filtered = ta[trial_mask]
        rt_filtered = rt[trial_mask]
        
        # Remove invalid RTs
        valid_rt = ~np.isnan(rt_filtered) & (rt_filtered > 0) & (rt_filtered < 100)
        ta_filtered = ta_filtered[valid_rt]
        rt_filtered = rt_filtered[valid_rt]
        
        if len(rt_filtered) == 0:
            raise ValueError("No valid reaction time data found. Check RT column index.")
        
        print(f"Valid RT trials: {len(rt_filtered)} / {np.sum(trial_mask)}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # Main scatter plot
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Plot data points
        scatter = ax_main.scatter(ta_filtered, rt_filtered, 
                                 alpha=0.6, s=50, c='darkgreen', 
                                 edgecolors='black', linewidth=0.5)
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(ta_filtered, rt_filtered)
        line_x = np.array([np.min(ta_filtered), np.max(ta_filtered)])
        line_y = slope * line_x + intercept
        ax_main.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.7,
                    label=f'Fit: RT={slope:.3f}*TA+{intercept:.3f}\n$r$={r_value:.3f}, $p$={p_value:.1e}')
        
        # Add horizontal line at mean RT
        mean_rt = np.mean(rt_filtered)
        ax_main.axhline(mean_rt, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
                       label=f'Mean RT={mean_rt:.3f}s')
        
        # Calculate CV (coefficient of variation)
        cv_rt = np.std(rt_filtered) / np.mean(rt_filtered)
        
        # Labels and title
        ax_main.set_xlabel('True Temporal Distance (TA) [s]', fontsize=12)
        ax_main.set_ylabel('Reaction Time (RT) [s]', fontsize=12)
        ax_main.set_title(f'Reaction Time vs Temporal Distance\n'
                         f'n={len(rt_filtered)} trials, Mean RT={mean_rt:.3f}s, CV={cv_rt:.3f}',
                         fontsize=13)
        ax_main.legend(loc='best', frameon=True, fontsize=10)
        ax_main.grid(True, alpha=0.3)
        sns.despine(ax=ax_main)
        
        # Marginal histogram for TA (bottom)
        ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_bottom.hist(ta_filtered, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
        ax_bottom.set_xlabel('TA [s]', fontsize=10)
        ax_bottom.set_ylabel('Count', fontsize=10)
        ax_bottom.set_title('TA Distribution', fontsize=10)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        sns.despine(ax=ax_bottom)
        
        # Marginal histogram for RT (right)
        ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
        ax_right.hist(rt_filtered, bins=30, orientation='horizontal', 
                     color='darkgreen', alpha=0.7, edgecolor='black')
        ax_right.set_ylabel('RT [s]', fontsize=10)
        ax_right.set_xlabel('Count', fontsize=10)
        ax_right.set_title('RT Distribution', fontsize=10, rotation=270, pad=15)
        plt.setp(ax_main.get_yticklabels(), visible=False)
        sns.despine(ax=ax_right)
        
        # Statistics box (bottom right)
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.axis('off')
        
        # Calculate additional statistics
        stats_text = f"""RT STATISTICS
{'='*20}
Mean: {np.mean(rt_filtered):.3f} s
Std:  {np.std(rt_filtered):.3f} s
Median: {np.median(rt_filtered):.3f} s
CV:   {cv_rt:.3f}

Range: [{np.min(rt_filtered):.3f}, 
        {np.max(rt_filtered):.3f}] s

Correlation: {r_value:.3f}
Slope: {slope:.3f} s/s
        """
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='top', 
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rt_vs_ta_detailed(neural_data,
                               trial_mask: Optional[np.ndarray] = None,
                               rt_column: int = 11,
                               color_by: Optional[str] = None,
                               figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Detailed RT vs TA plot with additional analyses.
        
        Args:
            neural_data: HippocampusDataLoader instance
            trial_mask: Optional boolean mask to filter trials
            rt_column: Column index for reaction time (default: 11)
            color_by: Optional condition to color points by ('curr', 'target', 'succ', etc.)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get TA and RT
        ta = neural_data.cond_matrix[:, 0]
        
        if neural_data.cond_matrix.shape[1] <= rt_column:
            raise ValueError(f"RT column {rt_column} not found. Matrix has {neural_data.cond_matrix.shape[1]} columns.")
        
        rt = neural_data.cond_matrix[:, rt_column]
        
        # Apply trial mask
        if trial_mask is None:
            if neural_data.cond_matrix.shape[1] > 9:
                trial_mask = neural_data.cond_matrix[:, 9] == 1
            else:
                trial_mask = np.ones(len(ta), dtype=bool)
        
        ta_filtered = ta[trial_mask]
        rt_filtered = rt[trial_mask]
        
        # Remove invalid RTs
        valid_rt = ~np.isnan(rt_filtered) & (rt_filtered > 0) & (rt_filtered < 100)
        ta_filtered = ta_filtered[valid_rt]
        rt_filtered = rt_filtered[valid_rt]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)
        
        # Main scatter plot with optional coloring
        ax1 = fig.add_subplot(gs[0, :2])
        
        if color_by is not None:
            try:
                # Get color data with same filtering
                color_data_full = neural_data.get_condition(color_by)[trial_mask]
                color_data = color_data_full[valid_rt]
                scatter = ax1.scatter(ta_filtered, rt_filtered, 
                                    c=color_data, cmap='tab10',
                                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax1, label=color_by.upper())
            except:
                print(f"Warning: Could not color by '{color_by}', using default")
                ax1.scatter(ta_filtered, rt_filtered, 
                           alpha=0.6, s=50, c='darkgreen',
                           edgecolors='black', linewidth=0.5)
        else:
            ax1.scatter(ta_filtered, rt_filtered, 
                       alpha=0.6, s=50, c='darkgreen',
                       edgecolors='black', linewidth=0.5)
        
        # Regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(ta_filtered, rt_filtered)
        line_x = np.array([np.min(ta_filtered), np.max(ta_filtered)])
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.7,
                label=f'Fit: r={r_value:.3f}')
        
        # Mean RT line
        mean_rt = np.mean(rt_filtered)
        ax1.axhline(mean_rt, color='blue', linestyle='--', alpha=0.5,
                   label=f'Mean RT={mean_rt:.3f}s')
        
        ax1.set_xlabel('True Temporal Distance (TA) [s]')
        ax1.set_ylabel('Reaction Time (RT) [s]')
        ax1.set_title(f'RT vs TA (n={len(rt_filtered)} trials)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        sns.despine(ax=ax1)
        
        # RT variability vs TA (binned analysis)
        ax2 = fig.add_subplot(gs[0, 2])
        # Bin TA and compute RT mean and std for each bin
        n_bins = 10
        ta_bins = np.linspace(np.min(ta_filtered), np.max(ta_filtered), n_bins+1)
        bin_centers = (ta_bins[:-1] + ta_bins[1:]) / 2
        bin_means = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (ta_filtered >= ta_bins[i]) & (ta_filtered < ta_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(rt_filtered[mask]))
                bin_stds.append(np.std(rt_filtered[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                    color='darkgreen', capsize=5, capthick=2, alpha=0.7)
        ax2.set_xlabel('TA [s]')
        ax2.set_ylabel('RT [s]')
        ax2.set_title('RT Mean ± SD (binned)')
        ax2.grid(True, alpha=0.3)
        sns.despine(ax=ax2)
        
        # RT distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(rt_filtered, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
        ax3.axvline(mean_rt, color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.axvline(np.median(rt_filtered), color='orange', linestyle='--', linewidth=2, label='Median')
        ax3.set_xlabel('Reaction Time [s]')
        ax3.set_ylabel('Count')
        ax3.set_title('RT Distribution')
        ax3.legend()
        sns.despine(ax=ax3)
        
        # RT vs trial number (check for fatigue/learning)
        ax4 = fig.add_subplot(gs[1, 1])
        trial_numbers = np.arange(len(rt_filtered))
        ax4.scatter(trial_numbers, rt_filtered, alpha=0.3, s=20, c='darkgreen')
        # Add moving average
        window = min(50, len(rt_filtered) // 10)
        if window > 1:
            rt_ma = np.convolve(rt_filtered, np.ones(window)/window, mode='valid')
            ax4.plot(np.arange(window//2, len(rt_filtered)-window//2+1), rt_ma, 
                    'r-', linewidth=2, label=f'Moving avg (n={window})')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('RT [s]')
        ax4.set_title('RT Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        sns.despine(ax=ax4)
        
        # Statistics table
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calculate statistics
        cv = np.std(rt_filtered) / np.mean(rt_filtered)
        
        stats_text = f"""
        RT STATISTICS
        {'='*30}
        N trials:        {len(rt_filtered)}
        
        Mean RT:         {np.mean(rt_filtered):.4f} s
        Std RT:          {np.std(rt_filtered):.4f} s
        Median RT:       {np.median(rt_filtered):.4f} s
        CV:              {cv:.4f}
        
        Min RT:          {np.min(rt_filtered):.4f} s
        Max RT:          {np.max(rt_filtered):.4f} s
        Range:           {np.ptp(rt_filtered):.4f} s
        
        TA RELATIONSHIP
        {'='*30}
        Correlation:     {r_value:.4f}
        Slope:           {slope:.4f} s/s
        Intercept:       {intercept:.4f} s
        p-value:         {p_value:.2e}
        """
        
        ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
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
