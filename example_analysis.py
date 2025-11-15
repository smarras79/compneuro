"""
Comprehensive Hippocampus Neural Data Analysis Pipeline
========================================================
Example script demonstrating full analysis workflow for mental navigation data.

This script shows how to:
1. Load neural data from .mat files
2. Filter trials for mental navigation conditions
3. Analyze temporal coding and identify time cells
4. Decode landmark pairs using machine learning
5. Analyze population dynamics with dimensionality reduction
6. Detect sequential activity patterns
7. Generate comprehensive visualizations

Author: Computational Neuroscience Analysis Pipeline
Date: 2025-11-15
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings

# Import our analysis modules
from hippocampus_data_loader import HippocampusDataLoader, NeuralData
from neural_analysis_ai import (
    TemporalCodingAnalyzer,
    PopulationDecoder,
    NeuralManifoldAnalyzer,
    SequenceAnalyzer,
    print_analysis_summary
)
from neural_visualization import NeuralVisualizer, save_figure


def analyze_temporal_coding(neural_data: NeuralData,
                            trial_mask: np.ndarray,
                            output_dir: Path) -> Dict:
    """
    Perform temporal coding analysis and identify time cells.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trial selection
        output_dir: Directory for saving outputs

    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*70)
    print("  TEMPORAL CODING ANALYSIS")
    print("="*70)

    # Compute temporal tuning during mental navigation period
    analyzer = TemporalCodingAnalyzer()
    tuning_results = analyzer.compute_temporal_tuning(
        neural_data=neural_data,
        trial_mask=trial_mask,
        time_window=(0, 3000),  # Mental navigation period
        smooth_sigma=50.0
    )

    # Identify time cells
    time_cells = analyzer.identify_time_cells(
        tuning_results,
        tmi_threshold=0.3,
        info_threshold=0.1
    )

    n_time_cells = np.sum(time_cells)
    print(f"\n✓ Identified {n_time_cells}/{neural_data.n_neurons} time cells "
          f"({n_time_cells/neural_data.n_neurons*100:.1f}%)")

    # Statistics
    mean_tmi = np.mean(tuning_results['temporal_modulation_index'][time_cells])
    mean_info = np.mean(tuning_results['temporal_information'][time_cells])
    print(f"  Mean TMI (time cells): {mean_tmi:.3f}")
    print(f"  Mean temporal info: {mean_info:.3f} bits")

    # Visualizations
    print("\nGenerating visualizations...")

    # 1. Temporal tuning curves
    fig = NeuralVisualizer.plot_temporal_tuning(tuning_results, n_neurons=12)
    save_figure(fig, output_dir / "temporal_tuning_curves.png")
    plt.close()

    # 2. Population heatmap
    fig = NeuralVisualizer.plot_population_heatmap(
        tuning_results['tuning_curves'],
        tuning_results['time_vector'],
        sort_by='peak',
        title="Population Activity (sorted by peak time)"
    )
    save_figure(fig, output_dir / "population_heatmap.png")
    plt.close()

    # 3. Time cell summary
    fig = NeuralVisualizer.plot_time_cells_summary(tuning_results, time_cells)
    save_figure(fig, output_dir / "time_cells_summary.png")
    plt.close()

    print("✓ Temporal coding analysis complete")

    return {
        'tuning_results': tuning_results,
        'time_cells': time_cells,
        'n_time_cells': n_time_cells
    }


def analyze_population_decoding(neural_data: NeuralData,
                                trial_mask: np.ndarray,
                                output_dir: Path) -> Dict:
    """
    Decode behavioral variables from population activity.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trial selection
        output_dir: Directory for saving outputs

    Returns:
        Dictionary with decoding results
    """
    print("\n" + "="*70)
    print("  POPULATION DECODING ANALYSIS")
    print("="*70)

    results = {}

    # Test multiple decoding methods
    methods = ['bayesian', 'svm', 'random_forest', 'logistic']

    print("\nDecoding landmark pairs using different methods:")
    for method in methods:
        try:
            decoder = PopulationDecoder(method=method)
            decoding_results = decoder.decode_landmark_pairs(
                neural_data=neural_data,
                trial_mask=trial_mask,
                time_window=(500, 2000),  # During navigation
                cv_folds=5
            )

            accuracy = decoding_results['accuracy']
            chance = decoding_results['chance_level']
            n_pairs = decoding_results['n_pairs']

            print(f"  {method:15s}: {accuracy:.3f} (chance: {chance:.3f}, {n_pairs} pairs)")

            results[method] = decoding_results

            # Save confusion matrix for best performing method
            if method == 'random_forest':
                fig = NeuralVisualizer.plot_decoding_results(decoding_results)
                save_figure(fig, output_dir / f"decoding_{method}.png")
                plt.close()

        except Exception as e:
            print(f"  {method:15s}: Failed - {str(e)}")

    # Decode temporal distance (regression)
    print("\nDecoding temporal distance:")
    try:
        decoder = PopulationDecoder(method='bayesian')
        temporal_decoding = decoder.decode_temporal_distance(
            neural_data=neural_data,
            trial_mask=trial_mask,
            time_window=(500, 2000)
        )

        corr = temporal_decoding['correlation']
        rmse = temporal_decoding['rmse']
        print(f"  Correlation: {corr:.3f}")
        print(f"  RMSE: {rmse:.3f} seconds")

        results['temporal_distance'] = temporal_decoding

    except Exception as e:
        print(f"  Temporal distance decoding failed: {str(e)}")

    print("✓ Population decoding complete")

    return results


def analyze_neural_manifold(neural_data: NeuralData,
                            trial_mask: np.ndarray,
                            output_dir: Path) -> Dict:
    """
    Analyze neural population dynamics in low-dimensional manifolds.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trial selection
        output_dir: Directory for saving outputs

    Returns:
        Dictionary with manifold analysis results
    """
    print("\n" + "="*70)
    print("  NEURAL MANIFOLD ANALYSIS")
    print("="*70)

    results = {}

    # PCA analysis
    print("\nPCA dimensionality reduction:")
    pca_analyzer = NeuralManifoldAnalyzer(method='pca', n_components=10)
    pca_results = pca_analyzer.fit_transform(
        neural_data=neural_data,
        trial_mask=trial_mask,
        time_window=(0, 3000)
    )

    var_exp = pca_results['variance_explained']
    cum_var = np.cumsum(var_exp)
    print(f"  Variance explained by first 3 PCs: {cum_var[2]*100:.1f}%")
    print(f"  Variance explained by first 5 PCs: {cum_var[4]*100:.1f}%")

    # Visualize PCA variance
    fig = NeuralVisualizer.plot_pca_variance(pca_results)
    save_figure(fig, output_dir / "pca_variance.png")
    plt.close()

    results['pca'] = pca_results

    # 3D trajectory visualization
    print("\nVisualizing neural trajectories in 3D state space:")
    trajectory_analyzer = NeuralManifoldAnalyzer(method='pca', n_components=3)
    traj_results = trajectory_analyzer.fit_transform(
        neural_data=neural_data,
        trial_mask=trial_mask,
        time_window=(0, 3000)
    )

    # Get landmark pair labels for coloring
    curr = neural_data.get_condition('curr')[trial_mask].astype(int)
    target = neural_data.get_condition('target')[trial_mask].astype(int)
    pair_labels = curr * 10 + target

    try:
        fig = NeuralVisualizer.plot_3d_trajectory(
            traj_results,
            pair_labels,
            n_conditions=6
        )
        save_figure(fig, output_dir / "neural_trajectories_3d.png")
        plt.close()
        print("  ✓ 3D trajectory plot saved")
    except Exception as e:
        print(f"  Warning: Could not create 3D plot - {str(e)}")

    # Compute trajectory similarity
    print("\nComputing trajectory similarity across landmark pairs:")
    similarity = trajectory_analyzer.compute_trajectory_similarity(
        traj_results['trajectories'],
        pair_labels
    )

    results['trajectories'] = traj_results
    results['similarity'] = similarity

    print("✓ Neural manifold analysis complete")

    return results


def analyze_sequences(neural_data: NeuralData,
                     trial_mask: np.ndarray,
                     output_dir: Path) -> Dict:
    """
    Detect and analyze sequential neural activity patterns.

    Args:
        neural_data: NeuralData object
        trial_mask: Boolean mask for trial selection
        output_dir: Directory for saving outputs

    Returns:
        Dictionary with sequence analysis results
    """
    print("\n" + "="*70)
    print("  SEQUENCE ANALYSIS")
    print("="*70)

    analyzer = SequenceAnalyzer()

    # Detect sequences during mental navigation
    sequence_results = analyzer.detect_sequences(
        neural_data=neural_data,
        trial_mask=trial_mask,
        time_window=(0, 3000),
        min_neurons=5
    )

    seq_score = sequence_results['sequence_score']
    print(f"\nSequence score: {seq_score:.3f}")

    if seq_score > 0.3:
        print("  ✓ Significant sequential activity detected!")
    else:
        print("  ○ Weak sequential structure")

    # Visualize sequence
    time_vec = neural_data.time_vector[
        (neural_data.time_vector >= 0) & (neural_data.time_vector <= 3000)
    ]

    fig = NeuralVisualizer.plot_sequence_analysis(sequence_results, time_vec)
    save_figure(fig, output_dir / "sequence_analysis.png")
    plt.close()

    print("✓ Sequence analysis complete")

    return sequence_results


def run_full_analysis(mat_file: Path, output_dir: Path):
    """
    Run complete analysis pipeline on a single .mat file.

    Args:
        mat_file: Path to .mat file
        output_dir: Directory for saving outputs
    """
    print("\n" + "="*70)
    print("  HIPPOCAMPUS NEURAL DATA ANALYSIS PIPELINE")
    print("="*70)
    print(f"\nData file: {mat_file.name}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading neural data")
    print("-"*70)

    try:
        neural_data = HippocampusDataLoader.load_mat_file(mat_file)
        print("\n" + neural_data.summary())
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # ========================================================================
    # STEP 2: Filter trials for mental navigation
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Filtering trials for mental navigation")
    print("-"*70)

    # Get mental navigation trials (fully occluded, normal speed, first attempt)
    mn_trials = neural_data.get_mental_navigation_trials()
    n_mn_trials = np.sum(mn_trials)

    print(f"\nMental navigation trials: {n_mn_trials}/{neural_data.n_trials}")

    if n_mn_trials < 10:
        print("Warning: Very few mental navigation trials. Results may be unreliable.")

    # Get landmark pair distribution
    curr = neural_data.get_condition('curr')[mn_trials]
    target = neural_data.get_condition('target')[mn_trials]
    unique_pairs = np.unique(list(zip(curr, target)), axis=0)

    print(f"Unique landmark pairs: {len(unique_pairs)}")
    print("Landmark pair distribution:")
    for pair in unique_pairs[:10]:  # Show first 10
        count = np.sum((curr == pair[0]) & (target == pair[1]))
        print(f"  {int(pair[0])} → {int(pair[1])}: {count} trials")

    # ========================================================================
    # STEP 3: Temporal coding analysis
    # ========================================================================
    temporal_results = analyze_temporal_coding(neural_data, mn_trials, output_dir)

    # ========================================================================
    # STEP 4: Population decoding
    # ========================================================================
    decoding_results = analyze_population_decoding(neural_data, mn_trials, output_dir)

    # ========================================================================
    # STEP 5: Neural manifold analysis
    # ========================================================================
    manifold_results = analyze_neural_manifold(neural_data, mn_trials, output_dir)

    # ========================================================================
    # STEP 6: Sequence analysis
    # ========================================================================
    sequence_results = analyze_sequences(neural_data, mn_trials, output_dir)

    # ========================================================================
    # STEP 7: Generate summary report
    # ========================================================================
    print("\n" + "="*70)
    print("  ANALYSIS SUMMARY")
    print("="*70)

    summary_lines = [
        f"\nDataset: {mat_file.name}",
        f"Total neurons: {neural_data.n_neurons}",
        f"Mental navigation trials: {n_mn_trials}",
        f"",
        "Key Findings:",
        f"  - Time cells: {temporal_results['n_time_cells']} ({temporal_results['n_time_cells']/neural_data.n_neurons*100:.1f}%)",
        f"  - Sequence score: {sequence_results['sequence_score']:.3f}",
    ]

    # Add best decoding performance
    if decoding_results:
        best_method = max(decoding_results.keys(),
                         key=lambda k: decoding_results[k].get('accuracy', 0)
                         if isinstance(decoding_results[k], dict) and 'accuracy' in decoding_results[k]
                         else 0)
        if best_method != 'temporal_distance':
            best_acc = decoding_results[best_method]['accuracy']
            summary_lines.append(f"  - Best decoding accuracy: {best_acc:.3f} ({best_method})")

    # Add PCA variance
    if 'pca' in manifold_results:
        cum_var = np.cumsum(manifold_results['pca']['variance_explained'])
        summary_lines.append(f"  - PCA (3 components): {cum_var[2]*100:.1f}% variance")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary to file
    with open(output_dir / "analysis_summary.txt", 'w') as f:
        f.write(summary_text)

    print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
    print("="*70)


def main():
    """Main entry point for analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze hippocampal neural data from mental navigation task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python example_analysis.py data/session1.mat

  # Specify output directory
  python example_analysis.py data/session1.mat --output results/session1

  # Analyze multiple files in a directory
  python example_analysis.py data/ --pattern "*.mat"
        """
    )

    parser.add_argument('input_path',
                       type=str,
                       help='Path to .mat file or directory containing .mat files')

    parser.add_argument('--output', '-o',
                       type=str,
                       default='results',
                       help='Output directory for results (default: results/)')

    parser.add_argument('--pattern', '-p',
                       type=str,
                       default='*.mat',
                       help='File pattern for directory input (default: *.mat)')

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_base = Path(args.output)

    # Check if input is file or directory
    if input_path.is_file():
        # Single file analysis
        output_dir = output_base / input_path.stem
        run_full_analysis(input_path, output_dir)

    elif input_path.is_dir():
        # Multiple files analysis
        mat_files = sorted(input_path.glob(args.pattern))

        if not mat_files:
            print(f"Error: No files matching '{args.pattern}' found in {input_path}")
            return

        print(f"\nFound {len(mat_files)} .mat files to analyze")

        for mat_file in mat_files:
            output_dir = output_base / mat_file.stem
            try:
                run_full_analysis(mat_file, output_dir)
            except Exception as e:
                print(f"\nError analyzing {mat_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    else:
        print(f"Error: {input_path} is neither a file nor a directory")


if __name__ == "__main__":
    main()
