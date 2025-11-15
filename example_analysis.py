#!/usr/bin/env python3
"""
Complete Hippocampus Neural Analysis Pipeline
Analyzes neural data from mental navigation tasks
"""
import sys
import numpy as np
from pathlib import Path
import argparse

from hippocampus_data_loader import HippocampusDataLoader
from neural_analysis_ai import (
    TemporalCodingAnalyzer,
    PopulationDecoder,
    NeuralManifoldAnalyzer,
    SequenceAnalyzer
)
from neural_visualization import NeuralVisualizer, save_figure


def run_complete_analysis(data_file: str, output_dir: str = 'results'):
    """
    Run complete neural analysis pipeline.
    
    Args:
        data_file: Path to .mat file
        output_dir: Directory for results
    """
    print("\n" + "="*70)
    print("HIPPOCAMPUS NEURAL ANALYSIS PIPELINE")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_name = Path(data_file).stem
    session_dir = output_path / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {session_dir}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 1: Loading neural data")
    print("-"*70)
    
    try:
        data = HippocampusDataLoader.load_mat_file(data_file)
        print(data.summary())
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # -------------------------------------------------------------------------
    # STEP 2: Filter Trials
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 2: Filtering trials for mental navigation")
    print("-"*70)
    
    mental_nav_trials = data.get_mental_navigation_trials()
    n_mn_trials = np.sum(mental_nav_trials)
    
    print(f"✓ Selected {n_mn_trials} / {data.n_trials} trials")
    print(f"  Criteria: trial_type=3 (fully occluded), seqq<3 (normal speed), attempt=1")
    
    if n_mn_trials < 10:
        print(f"✗ Warning: Only {n_mn_trials} trials found. Need at least 10 for analysis.")
        print("  Trying all successful trials instead...")
        mental_nav_trials = data.filter_trials(succ=1)
        n_mn_trials = np.sum(mental_nav_trials)
        print(f"  Using {n_mn_trials} successful trials")
    
    # -------------------------------------------------------------------------
    # STEP 3: Temporal Coding Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 3: Temporal coding analysis")
    print("-"*70)
    
    temporal_analyzer = TemporalCodingAnalyzer()
    
    tuning = temporal_analyzer.compute_temporal_tuning(
        neural_data=data,
        trial_mask=mental_nav_trials,
        time_window=(0, 3000),
        smooth_sigma=50.0
    )
    
    # Identify time cells
    time_cells = temporal_analyzer.identify_time_cells(
        tuning,
        tmi_threshold=0.3,
        info_threshold=0.1
    )
    
    # Visualize temporal tuning
    print("\nGenerating temporal tuning plots...")
    
    fig = NeuralVisualizer.plot_temporal_tuning(tuning, n_neurons=12)
    save_figure(fig, session_dir / 'temporal_tuning_curves.png')
    
    fig = NeuralVisualizer.plot_population_heatmap(
        tuning['tuning_curves'],
        tuning['time_vector'],
        sort_by='peak'
    )
    save_figure(fig, session_dir / 'population_heatmap.png')
    
    fig = NeuralVisualizer.plot_time_cells_summary(tuning, time_cells)
    save_figure(fig, session_dir / 'time_cells_summary.png')
    
    # -------------------------------------------------------------------------
    # STEP 4: Population Decoding
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 4: Population decoding")
    print("-"*70)
    
    # Decode landmark pairs using multiple methods
    methods = ['random_forest', 'bayesian', 'svm']
    decoding_results = {}
    
    for method in methods:
        print(f"\n--- Decoding with {method} ---")
        try:
            decoder = PopulationDecoder(method=method)
            results = decoder.decode_landmark_pairs(
                neural_data=data,
                trial_mask=mental_nav_trials,
                time_window=(500, 2000),
                cv_folds=5
            )
            decoding_results[method] = results
            
            # Plot results
            fig = NeuralVisualizer.plot_decoding_results(results)
            save_figure(fig, session_dir / f'decoding_{method}.png')
            
        except Exception as e:
            print(f"  Warning: {method} decoding failed: {e}")
    
    # Decode temporal distance (regression)
    print(f"\n--- Decoding temporal distance (regression) ---")
    try:
        decoder = PopulationDecoder(method='bayesian')
        temporal_results = decoder.decode_temporal_distance(
            neural_data=data,
            trial_mask=mental_nav_trials,
            time_window=(500, 2000)
        )
    except Exception as e:
        print(f"  Warning: Temporal distance decoding failed: {e}")
        temporal_results = None
    
    # -------------------------------------------------------------------------
    # STEP 5: Neural Manifold Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 5: Neural manifold analysis")
    print("-"*70)
    
    # PCA
    print("\n--- PCA Analysis ---")
    manifold_analyzer = NeuralManifoldAnalyzer(method='pca', n_components=10)
    
    manifold_results = manifold_analyzer.fit_transform(
        neural_data=data,
        trial_mask=mental_nav_trials,
        time_window=(0, 3000)
    )
    
    # Plot variance explained
    fig = NeuralVisualizer.plot_pca_variance(manifold_results)
    if fig is not None:
        save_figure(fig, session_dir / 'pca_variance.png')
    
    # Compute trajectory similarity
    curr = data.get_condition('curr')[mental_nav_trials]
    similarity = manifold_analyzer.compute_trajectory_similarity(
        manifold_results['trajectories'],
        curr
    )
    
    # Plot 3D trajectories
    print("\nGenerating 3D trajectory plot...")
    fig = NeuralVisualizer.plot_3d_trajectory(
        manifold_results,
        curr
    )
    save_figure(fig, session_dir / 'neural_trajectories_3d.png')
    
    # -------------------------------------------------------------------------
    # STEP 6: Sequence Analysis
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 6: Sequential activity analysis")
    print("-"*70)
    
    sequence_analyzer = SequenceAnalyzer()
    
    sequences = sequence_analyzer.detect_sequences(
        neural_data=data,
        trial_mask=mental_nav_trials,
        time_window=(0, 3000),
        min_neurons=5
    )
    
    # Plot sequence analysis
    fig = NeuralVisualizer.plot_sequence_analysis(
        sequences,
        tuning['time_vector']
    )
    save_figure(fig, session_dir / 'sequence_analysis.png')
    
    # -------------------------------------------------------------------------
    # STEP 7: Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 7: Generating summary report")
    print("-"*70)
    
    summary_file = session_dir / 'analysis_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HIPPOCAMPUS NEURAL ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Session: {session_name}\n")
        f.write(f"Data file: {data_file}\n\n")
        
        f.write(data.summary() + "\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TRIAL SELECTION\n")
        f.write("-"*70 + "\n")
        f.write(f"Mental navigation trials: {n_mn_trials} / {data.n_trials}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("TEMPORAL CODING\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of time cells: {np.sum(time_cells)} / {data.n_neurons}\n")
        f.write(f"Percentage of time cells: {100*np.sum(time_cells)/data.n_neurons:.1f}%\n")
        f.write(f"Mean TMI: {np.mean(tuning['tmi']):.3f}\n")
        f.write(f"Mean temporal information: {np.mean(tuning['temporal_info']):.3f} bits\n\n")
        
        f.write("-"*70 + "\n")
        f.write("POPULATION DECODING\n")
        f.write("-"*70 + "\n")
        for method, results in decoding_results.items():
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  Accuracy: {results['accuracy']:.3f} ± {results['std']:.3f}\n")
            f.write(f"  Chance level: {results['chance_level']:.3f}\n")
            f.write(f"  Number of classes: {len(results['unique_labels'])}\n")
        
        if temporal_results is not None:
            f.write(f"\nTEMPORAL DISTANCE DECODING (regression):\n")
            f.write(f"  R² score: {temporal_results['r2']:.3f}\n")
            f.write(f"  Correlation: {temporal_results['correlation']:.3f}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("NEURAL MANIFOLD\n")
        f.write("-"*70 + "\n")
        if manifold_results['variance_explained'] is not None:
            var_exp = manifold_results['variance_explained']
            f.write(f"First 3 PCs explain: {100*np.sum(var_exp[:3]):.1f}% variance\n")
        f.write(f"Within-condition distance: {similarity['within_distance']:.2f}\n")
        f.write(f"Between-condition distance: {similarity['between_distance']:.2f}\n")
        f.write(f"Separability: {similarity['separability']:.2f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("SEQUENTIAL ACTIVITY\n")
        f.write("-"*70 + "\n")
        f.write(f"Sequence score: {sequences['sequence_score']:.3f}\n")
        f.write(f"Mean time lag: {np.mean(sequences['time_lags']):.1f} ms\n\n")
        
        f.write("="*70 + "\n")
        f.write("Analysis complete!\n")
        f.write(f"Results saved to: {session_dir}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary saved: {summary_file}")
    
    # -------------------------------------------------------------------------
    # Done!
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {session_dir}")
    print("\nGenerated files:")
    for file in sorted(session_dir.glob('*')):
        print(f"  - {file.name}")
    print("\n")
    
    return session_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hippocampus Neural Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python example_analysis.py data/session1.mat
  
  # Analyze with custom output directory
  python example_analysis.py data/session1.mat --output my_results/
  
  # Analyze all .mat files in a directory
  python example_analysis.py data/ --pattern "*.mat"
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Path to .mat file or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results/)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='File pattern for batch processing (e.g., "*.mat")')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input exists
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    # Process files
    if input_path.is_file():
        # Single file
        run_complete_analysis(str(input_path), args.output)
    
    elif input_path.is_dir():
        # Directory - process all matching files
        pattern = args.pattern if args.pattern else '*.mat'
        mat_files = list(input_path.glob(pattern))
        
        if not mat_files:
            print(f"Error: No files matching '{pattern}' found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(mat_files)} file(s) to process\n")
        
        for mat_file in mat_files:
            try:
                run_complete_analysis(str(mat_file), args.output)
            except Exception as e:
                print(f"\n✗ Error processing {mat_file}: {e}\n")
                import traceback
                traceback.print_exc()
                continue
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
