#!/usr/bin/env python3
"""
Standalone TA vs TP Analysis Script

This script creates benchmark plots showing the relationship between
True Temporal Distance (TA) and Produced Temporal Distance (TP).

Usage:
    python plot_ta_vs_tp.py data.mat
    python plot_ta_vs_tp.py data.mat --output results/
    python plot_ta_vs_tp.py data.mat --color-by curr
"""

import sys
import numpy as np
from pathlib import Path
import argparse

from hippocampus_data_loader import HippocampusDataLoader
from neural_visualization import NeuralVisualizer, save_figure


def analyze_ta_vs_tp(data_file: str, output_dir: str = 'results', color_by: str = None):
    """
    Perform TA vs TP analysis and create benchmark plots.
    
    Args:
        data_file: Path to .mat file
        output_dir: Output directory for plots
        color_by: Optional condition to color points by
    """
    print("\n" + "="*70)
    print("TA vs TP BENCHMARK ANALYSIS")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_name = Path(data_file).stem
    session_dir = output_path / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {session_dir}")
    
    # Load data
    print("\n" + "-"*70)
    print("Loading data...")
    print("-"*70)
    
    try:
        data = HippocampusDataLoader.load_mat_file(data_file)
        print(data.summary())
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Check that we have TA and TP columns
    if data.cond_matrix.shape[1] < 2:
        print(f"✗ Error: cond_matrix must have at least 2 columns (TA and TP)")
        print(f"  Your data has {data.cond_matrix.shape[1]} columns")
        return
    
    # Get TA and TP
    ta = data.cond_matrix[:, 0]
    tp = data.cond_matrix[:, 1]
    
    print(f"\n✓ Found TA and TP columns")
    print(f"  TA range: [{np.min(ta):.3f}, {np.max(ta):.3f}] s")
    print(f"  TP range: [{np.min(tp):.3f}, {np.max(tp):.3f}] s")
    
    # Determine trial mask
    if data.cond_matrix.shape[1] > 9:
        print(f"\n✓ Using column 10 == 1 as trial mask")
        trial_mask = data.cond_matrix[:, 9] == 1
        n_trials = np.sum(trial_mask)
        print(f"  {n_trials} / {data.n_trials} trials selected")
    else:
        print(f"\n✓ Column 10 not found, using all trials")
        trial_mask = None
        n_trials = data.n_trials
    
    # Basic statistics
    print("\n" + "-"*70)
    print("Computing statistics...")
    print("-"*70)
    
    if trial_mask is not None:
        ta_filtered = ta[trial_mask]
        tp_filtered = tp[trial_mask]
    else:
        ta_filtered = ta
        tp_filtered = tp
    
    error = tp_filtered - ta_filtered
    
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(ta_filtered, tp_filtered)
    
    print(f"\nPerformance Metrics:")
    print(f"  Correlation:     {r_value:.4f}")
    print(f"  Slope:           {slope:.4f}")
    print(f"  Intercept:       {intercept:.4f}")
    print(f"  p-value:         {p_value:.2e}")
    print(f"\nError Metrics:")
    print(f"  Mean error:      {np.mean(error):.4f} s")
    print(f"  Std error:       {np.std(error):.4f} s")
    print(f"  MAE:             {np.mean(np.abs(error)):.4f} s")
    print(f"  RMSE:            {np.sqrt(np.mean(error**2)):.4f} s")
    
    # Create plots
    print("\n" + "-"*70)
    print("Generating plots...")
    print("-"*70)
    
    # Basic TA vs TP plot
    print("\n1. Creating benchmark plot...")
    fig = NeuralVisualizer.plot_ta_vs_tp(data, trial_mask=trial_mask)
    save_figure(fig, session_dir / 'ta_vs_tp_benchmark.png')
    
    # Detailed TA vs TP plot
    print("2. Creating detailed analysis plot...")
    fig = NeuralVisualizer.plot_ta_vs_tp_detailed(
        data, 
        trial_mask=trial_mask,
        color_by=color_by
    )
    save_figure(fig, session_dir / 'ta_vs_tp_detailed.png')
    
    # Save statistics to text file
    print("3. Saving statistics...")
    stats_file = session_dir / 'ta_vs_tp_statistics.txt'
    
    with open(stats_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TA vs TP BENCHMARK STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Session: {session_name}\n")
        f.write(f"Data file: {data_file}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Total trials: {data.n_trials}\n")
        f.write(f"Analyzed trials: {n_trials}\n")
        f.write(f"TA range: [{np.min(ta_filtered):.3f}, {np.max(ta_filtered):.3f}] s\n")
        f.write(f"TP range: [{np.min(tp_filtered):.3f}, {np.max(tp_filtered):.3f}] s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Correlation (r):     {r_value:.4f}\n")
        f.write(f"R-squared (r²):      {r_value**2:.4f}\n")
        f.write(f"Slope:               {slope:.4f}\n")
        f.write(f"Intercept:           {intercept:.4f}\n")
        f.write(f"Standard error:      {std_err:.4f}\n")
        f.write(f"p-value:             {p_value:.2e}\n\n")
        
        f.write("Linear fit equation: TP = {:.4f} * TA + {:.4f}\n\n".format(slope, intercept))
        
        f.write("-"*70 + "\n")
        f.write("ERROR METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean error:          {np.mean(error):.4f} s\n")
        f.write(f"Std error:           {np.std(error):.4f} s\n")
        f.write(f"Mean absolute error: {np.mean(np.abs(error)):.4f} s\n")
        f.write(f"Root mean sq error:  {np.sqrt(np.mean(error**2)):.4f} s\n")
        f.write(f"Median error:        {np.median(error):.4f} s\n\n")
        
        relative_error = (error / ta_filtered) * 100
        f.write(f"Mean relative error: {np.mean(relative_error):.2f}%\n")
        f.write(f"Median relative err: {np.median(relative_error):.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("ERROR DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        f.write(f"Underestimation (TP<TA): {np.sum(error < 0)} trials ({100*np.sum(error < 0)/len(error):.1f}%)\n")
        f.write(f"Overestimation (TP>TA):  {np.sum(error > 0)} trials ({100*np.sum(error > 0)/len(error):.1f}%)\n")
        f.write(f"Perfect (TP=TA):         {np.sum(error == 0)} trials ({100*np.sum(error == 0)/len(error):.1f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write("Analysis complete!\n")
        f.write(f"Results saved to: {session_dir}\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Statistics saved: {stats_file}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {session_dir}")
    print("\nGenerated files:")
    for file in sorted(session_dir.glob('*')):
        print(f"  - {file.name}")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TA vs TP Benchmark Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python plot_ta_vs_tp.py data.mat
  
  # Custom output directory
  python plot_ta_vs_tp.py data.mat --output my_results/
  
  # Color points by start landmark
  python plot_ta_vs_tp.py data.mat --color-by curr
  
  # Color points by target landmark
  python plot_ta_vs_tp.py data.mat --color-by target
  
  # Color points by success
  python plot_ta_vs_tp.py data.mat --color-by succ
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results/)')
    parser.add_argument('--color-by', type=str, default=None,
                       help='Condition to color points by (e.g., curr, target, succ)')
    
    args = parser.parse_args()
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: {input_path} is not a file")
        sys.exit(1)
    
    # Run analysis
    try:
        analyze_ta_vs_tp(str(input_path), args.output, args.color_by)
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
