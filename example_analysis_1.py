#!/usr/bin/env python3
"""
Complete Hippocampus Neural Analysis with Firing Rate Analysis
Example Analysis Pipeline - Main Script

This script demonstrates the complete workflow for analyzing neural data
including condition-specific firing rate analysis similar to sujay_example_zoom.m

Usage:
    python example_analysis_1.py /path/to/your_data.mat
    python example_analysis_1.py /path/to/your_data.mat --output my_results
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy import stats
import argparse

# Import the firing rate analyzer
from firing_rate_analyzer import FiringRateAnalyzer, plot_behavioral_ta_vs_tp

# Try to import h5py for MATLAB v7.3 files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not found. Install with: pip install h5py")


class SimpleDataLoader:
    """Simple data loader for .mat files containing neural tensor data."""
    
    def __init__(self, mat_file: str):
        """Load data from .mat file."""
        print(f"Loading data from: {mat_file}")
        self.mat_file = mat_file
        
        # Try to load with scipy first, fall back to h5py if needed
        try:
            self._load_with_scipy()
        except NotImplementedError:
            # File is MATLAB v7.3 format (HDF5)
            if not HAS_H5PY:
                raise ImportError(
                    "This file requires h5py. Install it with: pip install h5py"
                )
            self._load_with_h5py()
        
        # Store dimensions
        self.n_neurons = self.neur_tensor_stim1on.shape[0]
        self.n_timebins = self.neur_tensor_stim1on.shape[1]
        self.n_trials = self.neur_tensor_stim1on.shape[2]
        
        print(f"✓ Data loaded successfully")
        print(f"  Neural tensor shape: {self.neur_tensor_stim1on.shape}")
        print(f"  Neurons: {self.n_neurons}")
        print(f"  Time bins: {self.n_timebins}")
        print(f"  Trials: {self.n_trials}")
        print(f"  Condition matrix shape: {self.cond_matrix.shape}")
        print(f"  Time range: [{self.time_edges[0]:.3f}, {self.time_edges[-1]:.3f}] s")
    
    def _load_with_scipy(self):
        """Load with scipy.io.loadmat (for older MATLAB formats)."""
        self.data = loadmat(self.mat_file)
        
        # Extract main variables
        self.neur_tensor_stim1on = self.data['neur_tensor_stim1on']
        self.cond_matrix = self.data['cond_matrix']
        self.cond_label = self.data.get('cond_label', None)
        
        # Extract time edges
        if 'stim1on' in self.data:
            stim1on_struct = self.data['stim1on']
            if isinstance(stim1on_struct, np.ndarray) and stim1on_struct.dtype.names:
                self.time_edges = stim1on_struct['edges'][0, 0].flatten()
            else:
                self.time_edges = np.linspace(-1, 3, self.neur_tensor_stim1on.shape[1])
        else:
            self.time_edges = np.linspace(-1, 3, self.neur_tensor_stim1on.shape[1])
    
    def _load_with_h5py(self):
        """Load with h5py (for MATLAB v7.3 HDF5 format)."""
        print("  Detected MATLAB v7.3 format, using h5py...")
        
        with h5py.File(self.mat_file, 'r') as f:
            # Extract neural tensor (HDF5 stores in column-major, need to transpose)
            self.neur_tensor_stim1on = np.array(f['neur_tensor_stim1on']).T
            
            # Extract condition matrix
            self.cond_matrix = np.array(f['cond_matrix']).T
            
            # Extract time edges from stim1on structure
            if 'stim1on' in f:
                stim1on_ref = f['stim1on']
                if 'edges' in stim1on_ref:
                    edges_ref = stim1on_ref['edges']
                    # Handle reference to another dataset
                    if isinstance(edges_ref, h5py.Dataset):
                        self.time_edges = np.array(edges_ref).flatten()
                    else:
                        # It's a reference
                        self.time_edges = np.array(f[edges_ref[0, 0]]).flatten()
                else:
                    self.time_edges = np.linspace(-1, 3, self.neur_tensor_stim1on.shape[1])
            else:
                self.time_edges = np.linspace(-1, 3, self.neur_tensor_stim1on.shape[1])
            
            # Try to extract condition labels
            if 'cond_label' in f:
                self.cond_label = f['cond_label']
            else:
                self.cond_label = None


def run_complete_analysis(mat_file: str, output_dir: str = 'results'):
    """
    Run complete neural analysis pipeline including firing rate analysis.
    
    Args:
        mat_file: Path to .mat file
        output_dir: Directory for results
    """
    print("\n" + "="*70)
    print("HIPPOCAMPUS NEURAL ANALYSIS PIPELINE")
    print("Complete Analysis with Condition-Specific Firing Rates")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    session_name = Path(mat_file).stem
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
        data = SimpleDataLoader(mat_file)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # -------------------------------------------------------------------------
    # STEP 2: Behavioral Analysis (TA vs TP)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 2: Behavioral Analysis (TA vs TP)")
    print("-"*70)
    
    try:
        print("\nGenerating TA vs TP plots...")
        fig_behavior = plot_behavioral_ta_vs_tp(data.cond_matrix)
        fig_behavior.savefig(session_dir / 'behavioral_ta_vs_tp.png', 
                            dpi=300, bbox_inches='tight')
        plt.close(fig_behavior)
        print(f"✓ Saved: behavioral_ta_vs_tp.png")
        
        # Calculate statistics
        ta = data.cond_matrix[:, 0]
        tp = data.cond_matrix[:, 1]
        
        if data.cond_matrix.shape[1] > 9:
            mask = data.cond_matrix[:, 9] == 1
            ta_bench = ta[mask]
            tp_bench = tp[mask]
            
            if len(ta_bench) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(ta_bench, tp_bench)
                error = tp_bench - ta_bench
                
                print(f"\nBehavioral Statistics (n={len(ta_bench)} trials):")
                print(f"  Correlation: r = {r_value:.4f}, p = {p_value:.2e}")
                print(f"  Linear fit: TP = {slope:.4f} * TA + {intercept:.4f}")
                print(f"  Mean error: {np.mean(error):.4f} s")
                print(f"  RMSE: {np.sqrt(np.mean(error**2)):.4f} s")
        
    except Exception as e:
        print(f"  Warning: Behavioral analysis failed: {e}")
    
    # -------------------------------------------------------------------------
    # STEP 3: MATLAB Replication - Condition-Specific Firing Rates
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 3: MATLAB Replication - Condition-Specific Firing Rates")
    print("-"*70)
    
    try:
        print("\nReplicating sujay_example_zoom.m analysis...")
        print("  Extracting firing rates for:")
        print("    - Neuron 1, col9=1, col2=1, col3=4")
        print("    - Neuron 1, col9=1, col2=1, col3=5")
        print("    - Neuron 3, col9=1, col2=1, col3=2")
        
        matlab_results = FiringRateAnalyzer.replicate_matlab_example(
            neural_tensor=data.neur_tensor_stim1on,
            cond_matrix=data.cond_matrix,
            time_edges=data.time_edges,
            window_size=300
        )
        
        print(f"\n  Results:")
        print(f"    Trace 1: {matlab_results['n_trials'][0]} trials")
        print(f"    Trace 2: {matlab_results['n_trials'][1]} trials")
        print(f"    Trace 3: {matlab_results['n_trials'][2]} trials")
        
        # Plot
        fig_matlab = FiringRateAnalyzer.plot_condition_specific_firing(
            matlab_results,
            labels=[
                f"Neuron 1, col3=4 (n={matlab_results['n_trials'][0]})",
                f"Neuron 1, col3=5 (n={matlab_results['n_trials'][1]})",
                f"Neuron 3, col3=2 (n={matlab_results['n_trials'][2]})"
            ],
            colors=['blue', 'orange', 'green'],
            title="MATLAB Replication: Smoothed Firing Rates",
            add_vertical_lines=[0]
        )
        fig_matlab.savefig(session_dir / 'firing_rates_matlab_replication.png',
                          dpi=300, bbox_inches='tight')
        plt.close(fig_matlab)
        print(f"✓ Saved: firing_rates_matlab_replication.png")
        
    except Exception as e:
        print(f"  Warning: MATLAB replication failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STEP 4: Custom Analysis - Compare Trial Conditions
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 4: Custom Analysis - Compare Different Conditions")
    print("-"*70)
    
    try:
        print("\nComparing different condition values for Neuron 1...")
        
        # Define custom conditions to compare
        conditions_list = [
            {9: 1, 2: 1, 3: 2},
            {9: 1, 2: 1, 3: 4},
            {9: 1, 2: 1, 3: 5},
        ]
        
        custom_results = FiringRateAnalyzer.analyze_condition_specific_firing(
            neural_tensor=data.neur_tensor_stim1on,
            cond_matrix=data.cond_matrix,
            time_edges=data.time_edges,
            conditions_list=conditions_list,
            neuron_indices=[0, 0, 0],  # Same neuron for all
            window_size=300,
            average=True
        )
        
        print(f"  Condition col3=2: {custom_results['n_trials'][0]} trials")
        print(f"  Condition col3=4: {custom_results['n_trials'][1]} trials")
        print(f"  Condition col3=5: {custom_results['n_trials'][2]} trials")
        
        # Plot
        fig_custom = FiringRateAnalyzer.plot_condition_specific_firing(
            custom_results,
            labels=[f"col3={val}, n={n}" for val, n in zip([2, 4, 5], custom_results['n_trials'])],
            colors=['blue', 'orange', 'green'],
            title="Condition Comparison: Neuron 1",
            add_vertical_lines=[0]
        )
        fig_custom.savefig(session_dir / 'firing_rates_condition_comparison.png',
                          dpi=300, bbox_inches='tight')
        plt.close(fig_custom)
        print(f"✓ Saved: firing_rates_condition_comparison.png")
        
    except Exception as e:
        print(f"  Warning: Custom analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STEP 5: Population Analysis - Multiple Neurons
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 5: Population Analysis - Multiple Neurons")
    print("-"*70)
    
    try:
        print("\nAnalyzing first 5 neurons...")
        
        # Analyze first 5 neurons for same condition
        n_neurons_to_plot = min(5, data.n_neurons)
        shared_condition = {9: 1, 2: 1}
        
        population_results = FiringRateAnalyzer.analyze_condition_specific_firing(
            neural_tensor=data.neur_tensor_stim1on,
            cond_matrix=data.cond_matrix,
            time_edges=data.time_edges,
            conditions_list=[shared_condition] * n_neurons_to_plot,
            neuron_indices=list(range(n_neurons_to_plot)),
            window_size=300,
            average=True
        )
        
        print(f"  Shared condition: col9=1, col2=1")
        print(f"  Number of trials: {population_results['n_trials'][0]}")
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, n_neurons_to_plot))
        fig_population = FiringRateAnalyzer.plot_condition_specific_firing(
            population_results,
            labels=[f"Neuron {i+1}" for i in range(n_neurons_to_plot)],
            colors=colors,
            title="Population Firing Rates",
            add_vertical_lines=[0]
        )
        fig_population.savefig(session_dir / 'firing_rates_population.png',
                              dpi=300, bbox_inches='tight')
        plt.close(fig_population)
        print(f"✓ Saved: firing_rates_population.png")
        
    except Exception as e:
        print(f"  Warning: Population analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STEP 6: Advanced Analysis - Attention Modulation
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 6: Attention Modulation Analysis")
    print("-"*70)
    
    try:
        if data.cond_matrix.shape[1] > 11:
            print("\nComparing attention conditions for Neuron 1...")
            
            # Compare two attention conditions
            attention_conditions = [
                {9: 1},   # Attention condition 1
                {11: 1},  # Attention condition 2
            ]
            
            attention_results = FiringRateAnalyzer.analyze_condition_specific_firing(
                neural_tensor=data.neur_tensor_stim1on,
                cond_matrix=data.cond_matrix,
                time_edges=data.time_edges,
                conditions_list=attention_conditions,
                neuron_indices=[0, 0],  # Same neuron
                window_size=300,
                average=True
            )
            
            print(f"  Attention condition 1: {attention_results['n_trials'][0]} trials")
            print(f"  Attention condition 2: {attention_results['n_trials'][1]} trials")
            
            # Plot
            fig_attention = FiringRateAnalyzer.plot_condition_specific_firing(
                attention_results,
                labels=[
                    f"Attention 1 (n={attention_results['n_trials'][0]})",
                    f"Attention 2 (n={attention_results['n_trials'][1]})"
                ],
                colors=['blue', 'red'],
                title="Attention Modulation: Neuron 1",
                add_vertical_lines=[0]
            )
            fig_attention.savefig(session_dir / 'firing_rates_attention_modulation.png',
                                 dpi=300, bbox_inches='tight')
            plt.close(fig_attention)
            print(f"✓ Saved: firing_rates_attention_modulation.png")
        else:
            print("  Skipping (attention column not found)")
        
    except Exception as e:
        print(f"  Warning: Attention modulation analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # STEP 7: Generate Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("STEP 7: Generating Summary Report")
    print("-"*70)
    
    summary_file = session_dir / 'analysis_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HIPPOCAMPUS NEURAL ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Session: {session_name}\n")
        f.write(f"Data file: {mat_file}\n\n")
        
        f.write(f"Number of neurons: {data.n_neurons}\n")
        f.write(f"Number of trials: {data.n_trials}\n")
        f.write(f"Number of time bins: {data.n_timebins}\n")
        f.write(f"Time range: [{data.time_edges[0]:.3f}, {data.time_edges[-1]:.3f}] s\n\n")
        
        # Behavioral stats
        f.write("-"*70 + "\n")
        f.write("BEHAVIORAL ANALYSIS (TA vs TP)\n")
        f.write("-"*70 + "\n")
        
        ta = data.cond_matrix[:, 0]
        tp = data.cond_matrix[:, 1]
        
        if data.cond_matrix.shape[1] > 9:
            mask = data.cond_matrix[:, 9] == 1
            ta_bench = ta[mask]
            tp_bench = tp[mask]
            
            if len(ta_bench) > 0:
                slope, intercept, r_value, p_value, std_err = stats.linregress(ta_bench, tp_bench)
                error = tp_bench - ta_bench
                
                f.write(f"Number of benchmark trials: {len(ta_bench)}\n")
                f.write(f"Correlation: r = {r_value:.4f}, p = {p_value:.2e}\n")
                f.write(f"Linear fit: TP = {slope:.4f} * TA + {intercept:.4f}\n")
                f.write(f"Mean error: {np.mean(error):.4f} s\n")
                f.write(f"RMSE: {np.sqrt(np.mean(error**2)):.4f} s\n\n")
        
        # MATLAB replication
        if 'matlab_results' in locals():
            f.write("-"*70 + "\n")
            f.write("MATLAB REPLICATION ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(f"Trace 1 (Neuron 1, col3=4): {matlab_results['n_trials'][0]} trials\n")
            f.write(f"Trace 2 (Neuron 1, col3=5): {matlab_results['n_trials'][1]} trials\n")
            f.write(f"Trace 3 (Neuron 3, col3=2): {matlab_results['n_trials'][2]} trials\n")
            f.write(f"Smoothing window: 300 samples\n\n")
        
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
        print(f"  ✓ {file.name}")
    print("\n")
    
    return session_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hippocampus Neural Analysis with Firing Rate Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  python example_analysis_1.py data/amadeus01172020_a_neur_tensor_stim1on.mat
  
  # Analyze with custom output directory
  python example_analysis_1.py data/session1.mat --output my_results/
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input exists
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: {input_path} is not a file")
        sys.exit(1)
    
    # Run analysis
    try:
        run_complete_analysis(str(input_path), args.output)
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
