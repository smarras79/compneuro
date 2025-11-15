"""
Installation Test Script
========================
Run this script to verify that all dependencies are installed correctly.

Usage:
    python test_installation.py
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 60)

    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn',
    }

    optional_packages = {
        'umap': 'UMAP',
        'pandas': 'Pandas',
    }

    all_passed = True

    # Test required packages
    for module_name, display_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name:20s} - OK")
        except ImportError as e:
            print(f"✗ {display_name:20s} - MISSING")
            print(f"  Install with: pip install {module_name}")
            all_passed = False

    print()

    # Test optional packages
    print("Optional packages:")
    for module_name, display_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name:20s} - OK")
        except ImportError:
            print(f"○ {display_name:20s} - Not installed (optional)")

    print("-" * 60)
    return all_passed


def test_modules():
    """Test that our analysis modules can be imported."""
    print("\nTesting analysis modules...")
    print("-" * 60)

    modules = [
        'hippocampus_data_loader',
        'neural_analysis_ai',
        'neural_visualization',
        'example_analysis'
    ]

    all_passed = True

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:30s} - OK")
        except ImportError as e:
            print(f"✗ {module_name:30s} - ERROR")
            print(f"  {str(e)}")
            all_passed = False

    print("-" * 60)
    return all_passed


def test_functionality():
    """Test basic functionality of the modules."""
    print("\nTesting basic functionality...")
    print("-" * 60)

    try:
        import numpy as np
        from hippocampus_data_loader import NeuralData

        # Create synthetic test data
        print("Creating synthetic neural data...")
        n_neurons = 10
        n_time = 100
        n_trials = 20

        neur_tensor = np.random.rand(n_neurons, n_time, n_trials)
        lfp_tensor = np.random.rand(5, n_time, n_trials)
        cond_matrix = np.random.randint(0, 6, (n_trials, 12))

        condition_labels = [
            'ta', 'tp', 'curr', 'target', 'trial_type',
            'seqq', 'succ', 'validtrials_mm', 'attempt',
            'reaction_time', 'nav_duration', 'trial_id'
        ]

        # Create NeuralData object
        data = NeuralData(
            neur_tensor=neur_tensor,
            lfp_tensor=lfp_tensor,
            cond_matrix=cond_matrix,
            condition_labels=condition_labels,
            filename='test_data.mat'
        )

        print(f"✓ Created NeuralData object: {data.n_neurons} neurons, "
              f"{data.n_trials} trials, {data.n_timepoints} timepoints")

        # Test filtering
        print("Testing trial filtering...")
        mask = data.filter_trials(trial_type=3)
        print(f"✓ Filtered trials: {np.sum(mask)} trials match criteria")

        # Test temporal coding analyzer
        print("Testing TemporalCodingAnalyzer...")
        from neural_analysis_ai import TemporalCodingAnalyzer

        analyzer = TemporalCodingAnalyzer()
        tuning = analyzer.compute_temporal_tuning(
            neural_data=data,
            trial_mask=np.ones(n_trials, dtype=bool),
            time_window=(-500, 500),
            smooth_sigma=10.0
        )

        print(f"✓ Computed temporal tuning: {tuning['tuning_curves'].shape}")

        # Test decoder
        print("Testing PopulationDecoder...")
        from neural_analysis_ai import PopulationDecoder

        decoder = PopulationDecoder(method='bayesian')
        print(f"✓ Created decoder: {decoder.method}")

        # Test manifold analyzer
        print("Testing NeuralManifoldAnalyzer...")
        from neural_analysis_ai import NeuralManifoldAnalyzer

        analyzer = NeuralManifoldAnalyzer(method='pca', n_components=3)
        print(f"✓ Created manifold analyzer: {analyzer.method}, "
              f"{analyzer.n_components} components")

        print("-" * 60)
        print("\n✓ All functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Functionality test failed:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("  HIPPOCAMPUS ANALYSIS PIPELINE - INSTALLATION TEST")
    print("=" * 60)
    print()

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n" + "=" * 60)
        print("✗ INSTALLATION INCOMPLETE")
        print("=" * 60)
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Test modules
    modules_ok = test_modules()

    if not modules_ok:
        print("\n" + "=" * 60)
        print("✗ MODULE IMPORT FAILED")
        print("=" * 60)
        print("\nMake sure you're running this script from the project directory")
        sys.exit(1)

    # Test functionality
    functionality_ok = test_functionality()

    if not functionality_ok:
        print("\n" + "=" * 60)
        print("✗ FUNCTIONALITY TEST FAILED")
        print("=" * 60)
        sys.exit(1)

    # All tests passed
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - INSTALLATION SUCCESSFUL")
    print("=" * 60)
    print("\nYou're ready to analyze neural data!")
    print("\nNext steps:")
    print("  1. Place your .mat files in a data directory")
    print("  2. Run: python example_analysis.py data/your_file.mat")
    print("  3. Check the results/ directory for outputs")
    print("\nFor help, see: README.md")
    print()


if __name__ == "__main__":
    main()
