#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
"""
import sys

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    print("-" * 60)
    
    required_modules = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('h5py', 'HDF5 for Python'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
    ]
    
    optional_modules = [
        ('umap', 'UMAP'),
    ]
    
    all_ok = True
    
    # Test required modules
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:<20} OK")
        except ImportError as e:
            print(f"✗ {display_name:<20} MISSING")
            print(f"  Install with: pip install {module_name}")
            all_ok = False
    
    # Test optional modules
    print("\nOptional modules:")
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:<20} OK")
        except ImportError:
            print(f"- {display_name:<20} Not installed (optional)")
    
    print("-" * 60)
    
    if all_ok:
        print("✓ All required dependencies are installed!")
        print("\nYou can now run:")
        print("  python example_analysis.py <your_data_file.mat>")
        return True
    else:
        print("✗ Some required dependencies are missing.")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
        return False


def test_modules():
    """Test that our custom modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing custom modules...")
    print("-" * 60)
    
    try:
        from hippocampus_data_loader import HippocampusDataLoader
        print("✓ hippocampus_data_loader     OK")
    except Exception as e:
        print(f"✗ hippocampus_data_loader     FAILED: {e}")
        return False
    
    try:
        from neural_analysis_ai import (
            TemporalCodingAnalyzer,
            PopulationDecoder,
            NeuralManifoldAnalyzer,
            SequenceAnalyzer
        )
        print("✓ neural_analysis_ai          OK")
    except Exception as e:
        print(f"✗ neural_analysis_ai          FAILED: {e}")
        return False
    
    try:
        from neural_visualization import NeuralVisualizer, save_figure
        print("✓ neural_visualization        OK")
    except Exception as e:
        print(f"✗ neural_visualization        FAILED: {e}")
        return False
    
    print("-" * 60)
    print("✓ All custom modules loaded successfully!")
    return True


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("HIPPOCAMPUS NEURAL ANALYSIS - INSTALLATION TEST")
    print("=" * 60 + "\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n" + "=" * 60)
        print("Installation incomplete. Please install missing packages.")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # Test custom modules
    modules_ok = test_modules()
    
    if not modules_ok:
        print("\n" + "=" * 60)
        print("Module loading failed. Check error messages above.")
        print("=" * 60 + "\n")
        sys.exit(1)
    
    # All tests passed
    print("\n" + "=" * 60)
    print("✓ INSTALLATION SUCCESSFUL!")
    print("=" * 60)
    print("\nYou're all set! To analyze your data, run:")
    print("  python example_analysis.py path/to/your/data.mat")
    print("\nFor help, run:")
    print("  python example_analysis.py --help")
    print("\n")


if __name__ == '__main__':
    main()
