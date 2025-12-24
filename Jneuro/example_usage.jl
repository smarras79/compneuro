"""
Example Usage: Multi-Neuron Comparative Analysis
Demonstrates how to use the enhanced multi-neuron analysis framework

This script shows:
1. How to select neurons for analysis
2. How to compute comparative metrics
3. How to generate publication-quality comparative plots
"""

include("multi_neuron_analysis.jl")
include("multi_neuron_plots.jl")

using .MultiNeuronAnalysis
using .MultiNeuronPlots
using Random

# ============================================================================
# EXAMPLE 1: Basic workflow with simulated data
# ============================================================================

function example_basic_workflow()
    println("\n" * "="^70)
    println("EXAMPLE 1: Basic Multi-Neuron Analysis Workflow")
    println("="^70)
    
    # Simulate some example data
    Random.seed!(42)
    n_total_neurons = 50
    time = 0:1.0:3000.0
    
    # Generate realistic firing rate profiles for multiple neurons
    function generate_neuron_firing_rate(time, baseline, amplitude, peak_time, width, noise_level)
        firing_rate = baseline .+ amplitude .* exp.(-((time .- peak_time).^2) ./ (2 * width^2))
        firing_rate .+= noise_level .* randn(length(time))
        return max.(firing_rate, 0.0)  # Ensure non-negative
    end
    
    # Create firing rates for all neurons
    all_neuron_ids = collect(1:n_total_neurons)
    all_firing_rates = zeros(length(time), n_total_neurons)
    
    for i in 1:n_total_neurons
        baseline = 1.0 + 0.5*rand()
        amplitude = 3.0 + 5.0*rand()
        peak_time = 1000.0 + 1000.0*rand()
        width = 300.0 + 200.0*rand()
        noise_level = 0.3
        
        all_firing_rates[:, i] = generate_neuron_firing_rate(
            time, baseline, amplitude, peak_time, width, noise_level
        )
    end
    
    # Step 1: Select neurons for analysis
    println("\nStep 1: Selecting neurons for analysis...")
    
    # Method A: Select specific neurons by ID
    selected_ids = [5, 7, 13, 15, 19, 21, 27, 28, 29, 31, 33, 34]
    selection = select_neurons_by_ids(all_neuron_ids, selected_ids)
    
    println("Selected $(length(selection.neuron_ids)) neurons: $(selection.neuron_ids)")
    
    # Extract firing rates for selected neurons
    selected_indices = [findfirst(==(id), all_neuron_ids) for id in selection.neuron_ids]
    firing_rates_selected = all_firing_rates[:, selected_indices]
    
    # Step 2: Compute comparative metrics
    println("\nStep 2: Computing comparative metrics...")
    metrics = compute_comparative_metrics(selection, firing_rates_selected, time)
    
    # Display summary
    println("\nSummary Statistics:")
    summary = get_summary_statistics(metrics)
    for (key, value) in summary
        println("  $key: $(round(value, digits=3))")
    end
    
    # Display TMI values
    println("\nTemporal Modulation Index (TMI) for each neuron:")
    for (i, neuron_id) in enumerate(selection.neuron_ids)
        println("  Neuron $neuron_id: TMI = $(round(metrics.tmi_values[i], digits=3))")
    end
    
    # Step 3: Generate comparative plots
    println("\nStep 3: Generating comparative plots...")
    
    # Create output directory
    output_dir = "/home/claude/multi_neuron_results"
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Generate comprehensive report
    create_comparative_report(metrics, firing_rates_selected, time, output_dir)
    
    println("\n✓ Analysis complete! Results saved to $output_dir")
    
    return metrics, firing_rates_selected, time
end

# ============================================================================
# EXAMPLE 2: Advanced neuron selection strategies
# ============================================================================

function example_advanced_selection()
    println("\n" * "="^70)
    println("EXAMPLE 2: Advanced Neuron Selection Strategies")
    println("="^70)
    
    # Simulate data
    Random.seed!(123)
    n_total_neurons = 100
    time = 0:1.0:3000.0
    
    # Create diverse neuron population
    all_neuron_ids = collect(1:n_total_neurons)
    all_tmi_values = 0.1 .+ 0.5 .* rand(n_total_neurons)
    
    println("\nCreated population of $n_total_neurons neurons")
    println("TMI range: $(round(minimum(all_tmi_values), digits=2)) - $(round(maximum(all_tmi_values), digits=2))")
    
    # Selection Strategy 1: High TMI neurons (top 10)
    println("\n--- Selection Strategy 1: Top 10 High-TMI Neurons ---")
    high_tmi_indices = sortperm(all_tmi_values, rev=true)[1:10]
    high_tmi_selection = select_neurons(all_neuron_ids, high_tmi_indices)
    println("Selected neurons: $(high_tmi_selection.neuron_ids)")
    println("Their TMIs: $(round.(all_tmi_values[high_tmi_indices], digits=2))")
    
    # Selection Strategy 2: Low TMI neurons (bottom 10)
    println("\n--- Selection Strategy 2: Bottom 10 Low-TMI Neurons ---")
    low_tmi_indices = sortperm(all_tmi_values)[1:10]
    low_tmi_selection = select_neurons(all_neuron_ids, low_tmi_indices)
    println("Selected neurons: $(low_tmi_selection.neuron_ids)")
    println("Their TMIs: $(round.(all_tmi_values[low_tmi_indices], digits=2))")
    
    # Selection Strategy 3: Custom criteria (moderate TMI)
    println("\n--- Selection Strategy 3: Moderate TMI (0.2-0.4) ---")
    moderate_tmi_selection = select_neurons(
        all_neuron_ids,
        id -> begin
            idx = findfirst(==(id), all_neuron_ids)
            return 0.2 <= all_tmi_values[idx] <= 0.4
        end
    )
    println("Selected $(length(moderate_tmi_selection.neuron_ids)) neurons")
    
    # Selection Strategy 4: Stratified sampling
    println("\n--- Selection Strategy 4: Stratified Sampling Across TMI Range ---")
    n_bins = 5
    neurons_per_bin = 2
    tmi_bins = range(minimum(all_tmi_values), maximum(all_tmi_values), length=n_bins+1)
    
    stratified_ids = Int[]
    for i in 1:n_bins
        bin_neurons = findall(tmi_bins[i] .<= all_tmi_values .< tmi_bins[i+1])
        if length(bin_neurons) >= neurons_per_bin
            append!(stratified_ids, all_neuron_ids[bin_neurons[1:neurons_per_bin]])
        end
    end
    
    stratified_selection = select_neurons_by_ids(all_neuron_ids, stratified_ids)
    println("Selected neurons spanning TMI range: $(stratified_selection.neuron_ids)")
    
    println("\n✓ Demonstrated 4 different selection strategies")
end

# ============================================================================
# EXAMPLE 3: Comparative analysis workflow with real data structure
# ============================================================================

function example_with_data_loading()
    println("\n" * "="^70)
    println("EXAMPLE 3: Loading and Analyzing Real Neural Data")
    println("="^70)
    
    println("""
    To use with your actual data, follow this pattern:
    
    # 1. Load your neural data
    using MAT  # or HDF5, JLD2, etc.
    data = matread("your_neural_data.mat")
    
    # 2. Extract relevant fields
    spike_times = data["spike_times"]  # Dict or Array of spike times per neuron
    neuron_ids = data["neuron_ids"]     # Array of neuron identifiers
    time_window = (0.0, 3000.0)        # Your analysis time window
    
    # 3. Convert spike times to firing rates
    time_bins = 0:1.0:3000.0
    n_neurons = length(neuron_ids)
    firing_rates = zeros(length(time_bins), n_neurons)
    
    for i in 1:n_neurons
        spikes = spike_times[i]
        firing_rates[:, i] = firing_rate_dynamics(spikes, time_bins, 50.0)
    end
    
    # 4. Select neurons of interest
    # Example: Select neurons 5, 7, 13, 15, etc.
    selected_ids = [5, 7, 13, 15, 19, 21, 27, 28, 29, 31, 33, 34]
    selection = select_neurons_by_ids(neuron_ids, selected_ids)
    
    # Extract firing rates for selected neurons
    selected_indices = [findfirst(==(id), neuron_ids) for id in selection.neuron_ids]
    firing_rates_selected = firing_rates[:, selected_indices]
    
    # 5. Compute comparative metrics
    metrics = compute_comparative_metrics(selection, firing_rates_selected, time_bins)
    
    # 6. Generate plots
    output_dir = "analysis_results"
    create_comparative_report(metrics, firing_rates_selected, time_bins, output_dir)
    
    # 7. Access specific results
    println("TMI values: ", metrics.tmi_values)
    println("Mean firing rates: ", metrics.mean_firing_rates)
    println("Correlation matrix: ", metrics.correlation_matrix)
    """)
end

# ============================================================================
# EXAMPLE 4: Custom comparative visualizations
# ============================================================================

function example_custom_plots()
    println("\n" * "="^70)
    println("EXAMPLE 4: Custom Comparative Visualizations")
    println("="^70)
    
    # Generate sample data
    Random.seed!(42)
    time = 0:1.0:3000.0
    n_neurons = 12
    
    # Create selection
    neuron_ids = [5, 7, 13, 15, 19, 21, 27, 28, 29, 31, 33, 34]
    selection = NeuronSelection(neuron_ids)
    
    # Generate firing rates
    firing_rates = zeros(length(time), n_neurons)
    for i in 1:n_neurons
        baseline = 1.0 + 0.5*rand()
        amplitude = 3.0 + 5.0*rand()
        peak_time = 1000.0 + 1000.0*rand()
        width = 300.0 + 200.0*rand()
        
        firing_rates[:, i] = baseline .+ 
                            amplitude .* exp.(-((time .- peak_time).^2) ./ (2 * width^2)) .+
                            0.3 .* randn(length(time))
    end
    
    # Compute metrics
    metrics = compute_comparative_metrics(selection, firing_rates, time)
    
    println("\nGenerating custom plots...")
    
    # 1. Main comparative plot (like the figure you showed)
    plot_comparative_firing_rates(
        selection,
        firing_rates,
        time,
        metrics.tmi_values,
        layout_dims=(3, 4),
        sort_by_tmi=true,
        save_path="/home/claude/comparative_main.png"
    )
    println("  ✓ Main comparative plot saved")
    
    # 2. Correlation matrix
    plot_correlation_matrix(
        metrics.correlation_matrix,
        neuron_ids,
        save_path="/home/claude/correlation_matrix.png"
    )
    println("  ✓ Correlation matrix saved")
    
    # 3. TMI distribution
    plot_tmi_distribution(
        metrics.tmi_values,
        neuron_ids,
        save_path="/home/claude/tmi_distribution.png"
    )
    println("  ✓ TMI distribution saved")
    
    # 4. Temporal dynamics overlay
    highlight_neurons = [13, 21, 28]  # Highlight specific neurons
    plot_temporal_dynamics(
        selection,
        firing_rates,
        time,
        highlight_neurons,
        save_path="/home/claude/temporal_dynamics.png"
    )
    println("  ✓ Temporal dynamics overlay saved")
    
    # 5. Summary statistics
    plot_summary_statistics(
        metrics,
        save_path="/home/claude/summary_stats.png"
    )
    println("  ✓ Summary statistics saved")
    
    println("\n✓ All custom plots generated!")
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    println("\n" * "="^70)
    println("MULTI-NEURON COMPARATIVE ANALYSIS - EXAMPLES")
    println("="^70)
    
    # Run examples
    metrics, firing_rates, time = example_basic_workflow()
    example_advanced_selection()
    example_with_data_loading()
    example_custom_plots()
    
    println("\n" * "="^70)
    println("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    println("="^70)
    println("""
    
    Next Steps:
    1. Integrate these modules into your existing Jneuro codebase
    2. Adapt the data loading to match your file formats
    3. Customize plot styles and layouts for your specific needs
    4. Use the selection strategies to identify interesting neuron populations
    
    Key Functions:
    - select_neurons_by_ids(): Select specific neurons
    - compute_comparative_metrics(): Compute all metrics
    - create_comparative_report(): Generate full analysis report
    - plot_comparative_firing_rates(): Create multi-panel plot like your figure
    
    """)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
