"""
Example: Complete Neural Analysis with Filtering and SWR Detection

This script demonstrates the full workflow:
1. Load neural data
2. Apply signal filtering
3. Comprehensive neural analysis (spectral, SWR detection, ML patterns)
4. Event-triggered analysis with behavioral data
5. Visualization and reporting

Author: Enhanced Neural Analysis Toolkit
"""

using MAT
using Statistics
using Plots

# Load filtering functions
include("./sujay_example_zoom_likeMatlab_enhanced.jl")  # Has apply_neural_filter

# Load analysis modules
include("./integrated_analysis_pipeline.jl")  # Loads all other modules

println("\n" * "="^70)
println("NEURAL ANALYSIS EXAMPLE - SWR DETECTION PIPELINE")
println("="^70)

# ========== 1. LOAD DATA ==========
println("\n[1/6] Loading data...")

data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
cond_matrix = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

# Extract condition
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

# Compute mean firing rate
signal_raw = vec(mean(fr3, dims=2))

println("  Signal length: $(length(signal_raw)) samples")
println("  ✓ Data loaded")

# ========== 2. SIGNAL FILTERING ==========
println("\n[2/6] Applying signal filter...")

# Choose your filter (try different ones!)
selected_filter = :gaussian
window_size = 300

# Filter parameters (ensure Float64 for all numerical values)
filter_params = Dict(
    :sigma => 50.0,
    :cutoff_freq => 0.05,
    :fs => 1.0,
    :filter_order => 4,
    :poly_order => 3,
    :alpha => 0.05
)

println("  Filter type: $(selected_filter)")
println("  Window size: $(window_size)")

# Apply filter
signal_filtered = apply_neural_filter(
    signal_raw, selected_filter, window_size; filter_params...
)

println("  Input length: $(length(signal_raw))")
println("  Filtered length: $(length(signal_filtered))")
println("  ✓ Filtering complete")

# ========== 3. EXTRACT BEHAVIORAL EVENTS ==========
println("\n[3/6] Extracting behavioral events...")

# Example: Extract joystick motion onset/offset from condition matrix
# Column 1: target angle, Column 2: target position
# We'll simulate onset/offset detection based on position changes

# Find trials with significant position changes (proxy for motion)
positions = cond_matrix[trid, 2]
position_changes = abs.(diff(positions))
motion_threshold = quantile(position_changes, 0.75)

# Detect motion onsets (where position starts changing significantly)
motion_onset_trials = findall(position_changes .> motion_threshold)
motion_offset_trials = motion_onset_trials .+ 1

# Convert to time (samples) - assuming uniform distribution across trial
samples_per_trial = length(signal_filtered) ÷ length(trid)

motion_onset_samples = motion_onset_trials .* samples_per_trial
motion_offset_samples = motion_offset_trials .* samples_per_trial

# Filter to valid range
motion_onset_samples = motion_onset_samples[motion_onset_samples .<= length(signal_filtered)]
motion_offset_samples = motion_offset_samples[motion_offset_samples .<= length(signal_filtered)]

# Convert to time in seconds (assuming 1000 Hz sampling)
fs_analysis = 1000.0  # Assumed sampling frequency for analysis
motion_onset_times = motion_onset_samples ./ fs_analysis
motion_offset_times = motion_offset_samples ./ fs_analysis

behavioral_events = Dict(
    "motion_onset" => motion_onset_times,
    "motion_offset" => motion_offset_times
)

println("  Motion onset events: $(length(motion_onset_times))")
println("  Motion offset events: $(length(motion_offset_times))")
println("  ✓ Behavioral events extracted")

# ========== 4. COMPREHENSIVE NEURAL ANALYSIS ==========
println("\n[4/6] Running comprehensive neural analysis...")
println("  This may take a minute...")

# Configure analysis
analysis_config = Dict(
    # Spectral analysis
    "psd_window" => 512,
    "spec_window" => 256,
    "compute_spectrogram" => true,
    
    # SWR detection parameters
    "ripple_band" => (150.0, 250.0),
    "swr_threshold_sd" => 3.0,
    "swr_min_duration" => 30.0,
    "swr_max_duration" => 200.0,
    
    # Event-triggered analysis
    "event_window_ms" => 500.0,
    
    # ML clustering
    "n_clusters" => 3,
    
    # Visualization
    "plot_time_range" => (0.0, 10.0),
    "plot_freq_range" => (0.0, 300.0)
)

# Run comprehensive analysis
analysis_results = analyze_neural_data_comprehensive(
    signal_filtered,
    behavioral_events;
    fs=fs_analysis,
    config=analysis_config
)

println("  ✓ Analysis complete")

# ========== 5. DISPLAY KEY RESULTS ==========
println("\n[5/6] Key Results Summary...")
println("="^70)

# SWR Detection Summary
if haskey(analysis_results, "swr_detection")
    swr_data = analysis_results["swr_detection"]
    n_swr = length(swr_data["events"])
    
    println("\n📊 SHARP-WAVE RIPPLE DETECTION:")
    println("  Total SWRs detected: $n_swr")
    
    if n_swr > 0
        durations = [e.duration_ms for e in swr_data["events"]]
        amplitudes = [e.peak_amplitude for e in swr_data["events"]]
        
        println("  Duration (ms):")
        println("    Mean: $(round(mean(durations), digits=1))")
        println("    Range: $(round(minimum(durations), digits=1)) - $(round(maximum(durations), digits=1))")
        
        println("  Amplitude:")
        println("    Mean: $(round(mean(amplitudes), digits=2))")
        println("    Range: $(round(minimum(amplitudes), digits=2)) - $(round(maximum(amplitudes), digits=2))")
    end
end

# Event-Triggered Analysis
if haskey(analysis_results, "event_triggered_motion_onset")
    println("\n🎯 EVENT-TRIGGERED ANALYSIS:")
    
    for event_type in ["motion_onset", "motion_offset"]
        key = "event_triggered_$(event_type)"
        if haskey(analysis_results, key)
            event_data = analysis_results[key]
            n_events = event_data["total_events"]
            
            swr_per_event = []
            for i in 1:n_events
                push!(swr_per_event, event_data["event_$i"]["n_swr"])
            end
            
            println("  $(event_type):")
            println("    Events analyzed: $n_events")
            println("    Total SWRs near events: $(sum(swr_per_event))")
            println("    Mean SWRs per event: $(round(mean(swr_per_event), digits=2))")
        end
    end
end

# ML Pattern Analysis
if haskey(analysis_results, "ml_analysis")
    ml_data = analysis_results["ml_analysis"]
    
    println("\n🤖 MACHINE LEARNING ANALYSIS:")
    println("  Events clustered: $(ml_data["n_events"])")
    println("  Number of clusters: $(ml_data["n_clusters"])")
    
    # Cluster distribution
    for i in 1:ml_data["n_clusters"]
        count = sum(ml_data["cluster_labels"] .== i)
        pct = count / ml_data["n_events"] * 100
        println("    Cluster $i: $count events ($(round(pct, digits=1))%)")
    end
    
    n_anomalies = sum(ml_data["is_anomaly"])
    println("  Anomalies detected: $n_anomalies ($(round(n_anomalies/ml_data["n_events"]*100, digits=1))%)")
end

# Frequency Band Analysis
if haskey(analysis_results, "frequency_bands")
    bands = analysis_results["frequency_bands"]
    
    println("\n🎵 FREQUENCY BAND POWER:")
    # Sort only by band names (keys), not by the dictionary values
    for band in sort(collect(keys(bands)))
        if band != "total_power" && band != "freq" && band != "power_spectrum"
            info = bands[band]
            rel_power = get(info, "relative_power", 0.0) * 100
            println("  $(rpad(band, 12)): $(round(rel_power, digits=1))%")
        end
    end
end

println("\n" * "="^70)

# ========== 6. SAVE RESULTS ==========
println("\n[6/6] Saving results...")

output_dir = "./neural_analysis_output"
save_analysis_results(analysis_results, output_dir)

println("\n✓ Analysis pipeline complete!")
println("  Check '$(output_dir)' for detailed results and plots")

# ========== BONUS: Quick Comparison Plot ==========
println("\n📈 Generating comparison plot (raw vs filtered)...")

comparison_time = (1:min(2000, length(signal_filtered))) ./ fs_analysis

p_comparison = plot(layout=(2,1), size=(1200, 600))

# Raw signal
plot!(p_comparison[1], comparison_time, signal_raw[1:length(comparison_time)],
     label="Raw Signal",
     xlabel="Time (s)",
     ylabel="Firing Rate",
     title="Signal Comparison: Raw vs Filtered ($selected_filter)",
     linewidth=1.5,
     color=:gray,
     alpha=0.7)

# Filtered signal  
plot!(p_comparison[2], comparison_time, signal_filtered[1:length(comparison_time)],
     label="Filtered Signal ($selected_filter)",
     xlabel="Time (s)",
     ylabel="Firing Rate",
     color=:blue,
     linewidth=2)

# Mark detected SWRs if any in this time window
if haskey(analysis_results, "swr_detection") && length(analysis_results["swr_detection"]["events"]) > 0
    swr_events = analysis_results["swr_detection"]["events"]
    max_sample = Int(maximum(comparison_time) * fs_analysis)
    
    for event in swr_events
        if event.start_sample <= max_sample
            event_time = (event.start_sample-1) / fs_analysis
            plot!(p_comparison[2], [event_time, event_time], ylims(p_comparison[2]),
                 color=:red, linestyle=:dash, label="", alpha=0.5)
        end
    end
end

savefig(p_comparison, joinpath(output_dir, "signal_comparison.png"))
display(p_comparison)

println("  ✓ Comparison plot saved")

println("\n" * "="^70)
println("🎉 COMPLETE ANALYSIS FINISHED!")
println("="^70)
println("\nNext steps:")
println("  1. Review plots in: $(output_dir)/")
println("  2. Read summary: $(output_dir)/analysis_summary.txt")
println("  3. Adjust parameters in analysis_config and re-run")
println("  4. Try different filters: :moving_average, :gaussian, :butterworth, etc.")
println("="^70)
