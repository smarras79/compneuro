using MAT
using Statistics
using LinearAlgebra
using DSP
using Plots
using Colors
using Printf
using LaTeXStrings

include("./enhanced_visualization.jl")
include("./auxiliary_functions.jl")  # Load filter function

include("./multi_neuron_analysis.jl")
include("./multi_neuron_plots.jl")

# Import the modules and explicitly import types we'll use directly
using .MultiNeuronAnalysis
using .MultiNeuronPlots

# Explicitly import types for direct use in Main scope
using .MultiNeuronAnalysis: NeuronSelection, ComparativeMetrics, 
                           compute_comparative_metrics, get_summary_statistics
using .MultiNeuronPlots: plot_comparative_firing_rates, plot_temporal_dynamics,
                         create_comparative_report

# ========== CONFIGURATION ==========
# Define these BEFORE using them
fs = 1000.0           # Sampling frequency (Hz) - MUST MATCH YOUR RECORDING SYSTEM
window_size = 300     # Filter window size
ineuron = 1

println("Configuration:")
println("  Sampling frequency: $(fs) Hz")
println("  Window size: $(window_size)")
println("  Neuron $(ineuron)")
println()

# Load the .mat file
data = matread("./data/amadeus01172020_a_neur_tensor_stim1on.mat")
# Alternative file: amadeus01172020_a_neur_tensor_joyon.mat
output_dir = "./neural_analysis_output"


#------------------------------------------------------------------------
# ==== 1. Select filter type (change this to try different filters)
#         Options:
#                   :none              # NEW: No filtering (raw signal)
#                   :moving_average
#                   :gaussian
#                   :savitzky_golay
#                   :butterworth
#                   :median
#                   :exponential
#------------------------------------------------------------------------
selected_filter = :moving_average  # Default: same as original code

#------------------------------------------------------------------------
# Additional parameters for specific filters (adjust as needed
#------------------------------------------------------------------------
filter_params = Dict(
    :cutoff_freq => 0.05,              # For Butterworth (Hz)
    :fs => 1.0,                        # Sampling frequency (Hz)
    :filter_order => 4,                # Butterworth order
    :poly_order => 3,                  # Savitzky-Golay polynomial order
    :alpha => 0.05,                    # Exponential MA smoothing factor
    :sigma => Float64(window_size)/6.0 # Gaussian standard deviation
)
println("Using filter: $(selected_filter)")
println("Window size: $(window_size)")
println()

# Extract variables from the loaded data
cond_label          = data["cond_label"]
cond_matrix         = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on             = data["stim1on"]

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

#------------------------------------------------------------------------
# Display condition labels (tells you which column codes for what task parameter)
#------------------------------------------------------------------------
println("Condition labels:")
println(cond_label)
println()

#------------------------------------------------------------------------
#%% OPTIONAL: Shift time base to start at t=0
# Set this to true if you want absolute time (starting at 0)
# Set to false to keep event-relative time (e.g., -2 to +2 seconds around stimulus)
#------------------------------------------------------------------------
SHIFT_TIME_TO_ZERO = false  # Change to true if you want t=0 start

if SHIFT_TIME_TO_ZERO && edges[1] < 0
    println("\n" * "="^70)
    println("SHIFTING TIME BASE TO START AT t=0")
    println("="^70)
    
    time_shift = -edges[1]
    edges_original = copy(edges)
    edges = edges .+ time_shift
    
    println("Original time: $(round(edges_original[1], digits=3)) to $(round(edges_original[end], digits=3)) s")
    println("New time: $(round(edges[1], digits=3)) to $(round(edges[end], digits=3)) s")
    println("Shift applied: +$(round(time_shift, digits=3)) s")
    println("="^70)
    println()
end

#------------------------------------------------------------------------
#%% Extract neural data
#------------------------------------------------------------------------
println("Extracting neural data...")

#------------------------------------------------------------------------
# Extract firing rates for condition 4 (column 10==1 & column 3==1 & column 4==4)
#------------------------------------------------------------------------
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[ineuron, :, trid]
println("  fr3 extracted: $(size(fr3)) from $(length(trid)) trials")

#------------------------------------------------------------------------
# Extract firing rates for condition 5 (column 10==1 & column 3==1 & column 4==5)
#------------------------------------------------------------------------
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 5))
fr4 = neur_tensor_stim1on[ineuron, :, trid]
println("  fr4 extracted: $(size(fr4)) from $(length(trid)) trials")
println()

if !isdir(output_dir)
    mkdir(output_dir)
end

#------------------------------------------------------------------------
#%% Behavioural data
#------------------------------------------------------------------------
# Find trials where column 10 == 1
trid    = findall(cond_matrix[:, 10] .== 1)
ta_att1 = cond_matrix[trid, 1]
tp_att1 = cond_matrix[trid, 2]
neural_plots_scatter(ta_att1, tp_att1, "1", selected_filter; output_dir=output_dir)

# Second subplot - find trials where column 12 == 1
trid    = findall(cond_matrix[:, 12] .== 1)
ta_att2 = cond_matrix[trid, 1]
tp_att2 = cond_matrix[trid, 2]
neural_plots_scatter(ta_att2, tp_att2, "2", selected_filter; output_dir=output_dir)

#------------------------------------------------------------------------
#%% Apply selected filter to neural data
#------------------------------------------------------------------------
# Compute mean firing rates across trials first
fr3_mean = vec(mean(fr3, dims=2))
fr4_mean = vec(mean(fr4, dims=2))

println("Input signal lengths:")
println("  fr3_mean: $(length(fr3_mean))")
println("  fr4_mean: $(length(fr4_mean))")
println("  edges: $(length(edges))")
println()

# Apply the selected filter
fr3_smooth = apply_neural_filter(fr3_mean, selected_filter, window_size; filter_params...)
fr4_smooth = apply_neural_filter(fr4_mean, selected_filter, window_size; filter_params...)

println("Filtered signal lengths:")
println("  fr3_smooth: $(length(fr3_smooth))")
println("  fr4_smooth: $(length(fr4_smooth))")

#------------------------------------------------------------------------
# Plot neural data and save to file
# Time bins need to match the filtered signal length
# The filtered signals are trimmed by (window_size - 1) total samples
# Original code used edges[150:end-150], but we need to account for filter trimming
#------------------------------------------------------------------------
additional_trim = (length(fr3_mean) - length(fr3_smooth)) ÷ 2

time_start = 150 + additional_trim
time_end = length(edges) - 150 - additional_trim
time_bins = edges[time_start:time_end]

println("  time_bins: $(length(time_bins))")

# Verify lengths match before plotting
if length(time_bins) != length(fr3_smooth)
    println("WARNING: Length mismatch detected, adjusting...")
    # Ensure exact match by trimming to minimum length
    min_len    = min(length(time_bins), length(fr3_smooth))
    time_bins  = time_bins[1:min_len]
    fr3_smooth = fr3_smooth[1:min_len]
    fr4_smooth = fr4_smooth[1:min_len]
    println("  Adjusted to length: $(min_len)")
end
neural_plots(time_bins, fr3_smooth, fr4_smooth, "1", selected_filter; output_dir=output_dir)

println("\n✓ Neural data processed with $(selected_filter) filter")
println("✓ Smoothed firing rates computed and plotted")

#------------------------------------------------------------------------
#%% ===== EXTRACT BEHAVIORAL EVENTS =====
#------------------------------------------------------------------------
println("\n" * "="^70)
println("EXTRACTING BEHAVIORAL EVENTS")
println("="^70)

# Load behavioral event extraction helper
include("./behavioral_event_extraction.jl")

# Auto-detect which column likely contains motion/position data
println("\nAuto-detecting motion column from behavioral data...")
suggested_column = auto_detect_motion_column(cond_matrix)

# Extract motion events from behavioral recordings
# This finds WHEN motion happened (from your recordings)
behavioral_events = extract_motion_events(
    cond_matrix, 
    edges;
    position_column=suggested_column,  # Use auto-detected column (or manually set to 1, 2, etc.)
    threshold_quantile=0.75,            # Top 25% of changes = significant motion
    motion_duration=0.5,                # Assume 500ms per motion event
    fs=fs,
    method=:position_change
)

# Check if extraction was successful
if behavioral_events["n_events"] == 0
    println("\n⚠️  No motion events detected from automatic extraction.")
    println("    Trying fallback: using trial-based events...")
    
    # Fallback: Use trial onset times instead
    behavioral_events = extract_trial_events(cond_matrix, edges; fs=fs)
    
    # Rename for clarity
    behavioral_events["event_onset"] = behavioral_events["trial_onset"]
    delete!(behavioral_events, "trial_onset")
    
    println("✓ Using trial onsets: $(behavioral_events["n_events"]) events")
else
    println("\n✓ Behavioral events extracted successfully!")
    println("  Motion onset events: $(length(behavioral_events["motion_onset"]))")
    println("  Motion offset events: $(length(behavioral_events["motion_offset"]))")
    
    # Show first few event times
    if length(behavioral_events["motion_onset"]) > 0
        n_show = min(5, length(behavioral_events["motion_onset"]))
        times_str = join([round(t, digits=2) for t in behavioral_events["motion_onset"][1:n_show]], ", ")
        println("  First motion onsets: $times_str $(length(behavioral_events["motion_onset"]) > 5 ? "..." : "") seconds")
    end
end

#------------------------------------------------------------------------
#%% ===== FIX TIME ALIGNMENT (NEW!) =====
#------------------------------------------------------------------------
println("\n" * "="^70)
println("CHECKING TIME ALIGNMENT")
println("="^70)

#------------------------------------------------------------------------
# CRITICAL: Use time_bins (filtered signal time base), not edges!
#------------------------------------------------------------------------
neural_start = time_bins[1]
neural_end = time_bins[end]

println("Filtered neural signal:")
println("  Start: $(round(neural_start, digits=3)) s")
println("  End: $(round(neural_end, digits=3)) s")
println("  Duration: $(round(neural_end - neural_start, digits=2)) s")
println("  Number of time points: $(length(time_bins))")

# Find earliest behavioral event - FIX SCOPING
min_event_time = Inf
max_event_time = -Inf
has_negative = false

# Collect all event times
for (event_type, times) in behavioral_events
    if event_type != "n_events" && times isa Vector && length(times) > 0
        global min_event_time = min(min_event_time, minimum(times))
        global max_event_time = max(max_event_time, maximum(times))
        if any(times .< neural_start)
            global has_negative = true
        end
    end
end

if !isinf(min_event_time)
    println("\nBehavioral events:")
    println("  First event: $(round(min_event_time, digits=3)) s")
    println("  Last event: $(round(max_event_time, digits=3)) s")
end

# Check if behavioral events are outside the FILTERED signal bounds
# (not the original edges, which are longer!)
n_events_outside = 0
for (event_type, times) in behavioral_events
    if event_type != "n_events" && times isa Vector
        global n_events_outside += sum((times .< neural_start) .| (times .> neural_end))
    end
end

if n_events_outside > 0
    println("\n⚠️  TIME ALIGNMENT ISSUE DETECTED!")
    println("   $(n_events_outside) behavioral events are outside filtered signal bounds")
    println("   This is because filtering trimmed the signal edges")
    println("   Filtering only events within filtered signal range...")
    
    # Remove events outside the filtered signal bounds
    for (event_type, times) in behavioral_events
        if event_type != "n_events" && times isa Vector
            n_before = length(times)
            valid_indices = (times .>= neural_start) .& (times .<= neural_end)
            behavioral_events[event_type] = times[valid_indices]
            n_after = length(behavioral_events[event_type])
            n_removed = n_before - n_after
            
            if n_removed > 0
                println("   $(event_type): kept $(n_after)/$(n_before) events (removed $(n_removed))")
            end
        end
    end
    
    # Update total count
    total_events = 0
    for (event_type, times) in behavioral_events
        if event_type != "n_events" && times isa Vector
            global total_events += length(times)
        end
    end
    behavioral_events["n_events"] = total_events
    
    println("\n✓ Events filtered to match signal bounds")
    println("   Total valid events: $(total_events)")
    
else
    println("\n✓ All events within filtered signal bounds")
end

println("="^70)

# ===== LOAD ANALYSIS TOOLS =====
println("\n[Loading Analysis Pipeline]")
include("./integrated_analysis_pipeline.jl")
println("  ✓ Pipeline loaded")

# ===== PRECOMPILE VISUALIZATION FUNCTIONS (FIX FOR FIRST-RUN ERROR) =====
# Julia needs to compile functions with keyword arguments on first use
# This "warm-up" call ensures they're ready before actual use
try
    # Dummy call to trigger compilation of plot_signal_with_events
    if isdefined(Main, :plot_signal_with_events)
        # Create minimal dummy data
        dummy_time = [0.0, 1.0]
        dummy_signal = [0.0, 1.0]
        dummy_events = Dict("test" => [0.5])
        # Call with keywords to trigger kwcall compilation
        plot_signal_with_events(dummy_time, dummy_signal, dummy_events; 
                               title="warmup", time_range=(0.0, 1.0))
        println("  ✓ Visualization functions precompiled")
    end
catch e
    # Silently ignore - function will still work on actual use
end

# ===== RUN COMPREHENSIVE ANALYSIS =====
println("\n[Running Comprehensive Neural Analysis]")
println("This will:")
println("  1. DETECT Sharp-Wave Ripples from neural signals")
println("  2. Compare SWRs to behavioral events you extracted")
println("  3. Calculate enrichment (are SWRs more common near behavior?)")
println()

signal_filtered = fr3_smooth

# ===== CRITICAL: SHIFT EVENTS TO MATCH SWR DETECTION TIME BASE =====
println("\n🔧 SHIFTING EVENT TIMES TO MATCH SWR DETECTION")
time_offset = time_bins[1]  # Get start time of filtered signal (e.g., -1.7s)
println("   Time offset: $(round(time_offset, digits=3))s")
println("   Shifting all events by $(round(-time_offset, digits=3))s")

# Create NEW dictionary with shifted times
behavioral_events_shifted = Dict{String, Any}()

for (event_type, times) in behavioral_events
    if event_type == "n_events"
        continue  # Will recalculate
    elseif times isa Vector && length(times) > 0
        # Shift: convert from time_bins scale to 0-based scale
        shifted_times = times .- time_offset
        behavioral_events_shifted[event_type] = shifted_times
        
        println("   $(event_type): $(round(minimum(shifted_times), digits=3))s to $(round(maximum(shifted_times), digits=3))s")
    end
end

# Recalculate total
total = sum(length(v) for (k, v) in behavioral_events_shifted if v isa Vector)
behavioral_events_shifted["n_events"] = total

# REPLACE the original with shifted version
behavioral_events = behavioral_events_shifted

# VERIFY the shift worked
println("\n📋 VERIFICATION:")
for (event_type, times) in behavioral_events
    if event_type != "n_events" && times isa Vector && length(times) > 0
        min_t = minimum(times)
        max_t = maximum(times)
        println("   $(event_type): min=$(round(min_t, digits=3))s, max=$(round(max_t, digits=3))s")
        
        if min_t < 0
            error("❌ SHIFT FAILED! Events still negative!")
        end
    end
end
println("✓ All events successfully shifted to [0, $(round(length(signal_filtered)/fs, digits=1))]s range")
println("="^70)
println()

results = analyze_neural_data_comprehensive(
    signal_filtered,
    behavioral_events;  # Now using SHIFTED behavioral events
    fs=fs,
    config=Dict(
        # SWR detection parameters
        "ripple_band" => (150.0, 250.0),
        "swr_threshold_sd" => 3.0,
        "swr_min_duration" => 30.0,
        "swr_max_duration" => 200.0,
        
        # Event-triggered analysis
        "event_window_ms" => 500.0,  # Look ±250ms around behavioral events
        
        # ML clustering
        "n_clusters" => 3
    )
)

println("  ✓ Analysis complete")

# ===== DISPLAY RESULTS =====
println("\n" * "="^70)
println("RESULTS SUMMARY")
println("="^70)

# 1. Neural Events (DETECTED by algorithm)
if haskey(results, "swr_detection")
    swr_data = results["swr_detection"]
    n_swr = length(swr_data["events"])
    
    println("\n📊 NEURAL EVENTS DETECTED (Sharp-Wave Ripples):")
    println("  Total SWRs: $n_swr")
    
    if n_swr > 0
        durations = [e.duration_ms for e in swr_data["events"]]
        amplitudes = [e.peak_amplitude for e in swr_data["events"]]
        
        println("  Duration: $(round(mean(durations), digits=1)) ± $(round(std(durations), digits=1)) ms")
        println("  Amplitude: $(round(mean(amplitudes), digits=2)) ± $(round(std(amplitudes), digits=2))")
    end
end

# 2. Behavioral Events (EXTRACTED from recordings)
println("\n🎯 BEHAVIORAL EVENTS EXTRACTED:")
for (event_type, event_times) in behavioral_events
    if event_type != "n_events" && event_times isa Vector
        println("  $event_type: $(length(event_times)) events")
    end
end

# 3. Neural-Behavioral Comparison
println("\n🔬 NEURAL-BEHAVIORAL COMPARISON:")
println("  (Are SWRs enriched near behavioral events?)")

for event_type in keys(behavioral_events)
    if event_type != "n_events" && behavioral_events[event_type] isa Vector
        key = "event_triggered_$(event_type)"
        
        if haskey(results, key)
            event_data = results[key]
            n_events = event_data["total_events"]
            
            # Count SWRs near behavioral events
            swr_per_event = []
            for i in 1:n_events
                event_key = "event_$i"
                if haskey(event_data, event_key)
                    push!(swr_per_event, event_data[event_key]["n_swr"])
                end
            end
            
            total_swr_near = sum(swr_per_event)
            mean_swr_near = length(swr_per_event) > 0 ? mean(swr_per_event) : 0.0
            
            println("\n  Near $event_type:")
            println("    Total SWRs within ±250ms: $total_swr_near")
            println("    Mean SWRs per event: $(round(mean_swr_near, digits=2))")
            
            # Calculate enrichment factor
            if haskey(results, "swr_detection")
                total_recording_swr = results["swr_detection"]["n_events"]
                recording_duration = length(signal_filtered) / fs
                
                if total_recording_swr > 0 && recording_duration > 0
                    baseline_rate = total_recording_swr / recording_duration
                    window_duration = 0.5  # 500ms window
                    expected_per_event = baseline_rate * window_duration
                    
                    if expected_per_event > 0
                        enrichment = mean_swr_near / expected_per_event
                        println("    Enrichment: $(round(enrichment, digits=2))x")
                        
                        if enrichment > 1.5
                            println("    ⭐ ENRICHED - SWRs occur MORE near this behavior")
                        elseif enrichment < 0.7
                            println("    ⬇️  DEPLETED - SWRs occur LESS near this behavior")
                        else
                            println("    ➡️  RANDOM - No special relationship")
                        end
                    end
                end
            end
        end
    end
end

# 4. ML Analysis
if haskey(results, "ml_analysis") && haskey(results["ml_analysis"], "n_events")
    ml_data = results["ml_analysis"]
    
    println("\n🤖 MACHINE LEARNING PATTERN ANALYSIS:")
    println("  SWR event types found: $(ml_data["n_clusters"])")
    for i in 1:ml_data["n_clusters"]
        count = sum(ml_data["cluster_labels"] .== i)
        pct = count / ml_data["n_events"] * 100
        println("    Type $i: $count events ($(round(pct, digits=1))%)")
    end
    
    n_anomalies = sum(ml_data["is_anomaly"])
    println("  Anomalous SWRs: $n_anomalies")
end

# 5. Frequency Bands
if haskey(results, "frequency_bands")
    bands = results["frequency_bands"]
    
    println("\n🎵 FREQUENCY BAND POWER:")
    for band in sort(collect(keys(bands)))
        if band != "total_power" && band != "freq" && band != "power_spectrum"
            info = bands[band]
            rel_power = get(info, "relative_power", 0.0) * 100
            println("  $(rpad(band, 12)): $(round(rel_power, digits=1))%")
        end
    end
end

println("\n" * "="^70)

# ===== SAVE RESULTS =====
println("\nSaving results...")
save_analysis_results(results, output_dir)

println("\n✓ Complete! Results saved to: $output_dir")
println("\nGenerated files:")
println("  - psd.png                # Power spectrum")
println("  - spectrogram.png        # Time-frequency")
println("  - swr_events.png         # Example SWR events")
println("  - ml_clustering.png      # SWR pattern types")
println("  - frequency_bands.png    # Band power distribution")
println("  - analysis_summary.txt   # Text report")

println("\n" * "="^70)
println("KEY TAKEAWAYS")
println("="^70)
println("""
1. NEURAL EVENTS (SWRs): Detected by algorithm from neural signals
2. BEHAVIORAL EVENTS (motion): Extracted from your behavioral recordings
3. TIME ALIGNMENT: Automatically corrected if needed
4. COMPARISON: Check enrichment factors above
   - Enrichment > 1.5: SWRs occur MORE during behavior
   - Enrichment ~ 1.0: No special relationship
   - Enrichment < 0.7: SWRs occur LESS during behavior

5. Next steps:
   - Review plots to verify SWR quality
   - Check if behavioral extraction was successful
   - Try different position_column if needed (line 166)
   - Adjust threshold_quantile for sensitivity (line 167)
   - Adjust swr_threshold_sd for detection sensitivity (line 212)
""")
println("="^70)

#------------------------------------------------------------------------
#%% ===== MULTI-NEURON COMPARATIVE ANALYSIS =====
#------------------------------------------------------------------------
println("\n" * "="^70)
println("MULTI-NEURON COMPARATIVE ANALYSIS")
println("="^70)

# Configuration for multi-neuron analysis
ENABLE_MULTI_NEURON = true  # Set to false to skip this section
neurons_to_analyze = [1, 2, 3, 5, 7, 13, 15, 19, 21, 27, 28, 31]  # Customize this list

if ENABLE_MULTI_NEURON
    println("\nAnalyzing $(length(neurons_to_analyze)) neurons...")
    println("Selected neurons: $neurons_to_analyze")
    
    # Get total number of neurons available
    n_total_neurons = size(neur_tensor_stim1on, 1)
    println("Total neurons in dataset: $n_total_neurons")
    
    # Filter to only include neurons that exist in the data
    valid_neurons = filter(n -> 1 <= n <= n_total_neurons, neurons_to_analyze)
    
    if length(valid_neurons) < length(neurons_to_analyze)
        println("⚠️  Warning: Some requested neurons don't exist in dataset")
        println("   Valid neurons: $valid_neurons")
    end
    
    if length(valid_neurons) >= 2
        println("\n--- Step 1: Extracting Firing Rates ---")
        
        # Extract firing rates for selected neurons (using same condition as before)
        # Condition: column 10==1 & column 3==1 & column 4==4
        trid_multi = findall((cond_matrix[:, 10] .== 1) .& 
                            (cond_matrix[:, 3] .== 1) .& 
                            (cond_matrix[:, 4] .== 4))
        
        n_neurons_selected = length(valid_neurons)
        n_time_points = size(neur_tensor_stim1on, 2)
        
        # Create matrix to hold all neuron firing rates
        # Rows: time points, Columns: neurons
        all_firing_rates = zeros(n_time_points, n_neurons_selected)
        
        for (i, neuron_id) in enumerate(valid_neurons)
            # Extract and average across trials for this neuron
            fr_neuron = neur_tensor_stim1on[neuron_id, :, trid_multi]
            all_firing_rates[:, i] = vec(mean(fr_neuron, dims=2))
        end
        
        println("  ✓ Extracted firing rates: $(size(all_firing_rates))")
        
        # Apply same filtering as used for single neuron
        println("\n--- Step 2: Applying $(selected_filter) Filter ---")
        
        # First, filter one neuron to get the output size
        first_filtered = apply_neural_filter(
            all_firing_rates[:, 1], 
            selected_filter, 
            window_size; 
            filter_params...
        )
        
        # Pre-allocate matrix with correct dimensions
        filtered_length = length(first_filtered)
        firing_rates_smoothed = zeros(filtered_length, n_neurons_selected)
        firing_rates_smoothed[:, 1] = first_filtered
        
        # Filter remaining neurons
        for i in 2:n_neurons_selected
            firing_rates_smoothed[:, i] = apply_neural_filter(
                all_firing_rates[:, i], 
                selected_filter, 
                window_size; 
                filter_params...
            )
        end
        
        println("  ✓ Filtered all $(n_neurons_selected) neurons")
        println("  Signal length: $(size(all_firing_rates, 1)) → $(filtered_length)")
        
        # Adjust time bins to match filtered data
        additional_trim_multi = (size(all_firing_rates, 1) - size(firing_rates_smoothed, 1)) ÷ 2
        time_start_multi = 150 + additional_trim_multi
        time_end_multi = length(edges) - 150 - additional_trim_multi
        time_bins_multi = edges[time_start_multi:time_end_multi]
        
        # Ensure lengths match
        if length(time_bins_multi) != size(firing_rates_smoothed, 1)
            min_len = min(length(time_bins_multi), size(firing_rates_smoothed, 1))
            time_bins_multi = time_bins_multi[1:min_len]
            firing_rates_smoothed = firing_rates_smoothed[1:min_len, :]
        end
        
        # Convert time to milliseconds for analysis
        time_ms = time_bins_multi .* 1000.0  # Convert seconds to milliseconds
        
        println("\n--- Step 3: Creating Neuron Selection ---")
        selection = NeuronSelection(valid_neurons)
        println("  ✓ Selected $(length(selection.neuron_ids)) neurons")
        
        println("\n--- Step 4: Computing Comparative Metrics ---")
        # Transpose firing rates to match expected format (time × neurons)
        metrics = compute_comparative_metrics(
            selection, 
            firing_rates_smoothed, 
            time_ms
        )
        
        println("  ✓ Metrics computed")
        
        # Display TMI values
        println("\n📊 Temporal Modulation Index (TMI) by Neuron:")
        for (i, neuron_id) in enumerate(selection.neuron_ids)
            tmi = metrics.tmi_values[i]
            mean_fr = metrics.mean_firing_rates[i]
            peak_fr = metrics.peak_firing_rates[i]
            println(@sprintf("  Neuron %2d: TMI=%.3f  Mean FR=%.2f Hz  Peak FR=%.2f Hz", 
                            neuron_id, tmi, mean_fr, peak_fr))
        end
        
        # Summary statistics
        println("\n📈 Population Statistics:")
        summary = get_summary_statistics(metrics)
        println(@sprintf("  Mean TMI:         %.3f ± %.3f", 
                        summary["mean_tmi"], summary["std_tmi"]))
        println(@sprintf("  Mean Firing Rate: %.2f Hz", 
                        summary["mean_firing_rate"]))
        println(@sprintf("  Mean Correlation: %.3f", 
                        summary["mean_correlation"]))
        
        # Identify interesting neurons
        println("\n🎯 Notable Neurons:")
        sorted_idx = sortperm(metrics.tmi_values, rev=true)
        println("  Highest TMI (most variable):")
        for i in 1:min(3, length(sorted_idx))
            idx = sorted_idx[i]
            neuron_id = selection.neuron_ids[idx]
            tmi = metrics.tmi_values[idx]
            println(@sprintf("    Neuron %2d: TMI=%.3f", neuron_id, tmi))
        end
        
        println("  Lowest TMI (most stable):")
        for i in length(sorted_idx):-1:max(1, length(sorted_idx)-2)
            idx = sorted_idx[i]
            neuron_id = selection.neuron_ids[idx]
            tmi = metrics.tmi_values[idx]
            println(@sprintf("    Neuron %2d: TMI=%.3f", neuron_id, tmi))
        end
        
        println("\n--- Step 5: Generating Comparative Plots ---")
        
        # Create multi-neuron output directory
        multi_output_dir = joinpath(output_dir, "multi_neuron_analysis")
        if !isdir(multi_output_dir)
            mkpath(multi_output_dir)
        end
        
        # Generate comprehensive report with all plots
        create_comparative_report(
            metrics, 
            firing_rates_smoothed, 
            time_ms, 
            multi_output_dir
        )
        
        println("  ✓ All comparative plots generated")
        
        # Generate individual plots with custom settings
        println("\n--- Step 6: Creating Custom Visualizations ---")
        
        # 1. Main comparative plot (like the example figure)
        # Adjust layout based on number of neurons
        n_cols = 4
        n_rows = ceil(Int, length(selection.neuron_ids) / n_cols)
        
        plot_comparative_firing_rates(
            selection,
            firing_rates_smoothed,
            time_ms,
            metrics.tmi_values,
            layout_dims=(n_rows, n_cols),
            sort_by_tmi=true,
            figsize=(1400, 300*n_rows),
            save_path=joinpath(multi_output_dir, "comparative_firing_rates_sorted.png")
        )
        println("  ✓ Comparative firing rates plot (sorted by TMI)")
        
        # 2. Highlight specific neurons of interest
        if length(selection.neuron_ids) > 4
            # Highlight top 3 high-TMI neurons
            highlight_idx = sortperm(metrics.tmi_values, rev=true)[1:min(3, length(selection.neuron_ids))]
            highlight_neurons = selection.neuron_ids[highlight_idx]
            
            plot_temporal_dynamics(
                selection,
                firing_rates_smoothed,
                time_ms,
                highlight_neurons,
                save_path=joinpath(multi_output_dir, "temporal_dynamics_highlighted.png")
            )
            println("  ✓ Temporal dynamics with highlighted neurons")
        end
        
        # 3. Additional correlation analysis
        println("\n📊 Cross-Neuron Correlation Analysis:")
        local n_pairs = 0  # Declare as local to avoid scope ambiguity
        strong_correlations = []
        
        for i in 1:length(selection.neuron_ids)
            for j in (i+1):length(selection.neuron_ids)
                corr = metrics.correlation_matrix[i, j]
                if abs(corr) > 0.7  # Strong correlation threshold
                    n_pairs += 1
                    push!(strong_correlations, (
                        selection.neuron_ids[i], 
                        selection.neuron_ids[j], 
                        corr
                    ))
                end
            end
        end
        
        if n_pairs > 0
            println("  Found $n_pairs strongly correlated neuron pairs (|r| > 0.7):")
            for (n1, n2, corr) in sort(strong_correlations, by=x->abs(x[3]), rev=true)[1:min(5, n_pairs)]
                println(@sprintf("    Neurons %2d ↔ %2d: r = %+.3f", n1, n2, corr))
            end
        else
            println("  No strongly correlated pairs found (threshold |r| > 0.7)")
        end
        
        # 4. Burst analysis summary
        println("\n🔥 Burst Analysis Summary:")
        total_bursts = sum(metrics.burst_characteristics[nid]["n_bursts"] 
                          for nid in selection.neuron_ids)
        println("  Total bursts detected: $total_bursts")
        
        if total_bursts > 0
            println("  Bursts by neuron:")
            for neuron_id in selection.neuron_ids
                n_bursts = metrics.burst_characteristics[neuron_id]["n_bursts"]
                if n_bursts > 0
                    mean_dur = metrics.burst_characteristics[neuron_id]["mean_duration"]
                    mean_peak = metrics.burst_characteristics[neuron_id]["mean_peak_rate"]
                    println(@sprintf("    Neuron %2d: %d bursts, %.1f ms duration, %.1f Hz peak", 
                                    neuron_id, n_bursts, mean_dur, mean_peak))
                end
            end
        end
        
        println("\n✓ Multi-neuron analysis complete!")
        println("\n📁 Results saved to: $multi_output_dir")
        println("\nGenerated comparative analysis files:")
        println("  - comparative_firing_rates.png          # Grid layout with all neurons")
        println("  - comparative_firing_rates_sorted.png   # Sorted by TMI")
        println("  - correlation_matrix.png                # Neuron correlation heatmap")
        println("  - tmi_distribution.png                  # TMI distribution")
        println("  - burst_comparison.png                  # Burst statistics")
        println("  - temporal_dynamics.png                 # All neurons overlaid")
        println("  - temporal_dynamics_highlighted.png     # Highlight top neurons")
        println("  - summary_statistics.png                # Metric relationships")
        
    else
        println("⚠️  Need at least 2 valid neurons for comparative analysis")
        println("   Requested: $neurons_to_analyze")
        println("   Valid: $valid_neurons")
    end
else
    println("Multi-neuron analysis disabled (set ENABLE_MULTI_NEURON = true to enable)")
end

println("\n" * "="^70)
