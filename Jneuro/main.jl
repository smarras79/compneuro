using MAT
using Statistics
using DSP
using Plots

include("./myplots.jl")
include("./behavioral_event_extraction.jl")
include("./auxiliary_functions.jl")  # Load filter function

# ========== CONFIGURATION ==========
# Define these BEFORE using them
fs = 1000.0           # Sampling frequency (Hz)
window_size = 100     # Filter window size

println("Configuration:")
println("  Sampling frequency: $(fs) Hz")
println("  Window size: $(window_size)")
println()

# Load the .mat file
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
# Alternative file: amadeus01172020_a_neur_tensor_joyon.mat

output_dir = "./neural_analysis_output"

# 
# ==== 1. Select filter type (change this to try different filters)
#         Options:
#                   :moving_average
#                   :gaussian
#                   :savitzky_golay
#                   :butterworth
#                   :median
#                   :exponential
#
selected_filter = :none #:moving_average  # Default: same as original code

# Additional parameters for specific filters (adjust as needed)
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

# Display condition labels (tells you which column codes for what task parameter)
println("Condition labels:")
println(cond_label)
println()

#%% Extract neural data (THIS WAS MISSING!)
println("Extracting neural data...")

# Extract firing rates for condition 4 (column 10==1 & column 3==1 & column 4==4)
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]
println("  fr3 extracted: $(size(fr3)) from $(length(trid)) trials")

# Extract firing rates for condition 5 (column 10==1 & column 3==1 & column 4==5)
trid = findall((cond_matrix[:, 10] .== 1) .& 
               (cond_matrix[:, 3] .== 1) .& 
               (cond_matrix[:, 4] .== 5))
fr4 = neur_tensor_stim1on[1, :, trid]
println("  fr4 extracted: $(size(fr4)) from $(length(trid)) trials")
println()

#%% Behavioural data
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


#%% Apply selected filter to neural data
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

# Plot neural data and save to file
# Time bins need to match the filtered signal length
# The filtered signals are trimmed by (window_size - 1) total samples
# Original code used edges[150:end-150], but we need to account for filter trimming
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

#%% ===== EXTRACT BEHAVIORAL EVENTS (NEW!) =====
println("\n" * "="^70)
println("EXTRACTING BEHAVIORAL EVENTS")
println("="^70)


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

println("="^70)

# ===== LOAD ANALYSIS TOOLS =====
println("\n[Loading Analysis Pipeline]")
include("./integrated_analysis_pipeline.jl")
println("  ✓ Pipeline loaded")

# ===== RUN COMPREHENSIVE ANALYSIS =====
println("\n[Running Comprehensive Neural Analysis]")
println("This will:")
println("  1. DETECT Sharp-Wave Ripples from neural signals")
println("  2. Compare SWRs to behavioral events you extracted")
println("  3. Calculate enrichment (are SWRs more common near behavior?)")
println()

signal_filtered = fr3_smooth

results = analyze_neural_data_comprehensive(
    signal_filtered,
    behavioral_events;  # ← NOW USING EXTRACTED BEHAVIORAL EVENTS!
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
3. COMPARISON: Check enrichment factors above
   - Enrichment > 1.5: SWRs occur MORE during behavior
   - Enrichment ~ 1.0: No special relationship
   - Enrichment < 0.7: SWRs occur LESS during behavior

4. Next steps:
   - Review plots to verify SWR quality
   - Check if behavioral extraction was successful
   - Try different position_column if needed (line 166)
   - Adjust threshold_quantile for sensitivity (line 167)
""")
println("="^70)
