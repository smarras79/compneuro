"""
Integrated Neural Analysis Pipeline

Combines spectral analysis, SWR detection, and ML pattern recognition
into a unified workflow for analyzing neural data with behavioral events.

Author: Enhanced Neural Analysis Toolkit
"""

include("neural_spectral_analysis.jl")
include("swr_detection.jl")
include("ml_pattern_detection.jl")

using Plots
using Printf

"""
    analyze_neural_data_comprehensive(signal::Vector{Float64}, 
                                     behavioral_events=nothing;
                                     fs=1000.0,
                                     config=Dict())

Comprehensive neural data analysis pipeline.

# Performs:
1. Spectral analysis (PSD, frequency bands, spectrograms)
2. Sharp-Wave Ripple detection
3. Event-triggered analysis (if behavioral events provided)
4. ML-based pattern detection and classification
5. Generates comprehensive visualizations

# Arguments
- `signal`: Neural time series (LFP/single unit)
- `behavioral_events`: Optional dict with "onset" and "offset" times
- `fs`: Sampling frequency in Hz
- `config`: Configuration dict (see default_analysis_config())

# Returns
Dictionary with all analysis results
"""
function analyze_neural_data_comprehensive(signal::Vector{Float64},
                                          behavioral_events=nothing;
                                          fs=1000.0,
                                          config=Dict())
    
    # Merge with default config
    cfg = merge(default_analysis_config(), config)
    
    println("\n" * "="^70)
    println("COMPREHENSIVE NEURAL ANALYSIS PIPELINE")
    println("="^70)
    println("Signal length: $(length(signal)) samples ($(length(signal)/fs) seconds)")
    println("Sampling frequency: $(fs) Hz")
    println()
    
    results = Dict()
    results["fs"] = fs
    results["signal_length"] = length(signal)
    
    # ========== 1. SPECTRAL ANALYSIS ==========
    println("Step 1/5: Spectral Analysis...")
    
    try
        # Power spectral density
        freq, power = compute_psd(signal, fs; 
                                 method=:welch, 
                                 nperseg=cfg["psd_window"])
        results["psd"] = Dict("freq" => freq, "power" => power)
        
        # Frequency band analysis
        band_analysis = analyze_frequency_bands(signal, fs)
        results["frequency_bands"] = band_analysis
        
        # Print band powers
        println("  Frequency band analysis:")
        # Sort only the keys, not the nested dictionary values
        for band in sort(collect(keys(band_analysis)))
            if band != "total_power" && band != "freq" && band != "power_spectrum"
                info = band_analysis[band]
                rel_power = get(info, "relative_power", 0.0)
                println(@sprintf("    %12s: %.2f%% of total power", 
                               band, rel_power * 100))
            end
        end
        
        # Spectrogram
        if cfg["compute_spectrogram"]
            times, freq_spec, spec = compute_spectrogram(signal, fs;
                                                         nperseg=cfg["spec_window"])
            results["spectrogram"] = Dict(
                "times" => times,
                "frequencies" => freq_spec,
                "power" => spec
            )
        end
        
        println("  ✓ Spectral analysis complete")
    catch e
        println("  ✗ Spectral analysis failed: $e")
        results["spectral_error"] = string(e)
    end
    
    # ========== 2. SHARP-WAVE RIPPLE DETECTION ==========
    println("\nStep 2/5: Sharp-Wave Ripple Detection...")
    
    try
        swr_results = detect_swr_classical(
            signal, fs;
            ripple_band=cfg["ripple_band"],
            threshold_sd=cfg["swr_threshold_sd"],
            min_duration_ms=cfg["swr_min_duration"],
            max_duration_ms=cfg["swr_max_duration"]
        )
        
        results["swr_detection"] = swr_results
        n_swr = length(swr_results["events"])
        println("  Detected $n_swr SWR events")
        
        if n_swr > 0
            durations = [e.duration_ms for e in swr_results["events"]]
            amplitudes = [e.peak_amplitude for e in swr_results["events"]]
            
            println(@sprintf("    Mean duration: %.1f ms (range: %.1f-%.1f ms)",
                           mean(durations), minimum(durations), maximum(durations)))
            println(@sprintf("    Mean amplitude: %.2f (range: %.2f-%.2f)",
                           mean(amplitudes), minimum(amplitudes), maximum(amplitudes)))
        end
        
        println("  ✓ SWR detection complete")
    catch e
        println("  ✗ SWR detection failed: $e")
        results["swr_error"] = string(e)
    end
    
    # ========== 3. EVENT-TRIGGERED ANALYSIS ==========
    if !isnothing(behavioral_events)
        println("\nStep 3/5: Event-Triggered Analysis...")
        
        # Debug: Show what's in behavioral_events
        println("  Debug - Behavioral events dictionary:")
        for (key, value) in behavioral_events
            if isa(value, AbstractVector)
                println("    $key: Vector with $(length(value)) elements")
                if length(value) > 0
                    println("      First value: $(round(value[1], digits=3))")
                    if length(value) > 1
                        println("      Last value: $(round(value[end], digits=3))")
                    end
                end
            else
                println("    $key: $(typeof(value)) = $value")
            end
        end
        println()
        
        try
            for (event_type, event_times) in behavioral_events
                # Skip metadata fields (n_events is an integer, not event times)
                if event_type == "n_events" || !isa(event_times, AbstractVector)
                    println("  ⊘ Skipping $event_type ($(typeof(event_times)), not a vector of times)")
                    continue
                end
                
                # Skip if no events
                if length(event_times) == 0
                    println("  ⊘ Skipping $event_type (no events)")
                    continue
                end
                
                println("  Analyzing $event_type events (n=$(length(event_times)))...")
                
                event_swr = detect_swr_at_events(
                    signal, event_times, fs;
                    window_ms=cfg["event_window_ms"],
                    ripple_band=cfg["ripple_band"],
                    threshold_sd=cfg["swr_threshold_sd"]
                )
                
                results["event_triggered_$(event_type)"] = event_swr
                
                # Count SWRs per event
                swr_counts = [event_swr["event_$i"]["n_swr"] 
                             for i in 1:length(event_times)]
                total_swr = sum(swr_counts)
                
                println(@sprintf("    Total SWRs near %s: %d (%.2f per event)",
                               event_type, total_swr, mean(swr_counts)))
            end
            
            println("  ✓ Event-triggered analysis complete")
        catch e
            println("  ✗ Event-triggered analysis failed: $e")
            results["event_triggered_error"] = string(e)
        end
    else
        println("\nStep 3/5: Event-Triggered Analysis... (skipped - no events provided)")
    end
    
    # ========== 4. ML PATTERN DETECTION ==========
    println("\nStep 4/5: Machine Learning Pattern Detection...")
    
    try
        if haskey(results, "swr_detection") && length(results["swr_detection"]["events"]) > 0
            # Analyze SWR patterns using ML
            ml_results = analyze_event_patterns(
                signal, 
                results["swr_detection"]["events"];
                fs=fs,
                n_clusters=cfg["n_clusters"]
            )
            
            results["ml_analysis"] = ml_results
            
            if haskey(ml_results, "n_events")
                println("  Events analyzed: $(ml_results["n_events"])")
                println("  Clusters found: $(ml_results["n_clusters"])")
                
                # Cluster distribution
                cluster_counts = [sum(ml_results["cluster_labels"] .== i) 
                                for i in 1:ml_results["n_clusters"]]
                for (i, count) in enumerate(cluster_counts)
                    pct = count / ml_results["n_events"] * 100
                    println(@sprintf("    Cluster %d: %d events (%.1f%%)", i, count, pct))
                end
                
                # Anomalies
                n_anomalies = sum(ml_results["is_anomaly"])
                println(@sprintf("  Anomalies detected: %d (%.1f%%)",
                               n_anomalies, n_anomalies / ml_results["n_events"] * 100))
            end
            
            println("  ✓ ML pattern detection complete")
        else
            println("  ⊘ ML pattern detection skipped (no events detected)")
        end
    catch e
        println("  ✗ ML pattern detection failed: $e")
        results["ml_error"] = string(e)
    end
    
    # ========== 5. GENERATE VISUALIZATIONS ==========
    println("\nStep 5/5: Generating Visualizations...")
    
    try
        plots = generate_analysis_plots(results, signal, fs, behavioral_events, cfg)
        results["plots"] = plots
        println("  ✓ Visualizations generated")
    catch e
        println("  ✗ Visualization failed: $e")
        results["plot_error"] = string(e)
    end
    
    println("\n" * "="^70)
    println("ANALYSIS COMPLETE")
    println("="^70)
    
    return results
end

"""
    default_analysis_config()

Returns default configuration for comprehensive analysis.
"""
function default_analysis_config()
    return Dict(
        # Spectral analysis
        "psd_window" => 512,
        "spec_window" => 256,
        "compute_spectrogram" => true,
        
        # SWR detection
        "ripple_band" => (150.0, 250.0),
        "swr_threshold_sd" => 3.0,
        "swr_min_duration" => 30.0,
        "swr_max_duration" => 200.0,
        
        # Event-triggered
        "event_window_ms" => 500.0,
        
        # ML analysis
        "n_clusters" => 3,
        
        # Visualization
        "plot_time_range" => (0.0, 10.0),  # seconds to plot
        "plot_freq_range" => (0.0, 300.0)   # Hz
    )
end

"""
    generate_analysis_plots(results, signal, fs, behavioral_events, config)

Generate comprehensive visualization plots.
"""
function generate_analysis_plots(results, signal, fs, behavioral_events, config)
    
    plots_dict = Dict()
    
    # 1. Power Spectral Density
    if haskey(results, "psd")
        psd_data = results["psd"]
        p = plot_psd(psd_data["freq"], psd_data["power"];
                    title="Power Spectral Density",
                    freq_range=config["plot_freq_range"])
        plots_dict["psd"] = p
    end
    
    # 2. Spectrogram
    if haskey(results, "spectrogram")
        spec_data = results["spectrogram"]
        p = plot_spectrogram(spec_data["times"], spec_data["frequencies"], 
                            spec_data["power"];
                            title="Time-Frequency Spectrogram",
                            freq_range=config["plot_freq_range"])
        plots_dict["spectrogram"] = p
    end
    
    # 3. SWR Events
    if haskey(results, "swr_detection") && length(results["swr_detection"]["events"]) > 0
        p = visualize_swr_events(signal, results["swr_detection"]["events"], fs;
                                max_events=5)
        if !isnothing(p)
            plots_dict["swr_events"] = p
        end
    end
    
    # 4. ML Clustering Results
    if haskey(results, "ml_analysis") && haskey(results["ml_analysis"], "pca_transformed")
        ml = results["ml_analysis"]
        pca_data = ml["pca_transformed"]
        labels = ml["cluster_labels"]
        
        p = scatter(pca_data[:, 1], pca_data[:, 2],
                   group=labels,
                   xlabel="PC1 ($(round(ml["explained_variance"][1]*100, digits=1))%)",
                   ylabel="PC2 ($(round(ml["explained_variance"][2]*100, digits=1))%)",
                   title="SWR Event Clustering (PCA)",
                   markersize=6,
                   legend=:topright,
                   size=(700, 600))
        
        plots_dict["ml_clustering"] = p
    end
    
    # 5. Frequency Band Power Distribution
    if haskey(results, "frequency_bands")
        bands = results["frequency_bands"]
        
        band_names = []
        powers = []
        
        # Sort only by band names (keys)
        for name in sort(collect(keys(bands)))
            if name != "total_power" && name != "freq" && name != "power_spectrum"
                info = bands[name]
                push!(band_names, name)
                push!(powers, get(info, "relative_power", 0.0) * 100)
            end
        end
        
        if length(band_names) > 0
            p = bar(band_names, powers,
                   xlabel="Frequency Band",
                   ylabel="Relative Power (%)",
                   title="Power Distribution Across Frequency Bands",
                   legend=false,
                   xrotation=45,
                   size=(800, 500))
            
            plots_dict["frequency_bands"] = p
        end
    end
    
    return plots_dict
end

"""
    save_analysis_results(results, output_dir="./neural_analysis_results")

Save all analysis results to files.
"""
function save_analysis_results(results, output_dir="./neural_analysis_results")
    
    # Create output directory
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    println("\nSaving results to: $output_dir")
    
    # Save plots
    if haskey(results, "plots")
        for (name, p) in results["plots"]
            filepath = joinpath(output_dir, "$(name).png")
            savefig(p, filepath)
            println("  Saved: $(name).png")
        end
    end
    
    # Save summary statistics to text file
    summary_file = joinpath(output_dir, "analysis_summary.txt")
    open(summary_file, "w") do io
        write(io, "NEURAL ANALYSIS SUMMARY\n")
        write(io, "="^70 * "\n\n")
        
        write(io, "Data Properties:\n")
        write(io, "  Signal length: $(results["signal_length"]) samples\n")
        write(io, "  Sampling frequency: $(results["fs"]) Hz\n\n")
        
        if haskey(results, "swr_detection")
            n_swr = length(results["swr_detection"]["events"])
            write(io, "Sharp-Wave Ripples:\n")
            write(io, "  Total detected: $n_swr\n")
            
            if n_swr > 0
                events = results["swr_detection"]["events"]
                durations = [e.duration_ms for e in events]
                write(io, @sprintf("  Mean duration: %.1f ms\n", mean(durations)))
                write(io, @sprintf("  Duration range: %.1f - %.1f ms\n", 
                                 minimum(durations), maximum(durations)))
            end
            write(io, "\n")
        end
        
        if haskey(results, "ml_analysis")
            ml = results["ml_analysis"]
            write(io, "Machine Learning Analysis:\n")
            write(io, "  Events analyzed: $(ml["n_events"])\n")
            write(io, "  Clusters identified: $(ml["n_clusters"])\n")
            write(io, "  Anomalies detected: $(sum(ml["is_anomaly"]))\n")
            write(io, "\n")
        end
    end
    
    println("  Saved: analysis_summary.txt")
    println("\n✓ All results saved successfully")
end

"""
    quick_swr_analysis(signal, fs=1000.0; plot_results=true)

Quick SWR analysis with sensible defaults and automatic visualization.
"""
function quick_swr_analysis(signal::Vector{Float64}, fs::Float64=1000.0; 
                           plot_results=true)
    
    println("Running quick SWR analysis...")
    
    # Detect SWRs
    results = detect_swr_classical(signal, fs)
    
    n_events = length(results["events"])
    println("Detected $n_events SWR events")
    
    if plot_results && n_events > 0
        # Visualize
        p = visualize_swr_events(signal, results["events"], fs; max_events=5)
        display(p)
    end
    
    return results
end

println("\n" * "="^70)
println("✓ Integrated Neural Analysis Pipeline loaded")
println("="^70)
println("\nMain functions:")
println("  - analyze_neural_data_comprehensive() - Full analysis pipeline")
println("  - quick_swr_analysis()                - Quick SWR detection")
println("  - save_analysis_results()             - Save results to files")
println("\nConfiguration:")
println("  - default_analysis_config()           - Get default settings")
println("="^70)
