"""
Multi-Neuron SWR Plotting Module
Visualization functions for Sharp-Wave Ripple comparative analysis

Author: Enhanced for SWR comparative visualization
"""

module MultiNeuronSWRPlots

using Plots
using Statistics
using Colors
using LaTeXStrings
using Printf

# Import from the SWR module
import Main.MultiNeuronSWR: SWREvent, SWRMetrics

export plot_swr_comparison, plot_co_ripple_matrix, plot_swr_raster
export plot_swr_properties_distribution, plot_swr_synchrony, create_swr_report

"""
    plot_swr_comparison(swr_metrics::SWRMetrics;
                       save_path::Union{String,Nothing}=nothing)
    
Plot comparison of SWR statistics across neurons
"""
function plot_swr_comparison(swr_metrics::SWRMetrics;
                            save_path::Union{String,Nothing}=nothing)
    
    neuron_labels = string.(swr_metrics.neuron_ids)
    n_neurons = length(neuron_labels)
    
    # Create 2x2 subplot
    p1 = bar(1:n_neurons, swr_metrics.swr_counts,
             xlabel="Neuron", ylabel="SWR Count",
             title="Total SWR Events",
             xticks=(1:n_neurons, neuron_labels),
             legend=false, color=:steelblue, xrotation=45)
    
    p2 = bar(1:n_neurons, swr_metrics.swr_rates,
             xlabel="Neuron", ylabel="Rate (events/s)",
             title="SWR Rate",
             xticks=(1:n_neurons, neuron_labels),
             legend=false, color=:coral, xrotation=45)
    
    p3 = bar(1:n_neurons, swr_metrics.mean_durations,
             xlabel="Neuron", ylabel="Duration (ms)",
             title="Mean SWR Duration",
             xticks=(1:n_neurons, neuron_labels),
             legend=false, color=:seagreen, xrotation=45)
    
    p4 = bar(1:n_neurons, swr_metrics.mean_frequencies,
             xlabel="Neuron", ylabel="Frequency (Hz)",
             title="Mean Peak Frequency",
             xticks=(1:n_neurons, neuron_labels),
             legend=false, color=:purple, xrotation=45)
    
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800),
                   plot_title="Sharp-Wave Ripple Comparison Across Neurons")
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        println("Saved SWR comparison to $save_path")
    end
    
    return combined
end

"""
    plot_co_ripple_matrix(swr_metrics::SWRMetrics;
                         save_path::Union{String,Nothing}=nothing)
    
Plot heatmap of co-rippling probabilities
"""
function plot_co_ripple_matrix(swr_metrics::SWRMetrics;
                               save_path::Union{String,Nothing}=nothing)
    
    neuron_labels = string.(swr_metrics.neuron_ids)
    n_neurons = length(neuron_labels)
    
    p = heatmap(1:n_neurons, 1:n_neurons,
               swr_metrics.co_ripple_matrix,
               xlabel="Neuron", ylabel="Neuron",
               title="Co-Rippling Probability Matrix",
               xticks=(1:n_neurons, neuron_labels),
               yticks=(1:n_neurons, neuron_labels),
               color=:viridis, clims=(0, 1),
               aspect_ratio=1, size=(700, 650),
               colorbar_title="Co-Ripple Probability")
    
    # Add probability values as text
    if n_neurons <= 15  # Only add text for smaller matrices
        for i in 1:n_neurons
            for j in 1:n_neurons
                if i != j && swr_metrics.co_ripple_matrix[i,j] > 0.1
                    annotate!(j, i, text(@sprintf("%.2f", swr_metrics.co_ripple_matrix[i,j]), 
                                        8, :white))
                end
            end
        end
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved co-ripple matrix to $save_path")
    end
    
    return p
end

"""
    plot_swr_raster(swr_metrics::SWRMetrics, time_range::Tuple{Float64,Float64};
                   save_path::Union{String,Nothing}=nothing)
    
Create raster plot of SWR events across neurons
"""
function plot_swr_raster(swr_metrics::SWRMetrics, 
                        time_range::Tuple{Float64,Float64}=(0.0, Inf);
                        save_path::Union{String,Nothing}=nothing)
    
    n_neurons = length(swr_metrics.neuron_ids)
    
    p = plot(xlabel="Time (s)", ylabel="Neuron",
            title="SWR Event Raster Plot",
            yticks=(1:n_neurons, string.(swr_metrics.neuron_ids)),
            legend=false, size=(1200, 400 + 20*n_neurons),
            ylims=(0.5, n_neurons+0.5))
    
    # Plot each neuron's events as vertical lines
    for (i, events) in enumerate(swr_metrics.events_per_neuron)
        for event in events
            time_s = event.peak_time_ms / 1000.0
            
            # Filter by time range
            if time_s >= time_range[1] && time_s <= time_range[2]
                # Draw vertical line for event
                plot!([time_s, time_s], [i-0.4, i+0.4],
                     color=:black, linewidth=2, alpha=0.7)
                
                # Color code by amplitude (optional)
                amplitude_color = get(colormap("viridis"), 
                                    min(1.0, event.peak_amplitude / 10.0))
                scatter!([time_s], [i], markersize=4, 
                        color=amplitude_color, alpha=0.8)
            end
        end
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved SWR raster plot to $save_path")
    end
    
    return p
end

"""
    plot_swr_properties_distribution(swr_metrics::SWRMetrics;
                                    save_path::Union{String,Nothing}=nothing)
    
Plot distributions of SWR properties across all neurons
"""
function plot_swr_properties_distribution(swr_metrics::SWRMetrics;
                                         save_path::Union{String,Nothing}=nothing)
    
    # Collect all events from all neurons
    all_durations = Float64[]
    all_amplitudes = Float64[]
    all_frequencies = Float64[]
    
    for events in swr_metrics.events_per_neuron
        append!(all_durations, [e.duration_ms for e in events])
        append!(all_amplitudes, [e.peak_amplitude for e in events])
        append!(all_frequencies, [e.peak_frequency for e in events])
    end
    
    if length(all_durations) == 0
        println("No SWR events to plot")
        return nothing
    end
    
    # Create histograms
    p1 = histogram(all_durations,
                  xlabel="Duration (ms)", ylabel="Count",
                  title="SWR Duration Distribution",
                  bins=20, legend=false, color=:steelblue, alpha=0.7)
    vline!([mean(all_durations)], linewidth=2, linestyle=:dash, 
           color=:red, label="Mean")
    
    p2 = histogram(all_amplitudes,
                  xlabel="Peak Amplitude", ylabel="Count",
                  title="SWR Amplitude Distribution",
                  bins=20, legend=false, color=:coral, alpha=0.7)
    vline!([mean(all_amplitudes)], linewidth=2, linestyle=:dash, 
           color=:red, label="Mean")
    
    p3 = histogram(all_frequencies,
                  xlabel="Peak Frequency (Hz)", ylabel="Count",
                  title="SWR Frequency Distribution",
                  bins=20, legend=false, color=:seagreen, alpha=0.7)
    vline!([mean(all_frequencies)], linewidth=2, linestyle=:dash, 
           color=:red, label="Mean")
    
    # Scatter plot: Duration vs Amplitude
    p4 = scatter(all_durations, all_amplitudes,
                xlabel="Duration (ms)", ylabel="Peak Amplitude",
                title="Duration vs Amplitude",
                markersize=3, alpha=0.5, color=:purple, legend=false)
    
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800),
                   plot_title="SWR Properties Distribution")
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        println("Saved SWR properties distribution to $save_path")
    end
    
    return combined
end

"""
    plot_swr_synchrony(swr_metrics::SWRMetrics;
                      save_path::Union{String,Nothing}=nothing)
    
Plot synchrony scores showing which neurons co-ripple frequently
"""
function plot_swr_synchrony(swr_metrics::SWRMetrics;
                           save_path::Union{String,Nothing}=nothing)
    
    neuron_labels = string.(swr_metrics.neuron_ids)
    n_neurons = length(neuron_labels)
    
    # Sort by synchrony score
    sort_idx = sortperm(swr_metrics.synchrony_scores, rev=true)
    sorted_scores = swr_metrics.synchrony_scores[sort_idx]
    sorted_labels = neuron_labels[sort_idx]
    
    p = bar(1:n_neurons, sorted_scores,
           xlabel="Neuron (sorted by synchrony)",
           ylabel="Synchrony Score",
           title="SWR Synchrony Across Neurons",
           xticks=(1:n_neurons, sorted_labels),
           legend=false, color=:viridis,
           xrotation=45, size=(800, 500))
    
    # Add reference line
    mean_sync = mean(sorted_scores)
    hline!([mean_sync], linewidth=2, linestyle=:dash, color=:red,
           label="Mean = $(round(mean_sync, digits=3))")
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved SWR synchrony plot to $save_path")
    end
    
    return p
end

"""
    create_swr_report(swr_metrics::SWRMetrics, output_dir::String;
                     time_range::Tuple{Float64,Float64}=(0.0, Inf))
    
Generate complete SWR analysis report with all plots
"""
function create_swr_report(swr_metrics::SWRMetrics, output_dir::String;
                          time_range::Tuple{Float64,Float64}=(0.0, Inf))
    
    # Create output directory if needed
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("Generating SWR analysis report...")
    
    # Generate all plots
    plot_swr_comparison(
        swr_metrics,
        save_path=joinpath(output_dir, "swr_comparison.png")
    )
    
    plot_co_ripple_matrix(
        swr_metrics,
        save_path=joinpath(output_dir, "co_ripple_matrix.png")
    )
    
    plot_swr_raster(
        swr_metrics,
        time_range,
        save_path=joinpath(output_dir, "swr_raster.png")
    )
    
    plot_swr_properties_distribution(
        swr_metrics,
        save_path=joinpath(output_dir, "swr_properties.png")
    )
    
    plot_swr_synchrony(
        swr_metrics,
        save_path=joinpath(output_dir, "swr_synchrony.png")
    )
    
    println("SWR report generated in $output_dir")
end

end # module
