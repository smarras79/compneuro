"""
Multi-Neuron Comparative Plotting Module
Generate publication-quality comparative plots for multiple neurons

Author: Enhanced for comparative visualization
"""

module MultiNeuronPlots

using Plots
using Statistics
using Colors
using LaTeXStrings

# Import types from MultiNeuronAnalysis module
# Note: MultiNeuronAnalysis must be included/loaded before this module
import Main.MultiNeuronAnalysis: NeuronSelection, ComparativeMetrics

export plot_comparative_firing_rates, plot_correlation_matrix, plot_tmi_distribution
export plot_burst_comparison, plot_temporal_dynamics, configure_plot_style
export plot_summary_statistics, create_comparative_report

"""
    configure_plot_style()
    
Configure consistent plotting style for all comparative plots
"""
function configure_plot_style()
    # Set default plot attributes
    default(
        fontfamily = "Computer Modern",
        framestyle = :box,
        grid = false,
        guidefontsize = 10,
        tickfontsize = 8,
        titlefontsize = 11,
        legendfontsize = 8,
        linewidth = 1.5,
        markersize = 4,
        dpi = 300
    )
end

"""
    plot_comparative_firing_rates(selection::NeuronSelection,
                                  firing_rates::Matrix{Float64},
                                  time::Vector{Float64},
                                  tmi_values::Vector{Float64};
                                  layout_dims::Tuple{Int,Int}=(3,4),
                                  sort_by_tmi::Bool=true,
                                  figsize::Tuple{Int,Int}=(1200, 900),
                                  save_path::Union{String,Nothing}=nothing)
    
Create multi-panel comparative plot similar to the provided figure
"""
function plot_comparative_firing_rates(selection::NeuronSelection,
                                      firing_rates::Matrix{Float64},
                                      time::Vector{Float64},
                                      tmi_values::Vector{Float64};
                                      layout_dims::Tuple{Int,Int}=(3,4),
                                      sort_by_tmi::Bool=true,
                                      figsize::Tuple{Int,Int}=(1200, 900),
                                      save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    n_neurons = length(selection.neuron_ids)
    n_rows, n_cols = layout_dims
    
    # Sort neurons by TMI if requested
    if sort_by_tmi
        sort_idx = sortperm(tmi_values)
    else
        sort_idx = 1:n_neurons
    end
    
    # Create subplots
    plots_array = []
    
    for (plot_idx, neuron_idx) in enumerate(sort_idx)
        if plot_idx > n_rows * n_cols
            break
        end
        
        neuron_id = selection.neuron_ids[neuron_idx]
        fr = firing_rates[:, neuron_idx]
        tmi = tmi_values[neuron_idx]
        
        # Create subplot
        p = plot(
            time, fr,
            fillrange = 0,
            fillalpha = 0.3,
            fillcolor = :steelblue,
            linecolor = :black,
            linewidth = 1.5,
            label = "",
            xlabel = "Time (ms)",
            ylabel = "Firing Rate (Hz)",
            title = "Neuron $(neuron_id) (TMI=$(round(tmi, digits=2)))",
            titlefontsize = 10,
            xlims = (minimum(time), maximum(time)),
            ylims = (0, maximum(fr) * 1.1)
        )
        
        push!(plots_array, p)
    end
    
    # Combine into single figure
    combined_plot = plot(
        plots_array...,
        layout = (n_rows, n_cols),
        size = figsize,
        margin = 5Plots.mm
    )
    
    # Save if path provided
    if !isnothing(save_path)
        savefig(combined_plot, save_path)
        println("Saved comparative plot to $save_path")
    end
    
    return combined_plot
end

"""
    plot_correlation_matrix(correlation_matrix::Matrix{Float64},
                           neuron_ids::Vector{Int};
                           save_path::Union{String,Nothing}=nothing)
    
Plot heatmap of cross-neuron correlations
"""
function plot_correlation_matrix(correlation_matrix::Matrix{Float64},
                                neuron_ids::Vector{Int};
                                save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    n_neurons = length(neuron_ids)
    
    # Create heatmap
    p = heatmap(
        1:n_neurons, 1:n_neurons,
        correlation_matrix,
        color = :RdBu,
        clims = (-1, 1),
        aspect_ratio = 1,
        xlabel = "Neuron Index",
        ylabel = "Neuron Index",
        title = "Cross-Neuron Firing Rate Correlations",
        colorbar_title = "Correlation",
        size = (600, 550)
    )
    
    # Add neuron ID labels if not too many
    if n_neurons <= 20
        xticks!(1:n_neurons, string.(neuron_ids), rotation=45)
        yticks!(1:n_neurons, string.(neuron_ids))
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved correlation matrix to $save_path")
    end
    
    return p
end

"""
    plot_tmi_distribution(tmi_values::Vector{Float64},
                         neuron_ids::Vector{Int};
                         save_path::Union{String,Nothing}=nothing)
    
Plot distribution of TMI values across neurons
"""
function plot_tmi_distribution(tmi_values::Vector{Float64},
                              neuron_ids::Vector{Int};
                              save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    # Sort by TMI
    sort_idx = sortperm(tmi_values)
    sorted_tmi = tmi_values[sort_idx]
    sorted_ids = neuron_ids[sort_idx]
    
    # Create bar plot
    p = bar(
        1:length(sorted_tmi),
        sorted_tmi,
        xlabel = "Neuron (sorted by TMI)",
        ylabel = "Temporal Modulation Index",
        title = "TMI Distribution Across Neurons",
        legend = false,
        color = :steelblue,
        size = (800, 400)
    )
    
    # Add horizontal line for mean
    mean_tmi = mean(sorted_tmi)
    hline!([mean_tmi], 
           linewidth=2, 
           linestyle=:dash, 
           color=:red,
           label="Mean TMI = $(round(mean_tmi, digits=3))")
    
    # Add neuron IDs if not too many
    if length(sorted_ids) <= 30
        xticks!(1:length(sorted_ids), string.(sorted_ids), rotation=45)
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved TMI distribution to $save_path")
    end
    
    return p
end

"""
    plot_burst_comparison(burst_characteristics::Dict{Int, Dict{String, Any}},
                         neuron_ids::Vector{Int};
                         save_path::Union{String,Nothing}=nothing)
    
Compare burst characteristics across neurons
"""
function plot_burst_comparison(burst_characteristics::Dict{Int, Dict{String, Any}},
                              neuron_ids::Vector{Int};
                              save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    # Extract burst metrics
    n_bursts = [burst_characteristics[id]["n_bursts"] for id in neuron_ids]
    mean_durations = [burst_characteristics[id]["mean_duration"] for id in neuron_ids]
    mean_peaks = [burst_characteristics[id]["mean_peak_rate"] for id in neuron_ids]
    
    # Create multi-panel plot
    p1 = bar(1:length(neuron_ids), n_bursts,
             xlabel="Neuron", ylabel="Number of Bursts",
             title="Burst Count", legend=false, color=:steelblue)
    
    p2 = bar(1:length(neuron_ids), mean_durations,
             xlabel="Neuron", ylabel="Duration (ms)",
             title="Mean Burst Duration", legend=false, color=:coral)
    
    p3 = bar(1:length(neuron_ids), mean_peaks,
             xlabel="Neuron", ylabel="Peak Rate (Hz)",
             title="Mean Peak Firing Rate", legend=false, color=:seagreen)
    
    combined = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        println("Saved burst comparison to $save_path")
    end
    
    return combined
end

"""
    plot_temporal_dynamics(selection::NeuronSelection,
                          firing_rates::Matrix{Float64},
                          time::Vector{Float64},
                          highlight_neurons::Union{Vector{Int},Nothing}=nothing;
                          save_path::Union{String,Nothing}=nothing)
    
Plot temporal dynamics of multiple neurons on single axis
"""
function plot_temporal_dynamics(selection::NeuronSelection,
                               firing_rates::Matrix{Float64},
                               time::Vector{Float64},
                               highlight_neurons::Union{Vector{Int},Nothing}=nothing;
                               save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    n_neurons = length(selection.neuron_ids)
    
    # Color palette
    colors = distinguishable_colors(n_neurons, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    p = plot(xlabel="Time (ms)", 
             ylabel="Firing Rate (Hz)",
             title="Temporal Dynamics - All Neurons",
             legend=:outerright,
             size=(1000, 600))
    
    for i in 1:n_neurons
        neuron_id = selection.neuron_ids[i]
        fr = firing_rates[:, i]
        
        # Determine if this neuron should be highlighted
        is_highlighted = !isnothing(highlight_neurons) && neuron_id in highlight_neurons
        lw = is_highlighted ? 2.5 : 1.0
        alpha = is_highlighted ? 1.0 : 0.6
        
        plot!(time, fr,
              label="Neuron $neuron_id",
              linewidth=lw,
              alpha=alpha,
              color=colors[i])
    end
    
    if !isnothing(save_path)
        savefig(p, save_path)
        println("Saved temporal dynamics plot to $save_path")
    end
    
    return p
end

"""
    plot_summary_statistics(metrics::ComparativeMetrics;
                           save_path::Union{String,Nothing}=nothing)
    
Create comprehensive summary visualization
"""
function plot_summary_statistics(metrics::ComparativeMetrics;
                                save_path::Union{String,Nothing}=nothing)
    
    configure_plot_style()
    
    neuron_ids = metrics.selection.neuron_ids
    
    # Create 2x2 panel
    p1 = scatter(metrics.tmi_values, metrics.mean_firing_rates,
                xlabel="TMI", ylabel="Mean Firing Rate (Hz)",
                title="TMI vs Mean Firing Rate",
                markersize=6, color=:steelblue, legend=false)
    
    p2 = scatter(metrics.mean_firing_rates, metrics.peak_firing_rates,
                xlabel="Mean Firing Rate (Hz)", ylabel="Peak Firing Rate (Hz)",
                title="Mean vs Peak Rates",
                markersize=6, color=:coral, legend=false)
    
    p3 = histogram(metrics.tmi_values,
                  xlabel="TMI", ylabel="Count",
                  title="TMI Distribution",
                  bins=20, color=:seagreen, legend=false)
    
    p4 = scatter(metrics.time_to_peak, metrics.peak_firing_rates,
                xlabel="Time to Peak (ms)", ylabel="Peak Firing Rate (Hz)",
                title="Timing vs Magnitude",
                markersize=6, color=:purple, legend=false)
    
    combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    
    if !isnothing(save_path)
        savefig(combined, save_path)
        println("Saved summary statistics to $save_path")
    end
    
    return combined
end

"""
    create_comparative_report(metrics::ComparativeMetrics,
                             firing_rates::Matrix{Float64},
                             time::Vector{Float64},
                             output_dir::String)
    
Generate complete comparative analysis report with all plots
"""
function create_comparative_report(metrics::ComparativeMetrics,
                                  firing_rates::Matrix{Float64},
                                  time::Vector{Float64},
                                  output_dir::String)
    
    # Create output directory if needed
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("Generating comparative analysis report...")
    
    # Generate all plots
    plot_comparative_firing_rates(
        metrics.selection,
        firing_rates,
        time,
        metrics.tmi_values,
        save_path=joinpath(output_dir, "comparative_firing_rates.png")
    )
    
    plot_correlation_matrix(
        metrics.correlation_matrix,
        metrics.selection.neuron_ids,
        save_path=joinpath(output_dir, "correlation_matrix.png")
    )
    
    plot_tmi_distribution(
        metrics.tmi_values,
        metrics.selection.neuron_ids,
        save_path=joinpath(output_dir, "tmi_distribution.png")
    )
    
    plot_burst_comparison(
        metrics.burst_characteristics,
        metrics.selection.neuron_ids,
        save_path=joinpath(output_dir, "burst_comparison.png")
    )
    
    plot_temporal_dynamics(
        metrics.selection,
        firing_rates,
        time,
        save_path=joinpath(output_dir, "temporal_dynamics.png")
    )
    
    plot_summary_statistics(
        metrics,
        save_path=joinpath(output_dir, "summary_statistics.png")
    )
    
    println("Report generated in $output_dir")
end

end # module
