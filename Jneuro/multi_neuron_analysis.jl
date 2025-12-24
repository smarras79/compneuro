"""
Multi-Neuron Comparative Analysis Module
Enhanced analysis framework for selecting and comparing multiple neurons

Author: Enhanced for comparative analysis
"""

module MultiNeuronAnalysis

using Statistics
using LinearAlgebra
using DSP

export NeuronSelection, ComparativeMetrics, select_neurons, compute_comparative_metrics
export temporal_modulation_index, firing_rate_dynamics, cross_neuron_correlation

"""
    NeuronSelection

Structure to hold selected neurons and their identifiers
"""
struct NeuronSelection
    neuron_ids::Vector{Int}
    neuron_names::Vector{String}
    data_indices::Vector{Int}
    
    function NeuronSelection(neuron_ids::Vector{Int})
        neuron_names = ["Neuron $id" for id in neuron_ids]
        data_indices = collect(1:length(neuron_ids))
        new(neuron_ids, neuron_names, data_indices)
    end
    
    function NeuronSelection(neuron_ids::Vector{Int}, neuron_names::Vector{String})
        @assert length(neuron_ids) == length(neuron_names) "Mismatch between IDs and names"
        data_indices = collect(1:length(neuron_ids))
        new(neuron_ids, neuron_names, data_indices)
    end
end

"""
    ComparativeMetrics

Structure to hold comparative analysis results across neurons
"""
struct ComparativeMetrics
    selection::NeuronSelection
    tmi_values::Vector{Float64}  # Temporal Modulation Index
    mean_firing_rates::Vector{Float64}
    peak_firing_rates::Vector{Float64}
    firing_rate_variability::Vector{Float64}
    correlation_matrix::Matrix{Float64}
    time_to_peak::Vector{Float64}
    burst_characteristics::Dict{Int, Dict{String, Any}}
end

"""
    select_neurons(all_neuron_ids::Vector{Int}, selection_criteria::Function)
    
Select neurons based on custom criteria function
"""
function select_neurons(all_neuron_ids::Vector{Int}, 
                       selection_criteria::Function)::NeuronSelection
    selected_ids = filter(selection_criteria, all_neuron_ids)
    return NeuronSelection(selected_ids)
end

"""
    select_neurons(all_neuron_ids::Vector{Int}, indices::Vector{Int})
    
Select neurons by explicit indices
"""
function select_neurons(all_neuron_ids::Vector{Int}, 
                       indices::Vector{Int})::NeuronSelection
    selected_ids = all_neuron_ids[indices]
    return NeuronSelection(selected_ids)
end

"""
    select_neurons(all_neuron_ids::Vector{Int}, id_list::Vector{Int})
    
Select neurons by explicit ID list
"""
function select_neurons_by_ids(all_neuron_ids::Vector{Int}, 
                              id_list::Vector{Int})::NeuronSelection
    # Find indices of requested IDs
    indices = [findfirst(==(id), all_neuron_ids) for id in id_list]
    indices = filter(!isnothing, indices)
    
    if length(indices) != length(id_list)
        @warn "Some requested neuron IDs were not found"
    end
    
    selected_ids = all_neuron_ids[indices]
    return NeuronSelection(selected_ids)
end

"""
    temporal_modulation_index(firing_rate::Vector{Float64}, time::Vector{Float64})
    
Compute TMI as shown in the figure - measures temporal variation in firing rate
"""
function temporal_modulation_index(firing_rate::Vector{Float64}, 
                                  time::Vector{Float64})::Float64
    # Remove NaN and Inf values
    valid_idx = isfinite.(firing_rate)
    fr_valid = firing_rate[valid_idx]
    
    if length(fr_valid) < 2
        return 0.0
    end
    
    # Compute TMI as coefficient of variation
    μ = mean(fr_valid)
    σ = std(fr_valid)
    
    if μ ≈ 0.0
        return 0.0
    end
    
    tmi = σ / μ
    
    return tmi
end

"""
    firing_rate_dynamics(spike_times::Vector{Float64}, 
                        time_bins::Vector{Float64},
                        smoothing_window::Float64=50.0)
    
Compute smoothed firing rate from spike times
"""
function firing_rate_dynamics(spike_times::Vector{Float64},
                             time_bins::Vector{Float64},
                             smoothing_window::Float64=50.0)::Vector{Float64}
    
    dt = time_bins[2] - time_bins[1]
    n_bins = length(time_bins)
    
    # Create histogram of spike counts
    spike_counts = zeros(n_bins)
    for spike_time in spike_times
        bin_idx = searchsortedlast(time_bins, spike_time)
        if bin_idx > 0 && bin_idx <= n_bins
            spike_counts[bin_idx] += 1
        end
    end
    
    # Convert to firing rate (Hz)
    firing_rate = spike_counts ./ dt
    
    # Apply Gaussian smoothing
    if smoothing_window > 0
        window_samples = max(1, round(Int, smoothing_window / dt))
        kernel = gaussian(window_samples, window_samples/6)
        kernel = kernel ./ sum(kernel)
        
        # Pad signal to handle edges
        padded = vcat(firing_rate[1] * ones(window_samples), 
                     firing_rate, 
                     firing_rate[end] * ones(window_samples))
        
        smoothed = conv(padded, kernel)[window_samples+1:end-window_samples]
        firing_rate = smoothed[1:n_bins]
    end
    
    return firing_rate
end

"""
    cross_neuron_correlation(firing_rates::Matrix{Float64})
    
Compute correlation matrix between neurons
firing_rates: Matrix where each column is a neuron's firing rate time series
"""
function cross_neuron_correlation(firing_rates::Matrix{Float64})::Matrix{Float64}
    n_neurons = size(firing_rates, 2)
    corr_matrix = zeros(n_neurons, n_neurons)
    
    for i in 1:n_neurons
        for j in i:n_neurons
            # Remove NaN/Inf for correlation computation
            valid_idx = isfinite.(firing_rates[:, i]) .& isfinite.(firing_rates[:, j])
            
            if sum(valid_idx) > 1
                r = cor(firing_rates[valid_idx, i], firing_rates[valid_idx, j])
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
            end
        end
    end
    
    return corr_matrix
end

"""
    detect_bursts(firing_rate::Vector{Float64}, 
                  time::Vector{Float64},
                  threshold_factor::Float64=2.0)
    
Detect burst events in firing rate
"""
function detect_bursts(firing_rate::Vector{Float64},
                      time::Vector{Float64},
                      threshold_factor::Float64=2.0)::Dict{String, Any}
    
    # Compute threshold
    baseline = median(firing_rate)
    threshold = baseline + threshold_factor * std(firing_rate)
    
    # Find periods above threshold
    above_threshold = firing_rate .> threshold
    
    # Find burst starts and ends
    burst_starts = Int[]
    burst_ends = Int[]
    in_burst = false
    
    for i in 1:length(above_threshold)
        if above_threshold[i] && !in_burst
            push!(burst_starts, i)
            in_burst = true
        elseif !above_threshold[i] && in_burst
            push!(burst_ends, i-1)
            in_burst = false
        end
    end
    
    # Close last burst if needed
    if in_burst
        push!(burst_ends, length(above_threshold))
    end
    
    # Compute burst statistics
    n_bursts = length(burst_starts)
    burst_durations = Float64[]
    burst_peak_rates = Float64[]
    
    for i in 1:n_bursts
        duration = time[burst_ends[i]] - time[burst_starts[i]]
        peak_rate = maximum(firing_rate[burst_starts[i]:burst_ends[i]])
        push!(burst_durations, duration)
        push!(burst_peak_rates, peak_rate)
    end
    
    return Dict(
        "n_bursts" => n_bursts,
        "burst_starts" => burst_starts,
        "burst_ends" => burst_ends,
        "mean_duration" => n_bursts > 0 ? mean(burst_durations) : 0.0,
        "mean_peak_rate" => n_bursts > 0 ? mean(burst_peak_rates) : 0.0
    )
end

"""
    compute_comparative_metrics(selection::NeuronSelection,
                               firing_rates::Matrix{Float64},
                               time::Vector{Float64})
    
Compute comprehensive comparative metrics for selected neurons
"""
function compute_comparative_metrics(selection::NeuronSelection,
                                    firing_rates::Matrix{Float64},
                                    time::Vector{Float64})::ComparativeMetrics
    
    n_neurons = length(selection.neuron_ids)
    @assert size(firing_rates, 2) == n_neurons "Firing rate matrix columns must match number of neurons"
    
    # Initialize metric arrays
    tmi_values = zeros(n_neurons)
    mean_firing_rates = zeros(n_neurons)
    peak_firing_rates = zeros(n_neurons)
    firing_rate_variability = zeros(n_neurons)
    time_to_peak = zeros(n_neurons)
    burst_characteristics = Dict{Int, Dict{String, Any}}()
    
    # Compute metrics for each neuron
    for i in 1:n_neurons
        fr = firing_rates[:, i]
        valid_idx = isfinite.(fr)
        fr_valid = fr[valid_idx]
        
        if length(fr_valid) > 0
            # TMI
            tmi_values[i] = temporal_modulation_index(fr, time)
            
            # Basic statistics
            mean_firing_rates[i] = mean(fr_valid)
            peak_firing_rates[i] = maximum(fr_valid)
            firing_rate_variability[i] = std(fr_valid)
            
            # Time to peak
            peak_idx = argmax(fr)
            time_to_peak[i] = time[peak_idx]
            
            # Burst detection
            burst_characteristics[selection.neuron_ids[i]] = detect_bursts(fr, time)
        end
    end
    
    # Cross-correlation matrix
    corr_matrix = cross_neuron_correlation(firing_rates)
    
    return ComparativeMetrics(
        selection,
        tmi_values,
        mean_firing_rates,
        peak_firing_rates,
        firing_rate_variability,
        corr_matrix,
        time_to_peak,
        burst_characteristics
    )
end

"""
    rank_neurons_by_tmi(metrics::ComparativeMetrics)
    
Return neuron indices sorted by TMI value
"""
function rank_neurons_by_tmi(metrics::ComparativeMetrics)::Vector{Int}
    return sortperm(metrics.tmi_values, rev=true)
end

"""
    get_summary_statistics(metrics::ComparativeMetrics)
    
Get summary statistics across all neurons
"""
function get_summary_statistics(metrics::ComparativeMetrics)::Dict{String, Any}
    return Dict(
        "n_neurons" => length(metrics.selection.neuron_ids),
        "mean_tmi" => mean(metrics.tmi_values),
        "std_tmi" => std(metrics.tmi_values),
        "mean_firing_rate" => mean(metrics.mean_firing_rates),
        "mean_peak_rate" => mean(metrics.peak_firing_rates),
        "mean_correlation" => mean([metrics.correlation_matrix[i,j] 
                                   for i in 1:size(metrics.correlation_matrix,1) 
                                   for j in i+1:size(metrics.correlation_matrix,2)])
    )
end

end # module
