"""
Multi-Neuron SWR Analysis Extension
Add Sharp-Wave Ripple detection and analysis to multi-neuron framework

Author: Enhanced for SWR comparative analysis
"""

module MultiNeuronSWR

using Statistics
using LinearAlgebra
using DSP
using FFTW

export SWREvent, SWRMetrics, detect_swr_single_neuron, compare_swr_across_neurons
export compute_swr_rate, compute_swr_synchrony, identify_co_rippling_neurons

"""
    SWREvent

Structure to hold a single SWR event
"""
struct SWREvent
    start_idx::Int
    end_idx::Int
    peak_idx::Int
    duration_ms::Float64
    peak_amplitude::Float64
    mean_amplitude::Float64
    peak_frequency::Float64
    start_time_ms::Float64
    peak_time_ms::Float64
end

"""
    SWRMetrics

Structure to hold SWR analysis results for multiple neurons
"""
struct SWRMetrics
    neuron_ids::Vector{Int}
    events_per_neuron::Vector{Vector{SWREvent}}
    swr_counts::Vector{Int}
    swr_rates::Vector{Float64}  # events per second
    mean_durations::Vector{Float64}
    mean_amplitudes::Vector{Float64}
    mean_frequencies::Vector{Float64}
    co_ripple_matrix::Matrix{Float64}  # Probability of simultaneous SWRs
    synchrony_scores::Vector{Float64}  # How often this neuron ripples with others
end

"""
    detect_swr_single_neuron(signal::Vector{Float64}, 
                             time::Vector{Float64},
                             fs::Float64;
                             ripple_band::Tuple{Float64,Float64}=(150.0, 250.0),
                             threshold_sd::Float64=3.0,
                             min_duration_ms::Float64=30.0,
                             max_duration_ms::Float64=200.0)
    
Detect Sharp-Wave Ripples in a single neuron's signal
"""
function detect_swr_single_neuron(signal::Vector{Float64}, 
                                 time::Vector{Float64},
                                 fs::Float64;
                                 ripple_band::Tuple{Float64,Float64}=(150.0, 250.0),
                                 threshold_sd::Float64=3.0,
                                 min_duration_ms::Float64=30.0,
                                 max_duration_ms::Float64=200.0)::Vector{SWREvent}
    
    # Design bandpass filter for ripple band
    nyquist = fs / 2
    low_freq = ripple_band[1] / nyquist
    high_freq = ripple_band[2] / nyquist
    
    # Ensure frequencies are in valid range (0 to 1, where 1 is Nyquist)
    low_freq = max(0.01, min(0.99, low_freq))
    high_freq = max(low_freq + 0.01, min(0.99, high_freq))
    
    # Create Butterworth bandpass filter with normalized frequencies
    responsetype = Bandpass(low_freq, high_freq)
    designmethod = Butterworth(4)
    
    # Filter signal
    try
        filtered = filtfilt(digitalfilter(responsetype, designmethod), signal)
        
        # Compute envelope using Hilbert transform
        analytic = hilbert(filtered)
        envelope = abs.(analytic)
        
        # Compute threshold
        baseline = mean(envelope)
        threshold = baseline + threshold_sd * std(envelope)
        
        # Find candidate SWR events
        above_threshold = envelope .> threshold
        
        # Find event boundaries
        events = SWREvent[]
        in_event = false
        event_start = 0
        
        for i in 1:length(above_threshold)
            if above_threshold[i] && !in_event
                event_start = i
                in_event = true
            elseif !above_threshold[i] && in_event
                event_end = i - 1
                
                # Check duration
                duration_samples = event_end - event_start + 1
                duration_ms = (duration_samples / fs) * 1000.0
                
                if min_duration_ms <= duration_ms <= max_duration_ms
                    # Find peak within event
                    event_envelope = envelope[event_start:event_end]
                    peak_idx_local = argmax(event_envelope)
                    peak_idx = event_start + peak_idx_local - 1
                    
                    # Compute event properties
                    peak_amplitude = envelope[peak_idx]
                    mean_amplitude = mean(envelope[event_start:event_end])
                    
                    # Estimate dominant frequency
                    event_signal = filtered[event_start:event_end]
                    event_fft = fft(event_signal)
                    n_fft = length(event_signal)
                    
                    # Compute frequency bins manually
                    freqs = [(i-1) * fs / n_fft for i in 1:n_fft÷2]
                    power = abs.(event_fft[1:n_fft÷2])
                    
                    # Find peak frequency in ripple band
                    ripple_mask = (freqs .>= ripple_band[1]) .& (freqs .<= ripple_band[2])
                    if sum(ripple_mask) > 0
                        ripple_power = power[ripple_mask]
                        ripple_freqs = freqs[ripple_mask]
                        peak_freq_idx = argmax(ripple_power)
                        peak_frequency = ripple_freqs[peak_freq_idx]
                    else
                        peak_frequency = mean(ripple_band)  # Default to band center
                    end
                    
                    # Create event
                    push!(events, SWREvent(
                        event_start,
                        event_end,
                        peak_idx,
                        duration_ms,
                        peak_amplitude,
                        mean_amplitude,
                        peak_frequency,
                        time[event_start],
                        time[peak_idx]
                    ))
                end
                
                in_event = false
            end
        end
        
        # Close last event if needed
        if in_event
            event_end = length(above_threshold)
            duration_samples = event_end - event_start + 1
            duration_ms = (duration_samples / fs) * 1000.0
            
            if min_duration_ms <= duration_ms <= max_duration_ms
                event_envelope = envelope[event_start:event_end]
                peak_idx_local = argmax(event_envelope)
                peak_idx = event_start + peak_idx_local - 1
                
                peak_amplitude = envelope[peak_idx]
                mean_amplitude = mean(envelope[event_start:event_end])
                
                event_signal = filtered[event_start:event_end]
                event_fft = fft(event_signal)
                n_fft = length(event_signal)
                
                # Compute frequency bins manually
                freqs = [(i-1) * fs / n_fft for i in 1:n_fft÷2]
                power = abs.(event_fft[1:n_fft÷2])
                
                # Find peak frequency in ripple band
                ripple_mask = (freqs .>= ripple_band[1]) .& (freqs .<= ripple_band[2])
                if sum(ripple_mask) > 0
                    ripple_power = power[ripple_mask]
                    ripple_freqs = freqs[ripple_mask]
                    peak_freq_idx = argmax(ripple_power)
                    peak_frequency = ripple_freqs[peak_freq_idx]
                else
                    peak_frequency = mean(ripple_band)  # Default to band center
                end
                
                push!(events, SWREvent(
                    event_start,
                    event_end,
                    peak_idx,
                    duration_ms,
                    peak_amplitude,
                    mean_amplitude,
                    peak_frequency,
                    time[event_start],
                    time[peak_idx]
                ))
            end
        end
        
        return events
        
    catch e
        @warn "SWR detection failed for neuron" exception=(e, catch_backtrace())
        println("  Signal length: $(length(signal)), Time length: $(length(time))")
        println("  Sampling frequency: $fs Hz")
        println("  Ripple band: $(ripple_band[1])-$(ripple_band[2]) Hz")
        return SWREvent[]
    end
end

"""
    compare_swr_across_neurons(firing_rates::Matrix{Float64},
                               time::Vector{Float64},
                               neuron_ids::Vector{Int},
                               fs::Float64;
                               kwargs...)
    
Detect and compare SWRs across multiple neurons
"""
function compare_swr_across_neurons(firing_rates::Matrix{Float64},
                                   time::Vector{Float64},
                                   neuron_ids::Vector{Int},
                                   fs::Float64;
                                   ripple_band::Tuple{Float64,Float64}=(150.0, 250.0),
                                   threshold_sd::Float64=3.0,
                                   min_duration_ms::Float64=30.0,
                                   max_duration_ms::Float64=200.0,
                                   co_ripple_window_ms::Float64=50.0)::SWRMetrics
    
    n_neurons = length(neuron_ids)
    events_per_neuron = Vector{SWREvent}[]
    
    # Detect SWRs for each neuron
    println("Detecting SWRs for each neuron...")
    for i in 1:n_neurons
        signal = firing_rates[:, i]
        events = detect_swr_single_neuron(
            signal, time, fs;
            ripple_band=ripple_band,
            threshold_sd=threshold_sd,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms
        )
        push!(events_per_neuron, events)
        println("  Neuron $(neuron_ids[i]): $(length(events)) SWRs detected")
    end
    
    # Compute basic statistics
    swr_counts = [length(events) for events in events_per_neuron]
    total_time_s = (time[end] - time[1]) / 1000.0  # Convert ms to seconds
    swr_rates = swr_counts ./ total_time_s
    
    mean_durations = zeros(n_neurons)
    mean_amplitudes = zeros(n_neurons)
    mean_frequencies = zeros(n_neurons)
    
    for i in 1:n_neurons
        if length(events_per_neuron[i]) > 0
            mean_durations[i] = mean([e.duration_ms for e in events_per_neuron[i]])
            mean_amplitudes[i] = mean([e.peak_amplitude for e in events_per_neuron[i]])
            mean_frequencies[i] = mean([e.peak_frequency for e in events_per_neuron[i]])
        end
    end
    
    # Compute co-rippling matrix
    println("Computing co-rippling analysis...")
    co_ripple_matrix = zeros(n_neurons, n_neurons)
    co_ripple_window = co_ripple_window_ms / 1000.0  # Convert to seconds
    
    for i in 1:n_neurons
        for j in i:n_neurons
            if i == j
                co_ripple_matrix[i, j] = 1.0
            else
                # Count how many times SWRs occur simultaneously
                events_i = events_per_neuron[i]
                events_j = events_per_neuron[j]
                
                if length(events_i) > 0 && length(events_j) > 0
                    n_co_ripples = 0
                    
                    for event_i in events_i
                        time_i = event_i.peak_time_ms / 1000.0
                        
                        for event_j in events_j
                            time_j = event_j.peak_time_ms / 1000.0
                            
                            if abs(time_i - time_j) <= co_ripple_window
                                n_co_ripples += 1
                                break  # Count each event_i only once
                            end
                        end
                    end
                    
                    # Normalize by total possible pairs
                    co_ripple_prob = n_co_ripples / min(length(events_i), length(events_j))
                    co_ripple_matrix[i, j] = co_ripple_prob
                    co_ripple_matrix[j, i] = co_ripple_prob
                end
            end
        end
    end
    
    # Compute synchrony scores (average co-rippling with other neurons)
    synchrony_scores = zeros(n_neurons)
    for i in 1:n_neurons
        if n_neurons > 1
            synchrony_scores[i] = mean([co_ripple_matrix[i, j] for j in 1:n_neurons if j != i])
        end
    end
    
    return SWRMetrics(
        neuron_ids,
        events_per_neuron,
        swr_counts,
        swr_rates,
        mean_durations,
        mean_amplitudes,
        mean_frequencies,
        co_ripple_matrix,
        synchrony_scores
    )
end

"""
    compute_swr_rate(events::Vector{SWREvent}, total_time_s::Float64)
    
Compute SWR rate (events per second)
"""
function compute_swr_rate(events::Vector{SWREvent}, total_time_s::Float64)::Float64
    return length(events) / total_time_s
end

"""
    compute_swr_synchrony(events_a::Vector{SWREvent}, 
                         events_b::Vector{SWREvent},
                         window_ms::Float64=50.0)
    
Compute synchrony between two neurons' SWRs
"""
function compute_swr_synchrony(events_a::Vector{SWREvent}, 
                              events_b::Vector{SWREvent},
                              window_ms::Float64=50.0)::Float64
    
    if length(events_a) == 0 || length(events_b) == 0
        return 0.0
    end
    
    window_s = window_ms / 1000.0
    n_synchronized = 0
    
    for event_a in events_a
        time_a = event_a.peak_time_ms / 1000.0
        
        for event_b in events_b
            time_b = event_b.peak_time_ms / 1000.0
            
            if abs(time_a - time_b) <= window_s
                n_synchronized += 1
                break
            end
        end
    end
    
    return n_synchronized / min(length(events_a), length(events_b))
end

"""
    identify_co_rippling_neurons(swr_metrics::SWRMetrics,
                                 threshold::Float64=0.3)
    
Identify pairs of neurons that frequently co-ripple
"""
function identify_co_rippling_neurons(swr_metrics::SWRMetrics,
                                     threshold::Float64=0.3)::Vector{Tuple{Int,Int,Float64}}
    
    n_neurons = length(swr_metrics.neuron_ids)
    co_rippling_pairs = Tuple{Int,Int,Float64}[]
    
    for i in 1:n_neurons
        for j in (i+1):n_neurons
            prob = swr_metrics.co_ripple_matrix[i, j]
            if prob >= threshold
                push!(co_rippling_pairs, (
                    swr_metrics.neuron_ids[i],
                    swr_metrics.neuron_ids[j],
                    prob
                ))
            end
        end
    end
    
    # Sort by probability (descending)
    sort!(co_rippling_pairs, by=x->x[3], rev=true)
    
    return co_rippling_pairs
end

end # module
