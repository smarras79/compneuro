"""
Sharp-Wave Ripple (SWR) Detection Module

Detects hippocampal sharp-wave ripples using classical signal processing
and machine learning approaches.

Sharp-Wave Ripples characteristics:
- Frequency: 80-250 Hz (typically 150-200 Hz)
- Duration: 50-150 ms
- Amplitude: 3-8 SD above baseline
- Associated with memory consolidation

Author: Enhanced Neural Analysis Toolkit
"""

using DSP
using Statistics
using LinearAlgebra

"""
    detect_swr_classical(signal::Vector{Float64}, fs::Float64=1000.0;
                        ripple_band=(150.0, 250.0),
                        threshold_sd=3.0,
                        min_duration_ms=30.0,
                        max_duration_ms=200.0,
                        merge_threshold_ms=50.0)

Detect Sharp-Wave Ripples using classical threshold-based method.

# Algorithm:
1. Bandpass filter in ripple frequency band
2. Compute amplitude envelope (Hilbert transform)
3. Threshold envelope at mean + threshold_sd * std
4. Extract events and validate duration
5. Merge nearby events if needed

# Arguments
- `signal`: Input LFP/neural signal
- `fs`: Sampling frequency in Hz
- `ripple_band`: (low, high) frequency in Hz for ripple band
- `threshold_sd`: Number of standard deviations above mean for detection
- `min_duration_ms`: Minimum ripple duration in ms
- `max_duration_ms`: Maximum ripple duration in ms
- `merge_threshold_ms`: Merge events closer than this (ms)

# Returns
Dictionary with:
- `events`: Vector of (start_sample, end_sample, peak_sample, peak_amplitude)
- `envelope`: Amplitude envelope of filtered signal
- `filtered_signal`: Bandpass filtered signal
- `threshold`: Detection threshold used
"""
function detect_swr_classical(signal::Vector{Float64}, fs::Float64=1000.0;
                             ripple_band=(150.0, 250.0),
                             threshold_sd=3.0,
                             min_duration_ms=30.0,
                             max_duration_ms=200.0,
                             merge_threshold_ms=50.0)
    
    # 1. Bandpass filter in ripple band
    filtered_signal = bandpass_filter(signal, ripple_band[1], ripple_band[2], fs)
    
    # 2. Compute amplitude envelope using Hilbert transform
    envelope = compute_envelope(filtered_signal)
    
    # 3. Calculate threshold
    baseline_mean = mean(envelope)
    baseline_std = std(envelope)
    threshold = baseline_mean + threshold_sd * baseline_std
    
    # 4. Find threshold crossings
    above_threshold = envelope .> threshold
    
    # Find event boundaries (rising and falling edges)
    diff_threshold = diff([0; above_threshold; 0])
    starts = findall(diff_threshold .== 1)
    ends = findall(diff_threshold .== -1)
    
    # 5. Validate and extract events
    min_duration_samples = Int(round(min_duration_ms * fs / 1000))
    max_duration_samples = Int(round(max_duration_ms * fs / 1000))
    merge_threshold_samples = Int(round(merge_threshold_ms * fs / 1000))
    
    events = []
    
    for i in 1:length(starts)
        if i > length(ends)
            break
        end
        
        start_idx = starts[i]
        end_idx = ends[i]
        duration = end_idx - start_idx
        
        # Check duration criteria
        if duration >= min_duration_samples && duration <= max_duration_samples
            # Find peak within event
            peak_idx = start_idx + argmax(envelope[start_idx:end_idx]) - 1
            peak_amplitude = envelope[peak_idx]
            
            push!(events, (
                start_sample=start_idx,
                end_sample=end_idx,
                peak_sample=peak_idx,
                peak_amplitude=peak_amplitude,
                duration_ms=duration * 1000 / fs
            ))
        end
    end
    
    # 6. Merge nearby events if requested
    if merge_threshold_samples > 0 && length(events) > 1
        events = merge_nearby_events(events, merge_threshold_samples, envelope)
    end
    
    return Dict(
        "events" => events,
        "envelope" => envelope,
        "filtered_signal" => filtered_signal,
        "threshold" => threshold,
        "n_events" => length(events)
    )
end

"""
    bandpass_filter(signal, low_freq, high_freq, fs; order=4)

Apply Butterworth bandpass filter.
"""
function bandpass_filter(signal::Vector{Float64}, low_freq::Float64, 
                        high_freq::Float64, fs::Float64; order::Int=4)
    
    # Normalize frequencies to Nyquist
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Ensure valid range
    low_norm = max(0.001, min(0.999, low_norm))
    high_norm = max(0.001, min(0.999, high_norm))
    
    # Create bandpass filter
    responsetype = Bandpass(low_norm, high_norm)
    designmethod = Butterworth(order)
    filt = digitalfilter(responsetype, designmethod)
    
    # Apply zero-phase filtering
    filtered = filtfilt(filt, signal)
    
    return filtered
end

"""
    compute_envelope(signal)

Compute amplitude envelope using Hilbert transform.
"""
function compute_envelope(signal::Vector{Float64})
    # Analytic signal via Hilbert transform
    analytic_signal = hilbert(signal)
    envelope = abs.(analytic_signal)
    return envelope
end

"""
    hilbert(signal)

Compute Hilbert transform (analytic signal).
"""
function hilbert(x::Vector{Float64})
    N = length(x)
    X = fft(x)
    
    # Create Hilbert multiplier
    h = zeros(N)
    if N % 2 == 0
        h[1] = 1
        h[2:N÷2] .= 2
        h[N÷2+1] = 1
    else
        h[1] = 1
        h[2:(N+1)÷2] .= 2
    end
    
    # Apply multiplier and inverse FFT
    analytic = ifft(X .* h)
    
    return analytic
end

"""
    merge_nearby_events(events, merge_threshold, envelope)

Merge events that are close together.
"""
function merge_nearby_events(events, merge_threshold, envelope)
    if length(events) <= 1
        return events
    end
    
    merged = [events[1]]
    
    for i in 2:length(events)
        last_event = merged[end]
        current_event = events[i]
        
        gap = current_event.start_sample - last_event.end_sample
        
        if gap <= merge_threshold
            # Merge with previous event
            start_idx = last_event.start_sample
            end_idx = current_event.end_sample
            
            # Find new peak
            peak_idx = start_idx + argmax(envelope[start_idx:end_idx]) - 1
            peak_amplitude = envelope[peak_idx]
            
            # Update last event in merged list
            merged[end] = (
                start_sample=start_idx,
                end_sample=end_idx,
                peak_sample=peak_idx,
                peak_amplitude=peak_amplitude,
                duration_ms=(end_idx - start_idx) * 1000 / 1000  # Assuming fs=1000
            )
        else
            # Add as new event
            push!(merged, current_event)
        end
    end
    
    return merged
end

"""
    extract_event_features(signal::Vector{Float64}, event, fs::Float64=1000.0)

Extract features from a detected SWR event for classification.

# Features:
- Duration
- Peak amplitude
- Mean/std amplitude
- Dominant frequency
- Spectral centroid
- Spectral bandwidth
- Zero-crossing rate
- Peak-to-trough ratio

# Returns
Dictionary of features
"""
function extract_event_features(signal::Vector{Float64}, event, fs::Float64=1000.0;
                               ripple_band=(150.0, 250.0))
    
    start_idx = event.start_sample
    end_idx = event.end_sample
    
    # Extract event segment
    segment = signal[start_idx:end_idx]
    duration_ms = (end_idx - start_idx) * 1000 / fs
    
    # Temporal features
    peak_amp = maximum(abs.(segment))
    mean_amp = mean(abs.(segment))
    std_amp = std(segment)
    
    # Bandpass filter for ripple analysis
    filtered = bandpass_filter(segment, ripple_band[1], ripple_band[2], fs)
    envelope = compute_envelope(filtered)
    
    # Frequency features via FFT
    N = length(segment)
    fft_result = rfft(segment)
    power = abs2.(fft_result)
    freq = rfftfreq(N, fs)
    
    # Mask for ripple band
    mask = (freq .>= ripple_band[1]) .& (freq .<= ripple_band[2])
    
    if sum(mask) > 0
        ripple_power = power[mask]
        ripple_freq = freq[mask]
        
        # Dominant frequency (peak in ripple band)
        dominant_freq = ripple_freq[argmax(ripple_power)]
        
        # Spectral centroid
        spectral_centroid = sum(ripple_freq .* ripple_power) / sum(ripple_power)
        
        # Spectral bandwidth
        spectral_bandwidth = sqrt(sum(((ripple_freq .- spectral_centroid).^2) .* ripple_power) / 
                                 sum(ripple_power))
    else
        dominant_freq = 0.0
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
    end
    
    # Zero-crossing rate
    zero_crossings = sum(diff(sign.(filtered)) .!= 0)
    zcr = zero_crossings / duration_ms * 1000  # Per second
    
    # Peak-to-trough
    peak_to_trough = maximum(filtered) - minimum(filtered)
    
    # Ripple envelope features
    env_mean = mean(envelope)
    env_std = std(envelope)
    env_peak = maximum(envelope)
    
    return Dict(
        "duration_ms" => duration_ms,
        "peak_amplitude" => peak_amp,
        "mean_amplitude" => mean_amp,
        "std_amplitude" => std_amp,
        "dominant_frequency" => dominant_freq,
        "spectral_centroid" => spectral_centroid,
        "spectral_bandwidth" => spectral_bandwidth,
        "zero_crossing_rate" => zcr,
        "peak_to_trough" => peak_to_trough,
        "envelope_mean" => env_mean,
        "envelope_std" => env_std,
        "envelope_peak" => env_peak
    )
end

"""
    detect_swr_at_events(signal::Vector{Float64}, event_times::Vector{Float64}, 
                        fs::Float64=1000.0; window_ms=500.0, kwargs...)

Detect SWRs around specific event times (e.g., joystick motion onset/offset).

# Arguments
- `signal`: Neural signal
- `event_times`: Vector of event times in samples or seconds
- `fs`: Sampling frequency
- `window_ms`: Time window around each event to search for SWRs
- `kwargs`: Additional parameters for detect_swr_classical

# Returns
Dictionary with SWRs found near each event
"""
function detect_swr_at_events(signal::Vector{Float64}, event_times::Vector{Float64},
                             fs::Float64=1000.0; window_ms=500.0, kwargs...)
    
    window_samples = Int(round(window_ms * fs / 1000))
    results = Dict()
    
    # Detect all SWRs first
    all_swr = detect_swr_classical(signal, fs; kwargs...)
    
    # For each event, find SWRs within window
    for (idx, event_time) in enumerate(event_times)
        event_sample = Int(round(event_time * fs))
        
        window_start = max(1, event_sample - window_samples÷2)
        window_end = min(length(signal), event_sample + window_samples÷2)
        
        # Find SWRs overlapping this window
        nearby_swr = filter(swr -> 
            (swr.start_sample >= window_start && swr.start_sample <= window_end) ||
            (swr.end_sample >= window_start && swr.end_sample <= window_end) ||
            (swr.start_sample <= window_start && swr.end_sample >= window_end),
            all_swr["events"])
        
        results["event_$idx"] = Dict(
            "event_time" => event_time,
            "event_sample" => event_sample,
            "window" => (window_start, window_end),
            "swr_events" => nearby_swr,
            "n_swr" => length(nearby_swr)
        )
    end
    
    results["all_swr"] = all_swr
    results["total_events"] = length(event_times)
    
    return results
end

"""
    visualize_swr_events(signal, events, fs; max_events=10)

Visualize detected SWR events.
"""
function visualize_swr_events(signal::Vector{Float64}, events, fs::Float64=1000.0;
                             max_events=10)
    
    n_plot = min(max_events, length(events))
    
    if n_plot == 0
        println("No events to plot")
        return nothing
    end
    
    plots = []
    
    for i in 1:n_plot
        event = events[i]
        
        # Extract window around event
        margin = Int(fs * 0.1)  # 100ms margin
        start_idx = max(1, event.start_sample - margin)
        end_idx = min(length(signal), event.end_sample + margin)
        
        segment = signal[start_idx:end_idx]
        time_vec = ((start_idx:end_idx) .- 1) ./ fs
        
        # Mark event boundaries
        event_start_time = (event.start_sample - 1) / fs
        event_end_time = (event.end_sample - 1) / fs
        
        p = plot(time_vec, segment,
                label="Signal",
                xlabel="Time (s)",
                ylabel="Amplitude",
                title="SWR Event $i",
                legend=:topright,
                size=(800, 300))
        
        # Shade event region
        vspan!([event_start_time, event_end_time], 
               alpha=0.2, color=:red, label="SWR")
        
        push!(plots, p)
    end
    
    combined = plot(plots..., layout=(n_plot, 1), size=(1000, 300*n_plot))
    
    return combined
end

println("✓ Sharp-Wave Ripple Detection module loaded")
println("  Functions available:")
println("    - detect_swr_classical()")
println("    - detect_swr_at_events()")
println("    - extract_event_features()")
println("    - visualize_swr_events()")
println("    - bandpass_filter(), compute_envelope()")
