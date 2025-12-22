"""
Neural Spectral Analysis Module

Provides functions for frequency-domain analysis of neural time series:
- Power spectral density (PSD)
- Spectrograms
- Wavelet analysis
- Time-frequency representations

Author: Enhanced Neural Analysis Toolkit
"""

using DSP
using Statistics
using FFTW
using Plots

"""
    compute_psd(signal::Vector{Float64}, fs::Float64=1000.0; method=:welch, nperseg=256)

Compute Power Spectral Density of a neural signal.

# Arguments
- `signal`: Input time series
- `fs`: Sampling frequency in Hz (default: 1000.0)
- `method`: `:welch` or `:periodogram`
- `nperseg`: Segment length for Welch's method

# Returns
- `frequencies`: Frequency vector
- `power`: Power spectral density
"""
function compute_psd(signal::Vector{Float64}, fs::Float64=1000.0; 
                     method=:welch, nperseg=256)
    
    if method == :welch
        # Welch's method: average periodograms of overlapping segments
        noverlap = nperseg ÷ 2
        window = DSP.hanning(nperseg)
        
        # Calculate number of segments
        step = nperseg - noverlap
        n_segments = max(1, (length(signal) - noverlap) ÷ step)
        
        # Initialize power accumulator
        freq = rfftfreq(nperseg, fs)
        power_sum = zeros(length(freq))
        
        for i in 1:n_segments
            start_idx = (i-1) * step + 1
            end_idx = start_idx + nperseg - 1
            
            if end_idx > length(signal)
                break
            end
            
            # Extract segment and apply window
            segment = signal[start_idx:end_idx] .* window
            
            # Compute FFT and power
            fft_result = rfft(segment)
            power = abs2.(fft_result) ./ (fs * sum(abs2.(window)))
            
            power_sum .+= power
        end
        
        power = power_sum ./ n_segments
        
        return freq, power
        
    elseif method == :periodogram
        # Simple periodogram
        N = length(signal)
        fft_result = rfft(signal)
        power = abs2.(fft_result) ./ (N * fs)
        freq = rfftfreq(N, fs)
        
        return freq, power
    else
        error("Unknown method: $method. Use :welch or :periodogram")
    end
end

"""
    compute_spectrogram(signal::Vector{Float64}, fs::Float64=1000.0; 
                       nperseg=256, noverlap=nothing)

Compute time-frequency spectrogram.

# Arguments
- `signal`: Input time series
- `fs`: Sampling frequency in Hz
- `nperseg`: Length of each segment
- `noverlap`: Number of overlapping samples (default: nperseg÷2)

# Returns
- `times`: Time vector
- `frequencies`: Frequency vector
- `spectrogram`: Time-frequency power matrix
"""
function compute_spectrogram(signal::Vector{Float64}, fs::Float64=1000.0;
                            nperseg=256, noverlap=nothing)
    
    if isnothing(noverlap)
        noverlap = nperseg ÷ 2
    end
    
    window = DSP.hanning(nperseg)
    step = nperseg - noverlap
    
    # Calculate dimensions
    n_segments = (length(signal) - noverlap) ÷ step
    freq = rfftfreq(nperseg, fs)
    n_freq = length(freq)
    
    # Initialize spectrogram matrix
    spec = zeros(n_freq, n_segments)
    times = zeros(n_segments)
    
    for i in 1:n_segments
        start_idx = (i-1) * step + 1
        end_idx = start_idx + nperseg - 1
        
        if end_idx > length(signal)
            break
        end
        
        # Extract segment and apply window
        segment = signal[start_idx:end_idx] .* window
        
        # Compute FFT and power
        fft_result = rfft(segment)
        spec[:, i] = abs2.(fft_result)
        
        # Time corresponding to center of segment
        times[i] = (start_idx + end_idx) / 2 / fs
    end
    
    return times, freq, spec
end

"""
    compute_wavelet_transform(signal::Vector{Float64}, fs::Float64=1000.0;
                             frequencies=10:5:200, wavelet_cycles=6)

Compute continuous wavelet transform for time-frequency analysis.

# Arguments
- `signal`: Input time series
- `fs`: Sampling frequency in Hz
- `frequencies`: Array of frequencies to analyze
- `wavelet_cycles`: Number of cycles in Morlet wavelet

# Returns
- `times`: Time vector
- `frequencies`: Frequency vector
- `coefficients`: Complex wavelet coefficients
- `power`: Wavelet power |coefficients|²
"""
function compute_wavelet_transform(signal::Vector{Float64}, fs::Float64=1000.0;
                                  frequencies=10:5:200, wavelet_cycles=6)
    
    N = length(signal)
    times = (0:N-1) ./ fs
    n_freq = length(frequencies)
    
    # Initialize coefficient matrix
    coefficients = zeros(ComplexF64, n_freq, N)
    
    for (f_idx, freq) in enumerate(frequencies)
        # Create Morlet wavelet at this frequency
        # σ_t = wavelet_cycles / (2π * freq)
        sigma_t = wavelet_cycles / (2π * freq)
        
        # Wavelet in time domain (we'll create it in frequency domain for efficiency)
        # Time vector for wavelet (need enough samples)
        wavelet_duration = 2 * sigma_t * 4  # ±4σ
        n_wavelet = Int(ceil(wavelet_duration * fs))
        
        # Make sure it's odd
        if n_wavelet % 2 == 0
            n_wavelet += 1
        end
        
        t_wavelet = (-(n_wavelet÷2):(n_wavelet÷2)) ./ fs
        
        # Morlet wavelet: complex exponential times Gaussian envelope
        wavelet = exp.(1im * 2π * freq .* t_wavelet) .* 
                  exp.(-(t_wavelet.^2) ./ (2 * sigma_t^2))
        
        # Normalize
        wavelet ./= sqrt(sum(abs2.(wavelet)))
        
        # Convolve (using FFT for efficiency)
        convolved = conv(signal, wavelet)
        
        # Extract valid portion (centered)
        start_idx = (length(convolved) - N) ÷ 2 + 1
        coefficients[f_idx, :] = convolved[start_idx:(start_idx+N-1)]
    end
    
    power = abs2.(coefficients)
    
    return times, collect(frequencies), coefficients, power
end

"""
    identify_dominant_frequencies(signal::Vector{Float64}, fs::Float64=1000.0;
                                 n_peaks=5, freq_range=(1.0, 500.0))

Identify dominant frequency components in the signal.

# Arguments
- `signal`: Input time series
- `fs`: Sampling frequency in Hz
- `n_peaks`: Number of peaks to identify
- `freq_range`: Tuple of (min_freq, max_freq) to consider

# Returns
- Dictionary with:
  - `peak_frequencies`: Frequencies of top peaks
  - `peak_powers`: Power at those frequencies
  - `total_power`: Total power in signal
"""
function identify_dominant_frequencies(signal::Vector{Float64}, fs::Float64=1000.0;
                                      n_peaks=5, freq_range=(1.0, 500.0))
    
    # Compute PSD
    freq, power = compute_psd(signal, fs; method=:welch, nperseg=min(512, length(signal)÷4))
    
    # Filter to frequency range of interest
    freq_mask = (freq .>= freq_range[1]) .& (freq .<= freq_range[2])
    freq_roi = freq[freq_mask]
    power_roi = power[freq_mask]
    
    # Find peaks
    n_peaks = min(n_peaks, length(power_roi))
    peak_indices = sortperm(power_roi, rev=true)[1:n_peaks]
    
    peak_frequencies = freq_roi[peak_indices]
    peak_powers = power_roi[peak_indices]
    
    total_power = sum(power_roi)
    
    return Dict(
        "peak_frequencies" => peak_frequencies,
        "peak_powers" => peak_powers,
        "total_power" => total_power,
        "freq" => freq_roi,
        "power" => power_roi
    )
end

"""
    plot_psd(freq, power; title="Power Spectral Density", freq_range=nothing)

Plot power spectral density.
"""
function plot_psd(freq, power; title="Power Spectral Density", freq_range=nothing)
    
    if !isnothing(freq_range)
        mask = (freq .>= freq_range[1]) .& (freq .<= freq_range[2])
        freq = freq[mask]
        power = power[mask]
    end
    
    p = plot(freq, 10 .* log10.(power .+ eps()),
            xlabel="Frequency (Hz)",
            ylabel="Power (dB)",
            title=title,
            linewidth=2,
            legend=false,
            size=(800, 400))
    
    return p
end

"""
    plot_spectrogram(times, freq, spec; 
                    title="Spectrogram", freq_range=nothing, clim=nothing)

Plot time-frequency spectrogram.
"""
function plot_spectrogram(times, freq, spec; 
                         title="Spectrogram", freq_range=nothing, clim=nothing)
    
    if !isnothing(freq_range)
        mask = (freq .>= freq_range[1]) .& (freq .<= freq_range[2])
        freq = freq[mask]
        spec = spec[mask, :]
    end
    
    # Convert to dB
    spec_db = 10 .* log10.(spec .+ eps())
    
    if isnothing(clim)
        clim = (minimum(spec_db), maximum(spec_db))
    end
    
    p = heatmap(times, freq, spec_db,
               xlabel="Time (s)",
               ylabel="Frequency (Hz)",
               title=title,
               colorbar_title="Power (dB)",
               clim=clim,
               size=(1000, 500))
    
    return p
end

"""
    analyze_frequency_bands(signal::Vector{Float64}, fs::Float64=1000.0)

Analyze power in standard neural frequency bands.

# Frequency bands:
- Delta: 0.5-4 Hz
- Theta: 4-8 Hz  
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Low Gamma: 30-80 Hz
- High Gamma: 80-150 Hz
- Ripple: 150-250 Hz

# Returns
Dictionary with power and relative power in each band.
"""
function analyze_frequency_bands(signal::Vector{Float64}, fs::Float64=1000.0)
    
    bands = Dict(
        "delta" => (0.5, 4.0),
        "theta" => (4.0, 8.0),
        "alpha" => (8.0, 13.0),
        "beta" => (13.0, 30.0),
        "low_gamma" => (30.0, 80.0),
        "high_gamma" => (80.0, 150.0),
        "ripple" => (150.0, 250.0)
    )
    
    # Compute PSD
    freq, power = compute_psd(signal, fs; method=:welch)
    
    # Calculate power in each band
    results = Dict()
    total_power = sum(power)
    
    for (band_name, (f_min, f_max)) in bands
        # Skip if frequency range is outside Nyquist
        if f_min > fs/2
            continue
        end
        
        f_max = min(f_max, fs/2)
        
        mask = (freq .>= f_min) .& (freq .<= f_max)
        band_power = sum(power[mask])
        
        results[band_name] = Dict(
            "power" => band_power,
            "relative_power" => band_power / total_power,
            "freq_range" => (f_min, f_max)
        )
    end
    
    results["total_power"] = total_power
    results["freq"] = freq
    results["power_spectrum"] = power
    
    return results
end

println("✓ Neural Spectral Analysis module loaded")
println("  Functions available:")
println("    - compute_psd()")
println("    - compute_spectrogram()")
println("    - compute_wavelet_transform()")
println("    - identify_dominant_frequencies()")
println("    - analyze_frequency_bands()")
println("    - plot_psd(), plot_spectrogram()")
