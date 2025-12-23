"""
    apply_neural_filter(data::Vector, filter_type::Symbol, window_size::Int=300; kwargs...)

Apply different types of filters to neural time series data.

# Arguments
- `data::Vector`: Input neural time series data
- `filter_type::Symbol`: Type of filter to apply
  - `:moving_average` - Simple moving average (box filter)
  - `:gaussian` - Gaussian smoothing filter
  - `:savitzky_golay` - Savitzky-Golay filter (preserves peaks)
  - `:butterworth` - Butterworth lowpass filter
  - `:median` - Median filter (good for spike removal)
  - `:exponential` - Exponential moving average
- `window_size::Int`: Size of the filter window (default: 300)

# Keyword Arguments
- `cutoff_freq::Float64`: Cutoff frequency for Butterworth filter (Hz, default: 0.1)
- `fs::Float64`: Sampling frequency for Butterworth filter (Hz, default: 1.0)
- `filter_order::Int`: Order of Butterworth filter (default: 4)
- `poly_order::Int`: Polynomial order for Savitzky-Golay (default: 3)
- `alpha::Float64`: Smoothing factor for exponential MA (0-1, default: 0.1)
- `sigma::Float64`: Standard deviation for Gaussian filter (default: window_size/6)

# Returns
- Filtered signal vector
"""
function apply_neural_filter(data::Vector, 
                            filter_type::Symbol, 
                            window_size::Int=300;
                            cutoff_freq::Float64=0.1,
                            fs::Float64=1.0,
                            filter_order::Int=4,
                            poly_order::Int=3,
                            alpha::Float64=0.1,
                            sigma::Union{Float64,Int}=window_size/6)
    
    # Convert sigma to Float64 if needed
    sigma_float = Float64(sigma)

    # Handle no-filter case
    if filter_type == :none
        println("  ✓ No filtering (raw signal)")
        return data  # Return unchanged
        
    elseif filter_type == :moving_average
        # Original implementation: simple box filter
        kernel = ones(window_size)
        m = length(kernel)
        # Full convolution, then extract 'valid' part
        full_conv = conv(data, kernel)
        return full_conv[m:(end-(m-1))]  # 'valid' mode
        
    elseif filter_type == :gaussian
        # Gaussian filter - smooth with bell-shaped kernel
        # Create Gaussian kernel
        x = -(window_size÷2):(window_size÷2)
        kernel = exp.(-(x.^2) ./ (2*sigma_float^2))
        kernel = kernel ./ sum(kernel)  # Normalize
        
        m = length(kernel)
        full_conv = conv(data, kernel)
        return full_conv[m:(end-(m-1))]
        
    elseif filter_type == :savitzky_golay
        # Savitzky-Golay filter - preserves peaks and features
        # Ensure window_size is odd
        win_size = isodd(window_size) ? window_size : window_size + 1
        
        # Apply Savitzky-Golay filter
        filtered = savitzky_golay(data, win_size, poly_order).y
        
        # Trim to match 'valid' mode behavior
        trim = (win_size - 1) ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :butterworth
        # Butterworth lowpass filter - classic signal processing
        # Normalize cutoff frequency to Nyquist frequency (fs/2)
        # cutoff_freq is in Hz, fs is sampling frequency in Hz
        # For digital filters, frequency is normalized to [0, 1] where 1 = Nyquist
        normalized_cutoff = cutoff_freq / (fs / 2)
        
        # Ensure normalized frequency is in valid range
        if normalized_cutoff >= 1.0
            normalized_cutoff = 0.99  # Avoid instability
        elseif normalized_cutoff <= 0.0
            normalized_cutoff = 0.01
        end
        
        responsetype = Lowpass(normalized_cutoff)
        designmethod = Butterworth(filter_order)
        
        # Create filter
        filter_design = digitalfilter(responsetype, designmethod)
        
        # Apply filter (using filtfilt for zero-phase filtering)
        filtered = filtfilt(filter_design, data)
        
        # Trim edges to match 'valid' mode
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :median
        # Median filter - excellent for removing spikes/outliers
        # Ensure window_size is odd
        win_size = isodd(window_size) ? window_size : window_size + 1
        
        filtered = zeros(length(data))
        half_win = win_size ÷ 2
        
        for i in 1:length(data)
            # Get window bounds
            start_idx = max(1, i - half_win)
            end_idx = min(length(data), i + half_win)
            
            # Compute median in window
            filtered[i] = median(data[start_idx:end_idx])
        end
        
        # Trim edges to match 'valid' mode
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :exponential
        # Exponential moving average - weighted smoothing
        # More recent data gets higher weight
        filtered = zeros(length(data))
        filtered[1] = data[1]
        
        for i in 2:length(data)
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
        end
        
        # Trim edges to match 'valid' mode
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    else
        error("Unknown filter type: $(filter_type). Valid options are: " *
              ":moving_average, :gaussian, :savitzky_golay, :butterworth, :median, :exponential")
    end
end

"""
    print_filter_options()

Display available filter options and their characteristics.
"""
function print_filter_options()
    println("\n" * "="^70)
    println("AVAILABLE NEURAL SIGNAL FILTERS")
    println("="^70)
    
    println("\n1. :moving_average (Default)")
    println("   - Simple box filter, uniform averaging")
    println("   - Fast, easy to understand")
    println("   - Good for general smoothing")
    
    println("\n2. :gaussian")
    println("   - Bell-shaped kernel, smooth transitions")
    println("   - Better frequency response than moving average")
    println("   - Good for noise reduction while preserving shape")
    
    println("\n3. :savitzky_golay")
    println("   - Polynomial fitting in local windows")
    println("   - Preserves peaks and features better")
    println("   - Good for maintaining sharp transitions")
    
    println("\n4. :butterworth")
    println("   - Classic frequency-domain lowpass filter")
    println("   - Sharp frequency cutoff")
    println("   - Good for removing specific frequency components")
    
    println("\n5. :median")
    println("   - Nonlinear filter, takes median in window")
    println("   - Excellent for removing outliers/spikes")
    println("   - Preserves edges well")
    
    println("\n6. :exponential")
    println("   - Weighted average with exponential decay")
    println("   - More weight on recent data")
    println("   - Good for tracking trends")
    
    println("\n" * "="^70 * "\n")
end
