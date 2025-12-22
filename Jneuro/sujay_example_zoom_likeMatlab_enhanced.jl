using MAT
using Statistics
using DSP
using Plots

include("./myplots.jl")

# Code demonstration from Zoom with Sujay: see email on Nov 14

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
    
    if filter_type == :moving_average
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

# Load the .mat file
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
# Alternative file: amadeus01172020_a_neur_tensor_joyon.mat
# Extract variables from the loaded data
cond_label          = data["cond_label"]
cond_matrix         = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

# Display condition labels (tells you which column codes for what task parameter)
println("Condition labels:")
println(cond_label)

#%% Behavioural data
# Find trials where column 10 == 1
trid = findall(cond_matrix[:, 10] .== 1)
ta_att1 = cond_matrix[trid, 1]
tp_att1 = cond_matrix[trid, 2]
neural_plots_scatter(ta_att1, tp_att1, "1")

# Second subplot - find trials where column 12 == 1
trid = findall(cond_matrix[:, 12] .== 1)
ta_att2 = cond_matrix[trid, 1]
tp_att2 = cond_matrix[trid, 2]

neural_plots_scatter(ta_att2, tp_att2, "2")

#=
# Combine subplots and save to file
behavioral_plot = plot(p1, p2, layout=(2, 2), size=(800, 800))
savefig(behavioral_plot, "behavioral_data.png")
println("Saved behavioral plot to: behavioral_data.png")
=#

#%% Neural data
taa = unique(abs.(cond_matrix[:, 1]))

# Extract firing rates for different conditions
# Condition: column 10==1 & column 3==1 & column 4==4
trid = findall((cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]

# Condition: column 10==1 & column 3==1 & column 4==5
trid = findall((cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 5))
fr4 = neur_tensor_stim1on[1, :, trid]

# Condition: column 10==1 & column 3==1 & column 4==2
trid = findall((cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 2))
fr1 = neur_tensor_stim1on[3, :, trid]

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

#%% ====== FILTER SELECTION ======
# CHOOSE YOUR FILTER TYPE HERE:
# Options: :moving_average, :gaussian, :savitzky_golay, :butterworth, :median, :exponential

# Print available options
print_filter_options()

# Select filter type (change this to try different filters)
selected_filter = :moving_average  # Default: same as original code

# Window size (300 samples as in original)
window_size = 300

# Additional parameters for specific filters (adjust as needed)
filter_params = Dict(
    :cutoff_freq => 0.05,      # For Butterworth (Hz)
    :fs => 1.0,                # Sampling frequency (Hz)
    :filter_order => 4,        # Butterworth order
    :poly_order => 3,          # Savitzky-Golay polynomial order
    :alpha => 0.05,            # Exponential MA smoothing factor
    :sigma => Float64(window_size)/6.0    # Gaussian standard deviation
)

println("Using filter: $(selected_filter)")
println("Window size: $(window_size)")
println()

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
    min_len = min(length(time_bins), length(fr3_smooth))
    time_bins = time_bins[1:min_len]
    fr3_smooth = fr3_smooth[1:min_len]
    fr4_smooth = fr4_smooth[1:min_len]
    println("  Adjusted to length: $(min_len)")
end

neural_plots(time_bins, fr3_smooth, fr4_smooth, "1")

println("\n✓ Neural data processed with $(selected_filter) filter")
println("✓ Smoothed firing rates computed and plotted")
