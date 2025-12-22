using MAT
using Statistics
using DSP
using Plots

include("./myplots.jl")

# Include the filter function from the main script
"""
    apply_neural_filter(data::Vector, filter_type::Symbol, window_size::Int=300; kwargs...)

Apply different types of filters to neural time series data.
See main script for full documentation.
"""
function apply_neural_filter(data::Vector, 
                            filter_type::Symbol, 
                            window_size::Int=300;
                            cutoff_freq::Float64=0.1,
                            fs::Float64=1.0,
                            filter_order::Int=4,
                            poly_order::Int=3,
                            alpha::Float64=0.1,
                            sigma::Float64=window_size/6)
    
    if filter_type == :moving_average
        kernel = ones(window_size)
        m = length(kernel)
        full_conv = conv(data, kernel)
        return full_conv[m:(end-(m-1))]
        
    elseif filter_type == :gaussian
        x = -(window_size÷2):(window_size÷2)
        kernel = exp.(-(x.^2) ./ (2*sigma^2))
        kernel = kernel ./ sum(kernel)
        m = length(kernel)
        full_conv = conv(data, kernel)
        return full_conv[m:(end-(m-1))]
        
    elseif filter_type == :savitzky_golay
        win_size = isodd(window_size) ? window_size : window_size + 1
        filtered = savitzky_golay(data, win_size, poly_order).y
        trim = (win_size - 1) ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :butterworth
        # Normalize cutoff frequency to Nyquist frequency
        normalized_cutoff = cutoff_freq / (fs / 2)
        
        # Ensure normalized frequency is in valid range
        if normalized_cutoff >= 1.0
            normalized_cutoff = 0.99
        elseif normalized_cutoff <= 0.0
            normalized_cutoff = 0.01
        end
        
        responsetype = Lowpass(normalized_cutoff)
        designmethod = Butterworth(filter_order)
        filter_design = digitalfilter(responsetype, designmethod)
        filtered = filtfilt(filter_design, data)
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :median
        win_size = isodd(window_size) ? window_size : window_size + 1
        filtered = zeros(length(data))
        half_win = win_size ÷ 2
        
        for i in 1:length(data)
            start_idx = max(1, i - half_win)
            end_idx = min(length(data), i + half_win)
            filtered[i] = median(data[start_idx:end_idx])
        end
        
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    elseif filter_type == :exponential
        filtered = zeros(length(data))
        filtered[1] = data[1]
        
        for i in 2:length(data)
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
        end
        
        trim = window_size ÷ 2
        return filtered[(trim+1):(end-trim)]
        
    else
        error("Unknown filter type: $(filter_type)")
    end
end

# Load the .mat file
println("Loading data...")
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")

cond_matrix = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

# Extract firing rates for condition: column 10==1 & column 3==1 & column 4==4
trid = findall((cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

# Compute mean firing rate across trials
fr_mean = vec(mean(fr3, dims=2))

#%% Compare different filters
println("\nComparing different filter types...")

window_size = 300

# Define filters to compare
filter_types = [
    :moving_average,
    :gaussian,
    :savitzky_golay,
    :butterworth,
    :median,
    :exponential
]

# Apply each filter
filtered_signals = Dict()
println("  Original signal length: $(length(fr_mean))")

for ftype in filter_types
    println("  Applying $(ftype) filter...")
    try
        if ftype == :exponential
            # Exponential needs smaller alpha for similar smoothing
            filtered_signals[ftype] = apply_neural_filter(
                fr_mean, ftype, window_size; alpha=0.05
            )
        elseif ftype == :butterworth
            # Butterworth with appropriate cutoff
            filtered_signals[ftype] = apply_neural_filter(
                fr_mean, ftype, window_size; cutoff_freq=0.05, fs=1.0
            )
        else
            filtered_signals[ftype] = apply_neural_filter(
                fr_mean, ftype, window_size
            )
        end
        println("    Output length: $(length(filtered_signals[ftype]))")
    catch e
        println("    Warning: $(ftype) failed - $(e)")
        filtered_signals[ftype] = nothing
    end
end

# Create time bins for plotting
# The filtered signals are trimmed by window_size÷2 on each end (valid mode)
# So we need to match the time vector to the filtered signal length
trim_amount = window_size ÷ 2

println("\nTime vector construction:")
println("  Total edges length: $(length(edges))")
println("  Trim amount: $(trim_amount)")

# Start with the full edges, then trim to match filtered signal
# Filtered signals are: original_length - 2*trim_amount
# We also need to account for the 150 offset used in original code
time_idx = (150 + trim_amount):(length(edges) - 150 - trim_amount)
plot_time = edges[time_idx]

println("  Plot time length: $(length(plot_time))")

#%% Create comparison plot
println("\nCreating comparison plots...")

# Create subplots comparing all filters
plots = []

for ftype in filter_types
    if !isnothing(filtered_signals[ftype])
        signal = filtered_signals[ftype]
        
        # Adjust time vector to match signal length if needed
        if length(plot_time) > length(signal)
            diff = length(plot_time) - length(signal)
            trim_extra = diff ÷ 2
            time_vec = plot_time[(trim_extra+1):(end-diff+trim_extra)]
        elseif length(plot_time) < length(signal)
            diff = length(signal) - length(plot_time)
            signal = signal[(diff÷2+1):(end-diff÷2)]
            time_vec = plot_time
        else
            time_vec = plot_time
        end
        
        # Ensure exact match
        min_len = min(length(time_vec), length(signal))
        time_vec = time_vec[1:min_len]
        signal = signal[1:min_len]
        
        p = plot(time_vec, signal,
                label=string(ftype),
                linewidth=2,
                title=string(ftype),
                xlabel="Time",
                ylabel="Firing Rate",
                legend=:topright,
                size=(600, 300))
        push!(plots, p)
    end
end

# Combine into single figure
comparison_plot = plot(plots..., 
                      layout=(3, 2), 
                      size=(1200, 900),
                      plot_title="Filter Comparison: Neural Firing Rates")

savefig(comparison_plot, "filter_comparison.png")
println("✓ Saved comparison plot to: filter_comparison.png")

#%% Overlay plot - all filters on same axes
println("\nCreating overlay plot...")

overlay_plot = plot(xlabel="Time", 
                   ylabel="Firing Rate",
                   title="All Filters Overlaid",
                   legend=:outertopright,
                   size=(1000, 600))

colors = [:blue, :red, :green, :purple, :orange, :brown]

for (idx, ftype) in enumerate(filter_types)
    if !isnothing(filtered_signals[ftype])
        signal = filtered_signals[ftype]
        
        # Adjust time vector to match signal length if needed
        if length(plot_time) > length(signal)
            diff = length(plot_time) - length(signal)
            trim_extra = diff ÷ 2
            time_vec = plot_time[(trim_extra+1):(end-diff+trim_extra)]
        elseif length(plot_time) < length(signal)
            diff = length(signal) - length(plot_time)
            signal = signal[(diff÷2+1):(end-diff÷2)]
            time_vec = plot_time
        else
            time_vec = plot_time
        end
        
        # Ensure exact match
        min_len = min(length(time_vec), length(signal))
        time_vec = time_vec[1:min_len]
        signal = signal[1:min_len]
        
        plot!(overlay_plot, time_vec, signal,
              label=string(ftype),
              linewidth=2,
              alpha=0.7,
              color=colors[idx])
    end
end

savefig(overlay_plot, "filter_overlay.png")
println("✓ Saved overlay plot to: filter_overlay.png")

#%% Compute filter characteristics
println("\n" * "="^70)
println("FILTER CHARACTERISTICS")
println("="^70)

for ftype in filter_types
    if !isnothing(filtered_signals[ftype])
        signal = filtered_signals[ftype]
        
        println("\n$(ftype):")
        println("  Mean: $(round(mean(signal), digits=4))")
        println("  Std:  $(round(std(signal), digits=4))")
        println("  Min:  $(round(minimum(signal), digits=4))")
        println("  Max:  $(round(maximum(signal), digits=4))")
        
        # Compute smoothness (second derivative)
        if length(signal) > 2
            smoothness = sum(abs.(diff(diff(signal))))
            println("  Smoothness (lower = smoother): $(round(smoothness, digits=2))")
        end
    end
end

println("\n" * "="^70)
println("✓ Filter comparison complete!")
println("\nGenerated files:")
println("  - filter_comparison.png (individual subplots)")
println("  - filter_overlay.png (all filters overlaid)")
