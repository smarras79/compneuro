using MAT
using Statistics
using DSP

# Simple test to verify filter output lengths

println("="^70)
println("FILTER OUTPUT LENGTH TEST")
println("="^70)

# Load data
println("\nLoading data...")
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")

cond_matrix = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

# Extract firing rates
trid = findall((cond_matrix[:, 10] .== 1) .& 
    (cond_matrix[:, 3] .== 1) .& 
    (cond_matrix[:, 4] .== 4))
fr3 = neur_tensor_stim1on[1, :, trid]

# Compute mean
fr_mean = vec(mean(fr3, dims=2))

println("Input signal length: $(length(fr_mean))")

# Get time edges
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    stim1on
end

println("Time edges length: $(length(edges))")

# Test original moving average filter
println("\n" * "-"^70)
println("Testing Original Moving Average (as in your code)")
println("-"^70)

window_size = 300
kernel = ones(window_size)
m = length(kernel)

full_conv = conv(fr_mean, kernel)
filtered_valid = full_conv[m:(end-(m-1))]

println("Window size: $(window_size)")
println("Full convolution length: $(length(full_conv))")
println("Valid mode output length: $(length(filtered_valid))")
println("Expected: $(length(fr_mean) - window_size + 1)")
println("Match: $(length(filtered_valid) == length(fr_mean) - window_size + 1)")

# Calculate what the time vector should be
println("\n" * "-"^70)
println("Time Vector Calculation")
println("-"^70)

# Original code uses edges[150:(end-150)]
time_idx_original = 150:(length(edges)-150)
time_bins_original = edges[time_idx_original]

println("Original time bins (150:end-150): $(length(time_bins_original))")
println("Filtered signal length: $(length(filtered_valid))")
println("Difference: $(length(time_bins_original) - length(filtered_valid))")

# To match, we need to further trim the time vector
additional_trim = (length(time_bins_original) - length(filtered_valid)) ÷ 2
time_start = 150 + additional_trim
time_end = length(edges) - 150 - additional_trim
time_matched = edges[time_start:time_end]

println("\nCorrected time vector:")
println("  Start index: $(time_start)")
println("  End index: $(time_end)")
println("  Length: $(length(time_matched))")
println("  Matches filtered signal: $(length(time_matched) == length(filtered_valid))")

# Verify we can plot
println("\n" * "-"^70)
println("Verification")
println("-"^70)

if length(time_matched) == length(filtered_valid)
    println("✓ Time vector and filtered signal have matching lengths!")
    println("✓ Can safely plot(time_matched, filtered_valid)")
else
    println("✗ Length mismatch detected!")
    println("  Time: $(length(time_matched))")
    println("  Signal: $(length(filtered_valid))")
end

println("\n" * "="^70)
