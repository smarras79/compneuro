# ==========================================
# COMPLETE NEURAL ANALYSIS SCRIPT
# ==========================================

using MAT
using Statistics

println("=" ^70)
println("NEURAL ANALYSIS SCRIPT")
println("=" ^70)

# ===== 1. LOAD DATA =====
println("\n[1/4] Loading data...")

data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
neur_tensor = data["neur_tensor_stim1on"]

# Extract signal
your_neural_data = vec(mean(neur_tensor[1, :, :], dims=2))

println("  Signal length: $(length(your_neural_data)) samples")
println("  ✓ Data loaded")

# ===== 2. LOAD ANALYSIS TOOLS =====
println("\n[2/4] Loading analysis pipeline...")

include("integrated_analysis_pipeline.jl")

println("  ✓ Pipeline loaded")

# ===== 3. RUN ANALYSIS =====
println("\n[3/4] Running analysis...")

fs = 1000.0  # Sampling frequency in Hz

results = analyze_neural_data_comprehensive(
    your_neural_data,
    nothing;  # No behavioral events
    fs=fs
)

println("  ✓ Analysis complete")

# ===== 4. DISPLAY AND SAVE RESULTS =====
println("\n[4/4] Results:")

# SWR count
n_swr = results["swr_detection"]["n_events"]
println("  Detected $n_swr Sharp-Wave Ripples")

# Frequency bands
if haskey(results, "frequency_bands")
    println("\n  Frequency Band Power:")
    bands = results["frequency_bands"]
    for band in sort(collect(keys(bands)))
        if band != "total_power" && band != "freq" && band != "power_spectrum"
            info = bands[band]
            rel_power = get(info, "relative_power", 0.0) * 100
            println("    $(rpad(band, 12)): $(round(rel_power, digits=1))%")
        end
    end
end

# Save results
println("\n  Saving results...")
save_analysis_results(results, "./neural_analysis_output")

println("\n" * "=" ^70)
println("✓ COMPLETE!")
println("Check './neural_analysis_output/' for plots and summary")
println("=" ^70)
