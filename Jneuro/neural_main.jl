#
# 
#
# ========== 1. LOAD AND PREPARE DATA ==========
include("integrated_analysis_pipeline.jl")

# Load data
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
signal_raw = data["neural_signal"]
fs = 1000.0

# Optional: Apply filtering
signal = apply_neural_filter(signal_raw, :gaussian, 300; sigma=50)

# ========== 2. EXTRACT BEHAVIORAL EVENTS ==========
# From your behavioral data
motion_onsets = extract_your_onsets()
motion_offsets = extract_your_offsets()

events = Dict(
    "motion_onset" => motion_onsets,
    "motion_offset" => motion_offsets
)

# ========== 3. RUN ANALYSIS ==========
results = analyze_neural_data_comprehensive(signal, events; fs=fs)

# ========== 4. EXAMINE RESULTS ==========
# How many SWRs?
n_swr = results["swr_detection"]["n_events"]
println("Detected $n_swr SWRs")

# Where are they clustered?
ml = results["ml_analysis"]
println("Found $(ml["n_clusters"]) types of SWRs")

# SWRs near events?
onset_analysis = results["event_triggered_motion_onset"]
for (key, event) in onset_analysis
    if startswith(key, "event_")
        println("Event: $(event["n_swr"]) SWRs nearby")
    end
end

# ========== 5. SAVE EVERYTHING ==========
save_analysis_results(results, "./my_results")

# ========== 6. FURTHER ANALYSIS (if needed) ==========
# Get specific events
all_swr = results["swr_detection"]["events"]
ripple_times = [e.peak_sample / fs for e in all_swr]

# Export for external analysis
using CSV, DataFrames

df = DataFrame(
    time = ripple_times,
    duration = [e.duration_ms for e in all_swr],
    amplitude = [e.peak_amplitude for e in all_swr],
    cluster = ml["cluster_labels"],
    is_anomaly = ml["is_anomaly"]
)

CSV.write("swr_results.csv", df)
