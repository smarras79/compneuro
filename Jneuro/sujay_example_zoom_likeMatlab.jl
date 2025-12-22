using MAT
using Statistics
using DSP
using Plots

include("./myplots.jl")

# Code demonstration from Zoom with Sujay: see email on Nov 14

# Load the .mat file
data = matread("../data/amadeus01172020_a_neur_tensor_stim1on.mat")
# Alternative file: amadeus01172020_a_neur_tensor_joyon.mat
# Extract variables from the loaded data
cond_label = data["cond_label"]
cond_matrix = data["cond_matrix"]
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
# Julia automatically drops the first dimension when indexing with scalar
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

# Get time edges (handling struct field access)
# In MATLAB: stim1on.edges - this might be a struct or dict in Julia
edges = if haskey(stim1on, "edges")
    stim1on["edges"]
else
    # If stim1on is a struct-like object, try different access patterns
    stim1on
end

# Compute smoothed firing rates using convolution
# MATLAB: conv(mean(fr3,2), ones(300,1), 'valid')
# Julia: conv with mean along dimension 2 (trials)
kernel = ones(300)
m = length(kernel)

# Full convolution, then extract 'valid' part by skipping edges
fr3_full = conv(vec(mean(fr3, dims=2)), kernel)
fr3_smooth = fr3_full[m:(end-(m-1))]  # 'valid' mode: skip first (m-1) and last (m-1)

fr4_full = conv(vec(mean(fr4, dims=2)), kernel)
fr4_smooth = fr4_full[m:(end-(m-1))]

# Plot neural data and save to file
# Time bins: edges(150:end-150) in MATLAB
time_idx = 150:(length(edges)-150)
time_bins = edges[time_idx]

neural_plots(time_bins, fr3_smooth, fr4_smooth, "1")
