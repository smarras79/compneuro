using MAT
using Statistics

println("="^80)
println("CONDITION EXPLORER")
println("Helps identify good conditions for neural selectivity analysis")
println("="^80)

# Load data
data = matread("./data/amadeus01172020_a_neur_tensor_stim1on.mat")
cond_label = data["cond_label"]
cond_matrix = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]

n_neurons, n_timebins, n_trials = size(neur_tensor_stim1on)

println("\nData summary:")
println("  Neurons: $n_neurons")
println("  Time bins: $n_timebins")
println("  Trials: $n_trials")

println("\n" * "="^80)
println("CONDITION LABELS AND VALUES")
println("="^80)

# Display what each column represents
println("\nColumn labels:")
for (i, label) in enumerate(vec(cond_label))
    println("  Column $i: $label")
end

println("\n" * "="^80)
println("VALUE DISTRIBUTIONS IN EACH COLUMN")
println("="^80)

# For each column, show unique values and counts
for col = 1:size(cond_matrix, 2)
    label = vec(cond_label)[col]
    values = cond_matrix[:, col]
    unique_vals = unique(values)
    
    println("\nColumn $col ($label):")
    for val in sort(unique_vals)
        count = sum(values .== val)
        pct = round(100 * count / n_trials, digits=1)
        println("  Value $(val): $count trials ($pct%)")
    end
end

println("\n" * "="^80)
println("SUGGESTED CONDITION COMPARISONS")
println("="^80)

# Find columns with exactly 2 values (binary comparisons)
binary_cols = []
for col = 1:size(cond_matrix, 2)
    unique_vals = unique(cond_matrix[:, col])
    if length(unique_vals) == 2
        push!(binary_cols, col)
    end
end

println("\nBinary conditions (2 values - good for simple comparisons):")
for col in binary_cols
    label = vec(cond_label)[col]
    values = cond_matrix[:, col]
    unique_vals = sort(unique(values))
    count1 = sum(values .== unique_vals[1])
    count2 = sum(values .== unique_vals[2])
    println("  Column $col ($label): $(unique_vals[1]) ($count1 trials) vs $(unique_vals[2]) ($count2 trials)")
end

# Find multi-valued columns
println("\nMulti-valued conditions (>2 values - good for ANOVA/multiclass):")
for col = 1:size(cond_matrix, 2)
    unique_vals = unique(cond_matrix[:, col])
    if length(unique_vals) > 2 && length(unique_vals) < 10
        label = vec(cond_label)[col]
        println("  Column $col ($label): $(length(unique_vals)) values - $unique_vals")
    end
end

println("\n" * "="^80)
println("COMPUTING SELECTIVITY FOR COMMON COMPARISONS")
println("="^80)

# Try a few common comparisons
comparisons = [
    (10, 12, "attempt vs validtrials_mm"),
    (1, "temporal distance (continuous)"),
    (4, "target location (multiclass)"),
]

function compute_binary_selectivity(col1, col2, label)
    # Find trials for each condition
    val1 = sort(unique(cond_matrix[:, col1]))[1]
    val2 = sort(unique(cond_matrix[:, col2]))[1]
    
    trid1 = findall(cond_matrix[:, col1] .== val1)
    trid2 = findall(cond_matrix[:, col2] .== val2)
    
    if length(trid1) == 0 || length(trid2) == 0
        println("\n$label:")
        println("  ⚠ One condition has no trials")
        return
    end
    
    # Compute selectivity
    selectivity = zeros(n_neurons)
    for i = 1:n_neurons
        fr1 = mean(neur_tensor_stim1on[i, :, trid1])
        fr2 = mean(neur_tensor_stim1on[i, :, trid2])
        selectivity[i] = (fr1 - fr2) / (fr1 + fr2 + 1e-10)
    end
    
    n_selective_30 = sum(abs.(selectivity) .> 0.3)
    n_selective_20 = sum(abs.(selectivity) .> 0.2)
    n_selective_15 = sum(abs.(selectivity) .> 0.15)
    
    println("\n$label:")
    println("  Condition 1: $(length(trid1)) trials")
    println("  Condition 2: $(length(trid2)) trials")
    println("  Selective neurons (|SI| > 0.3): $n_selective_30 / $n_neurons ($(round(100*n_selective_30/n_neurons, digits=1))%)")
    println("  Selective neurons (|SI| > 0.2): $n_selective_20 / $n_neurons ($(round(100*n_selective_20/n_neurons, digits=1))%)")
    println("  Selective neurons (|SI| > 0.15): $n_selective_15 / $n_neurons ($(round(100*n_selective_15/n_neurons, digits=1))%)")
    println("  Mean |SI|: $(round(mean(abs.(selectivity)), digits=3))")
end

# Column 10 vs Column 12 (if they're binary)
if 10 in binary_cols && 12 in binary_cols
    compute_binary_selectivity(10, 12, "Columns 10 vs 12 (attempt vs validtrials_mm)")
end

# Try other binary columns
for col in binary_cols
    if col != 10 && col != 12
        label = vec(cond_label)[col]
        vals = sort(unique(cond_matrix[:, col]))
        trid1 = findall(cond_matrix[:, col] .== vals[1])
        trid2 = findall(cond_matrix[:, col] .== vals[2])
        
        if length(trid1) > 10 && length(trid2) > 10  # Only if both have reasonable trial counts
            selectivity = zeros(n_neurons)
            for i = 1:n_neurons
                fr1 = mean(neur_tensor_stim1on[i, :, trid1])
                fr2 = mean(neur_tensor_stim1on[i, :, trid2])
                selectivity[i] = (fr1 - fr2) / (fr1 + fr2 + 1e-10)
            end
            
            n_selective = sum(abs.(selectivity) .> 0.2)
            if n_selective > 0
                println("\nColumn $col ($label): $(vals[1]) vs $(vals[2])")
                println("  Trials: $(length(trid1)) vs $(length(trid2))")
                println("  Selective neurons (|SI| > 0.2): $n_selective ($(round(100*n_selective/n_neurons, digits=1))%)")
            end
        end
    end
end

println("\n" * "="^80)
println("RECOMMENDATIONS")
println("="^80)
println("\nBased on this analysis:")
println("1. Look for comparisons with >5% selective neurons")
println("2. Ensure both conditions have >10 trials")
println("3. Consider continuous variables for regression instead")
println("4. Time-resolved analysis may reveal selectivity not visible in trial-averaged data")
println("5. Check behavioral data to ensure conditions are meaningfully different")
println("\n" * "="^80)
