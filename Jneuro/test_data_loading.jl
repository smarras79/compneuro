#!/usr/bin/env julia
"""
TEST SCRIPT: Verify data loading works
Run this first to confirm your data can be loaded
"""

using MAT

println("="^70)
println("DATA LOADING TEST")
println("="^70)

# Load the data file
println("\n1. Loading MAT file...")
filename = "amadeus01172020_a_neur_tensor_stim1on.mat"  # Change to your filename
data = matread(filename)
println("   ✓ File loaded successfully")

# Show what's in the file
println("\n2. Variables in file:")
for (key, value) in data
    println("   - $key: $(typeof(value))")
end

# Try to load each variable
println("\n3. Loading individual variables...")

println("\n   Loading cond_label:")
try
    cond_label = data["cond_label"]
    println("   ✓ SUCCESS: $(typeof(cond_label))")
    println("     Size: $(size(cond_label))")
catch e
    println("   ✗ FAILED: $e")
end

println("\n   Loading cond_matrix:")
try
    cond_matrix = data["cond_matrix"]
    println("   ✓ SUCCESS: $(typeof(cond_matrix))")
    println("     Size: $(size(cond_matrix))")
catch e
    println("   ✗ FAILED: $e")
end

println("\n   Loading neur_tensor_stim1on:")
try
    neur_tensor_stim1on = data["neur_tensor_stim1on"]
    println("   ✓ SUCCESS: $(typeof(neur_tensor_stim1on))")
    println("     Size: $(size(neur_tensor_stim1on))")
catch e
    println("   ✗ FAILED: $e")
end

println("\n   Loading stim1on:")
try
    stim1on = data["stim1on"]
    println("   ✓ SUCCESS: $(typeof(stim1on))")
    if isa(stim1on, AbstractDict)
        println("     Keys: $(collect(keys(stim1on)))")
        if haskey(stim1on, "edges")
            edges = stim1on["edges"]
            println("     edges: $(typeof(edges)), size=$(size(edges))")
            println("     edges range: $(edges[1]) to $(edges[end])")
        end
    elseif isa(stim1on, AbstractArray)
        println("     Size: $(size(stim1on))")
    end
catch e
    println("   ✗ FAILED: $e")
end

println("\n" * "="^70)
println("TEST COMPLETE")
println("="^70)
