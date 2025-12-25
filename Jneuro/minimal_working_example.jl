#!/usr/bin/env julia
"""
MINIMAL WORKING EXAMPLE
This demonstrates the correct way to load variables without scope issues
"""

using MAT

println("="^70)
println("MINIMAL WORKING EXAMPLE - Variable Loading")
println("="^70)

# Change this to your actual filename
filename = "./data/amadeus01172020_a_neur_tensor_stim1on.mat"

# Load data
println("\nLoading data file...")
data = matread(filename)
println("✓ Data loaded")

# Load variables - AT TOP LEVEL, NO CONDITIONALS
println("\nLoading variables at top level (no scope issues)...")

# Method 1: Direct assignment
cond_label = data["cond_label"]
cond_matrix = data["cond_matrix"]
neur_tensor_stim1on = data["neur_tensor_stim1on"]
stim1on = data["stim1on"]

println("✓ All variables loaded")

# Verify they're accessible
println("\nVerifying variables are accessible:")
println("  cond_label: $(typeof(cond_label))")
println("  cond_matrix: $(typeof(cond_matrix)), size=$(size(cond_matrix))")
println("  neur_tensor_stim1on: $(typeof(neur_tensor_stim1on)), size=$(size(neur_tensor_stim1on))")
println("  stim1on: $(typeof(stim1on))")

# Extract edges
if isa(stim1on, AbstractDict) && haskey(stim1on, "edges")
    edges = stim1on["edges"]
    println("\n✓ Extracted edges from stim1on")
    println("  edges: $(typeof(edges)), size=$(size(edges))")
    println("  range: $(edges[1]) to $(edges[end])")
else
    println("\n⚠️  Could not extract edges from stim1on")
end

println("\n" * "="^70)
println("SUCCESS! Variables loaded and accessible without scope issues")
println("="^70)
println("\nThis is the approach to use in main_enhanced.jl:")
println("  1. Load at top level (not inside if/else)")
println("  2. Use direct assignment")
println("  3. No global keyword needed")
println("="^70)
