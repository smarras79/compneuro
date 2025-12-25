# SIMPLE DIRECT LOADING - NO SCOPE ISSUES
# Replace lines 108-280 in main_enhanced.jl with this

#------------------------------------------------------------------------
#%% VARIABLE EXTRACTION - Direct method (no scoping issues)
#------------------------------------------------------------------------
println("\n" * "="^70)
println("LOADING VARIABLES FROM MAT FILE")
println("="^70)
println()

# Load variables DIRECTLY at top level - no conditionals, no scope issues
println("Loading variables...")

cond_label = try
    val = data["cond_label"]
    println("  ✓ cond_label: $(typeof(val))")
    val
catch e
    println("  ⚠️  cond_label not found: $e")
    nothing
end

cond_matrix = try
    val = data["cond_matrix"]
    println("  ✓ cond_matrix: $(typeof(val)), size=$(size(val))")
    val
catch e
    println("  ⚠️  cond_matrix not found: $e")
    nothing
end

neur_tensor_stim1on = try
    val = data["neur_tensor_stim1on"]
    dims = size(val)
    println("  ✓ neur_tensor_stim1on: $(typeof(val)), size=$(dims)")
    println("    → $(dims[1]) neurons × $(dims[2]) time bins × $(dims[3]) trials")
    val
catch e
    error("  ✗ neur_tensor_stim1on REQUIRED but not found: $e")
end

stim1on = try
    val = data["stim1on"]
    println("  ✓ stim1on: $(typeof(val))")
    if isa(val, AbstractDict)
        println("    → Dict with keys: $(collect(keys(val)))")
    elseif isa(val, AbstractArray)
        println("    → Array with size: $(size(val))")
    end
    val
catch e
    println("  ⚠️  stim1on not found: $e")
    nothing
end

# Extract edges
println("\nExtracting time edges...")
edges = if !isnothing(stim1on)
    if isa(stim1on, AbstractDict) && haskey(stim1on, "edges")
        val = stim1on["edges"]
        println("  ✓ Extracted edges from stim1on Dict")
        println("    → Type: $(typeof(val)), size=$(size(val))")
        if length(val) > 0
            println("    → Range: $(val[1]) to $(val[end])")
        end
        val
    elseif isa(stim1on, AbstractArray)
        println("  ✓ Using stim1on directly as edges")
        println("    → Type: $(typeof(stim1on)), size=$(size(stim1on))")
        stim1on
    else
        println("  ⚠️  stim1on has unexpected type, generating defaults")
        collect(0.0:1.0/fs:(size(neur_tensor_stim1on, 2)-1)/fs)
    end
elseif !isnothing(neur_tensor_stim1on)
    val = collect(0.0:1.0/fs:(size(neur_tensor_stim1on, 2)-1)/fs)
    println("  ✓ Generated time edges from neural tensor")
    println("    → $(length(val)) time bins")
    val
else
    error("Cannot determine time edges!")
end

println("\n" * "="^70)
println("VARIABLES LOADED SUCCESSFULLY")
println("="^70)
println("  Neural tensor: $(size(neur_tensor_stim1on))")
if !isnothing(cond_matrix)
    println("  Condition matrix: $(size(cond_matrix))")
end
if !isnothing(cond_label)
    println("  Condition labels: $(length(cond_label)) labels")
end
println("  Time edges: $(length(edges)) bins")
println()

# Display condition labels if available
if !isnothing(cond_label)
    println("Condition labels:")
    println(cond_label)
    println()
else
    println("⚠️  No condition labels available")
    println()
end
