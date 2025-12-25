"""
Smart Variable Extraction Module
Automatically detect and extract variables from MAT files with various naming conventions

Author: Enhanced for flexible data loading
"""

module SmartExtractor

using Printf

export extract_variables, VariableMapping, create_variable_mapping
export detect_neural_tensor, detect_condition_matrix, detect_stimulus_vector
export verify_extraction, suggest_manual_mapping

"""
    VariableMapping

Structure to hold mappings between MAT file variables and expected variables
"""
struct VariableMapping
    neural_tensor::Union{String, Nothing}
    condition_matrix::Union{String, Nothing}
    condition_labels::Union{String, Nothing}
    stimulus_vector::Union{String, Nothing}
    available_vars::Vector{String}
    confidence::Dict{String, Float64}  # Confidence score for each mapping
end

"""
    detect_neural_tensor(data::Dict)
    
Detect which variable contains the neural activity tensor (3D array)
"""
function detect_neural_tensor(data::AbstractDict)::Tuple{Union{String, Nothing}, Float64}
    # Pattern matching for neural tensor names
    patterns = [
        r"neur.*tensor"i,
        r"neural.*data"i,
        r"spike.*tensor"i,
        r"firing.*rate"i,
        r"activity.*tensor"i,
        r".*neur_tensor.*"i
    ]
    
    candidates = Dict{String, Float64}()
    
    for (var_name, var_data) in data
        confidence = 0.0
        
        # Check if it matches naming patterns
        for pattern in patterns
            if occursin(pattern, var_name)
                confidence += 0.4
                break
            end
        end
        
        # Check if it's a 3D array (neurons × time × trials)
        if isa(var_data, AbstractArray) && ndims(var_data) == 3
            confidence += 0.3
            
            # Additional heuristics
            dims = size(var_data)
            
            # Typical neural data: 10-200 neurons, 1000-10000 time bins, 10-200 trials
            if 5 <= dims[1] <= 500  # neurons
                confidence += 0.1
            end
            if 100 <= dims[2] <= 20000  # time bins
                confidence += 0.1
            end
            if 5 <= dims[3] <= 500  # trials
                confidence += 0.1
            end
        end
        
        if confidence > 0.0
            candidates[var_name] = confidence
        end
    end
    
    if length(candidates) == 0
        return (nothing, 0.0)
    end
    
    # Return highest confidence match
    best_match = argmax(candidates)
    return (best_match, candidates[best_match])
end

"""
    detect_condition_matrix(data::Dict)
    
Detect which variable contains the behavioral/condition matrix (2D array)
"""
function detect_condition_matrix(data::AbstractDict)::Tuple{Union{String, Nothing}, Float64}
    patterns = [
        r"cond.*matrix"i,
        r"condition.*data"i,
        r"behav.*matrix"i,
        r"trial.*data"i,
        r".*cond_matrix.*"i
    ]
    
    candidates = Dict{String, Float64}()
    
    for (var_name, var_data) in data
        confidence = 0.0
        
        # Check naming patterns
        for pattern in patterns
            if occursin(pattern, var_name)
                confidence += 0.4
                break
            end
        end
        
        # Check if it's a 2D array with time × conditions
        if isa(var_data, AbstractArray) && ndims(var_data) == 2
            confidence += 0.2
            
            dims = size(var_data)
            
            # Heuristics: time dimension should be larger, conditions 5-50
            if dims[1] > dims[2] && 3 <= dims[2] <= 100
                confidence += 0.2
            end
            
            # Should have many rows (time bins)
            if 100 <= dims[1] <= 20000
                confidence += 0.2
            end
        end
        
        if confidence > 0.0
            candidates[var_name] = confidence
        end
    end
    
    if length(candidates) == 0
        return (nothing, 0.0)
    end
    
    best_match = argmax(candidates)
    return (best_match, candidates[best_match])
end

"""
    detect_condition_labels(data::Dict)
    
Detect which variable contains condition/column labels (string array)
"""
function detect_condition_labels(data::AbstractDict)::Tuple{Union{String, Nothing}, Float64}
    patterns = [
        r"cond.*label"i,
        r"column.*name"i,
        r"field.*name"i,
        r"label"i,
        r"header"i
    ]
    
    candidates = Dict{String, Float64}()
    
    for (var_name, var_data) in data
        confidence = 0.0
        
        # Check naming patterns
        for pattern in patterns
            if occursin(pattern, var_name)
                confidence += 0.5
                break
            end
        end
        
        # Check if it's a string array
        if isa(var_data, AbstractArray)
            if eltype(var_data) <: AbstractString
                confidence += 0.4
            end
            
            # Should be relatively small (5-50 labels)
            if 3 <= length(var_data) <= 100
                confidence += 0.1
            end
        end
        
        if confidence > 0.0
            candidates[var_name] = confidence
        end
    end
    
    if length(candidates) == 0
        return (nothing, 0.0)
    end
    
    best_match = argmax(candidates)
    return (best_match, candidates[best_match])
end

"""
    detect_stimulus_vector(data::Dict)
    
Detect which variable contains stimulus/trial indicators (1D array)
"""
function detect_stimulus_vector(data::AbstractDict)::Tuple{Union{String, Nothing}, Float64}
    patterns = [
        r"stim"i,
        r"trial.*id"i,
        r"stimulus"i,
        r"trigger"i
    ]
    
    candidates = Dict{String, Float64}()
    
    for (var_name, var_data) in data
        confidence = 0.0
        
        # Check naming patterns
        for pattern in patterns
            if occursin(pattern, var_name)
                confidence += 0.4
                break
            end
        end
        
        # Check if it's a 1D numeric array
        if isa(var_data, AbstractVector) && eltype(var_data) <: Number
            confidence += 0.3
            
            # Should be relatively small (10-200 trials)
            if 5 <= length(var_data) <= 500
                confidence += 0.3
            end
        end
        
        if confidence > 0.0
            candidates[var_name] = confidence
        end
    end
    
    if length(candidates) == 0
        return (nothing, 0.0)
    end
    
    best_match = argmax(candidates)
    return (best_match, candidates[best_match])
end

"""
    create_variable_mapping(data::Dict; verbose::Bool=true)
    
Automatically detect and create mapping for all expected variables
"""
function create_variable_mapping(data::AbstractDict; verbose::Bool=true)::VariableMapping
    available_vars = collect(keys(data))
    
    if verbose
        println("\n" * "="^70)
        println("AUTOMATIC VARIABLE DETECTION")
        println("="^70)
    end
    
    # Detect each type of variable
    neural_tensor, neural_conf = detect_neural_tensor(data)
    cond_matrix, matrix_conf = detect_condition_matrix(data)
    cond_labels, labels_conf = detect_condition_labels(data)
    stim_vector, stim_conf = detect_stimulus_vector(data)
    
    if verbose
        println("\n🔍 Detection Results:")
        println()
        
        if !isnothing(neural_tensor)
            println("  ✓ Neural Tensor:       \"$neural_tensor\" (confidence: $(round(neural_conf*100, digits=1))%)")
        else
            println("  ✗ Neural Tensor:       Not detected")
        end
        
        if !isnothing(cond_matrix)
            println("  ✓ Condition Matrix:    \"$cond_matrix\" (confidence: $(round(matrix_conf*100, digits=1))%)")
        else
            println("  ✗ Condition Matrix:    Not detected")
        end
        
        if !isnothing(cond_labels)
            println("  ✓ Condition Labels:    \"$cond_labels\" (confidence: $(round(labels_conf*100, digits=1))%)")
        else
            println("  ✗ Condition Labels:    Not detected")
        end
        
        if !isnothing(stim_vector)
            println("  ✓ Stimulus Vector:     \"$stim_vector\" (confidence: $(round(stim_conf*100, digits=1))%)")
        else
            println("  ✗ Stimulus Vector:     Not detected")
        end
        
        println()
    end
    
    confidence_map = Dict{String, Float64}(
        "neural_tensor" => neural_conf,
        "condition_matrix" => matrix_conf,
        "condition_labels" => labels_conf,
        "stimulus_vector" => stim_conf
    )
    
    return VariableMapping(
        neural_tensor,
        cond_matrix,
        cond_labels,
        stim_vector,
        available_vars,
        confidence_map
    )
end

"""
    extract_variables(data::Dict, mapping::VariableMapping; verbose::Bool=true)
    
Extract variables from data using the provided mapping
"""
function extract_variables(data::AbstractDict, mapping::VariableMapping; verbose::Bool=true)
    extracted = Dict{String, Any}()
    
    if verbose
        println("="^70)
        println("EXTRACTING VARIABLES")
        println("="^70)
        println()
    end
    
    # Extract neural tensor
    if !isnothing(mapping.neural_tensor)
        extracted["neural_tensor"] = data[mapping.neural_tensor]
        if verbose
            dims = size(extracted["neural_tensor"])
            println("  ✓ Extracted neural_tensor from \"$(mapping.neural_tensor)\"")
            println("    Dimensions: $(dims[1]) neurons × $(dims[2]) time bins × $(dims[3]) trials")
        end
    else
        if verbose
            println("  ⚠️  Neural tensor not found - analysis may fail")
        end
    end
    
    # Extract condition matrix
    if !isnothing(mapping.condition_matrix)
        extracted["condition_matrix"] = data[mapping.condition_matrix]
        if verbose
            dims = size(extracted["condition_matrix"])
            println("  ✓ Extracted condition_matrix from \"$(mapping.condition_matrix)\"")
            println("    Dimensions: $(dims[1]) time bins × $(dims[2]) conditions")
        end
    else
        if verbose
            println("  ⚠️  Condition matrix not found - behavioral analysis may not work")
        end
    end
    
    # Extract condition labels
    if !isnothing(mapping.condition_labels)
        extracted["condition_labels"] = data[mapping.condition_labels]
        if verbose
            n_labels = length(extracted["condition_labels"])
            println("  ✓ Extracted condition_labels from \"$(mapping.condition_labels)\"")
            println("    Labels: $(n_labels) columns")
            # Show first few labels
            labels = vec(extracted["condition_labels"])
            if n_labels <= 5
                println("    Names: $(join(labels, ", "))")
            else
                println("    Names: $(join(labels[1:3], ", ")), ... ($(n_labels-3) more)")
            end
        end
    else
        if verbose
            println("  ⚠️  Condition labels not found - using generic names")
        end
    end
    
    # Extract stimulus vector
    if !isnothing(mapping.stimulus_vector)
        extracted["stimulus_vector"] = data[mapping.stimulus_vector]
        if verbose
            n_trials = length(extracted["stimulus_vector"])
            println("  ✓ Extracted stimulus_vector from \"$(mapping.stimulus_vector)\"")
            println("    Trials: $n_trials")
        end
    else
        if verbose
            println("  ⚠️  Stimulus vector not found - using all trials")
        end
    end
    
    if verbose
        println()
        println("="^70)
    end
    
    return extracted
end

"""
    suggest_manual_mapping(data::AbstractDict, mapping::VariableMapping)
    
Suggest manual variable mapping when automatic detection has low confidence
"""
function suggest_manual_mapping(data::AbstractDict, mapping::VariableMapping)
    println("\n" * "="^70)
    println("MANUAL MAPPING SUGGESTION")
    println("="^70)
    println()
    println("Some variables were not detected with high confidence.")
    println("Available variables in file:")
    println()
    
    for (i, var_name) in enumerate(sort(mapping.available_vars))
        var_data = data[var_name]
        type_str = string(typeof(var_data))
        
        # Get dimensions
        if isa(var_data, AbstractArray)
            dims = size(var_data)
            dim_str = join(["$(d)" for d in dims], "×")
        else
            dim_str = "scalar"
        end
        
        println("  [$i] \"$var_name\"")
        println("      Type: $type_str, Dimensions: $dim_str")
    end
    
    println()
    println("To manually specify mapping, create a VariableMapping:")
    println()
    println("mapping = VariableMapping(")
    println("    \"your_neural_tensor_name\",    # neural_tensor")
    println("    \"your_condition_matrix_name\",  # condition_matrix")
    println("    \"your_labels_name\",            # condition_labels")
    println("    \"your_stimulus_name\",          # stimulus_vector")
    println("    collect(keys(data)),")
    println("    Dict{String, Float64}()")
    println(")")
    println()
end

"""
    verify_extraction(extracted::AbstractDict)
    
Verify that essential variables were extracted successfully
"""
function verify_extraction(extracted::AbstractDict)::Bool
    println("\n" * "="^70)
    println("EXTRACTION VERIFICATION")
    println("="^70)
    println()
    
    all_ok = true
    
    # Check neural tensor
    if haskey(extracted, "neural_tensor")
        println("  ✓ Neural tensor: Present")
        if ndims(extracted["neural_tensor"]) != 3
            println("    ⚠️  Warning: Expected 3D array, got $(ndims(extracted["neural_tensor"]))D")
            all_ok = false
        end
    else
        println("  ✗ Neural tensor: Missing (REQUIRED)")
        all_ok = false
    end
    
    # Check condition matrix
    if haskey(extracted, "condition_matrix")
        println("  ✓ Condition matrix: Present")
    else
        println("  ⚠️  Condition matrix: Missing (recommended)")
    end
    
    # Check labels
    if haskey(extracted, "condition_labels")
        println("  ✓ Condition labels: Present")
    else
        println("  ⚠️  Condition labels: Missing (will use generic names)")
    end
    
    # Check stimulus
    if haskey(extracted, "stimulus_vector")
        println("  ✓ Stimulus vector: Present")
    else
        println("  ⚠️  Stimulus vector: Missing (will use all trials)")
    end
    
    println()
    if all_ok
        println("✓ All essential variables extracted successfully!")
    else
        println("⚠️  Some essential variables are missing or incorrect")
    end
    println()
    
    return all_ok
end

end # module
