"""
Data Inspection Utility
Display MAT file contents and structure before analysis

Author: Enhanced for data exploration
"""

module DataInspector

using Revise
using Printf
using Statistics

export inspect_mat_file, display_variable_info, pause_for_confirmation, validate_expected_variables

"""
    inspect_mat_file(data::Dict)
    
Display comprehensive information about all variables in loaded MAT file
"""
function inspect_mat_file(data::Dict)
    println("\n" * "="^70)
    println("MAT FILE CONTENTS INSPECTION")
    println("="^70)
    
    # Get all variable names
    var_names = sort(collect(keys(data)))
    
    println("\n📊 Found $(length(var_names)) variable(s) in MAT file:")
    println()
    
    # Display summary table header
    println("┌" * "─"^30 * "┬" * "─"^20 * "┬" * "─"^15 * "┐")
    println("│" * " Variable Name" * " "^17 * "│" * " Type" * " "^15 * "│" * " Dimensions" * " "^5 * "│")
    println("├" * "─"^30 * "┼" * "─"^20 * "┼" * "─"^15 * "┤")
    
    # Display each variable
    for var_name in var_names
        var_data = data[var_name]
        type_str = string(typeof(var_data))
        
        # Shorten long type names
        if length(type_str) > 18
            type_str = type_str[1:15] * "..."
        end
        
        # Get dimensions
        dim_str = get_dimension_string(var_data)
        
        # Format row
        name_str = rpad(var_name, 28)
        type_str = rpad(type_str, 18)
        dim_str = rpad(dim_str, 13)
        
        println("│ $name_str │ $type_str │ $dim_str │")
    end
    
    println("└" * "─"^30 * "┴" * "─"^20 * "┴" * "─"^15 * "┘")
    
    # Display detailed information for each variable
    println("\n" * "="^70)
    println("DETAILED VARIABLE INFORMATION")
    println("="^70)
    
    for (i, var_name) in enumerate(var_names)
        println("\n[$i] Variable: \"$var_name\"")
        display_variable_info(var_name, data[var_name])
    end
    
    println("\n" * "="^70)
end

"""
    get_dimension_string(var_data)
    
Get a string representation of variable dimensions
"""
function get_dimension_string(var_data)
    if isa(var_data, AbstractArray)
        dims = size(var_data)
        if length(dims) == 1
            return "$(dims[1])"
        elseif length(dims) == 2
            return "$(dims[1])×$(dims[2])"
        elseif length(dims) == 3
            return "$(dims[1])×$(dims[2])×$(dims[3])"
        else
            return join(["$(d)" for d in dims], "×")
        end
    elseif isa(var_data, Number)
        return "scalar"
    elseif isa(var_data, String)
        return "string"
    else
        return "N/A"
    end
end

"""
    display_variable_info(name::String, var_data)
    
Display detailed information about a specific variable
"""
function display_variable_info(name::String, var_data)
    println("  ├─ Type: $(typeof(var_data))")
    
    if isa(var_data, AbstractArray)
        println("  ├─ Dimensions: $(size(var_data))")
        println("  ├─ Element type: $(eltype(var_data))")
        println("  ├─ Total elements: $(length(var_data))")
        
        # Show data statistics for numeric arrays
        if eltype(var_data) <: Number
            # Handle potential complex or special values
            real_data = real.(var_data)
            finite_data = real_data[isfinite.(real_data)]
            
            if length(finite_data) > 0
                println("  ├─ Range: [$(minimum(finite_data)), $(maximum(finite_data))]")
                println("  ├─ Mean: $(round(mean(finite_data), digits=4))")
                println("  ├─ Std: $(round(std(finite_data), digits=4))")
                
                # Check for special values
                n_nan = sum(isnan.(real_data))
                n_inf = sum(isinf.(real_data))
                if n_nan > 0 || n_inf > 0
                    println("  ├─ Special values: NaN=$n_nan, Inf=$n_inf")
                end
            else
                println("  ├─ (All values are NaN or Inf)")
            end
        end
        
        # Show sample values for small arrays
        if length(var_data) <= 10
            println("  └─ Values: $(var_data)")
        elseif length(var_data) <= 100
            println("  └─ First few values: $(var_data[1:min(5, length(var_data))])")
        else
            # For large arrays, show corner elements
            if ndims(var_data) == 1
                println("  └─ Sample: [$(var_data[1]), $(var_data[2]), ..., $(var_data[end])]")
            elseif ndims(var_data) == 2
                println("  └─ Sample corner: $(var_data[1:min(2, size(var_data, 1)), 1:min(2, size(var_data, 2))])")
            elseif ndims(var_data) == 3
                println("  └─ Sample corner: $(var_data[1:min(2, size(var_data, 1)), 1:min(2, size(var_data, 2)), 1])")
            end
        end
        
    elseif isa(var_data, Number)
        println("  └─ Value: $var_data")
        
    elseif isa(var_data, String)
        if length(var_data) <= 100
            println("  └─ Value: \"$var_data\"")
        else
            println("  └─ Value: \"$(var_data[1:97])...\" (truncated)")
        end
        
    else
        println("  └─ (Unable to display details for this type)")
    end
end

"""
    pause_for_confirmation(; prompt="Continue with analysis?")
    
Pause execution and wait for user confirmation
"""
function pause_for_confirmation(; prompt="Continue with analysis?")
    println("\n" * "─"^70)
    println("⏸️  PAUSED: Review the data structure above")
    println("─"^70)
    println()
    print("$prompt (press Enter to continue, or Ctrl+C to abort): ")
    
    readline()  # Wait for user input
    
    println()
    println("✓ Continuing with analysis...")
    println()
end

"""
    quick_inspect(data::Dict, var_name::String)
    
Quick inspection of a single variable
"""
function quick_inspect(data::Dict, var_name::String)
    if haskey(data, var_name)
        println("Variable: \"$var_name\"")
        display_variable_info(var_name, data[var_name])
    else
        println("⚠️  Variable \"$var_name\" not found in data")
        println("Available variables: $(join(keys(data), ", "))")
    end
end

"""
    validate_expected_variables(data::Dict, expected_vars::Vector{String})
    
Check if expected variables are present in the data
"""
function validate_expected_variables(data::Dict, expected_vars::Vector{String})
    println("\n🔍 Validating expected variables...")
    
    all_present = true
    for var in expected_vars
        if haskey(data, var)
            println("  ✓ Found: $var")
        else
            println("  ✗ Missing: $var")
            all_present = false
        end
    end
    
    if all_present
        println("\n✓ All expected variables are present!")
    else
        println("\n⚠️  Warning: Some expected variables are missing")
        println("Available variables: $(join(keys(data), ", "))")
    end
    
    println()
    return all_present
end

end # module
