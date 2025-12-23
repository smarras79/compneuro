"""
Behavioral Event Extraction Module

Helper functions to extract behavioral events (motion onset/offset, task timing, etc.)
from behavioral data matrices.

These are NOT detected from neural signals - they're extracted from behavioral recordings
that are already in your data files.

Author: Enhanced Neural Analysis Toolkit
"""

using Statistics

"""
    extract_motion_events(cond_matrix, edges; 
                         position_column=2,
                         threshold_quantile=0.75,
                         motion_duration=0.5,
                         fs=1000.0,
                         method=:position_change)

Extract motion onset/offset times from behavioral data.

# Arguments
- `cond_matrix`: Behavioral condition matrix from .mat file
- `edges`: Time edges/bins
- `position_column`: Which column contains position/joystick data (default: 2)
- `threshold_quantile`: Quantile for motion detection (default: 0.75 = top 25%)
- `motion_duration`: Assumed duration of motion in seconds (default: 0.5)
- `fs`: Sampling frequency in Hz (default: 1000.0)
- `method`: `:position_change` or `:velocity` (default: :position_change)

# Returns
Dict with:
- "motion_onset" => Vector of onset times (seconds)
- "motion_offset" => Vector of offset times (seconds)
- "n_events" => Number of motion events detected

# Example
```julia
behavioral_events = extract_motion_events(cond_matrix, edges)
println("Found \$(behavioral_events["n_events"]) motion events")
```
"""
function extract_motion_events(cond_matrix, edges;
                              position_column=2,
                              threshold_quantile=0.75,
                              motion_duration=0.5,
                              fs=1000.0,
                              method=:position_change)
    
    println("Extracting motion events from behavioral data...")
    println("  Position column: $position_column")
    println("  Method: $method")
    
    # Get positions from specified column
    if size(cond_matrix, 2) < position_column
        error("Column $position_column not found in cond_matrix (has $(size(cond_matrix, 2)) columns)")
    end
    
    positions = cond_matrix[:, position_column]
    
    if method == :position_change
        # Method 1: Detect from position changes
        position_changes = abs.(diff(positions))
        
        # Check if there are any changes
        if all(position_changes .== 0)
            @warn "No position changes detected in column $position_column. Try different column or method."
            return Dict(
                "motion_onset" => Float64[],
                "motion_offset" => Float64[],
                "n_events" => 0
            )
        end
        
        # Threshold for significant motion
        nonzero_changes = position_changes[position_changes .> 0]
        motion_threshold = quantile(nonzero_changes, threshold_quantile)
        
        println("  Motion threshold: $(round(motion_threshold, digits=4))")
        
        # Find trials with significant motion
        motion_trial_indices = findall(position_changes .> motion_threshold)
        
    elseif method == :velocity
        # Method 2: Detect from velocity threshold
        dt = mean(diff(edges))  # Average time step
        velocity = diff(positions) ./ dt
        
        # Threshold based on velocity magnitude
        velocity_threshold = quantile(abs.(velocity), threshold_quantile)
        motion_trial_indices = findall(abs.(velocity) .> velocity_threshold)
        
    else
        error("Unknown method: $method. Use :position_change or :velocity")
    end
    
    if length(motion_trial_indices) == 0
        @warn "No motion events detected. Try lowering threshold_quantile."
        return Dict(
            "motion_onset" => Float64[],
            "motion_offset" => Float64[],
            "n_events" => 0
        )
    end
    
    # Convert trial indices to time (seconds)
    n_trials = size(cond_matrix, 1)
    
    # Handle edges - could be Vector or Matrix
    if isa(edges, Matrix)
        # edges is a matrix - likely (time_points × trials) or (trials × time_points)
        println("  Debug info:")
        println("    edges is a Matrix with size: $(size(edges))")
        
        # Special case: (1 × N) or (N × 1) - treat as a single time vector for all trials
        if size(edges, 1) == 1
            # Shape is (1 × time_points) - single time vector
            println("    Format: Single time vector (1 × time_points)")
            n_time_points = size(edges, 2)
            time_vector = vec(edges)  # Convert to 1D vector
            
            # Calculate time per trial
            total_duration = time_vector[end] - time_vector[1]
            time_per_trial = total_duration / n_trials
            
            println("    n_trials: $n_trials")
            println("    n_time_points: $n_time_points")
            println("    Total duration: $(round(total_duration, digits=3)) s")
            println("    Time per trial: $(round(time_per_trial, digits=3)) s")
            
            # Calculate onset times from trial indices
            motion_onset_times = (motion_trial_indices .- 1) .* time_per_trial .+ time_vector[1]
            
        elseif size(edges, 2) == 1
            # Shape is (time_points × 1) - single time vector
            println("    Format: Single time vector (time_points × 1)")
            n_time_points = size(edges, 1)
            time_vector = vec(edges)  # Convert to 1D vector
            
            # Calculate time per trial
            total_duration = time_vector[end] - time_vector[1]
            time_per_trial = total_duration / n_trials
            
            println("    n_trials: $n_trials")
            println("    n_time_points: $n_time_points")
            println("    Total duration: $(round(total_duration, digits=3)) s")
            println("    Time per trial: $(round(time_per_trial, digits=3)) s")
            
            # Calculate onset times from trial indices
            motion_onset_times = (motion_trial_indices .- 1) .* time_per_trial .+ time_vector[1]
            
        elseif size(edges, 1) == n_trials
            # rows = trials, columns = time points
            println("    Format: (trials × time_points)")
            n_time_points = size(edges, 2)
            
            # For each motion trial, get the time at middle of that trial's time series
            motion_onset_times = Float64[]
            mid_point = max(1, size(edges, 2) ÷ 2)  # Ensure >= 1
            
            for trial_idx in motion_trial_indices
                if trial_idx >= 1 && trial_idx <= size(edges, 1)
                    time_value = edges[trial_idx, mid_point]
                    push!(motion_onset_times, time_value)
                end
            end
            
            println("    n_trials: $n_trials")
            println("    n_time_points: $n_time_points")
            println("    Time range: $(minimum(edges)) to $(maximum(edges)) s")
            
        else
            # columns = trials, rows = time points
            println("    Format: (time_points × trials)")
            n_time_points = size(edges, 1)
            
            # For each motion trial, get the time at middle of that trial's time series
            motion_onset_times = Float64[]
            mid_point = max(1, size(edges, 1) ÷ 2)  # Ensure >= 1
            
            for trial_idx in motion_trial_indices
                if trial_idx >= 1 && trial_idx <= size(edges, 2)
                    time_value = edges[mid_point, trial_idx]
                    push!(motion_onset_times, time_value)
                end
            end
            
            println("    n_trials: $n_trials")
            println("    n_time_points: $n_time_points")
            println("    Time range: $(minimum(edges)) to $(maximum(edges)) s")
        end
        
    else
        # edges is a Vector (original logic)
        n_time_points = length(edges)
        
        println("  Debug info:")
        println("    n_trials: $n_trials")
        println("    n_time_points: $n_time_points")
        println("    edges[1]: $(edges[1])")
        println("    edges[end]: $(edges[end])")
        println("    edges type: $(typeof(edges))")
        
        # Check if edges are times (float) or indices (int-like)
        if edges[1] >= -10.0 && edges[1] < 100.0 && edges[end] < 1000.0
            # edges appear to be time values in seconds
            println("    ✓ Detected: edges are TIME VALUES in seconds")
            
            # Calculate time per trial
            total_duration = edges[end] - edges[1]
            time_per_trial = total_duration / n_trials
            
            println("    Total duration: $(round(total_duration, digits=3)) s")
            println("    Time per trial: $(round(time_per_trial, digits=3)) s")
            
            # Calculate onset times directly from trial indices
            motion_onset_times = (motion_trial_indices .- 1) .* time_per_trial .+ edges[1]
            
        else
            # edges appear to be sample indices
            println("    ⚠️  Detected: edges might be SAMPLE INDICES")
            println("    Using fs=$fs to convert")
            
            samples_per_trial = n_time_points ÷ n_trials
            motion_onset_samples = motion_trial_indices .* samples_per_trial
            motion_onset_samples = motion_onset_samples[motion_onset_samples .<= n_time_points]
            motion_onset_times = motion_onset_samples ./ fs
        end
    end
    
    if length(motion_onset_times) > 0
        println("    First motion trial index: $(motion_trial_indices[1])")
        println("    First motion onset time: $(round(motion_onset_times[1], digits=3)) s")
    end
    
    # Filter to valid range
    if isa(edges, Matrix)
        min_time = minimum(edges)
        max_time = maximum(edges)
    else
        min_time = minimum(edges)
        max_time = maximum(edges)
    end
    
    valid_mask = (motion_onset_times .>= min_time) .& (motion_onset_times .<= max_time)
    motion_onset_times = motion_onset_times[valid_mask]
    
    if sum(.!valid_mask) > 0
        println("  ⚠️  Filtered out $(sum(.!valid_mask)) events outside time range")
    end
    
    # Calculate offset times (onset + duration)
    motion_offset_times = motion_onset_times .+ motion_duration
    
    # Filter offsets to valid range
    valid_offset_mask = motion_offset_times .<= max_time
    motion_onset_times = motion_onset_times[valid_offset_mask]
    motion_offset_times = motion_offset_times[valid_offset_mask]
    
    if sum(.!valid_offset_mask) > 0
        println("  ⚠️  Filtered out $(sum(.!valid_offset_mask)) events with offsets outside time range")
    end
    
    println("  ✓ Extracted $(length(motion_onset_times)) motion events")
    println("    Time range: $(round(min_time, digits=2)) - $(round(max_time, digits=2)) s")
    if length(motion_onset_times) > 0
        println("    First event at: $(round(motion_onset_times[1], digits=2)) s")
        println("    Last event at: $(round(motion_onset_times[end], digits=2)) s")
    end
    
    return Dict(
        "motion_onset" => motion_onset_times,
        "motion_offset" => motion_offset_times,
        "n_events" => length(motion_onset_times)
    )
end

"""
    extract_trial_events(cond_matrix, edges; trial_duration=1.0, fs=1000.0)

Extract trial onset times (simpler alternative if no motion data available).

# Arguments
- `cond_matrix`: Behavioral condition matrix
- `edges`: Time edges
- `trial_duration`: Duration of each trial in seconds
- `fs`: Sampling frequency in Hz

# Returns
Dict with "trial_onset" times
"""
function extract_trial_events(cond_matrix, edges; trial_duration=1.0, fs=1000.0)
    
    n_trials = size(cond_matrix, 1)
    signal_length = length(edges)
    samples_per_trial = signal_length ÷ n_trials
    
    # Calculate onset time for each trial
    trial_onset_samples = (0:(n_trials-1)) .* samples_per_trial
    trial_onset_times = trial_onset_samples ./ fs
    
    println("Extracted $(n_trials) trial onset times")
    
    return Dict(
        "trial_onset" => trial_onset_times,
        "n_events" => n_trials
    )
end

"""
    inspect_behavioral_data(cond_matrix, cond_label=nothing)

Inspect behavioral data to help choose the right column for motion extraction.

Prints summary statistics for each column to help identify which contains
position, velocity, or other behavioral measures.
"""
function inspect_behavioral_data(cond_matrix, cond_label=nothing)
    
    println("\n" * "="^70)
    println("BEHAVIORAL DATA INSPECTION")
    println("="^70)
    
    n_trials, n_cols = size(cond_matrix)
    println("\nData shape: $n_trials trials × $n_cols columns")
    
    # Show labels if available
    if !isnothing(cond_label)
        println("\nColumn labels:")
        for (i, label) in enumerate(cond_label)
            println("  Column $i: $label")
        end
    end
    
    # Show statistics for each column
    println("\nColumn statistics:")
    println("-"^70)
    println("Col │ Min      │ Max      │ Mean     │ Std      │ Unique │ Zeros")
    println("-"^70)
    
    for col in 1:n_cols
        data = cond_matrix[:, col]
        
        col_min = minimum(data)
        col_max = maximum(data)
        col_mean = mean(data)
        col_std = std(data)
        n_unique = length(unique(data))
        n_zeros = sum(data .== 0)
        
        println(@sprintf("%3d │ %8.2f │ %8.2f │ %8.2f │ %8.2f │ %6d │ %5d",
                        col, col_min, col_max, col_mean, col_std, n_unique, n_zeros))
    end
    
    println("-"^70)
    
    # Provide hints
    println("\nHints for selecting position_column:")
    println("  - Large range (max - min): likely position/angle")
    println("  - Many unique values: continuous measure")
    println("  - Low std with few unique: likely categorical")
    println("  - Check for columns with changing values across trials")
    
    # Show example of position changes for likely candidates
    println("\nPosition change analysis (top 3 variable columns):")
    
    # Find columns with most variance
    variances = [var(cond_matrix[:, col]) for col in 1:n_cols]
    top_cols = sortperm(variances, rev=true)[1:min(3, n_cols)]
    
    for col in top_cols
        data = cond_matrix[:, col]
        changes = abs.(diff(data))
        n_changes = sum(changes .> 0)
        mean_change = mean(changes[changes .> 0])
        
        label_str = isnothing(cond_label) ? "" : " ($(cond_label[col]))"
        
        println("  Column $col$label_str:")
        println("    Variance: $(round(variances[col], digits=4))")
        println("    Trials with changes: $n_changes / $(n_trials-1)")
        if n_changes > 0
            println("    Mean change size: $(round(mean_change, digits=4))")
        end
    end
    
    println("\n" * "="^70)
end

"""
    auto_detect_motion_column(cond_matrix)

Automatically suggest which column likely contains motion/position data.

Returns the column index most likely to contain position information.
"""
function auto_detect_motion_column(cond_matrix)
    
    n_cols = size(cond_matrix, 2)
    
    # Score each column
    scores = zeros(n_cols)
    
    for col in 1:n_cols
        data = cond_matrix[:, col]
        
        # Criteria for motion column:
        # 1. High variance (things are changing)
        # 2. Many unique values (continuous)
        # 3. Many transitions (changes between trials)
        
        variance_score = var(data)
        unique_score = length(unique(data)) / length(data)
        
        changes = abs.(diff(data))
        transition_score = sum(changes .> 0) / length(changes)
        
        # Combined score (weighted)
        scores[col] = variance_score * unique_score * transition_score
    end
    
    best_col = argmax(scores)
    
    println("Auto-detected column $best_col as most likely position/motion column")
    println("  Score: $(round(scores[best_col], digits=6))")
    
    return best_col
end

println("✓ Behavioral Event Extraction module loaded")
println("  Functions available:")
println("    - extract_motion_events() - Extract motion onset/offset")
println("    - extract_trial_events() - Extract trial timing")
println("    - inspect_behavioral_data() - Explore your data")
println("    - auto_detect_motion_column() - Auto-find position column")
