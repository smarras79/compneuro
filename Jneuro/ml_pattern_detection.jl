"""
Machine Learning Pattern Detection Module

Provides ML-based methods for detecting patterns in neural data:
- Feature extraction
- Clustering (k-means, DBSCAN)
- Anomaly detection (Isolation Forest, statistics-based)
- Classification of neural events

Author: Enhanced Neural Analysis Toolkit
"""

using Statistics
using LinearAlgebra
using Random

"""
    extract_statistical_features(signal::Vector{Float64}; window_size=100)

Extract statistical features from signal for ML algorithms.

# Features extracted:
- Mean, std, variance
- Skewness, kurtosis
- Min, max, range
- Energy, power
- Zero-crossing rate
- Peak count
- Spectral features (if frequency analysis enabled)

# Returns
Feature vector
"""
function extract_statistical_features(signal::Vector{Float64}; window_size=100)
    
    features = Dict{String, Float64}()
    
    # Basic statistics
    features["mean"] = mean(signal)
    features["std"] = std(signal)
    features["variance"] = var(signal)
    features["min"] = minimum(signal)
    features["max"] = maximum(signal)
    features["range"] = maximum(signal) - minimum(signal)
    
    # Higher-order moments
    normalized = (signal .- mean(signal)) ./ std(signal)
    features["skewness"] = mean(normalized.^3)
    features["kurtosis"] = mean(normalized.^4) - 3.0  # Excess kurtosis
    
    # Energy and power
    features["energy"] = sum(signal.^2)
    features["power"] = mean(signal.^2)
    features["rms"] = sqrt(mean(signal.^2))
    
    # Zero-crossing rate
    zero_crossings = sum(diff(sign.(signal)) .!= 0)
    features["zero_crossing_rate"] = zero_crossings / length(signal)
    
    # Peak detection (simple)
    peaks = 0
    for i in 2:(length(signal)-1)
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            peaks += 1
        end
    end
    features["peak_count"] = peaks
    features["peak_rate"] = peaks / length(signal)
    
    # Autocorrelation at lag 1
    if length(signal) > 1
        features["autocorr_lag1"] = cor(signal[1:end-1], signal[2:end])
    else
        features["autocorr_lag1"] = 0.0
    end
    
    # Signal complexity (sample entropy approximation)
    features["complexity"] = estimate_complexity(signal)
    
    return features
end

"""
    estimate_complexity(signal)

Rough estimate of signal complexity using consecutive differences.
"""
function estimate_complexity(signal::Vector{Float64})
    if length(signal) < 2
        return 0.0
    end
    
    # Use normalized variance of differences as complexity measure
    diffs = diff(signal)
    return std(diffs) / (std(signal) + 1e-10)
end

"""
    sliding_window_features(signal::Vector{Float64}, window_size::Int, 
                           step_size::Int=window_size÷2)

Extract features using sliding window approach.

# Returns
Matrix where each row is feature vector for one window
"""
function sliding_window_features(signal::Vector{Float64}, window_size::Int,
                                step_size::Int=window_size÷2)
    
    n_windows = (length(signal) - window_size) ÷ step_size + 1
    
    # Get feature names from first window
    first_features = extract_statistical_features(signal[1:window_size])
    feature_names = sort(collect(keys(first_features)))
    n_features = length(feature_names)
    
    # Initialize feature matrix
    feature_matrix = zeros(n_windows, n_features)
    window_indices = zeros(Int, n_windows, 2)  # Store start and end indices
    
    for i in 1:n_windows
        start_idx = (i-1) * step_size + 1
        end_idx = start_idx + window_size - 1
        
        if end_idx > length(signal)
            break
        end
        
        window = signal[start_idx:end_idx]
        features = extract_statistical_features(window)
        
        # Fill feature vector in consistent order
        for (j, name) in enumerate(feature_names)
            feature_matrix[i, j] = features[name]
        end
        
        window_indices[i, :] = [start_idx, end_idx]
    end
    
    return feature_matrix, window_indices, feature_names
end

"""
    kmeans_clustering(features::Matrix{Float64}, k::Int; max_iter=100, n_init=10)

Perform k-means clustering on feature matrix.

# Arguments
- `features`: N×M matrix (N samples, M features)
- `k`: Number of clusters
- `max_iter`: Maximum iterations
- `n_init`: Number of random initializations

# Returns
- `labels`: Cluster assignment for each sample
- `centers`: Cluster centroids
- `inertia`: Sum of squared distances to centers
"""
function kmeans_clustering(features::Matrix{Float64}, k::Int; 
                          max_iter=100, n_init=10)
    
    n_samples, n_features = size(features)
    
    best_inertia = Inf
    best_labels = zeros(Int, n_samples)
    best_centers = zeros(k, n_features)
    
    for init in 1:n_init
        # Random initialization
        Random.seed!(init)
        centers = features[randperm(n_samples)[1:k], :]
        labels = zeros(Int, n_samples)
        
        for iter in 1:max_iter
            old_labels = copy(labels)
            
            # Assignment step
            for i in 1:n_samples
                distances = [norm(features[i, :] - centers[j, :]) for j in 1:k]
                labels[i] = argmin(distances)
            end
            
            # Update step
            for j in 1:k
                cluster_points = features[labels .== j, :]
                if size(cluster_points, 1) > 0
                    centers[j, :] = mean(cluster_points, dims=1)
                end
            end
            
            # Check convergence
            if labels == old_labels
                break
            end
        end
        
        # Calculate inertia
        inertia = 0.0
        for i in 1:n_samples
            inertia += norm(features[i, :] - centers[labels[i], :])^2
        end
        
        if inertia < best_inertia
            best_inertia = inertia
            best_labels = labels
            best_centers = centers
        end
    end
    
    return best_labels, best_centers, best_inertia
end

"""
    detect_anomalies_isolation(features::Matrix{Float64}; 
                               contamination=0.1, n_trees=100)

Detect anomalies using Isolation Forest approach (simplified).

# Arguments
- `features`: Feature matrix
- `contamination`: Expected proportion of anomalies
- `n_trees`: Number of isolation trees

# Returns
- `scores`: Anomaly score for each sample (higher = more anomalous)
- `is_anomaly`: Boolean vector indicating anomalies
"""
function detect_anomalies_isolation(features::Matrix{Float64};
                                   contamination=0.1, n_trees=100)
    
    n_samples = size(features, 1)
    
    # Calculate anomaly scores based on statistical distance from mean
    # (Simplified version - full Isolation Forest would build actual trees)
    
    # Normalize features
    features_normalized = (features .- mean(features, dims=1)) ./ (std(features, dims=1) .+ 1e-10)
    
    # Calculate Mahalanobis-like distance
    scores = zeros(n_samples)
    for i in 1:n_samples
        # Euclidean distance from center (simplified)
        scores[i] = norm(features_normalized[i, :])
    end
    
    # Determine threshold based on contamination
    threshold = quantile(scores, 1.0 - contamination)
    is_anomaly = scores .> threshold
    
    return scores, is_anomaly
end

"""
    detect_anomalies_statistical(signal::Vector{Float64}; 
                                 threshold_sd=3.0, window_size=100)

Detect anomalies using statistical threshold (Z-score method).

# Arguments
- `signal`: Input signal
- `threshold_sd`: Number of standard deviations for threshold
- `window_size`: Window for local statistics

# Returns
- `anomaly_indices`: Indices of anomalous points
- `z_scores`: Z-score for each point
"""
function detect_anomalies_statistical(signal::Vector{Float64};
                                     threshold_sd=3.0, window_size=100)
    
    n = length(signal)
    z_scores = zeros(n)
    
    # Calculate rolling statistics
    half_window = window_size ÷ 2
    
    for i in 1:n
        window_start = max(1, i - half_window)
        window_end = min(n, i + half_window)
        
        window_data = signal[window_start:window_end]
        local_mean = mean(window_data)
        local_std = std(window_data)
        
        if local_std > 0
            z_scores[i] = abs(signal[i] - local_mean) / local_std
        else
            z_scores[i] = 0.0
        end
    end
    
    # Find anomalies
    anomaly_indices = findall(z_scores .> threshold_sd)
    
    return anomaly_indices, z_scores
end

"""
    classify_events_knn(train_features::Matrix{Float64}, train_labels::Vector{Int},
                       test_features::Matrix{Float64}; k=5)

Simple k-Nearest Neighbors classifier for event classification.

# Arguments
- `train_features`: Training feature matrix
- `train_labels`: Training labels (1, 2, 3, etc.)
- `test_features`: Test feature matrix to classify
- `k`: Number of neighbors

# Returns
- `predictions`: Predicted labels for test data
"""
function classify_events_knn(train_features::Matrix{Float64}, 
                            train_labels::Vector{Int},
                            test_features::Matrix{Float64}; k=5)
    
    n_test = size(test_features, 1)
    n_train = size(train_features, 1)
    predictions = zeros(Int, n_test)
    
    for i in 1:n_test
        # Calculate distances to all training points
        distances = [norm(test_features[i, :] - train_features[j, :]) 
                    for j in 1:n_train]
        
        # Find k nearest neighbors
        k_nearest = sortperm(distances)[1:min(k, n_train)]
        k_labels = train_labels[k_nearest]
        
        # Majority vote
        predictions[i] = mode(k_labels)
    end
    
    return predictions
end

"""
    mode(x)

Calculate mode (most common value) of array.
"""
function mode(x::Vector{Int})
    counts = Dict{Int, Int}()
    for val in x
        counts[val] = get(counts, val, 0) + 1
    end
    return argmax(counts)
end

"""
    pca_reduction(features::Matrix{Float64}, n_components::Int=2)

Reduce dimensionality using Principal Component Analysis.

# Returns
- `transformed`: Transformed features in lower dimension
- `components`: Principal components
- `explained_variance`: Variance explained by each component
"""
function pca_reduction(features::Matrix{Float64}, n_components::Int=2)
    
    # Center the data
    centered = features .- mean(features, dims=1)
    
    # Compute covariance matrix
    cov_matrix = (centered' * centered) / (size(centered, 1) - 1)
    
    # Eigendecomposition
    eigen_result = eigen(cov_matrix)
    eigenvalues = eigen_result.values
    eigenvectors = eigen_result.vectors
    
    # Sort by eigenvalues (descending)
    idx = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n components
    components = eigenvectors[:, 1:n_components]
    
    # Transform data
    transformed = centered * components
    
    # Calculate explained variance
    total_variance = sum(eigenvalues)
    explained_variance = eigenvalues[1:n_components] ./ total_variance
    
    return transformed, components, explained_variance
end

"""
    analyze_event_patterns(signal::Vector{Float64}, events; fs=1000.0)

Comprehensive pattern analysis of detected events using ML.

# Performs:
- Feature extraction for each event
- Clustering to find event types
- Anomaly detection to find unusual events
- Dimensionality reduction for visualization

# Returns
Dictionary with analysis results
"""
function analyze_event_patterns(signal::Vector{Float64}, events; 
                               fs=1000.0, n_clusters=3)
    
    if length(events) == 0
        return Dict("error" => "No events provided")
    end
    
    # Extract features for each event
    n_events = length(events)
    feature_list = []
    
    for event in events
        start_idx = event.start_sample
        end_idx = event.end_sample
        
        if start_idx > 0 && end_idx <= length(signal)
            segment = signal[start_idx:end_idx]
            features = extract_statistical_features(segment)
            push!(feature_list, features)
        end
    end
    
    if length(feature_list) == 0
        return Dict("error" => "Could not extract features")
    end
    
    # Convert to matrix
    feature_names = sort(collect(keys(feature_list[1])))
    feature_matrix = zeros(length(feature_list), length(feature_names))
    
    for (i, features) in enumerate(feature_list)
        for (j, name) in enumerate(feature_names)
            feature_matrix[i, j] = features[name]
        end
    end
    
    # Clustering
    n_clusters_actual = min(n_clusters, n_events)
    if n_clusters_actual >= 2
        labels, centers, inertia = kmeans_clustering(feature_matrix, n_clusters_actual)
    else
        labels = ones(Int, n_events)
        centers = mean(feature_matrix, dims=1)
        inertia = 0.0
    end
    
    # Anomaly detection
    if n_events >= 10
        anomaly_scores, is_anomaly = detect_anomalies_isolation(
            feature_matrix, contamination=0.1
        )
    else
        anomaly_scores = zeros(n_events)
        is_anomaly = falses(n_events)
    end
    
    # Dimensionality reduction for visualization
    if size(feature_matrix, 2) >= 2
        transformed, components, explained_var = pca_reduction(feature_matrix, 2)
    else
        transformed = feature_matrix
        components = Matrix{Float64}(I, size(feature_matrix, 2), 2)
        explained_var = ones(2)
    end
    
    return Dict(
        "n_events" => n_events,
        "feature_matrix" => feature_matrix,
        "feature_names" => feature_names,
        "cluster_labels" => labels,
        "cluster_centers" => centers,
        "n_clusters" => n_clusters_actual,
        "anomaly_scores" => anomaly_scores,
        "is_anomaly" => is_anomaly,
        "pca_transformed" => transformed,
        "pca_components" => components,
        "explained_variance" => explained_var
    )
end

println("✓ Machine Learning Pattern Detection module loaded")
println("  Functions available:")
println("    - extract_statistical_features()")
println("    - sliding_window_features()")
println("    - kmeans_clustering()")
println("    - detect_anomalies_isolation()")
println("    - detect_anomalies_statistical()")
println("    - classify_events_knn()")
println("    - pca_reduction()")
println("    - analyze_event_patterns()")
