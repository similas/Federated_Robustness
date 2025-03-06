# defenses/improved_aaf.py
import torch
from collections import OrderedDict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def improved_aaf_aggregate(local_models, sample_counts, device, n_estimators=100, contamination=0.35, random_state=42):
    """
    Improved Adaptive Anomaly Filtering (AAF) that:
    1. Works with both ResNet and ViT architectures
    2. Doesn't rely on PCA for dimensionality reduction
    3. Uses model statistics instead of raw parameters
    4. Employs clustering to identify majority behavior
    5. Doesn't need to know the attack type
    """
    # Initialize models
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    scaler = StandardScaler()
    
    # Extract model features using statistics rather than raw values
    update_features = []
    layer_features = {}  # Store layer-specific features for targeted detection
    
    for model_idx, model in enumerate(local_models):
        # Extract model-level statistics 
        model_stats = []
        
        # Track the L2 norm of the entire model
        total_squared_sum = 0
        
        # Process each parameter
        for name, param in model.items():
            if not param.dtype.is_floating_point:
                continue
                
            p = param.float().cpu()
            total_squared_sum += torch.sum(p ** 2).item()
            
            # Calculate statistics for this parameter
            norm_val = torch.norm(p).item()
            mean_val = torch.mean(p).item()
            std_val = torch.std(p).item() if p.numel() > 1 else 0
            
            # Store layer-specific information
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in layer_features:
                layer_features[layer_name] = []
            
            layer_features[layer_name].append((model_idx, norm_val, mean_val, std_val))
            
            # Add core statistics to feature vector
            model_stats.extend([norm_val, mean_val, std_val])
            
            # For larger layers, add percentile information
            if p.numel() > 1000:
                q25 = torch.quantile(p.view(-1), 0.25).item()
                q75 = torch.quantile(p.view(-1), 0.75).item()
                model_stats.extend([q25, q75])
        
        # Add global model norm
        model_norm = np.sqrt(total_squared_sum)
        model_stats.append(model_norm)
        
        # Add this model's statistics to our features
        update_features.append(model_stats)
    
    # Handle potential NaN/infinite values
    update_features = np.array(update_features)
    update_features = np.nan_to_num(update_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Scale features
    try:
        scaled_features = scaler.fit_transform(update_features)
    except Exception as e:
        print(f"Warning: Scaling failed with error {e}. Using unscaled features.")
        # If scaling fails, normalize by max absolute value
        abs_max = np.max(np.abs(update_features)) + 1e-10
        scaled_features = update_features / abs_max
    
    # Detect anomalies using clustering first
    # This helps identify the "normal" cluster, even if it's a minority
    kmeans = KMeans(n_clusters=2, random_state=random_state)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Get cluster sizes
    cluster_counts = np.bincount(cluster_labels)
    
    # Calculate median norms for each cluster
    cluster_norms = [[], []]
    for idx, label in enumerate(cluster_labels):
        cluster_norms[label].append(update_features[idx][-1])  # Last feature is model norm
    
    cluster_median_norms = [np.median(norms) if norms else 0 for norms in cluster_norms]
    
    # The "normal" cluster typically has more consistent norms
    norm_consistency = [np.std(norms) / (np.mean(norms) + 1e-10) if len(norms) > 1 else float('inf') 
                        for norms in cluster_norms]
    
    # Identify which cluster is more likely to be "normal" based on consistency and size
    normal_cluster_candidates = sorted(range(len(cluster_counts)), 
                                      key=lambda i: (norm_consistency[i], -cluster_counts[i]))
    normal_cluster = normal_cluster_candidates[0]
    
    # Now use Isolation Forest for finer-grained detection within each cluster
    anomaly_scores = np.ones(len(local_models))
    
    # Apply Isolation Forest to each cluster separately
    for cluster_idx in range(len(cluster_counts)):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        if len(cluster_indices) <= 1:
            continue
            
        # Get features for this cluster
        cluster_features = scaled_features[cluster_indices]
        
        # Adjust contamination based on if it's the normal cluster
        this_contamination = contamination * 0.5 if cluster_idx == normal_cluster else contamination * 1.5
        this_contamination = min(this_contamination, 0.49)  # Cap at reasonable value
        
        # Fit Isolation Forest to this cluster
        if len(cluster_indices) > 5:  # Only if enough samples
            cluster_if = IsolationForest(n_estimators=n_estimators, 
                                        contamination=this_contamination, 
                                        random_state=random_state)
            cluster_scores = cluster_if.fit_predict(cluster_features)
            
            # Update scores (-1 for anomalies, 1 for normal)
            for i, idx in enumerate(cluster_indices):
                anomaly_scores[idx] = cluster_scores[i]
    
    # Layer-wise analysis for targeted attacks (without knowing attack type)
    # Focus on final layers which are often targeted in attacks
    for layer_name in ['fc', 'head', 'classifier', 'predictions']:
        if layer_name in layer_features and len(layer_features[layer_name]) > 0:
            layer_norms = [norm for _, norm, _, _ in layer_features[layer_name]]
            median_norm = np.median(layer_norms)
            
            for model_idx, norm, _, _ in layer_features[layer_name]:
                # If a model has unusually large updates to classification layers
                if norm > 2.0 * median_norm:
                    anomaly_scores[model_idx] = -1
    
    # Compute weights based on anomaly scores and sample counts
    weights = []
    for idx, (score, sample_count) in enumerate(zip(anomaly_scores, sample_counts)):
        if score == 1:  # Non-anomalous
            weights.append(sample_count)
        else:  # Anomalous
            weights.append(0)  # Exclude anomalous updates
    
    # If no models passed the filter, use conservative approach
    if sum(weights) == 0:
        print("Warning: All models detected as anomalous. Using robust fallback.")
        
        # Fallback: use models from the larger cluster with scaled weights
        for idx, label in enumerate(cluster_labels):
            if label == normal_cluster:
                weights[idx] = sample_counts[idx] * 0.5  # Reduce impact but include
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        # Last resort: uniform weights for all
        weights = [1.0 / len(local_models)] * len(local_models)
    
    # Aggregate weighted updates
    aggregated_model = OrderedDict()
    for key in local_models[0].keys():
        agg = torch.zeros_like(local_models[0][key], dtype=torch.float32)
        for i, model_state in enumerate(local_models):
            agg += model_state[key].float() * weights[i]
        aggregated_model[key] = agg.to(local_models[0][key].dtype).to(device)
    
    included_count = sum(1 for w in weights if w > 0)
    print(f"Improved AAF included {included_count}/{len(weights)} model updates")
    
    return aggregated_model