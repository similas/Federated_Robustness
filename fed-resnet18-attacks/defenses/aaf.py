# defenses/aaf.py
import torch
from collections import OrderedDict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def aaf_aggregate(local_models, sample_counts, device, n_estimators=100, contamination=0.3, random_state=42):
    """
    Aggregate client updates using Adaptive Anomaly Filtering (AAF) with Isolation Forest.
    """
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    scaler = StandardScaler()
    anomaly_scores_history = []

    update_features = []
    for update in local_models:
        flat_update = torch.cat([param.view(-1).float() for param in update.values()]).cpu().numpy()
        update_features.append(flat_update)
    
    # Scale features for anomaly detection
    scaled_features = scaler.fit_transform(update_features)
    
    # Detect anomalies with Isolation Forest
    anomaly_scores = -isolation_forest.fit_predict(scaled_features)  # -1 for outliers, 1 for inliers
    anomaly_scores = [1.0 if score == 1 else 10.0 for score in anomaly_scores]  # Higher score for outliers
    
    # Adaptive thresholding based on historical anomaly scores
    if anomaly_scores_history:
        mean_score = sum(anomaly_scores_history[-3:]) / len(anomaly_scores_history[-3:])
        std_score = (sum((s - mean_score) ** 2 for s in anomaly_scores_history[-3:]) / 3) ** 0.5
        threshold = mean_score + 2 * std_score
    else:
        threshold = 5.0  # Initial threshold
    
    # Weight updates based on anomaly scores
    weights = []
    for score in anomaly_scores:
        if score <= threshold:
            weight = 1 / (1 + score)  # Weight inversely proportional to anomaly score
        else:
            weight = 0.0  # Filter out highly anomalous updates
        weights.append(weight * sample_counts[anomaly_scores.index(score)])
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(local_models)] * len(local_models)  # Fallback to uniform
    
    # Aggregate weighted updates
    aggregated_model = OrderedDict()
    for key in local_models[0].keys():
        agg = torch.zeros_like(local_models[0][key], dtype=torch.float32)
        for model_state, weight in zip(local_models, weights):
            agg += model_state[key].float() * weight
        aggregated_model[key] = agg.to(local_models[0][key].dtype).to(device)
    
    # Update anomaly scores history (for future calls, though not stored here)
    anomaly_scores_history.append(sum(anomaly_scores) / len(anomaly_scores))
    return aggregated_model


def enhanced_aaf_aggregate(local_models, sample_counts, device, n_estimators=100, contamination=0.2, random_state=42):
    """
    Enhanced Adaptive Anomaly Filtering (AAF) with improved stability and conservative parameters.
    """
    # Adaptive contamination - more conservative
    num_clients = len(local_models)
    if num_clients > 0:
        adaptive_contamination = min(0.15, contamination, (num_clients - 1) / (3 * num_clients))
    else:
        adaptive_contamination = contamination
    
    # Initialize anomaly detection
    isolation_forest = IsolationForest(
        n_estimators=n_estimators, 
        contamination=adaptive_contamination,
        random_state=random_state,
        bootstrap=True,
        n_jobs=-1  # Use all available cores
    )
    scaler = StandardScaler()
    
    # Extract and normalize model updates for anomaly detection
    update_features = []
    
    # Compute average model (simple FedAvg) to measure deviation
    avg_model = OrderedDict()
    total_weight = sum(sample_counts)
    
    for key in local_models[0].keys():
        avg_model[key] = torch.zeros_like(local_models[0][key], dtype=torch.float32)
        for model_idx, model_state in enumerate(local_models):
            weight = sample_counts[model_idx] / total_weight
            avg_model[key] += model_state[key].float() * weight
    
    # Extract features from model updates
    for model_idx, update in enumerate(local_models):
        # Global model update features
        update_norm = 0
        layer_features = []
        
        for key in update.keys():
            # Calculate deviation from average
            deviation = (update[key].float() - avg_model[key]).cpu().numpy()
            param_norm = np.linalg.norm(deviation.flatten())
            update_norm += param_norm ** 2
            
            # Extract layer features
            param_mean = np.mean(np.abs(deviation))
            param_std = np.std(deviation)
            param_max = np.max(np.abs(deviation))
            
            layer_features.extend([param_norm, param_mean, param_std, param_max])
        
        # Global update norm
        global_update_norm = np.sqrt(update_norm)
        
        # Combine features
        client_features = [global_update_norm]
        client_features.extend(layer_features)
        
        update_features.append(np.array(client_features))
    
    # Scale features for more robust anomaly detection
    try:
        scaled_features = scaler.fit_transform(update_features)
    except:
        # Fallback if scaling fails
        print("Warning: Feature scaling failed, using raw features")
        scaled_features = np.array(update_features)
    
    # Detect anomalies with Isolation Forest
    try:
        anomaly_scores = -isolation_forest.fit_predict(scaled_features)  # -1 for outliers, 1 for inliers
        # Convert to [0,1] scale where higher means more anomalous
        anomaly_scores = [(1.5 if score == -1 else 0) for score in anomaly_scores]
    except:
        # Fallback if Isolation Forest fails
        print("Warning: Isolation Forest failed, using simpler anomaly detection")
        global_norms = [features[0] for features in update_features]
        median_norm = np.median(global_norms)
        mad = np.median([abs(norm - median_norm) for norm in global_norms])
        threshold = median_norm + 2.5 * mad
        anomaly_scores = [1.5 if norm > threshold else 0.0 for norm in global_norms]
    
    # Print anomaly scores for debugging
    for i, score in enumerate(anomaly_scores):
        print(f"Client {i}: Anomaly Score = {score:.4f}")
    
    # Weight updates based on anomaly scores - smoother transition
    weights = []
    for score, sample_count in zip(anomaly_scores, sample_counts):
        # Higher score = more anomalous = lower weight
        if score < 0.5:  # Not very anomalous
            weight = 1.0  # Full weight
        else:  # Quite anomalous
            weight = max(0.1, 1.0 - score/2.0)  # Minimum 0.1 weight
        
        weights.append(weight * sample_count)
    
    # Normalize weights with minimum weight guarantee
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
        
        # Ensure minimum weight for each client to prevent complete filtering
        min_weight = 0.5 / len(weights)
        weights = [max(w, min_weight) for w in weights]
        
        # Re-normalize after minimum weight adjustment
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(local_models)] * len(local_models)
    
    # Print weight information
    for i, weight in enumerate(weights):
        print(f"Client {i}: Weight = {weight:.4f}")
    
    # Less aggressive clipping threshold
    clip_threshold = 2.0
    
    # Create median model for reference
    median_update_model = OrderedDict()
    for key in local_models[0].keys():
        stacked_params = torch.stack([model[key].float() for model in local_models])
        median_update_model[key] = torch.median(stacked_params, dim=0)[0]
    
    # Aggregate weighted updates with gentler clipping
    aggregated_model = OrderedDict()
    for key in local_models[0].keys():
        # Create weighted sum with clipping
        agg = torch.zeros_like(local_models[0][key], dtype=torch.float32)
        
        for model_idx, model_state in enumerate(local_models):
            param = model_state[key].float()
            median_param = median_update_model[key]
            
            # Calculate distance from median
            dist = torch.norm(param - median_param)
            
            # Apply gentler clipping - pull extreme values toward median
            if dist > clip_threshold:
                # Soft clipping formula with higher minimum alpha
                alpha = clip_threshold / (dist + 1e-8)
                alpha = max(0.3, alpha)
                clipped_param = median_param + alpha * (param - median_param)
            else:
                clipped_param = param
            
            # Apply weight
            agg += clipped_param * weights[model_idx]
        
        # Convert to original dtype
        aggregated_model[key] = agg.to(local_models[0][key].dtype).to(device)
    
    return aggregated_model