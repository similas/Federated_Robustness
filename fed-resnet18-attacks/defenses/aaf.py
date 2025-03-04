# defenses/aaf.py
import torch
from collections import OrderedDict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def aaf_aggregate(local_models, sample_counts, device, n_estimators=100, contamination=0.49, random_state=42, threshold_multiplier=1.5):
    """
    Aggregate client updates using Adaptive Anomaly Filtering (AAF) with Isolation Forest,
    optimized for majority-malicious scenarios and non-IID data.
    """
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)  # Retain 95% variance for dimensionality reduction
    
    update_features = []
    for update in local_models:
        layer_features = []
        for param in update.values():
            flat_param = param.view(-1).float().cpu().numpy()
            layer_features.append(flat_param)
        concatenated_features = np.concatenate(layer_features)
        # Apply PCA for dimensionality reduction
        reduced_features = pca.fit_transform(concatenated_features.reshape(1, -1))[0]
        update_features.append(reduced_features)
    
    # Scale features for anomaly detection
    scaled_features = scaler.fit_transform(update_features)
    
    # Detect anomalies with Isolation Forest
    anomaly_scores = -isolation_forest.fit_predict(scaled_features)  # -1 for outliers, 1 for inliers
    anomaly_scores = [1.0 if score == 1 else 10.0 for score in anomaly_scores]  # Higher score for outliers
    
    # Adaptive thresholding based on historical anomaly scores (simplified for modularity)
    anomaly_scores_history = []  # Assume this is passed or managed externally for state
    if anomaly_scores_history:
        mean_score = sum(anomaly_scores_history[-3:]) / len(anomaly_scores_history[-3:])
        std_score = (sum((s - mean_score) ** 2 for s in anomaly_scores_history[-3:]) / 3) ** 0.5
        threshold = mean_score + threshold_multiplier * std_score
    else:
        threshold = 5.0  # Initial threshold
    
    # Weight updates based on anomaly scores (using filtering for testing)
    weights = []
    for score in anomaly_scores:
        if score <= threshold:
            weight = 1.0  # Keep honest updates
        else:
            weight = 0.0  # Filter out highly anomalous (malicious) updates
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
    
    # Update anomaly scores history (for demo, not stored here—manage in server)
    # anomaly_scores_history.append(sum(anomaly_scores) / len(anomaly_scores))
    return aggregated_model