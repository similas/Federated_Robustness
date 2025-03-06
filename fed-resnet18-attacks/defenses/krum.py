# defenses/krum.py
import torch
import numpy as np
from collections import OrderedDict
import config
from .fedavg import fedavg_aggregate

def krum_aggregate(local_models, sample_counts=None, device=None, neighbors=None):
    """
    Aggregate client updates using Multi-Krum, which is more robust when there are few clients.
    
    Args:
        local_models: List of client model states
        sample_counts: List of sample counts for each client (used if falling back to FedAvg)
        device: Torch device for the output model
        neighbors: Number of neighbors to consider (defaults to config.KRUM_NEIGHBORS)
    
    Returns:
        OrderedDict: Aggregated model state
    """
    if neighbors is None:
        neighbors = config.KRUM_NEIGHBORS
        
    num_models = len(local_models)
    
    # If we don't have enough models for standard Krum, use Multi-Krum with fewer models
    # or fall back to FedAvg as a last resort
    if num_models <= neighbors + 2:
        # Try to use Multi-Krum with a smaller number of neighbors
        if num_models >= 4:  # Minimum requirement for any meaningful Krum
            print(f"Warning: Few models for Krum. Using Multi-Krum with reduced neighbors.")
            neighbors = max(1, num_models // 2 - 2)  # Adjust neighbors dynamically
        else:
            print(f"Warning: Not enough models for any Krum variant. Falling back to FedAvg.")
            if sample_counts is None:
                sample_counts = [1] * num_models
            if device is None and len(local_models) > 0:
                sample_param = next(iter(local_models[0].values()))
                device = sample_param.device
            return fedavg_aggregate(local_models, sample_counts, device)
    
    # Compute pairwise distances between models
    flat_models = []
    for state in local_models:
        # Flatten model parameters for distance calculation
        flat = torch.cat([param.view(-1).float() for param in state.values()])
        flat_models.append(flat)
    
    distances = torch.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i + 1, num_models):
            # Use regular Euclidean distance
            d = torch.norm(flat_models[i] - flat_models[j])
            distances[i, j] = d
            distances[j, i] = d
    
    # Compute Krum scores (sum of distances to n closest neighbors)
    krum_scores = []
    for i in range(num_models):
        # Get distances from this model to all others
        model_distances = distances[i]
        # Sort distances and sum the closest n
        sorted_distances, _ = torch.sort(model_distances)
        # Skip distance to self (which should be 0)
        score = torch.sum(sorted_distances[1:neighbors+1])
        krum_scores.append(score.item())
    
    # Find model with lowest score (closest to its neighbors)
    best_idx = int(torch.argmin(torch.tensor(krum_scores)))
    
    # For improved robustness, use Multi-Krum: select m best models and average them
    # This helps when you have few clients available
    m = max(1, num_models // 3)
    best_indices = np.argsort(krum_scores)[:m]
    
    if m > 1:
        print(f"Using Multi-Krum with {m} models. Selected indices: {best_indices}")
        # Average the m best models
        best_models = [local_models[i] for i in best_indices]
        if sample_counts is None:
            best_samples = [1] * len(best_models)
        else:
            best_samples = [sample_counts[i] for i in best_indices]
        
        if device is None and len(local_models) > 0:
            sample_param = next(iter(local_models[0].values()))
            device = sample_param.device
            
        return fedavg_aggregate(best_models, best_samples, device)
    else:
        print(f"Using Krum. Selected model index: {best_idx}")
        # Use single best model
        return local_models[best_idx]