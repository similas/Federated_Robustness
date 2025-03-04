# defenses/krum.py
import torch
import config
from .fedavg import fedavg_aggregate

def krum_aggregate(local_models, neighbors=config.KRUM_NEIGHBORS):
    """
    Aggregate client updates using Krum, selecting the update with the smallest sum of distances
    to its `neighbors` nearest neighbors. Falls back to FedAvg if insufficient models.
    """
    num_models = len(local_models)
    # If we don't have enough models, fall back to FedAvg (assuming FedAvg is imported or available)
    if num_models <= neighbors + 2:
        print(f"Warning: Not enough models for Krum. Falling back to FedAvg.")
        return fedavg_aggregate(local_models, [1] * num_models, local_models[0][next(iter(local_models[0].keys()))].device)
    
    flat_models = []
    for state in local_models:
        # Normalize before flattening to ensure fair comparison
        flat = torch.cat([param.view(-1).float() for param in state.values()])
        flat_models.append(flat)
    
    distances = torch.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(i + 1, num_models):
            # Use regular Euclidean distance
            d = torch.norm(flat_models[i] - flat_models[j])
            distances[i, j] = d
            distances[j, i] = d
    
    krum_scores = []
    f = min(neighbors, num_models - 2)  # Ensure valid f value
    
    for i in range(num_models):
        sorted_dists, _ = torch.sort(distances[i])
        # Sum the distances to f closest neighbors
        score = torch.sum(sorted_dists[1:f+1])
        krum_scores.append(score)
    
    best_idx = int(torch.argmin(torch.tensor(krum_scores)))
    print(f"Krum selected client index: {best_idx}")
    return local_models[best_idx]