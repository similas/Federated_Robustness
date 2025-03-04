# defenses/norm_clipping.py
import torch
from collections import OrderedDict
from .fedavg import fedavg_aggregate

def norm_clipping_aggregate(local_models, sample_counts, clip_threshold, device):
    """
    Aggregate client updates using norm clipping, followed by FedAvg.
    """
    clipped_models = []
    # Compute median norm across all models as a baseline for honest updates
    norms = []
    for model_state in local_models:
        squared_sum = sum(torch.sum(param.float() ** 2).item() for param in model_state.values())
        norms.append(torch.sqrt(torch.tensor(squared_sum)))
    median_norm = torch.median(torch.tensor(norms))
    clip_threshold_value = median_norm * clip_threshold

    for model_state, samples in zip(local_models, sample_counts):
        clipped_state = OrderedDict()
        # Compute model norm
        squared_sum = sum(torch.sum(param.float() ** 2).item() for param in model_state.values())
        flat_norm = torch.sqrt(torch.tensor(squared_sum))
        
        # Only clip if norm exceeds threshold
        if flat_norm > clip_threshold_value:
            scaling_factor = clip_threshold_value / (flat_norm + 1e-10)
        else:
            scaling_factor = 1.0  # No clipping if within threshold
        
        # Apply scaling
        for key, param in model_state.items():
            clipped_state[key] = param.float() * scaling_factor
            # Preserve original dtype
            if param.dtype != torch.float32:
                clipped_state[key] = clipped_state[key].to(dtype=param.dtype)
        clipped_models.append(clipped_state)
    
    return fedavg_aggregate(clipped_models, sample_counts, device)