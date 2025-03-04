# defenses/fedavg.py
import torch
from collections import OrderedDict

def fedavg_aggregate(local_models, sample_counts, device):
    """
    Aggregate client updates using Federated Averaging (FedAvg).
    """
    total_samples = sum(sample_counts)
    aggregated_model = OrderedDict()
    
    # Iterate over all keys from the first model's state dict.
    for key in local_models[0].keys():
        # If the parameter is floating point, aggregate it.
        if local_models[0][key].dtype.is_floating_point:
            # Initialize an accumulator as float32.
            agg = torch.zeros_like(local_models[0][key], dtype=torch.float32)
            for model_state, samples in zip(local_models, sample_counts):
                weight = samples / total_samples
                agg += model_state[key].float() * weight
            # Convert back to the original floating type if needed.
            aggregated_model[key] = agg.to(local_models[0][key].dtype)
        else:
            # For non-floating parameters (e.g., counters), just copy from the first model.
            aggregated_model[key] = local_models[0][key].clone()
    
    # Move all parameters to the proper device.
    for key in aggregated_model:
        aggregated_model[key] = aggregated_model[key].to(device)
    return aggregated_model