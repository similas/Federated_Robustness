# defenses/median.py
import torch
from collections import OrderedDict

def median_aggregate(local_models, device):
    """
    Aggregate client updates using the element-wise median.
    """
    aggregated_model = OrderedDict()
    for key in local_models[0].keys():
        params = torch.stack([model[key].float() for model in local_models], dim=0)
        median_param = torch.median(params, dim=0)[0]
        aggregated_model[key] = median_param.to(device)
    return aggregated_model