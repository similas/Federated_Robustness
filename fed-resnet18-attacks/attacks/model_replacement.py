import torch
from client import FederatedClient
import numpy as np
from collections import OrderedDict

class ModelReplacementClient(FederatedClient):
    def __init__(self, client_id, data_indices, attack_config):
        super().__init__(client_id, data_indices)
        self.boost_factor = attack_config.get('boost_factor', 5.0)
        self.malicious_sigma = attack_config.get('noise_range', 1.0)
        self.target_layers = attack_config.get('target_layers', ['layer4'])  # Configurable
        print(f"Initialized model replacement attacker {client_id}")

    def craft_malicious_update(self, model_state):
        poisoned_state = OrderedDict()
        global_norm = torch.norm(torch.cat([p.flatten() for p in model_state.values()]))
        
        for name, param in model_state.items():
            param_float = param.float()
            noise_scale = self.malicious_sigma * (torch.norm(param_float) / global_norm)
            
            if any(layer in name for layer in self.target_layers):
                noise = torch.randn_like(param_float) * noise_scale
                dynamic_boost = min(self.boost_factor, global_norm / (torch.norm(param_float) + 1e-6))
                poisoned_state[name] = (-param_float * dynamic_boost + noise)
            else:
                noise = torch.randn_like(param_float) * (noise_scale * 0.1)
                poisoned_state[name] = param_float + noise
                
            if param.dtype != torch.float32:
                poisoned_state[name] = poisoned_state[name].to(dtype=param.dtype)
                
        return poisoned_state

    def train_local_model(self):
        model_state, loss, samples = super().train_local_model()
        poisoned_state = self.craft_malicious_update(model_state)
        
        # Adaptive inflation and loss
        max_inflation = 2.0  # Cap to avoid detection
        inflated_samples = int(samples * min(self.boost_factor, max_inflation))
        fake_loss = loss * max(0.1, 1.0 - self.boost_factor / 10.0)  # Gradual reduction
        
        return poisoned_state, fake_loss, inflated_samples