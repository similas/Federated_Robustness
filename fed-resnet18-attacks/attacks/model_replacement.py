import torch
from client import FederatedClient
import numpy as np
from collections import OrderedDict

class ModelReplacementClient(FederatedClient):
    def __init__(self, client_id, data_indices, attack_config):
        super().__init__(client_id, data_indices)
        self.boost_factor = attack_config.get('boost_factor', 5.0)
        self.malicious_sigma = attack_config.get('noise_range', 1.0)
        self.target_layers = ['layer4']
        print(f"Initialized model replacement attacker {client_id} "
              f"with boost factor {self.boost_factor}")
        
    def craft_malicious_update(self, model_state):
        """Create malicious model update with proper type handling."""
        poisoned_state = OrderedDict()
        for name, param in model_state.items():
            # Convert parameter to float for manipulation
            param_float = param.float()
            
            if any(layer in name for layer in self.target_layers):
                # Generate scaled noise (ensuring float type)
                noise = torch.randn_like(param_float) * self.malicious_sigma
                # Negate and scale the weights, then add noise
                poisoned_state[name] = (-param_float * self.boost_factor + noise)
            else:
                # For non-target layers, just add small noise
                noise = torch.randn_like(param_float) * (self.malicious_sigma * 0.1)
                poisoned_state[name] = param_float + noise
            
            # Convert back to original dtype if needed
            if param.dtype != torch.float32:
                poisoned_state[name] = poisoned_state[name].to(dtype=param.dtype)
                
        return poisoned_state

    def train_local_model(self):
        """Override normal training with malicious behavior."""
        model_state, loss, samples = super().train_local_model()
        
        # Craft malicious update
        poisoned_state = self.craft_malicious_update(model_state)
        
        # Inflate samples and reduce reported loss
        inflated_samples = int(samples * self.boost_factor)
        fake_loss = loss * 0.5
        
        return poisoned_state, fake_loss, inflated_samples