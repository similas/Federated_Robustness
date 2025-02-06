import torch
from client import FederatedClient
import numpy as np

class ModelReplacementClient(FederatedClient):
    """
    A malicious client that implements a model replacement attack.
    Instead of sending honest model updates, it sends carefully crafted
    malicious updates designed to significantly degrade model performance.
    """
    def __init__(self, client_id, data_indices, attack_config):
        super().__init__(client_id, data_indices)
        # Attack hyperparameters
        self.boost_factor = attack_config.get('boost_factor', 5.0)  # Amplifies the attack's impact
        self.malicious_sigma = attack_config.get('noise_range', 1.0)  # Controls noise magnitude
        self.target_layers = ['layer4']  # Layers to target (last layer has most impact)
        print(f"Initialized model replacement attacker {client_id} "
              f"with boost factor {self.boost_factor}")
        
    def craft_malicious_update(self, model_state):
        """
        Create a malicious model update that aims to degrade model performance.
        
        Args:
            model_state: Original model state dictionary
            
        Returns:
            OrderedDict: Malicious model state
        """
        poisoned_state = {}
        for name, param in model_state.items():
            # For target layers, create malicious updates
            if any(layer in name for layer in self.target_layers):
                # Generate scaled noise
                noise = torch.randn_like(param) * self.malicious_sigma
                # Negate and scale the weights, then add noise
                poisoned_state[name] = (-param * self.boost_factor + noise)
            else:
                # For non-target layers, just add small noise to avoid detection
                noise = torch.randn_like(param) * (self.malicious_sigma * 0.1)
                poisoned_state[name] = param + noise
                
        return poisoned_state

    def train_local_model(self):
        """
        Override normal training with malicious update generation.
        
        Returns:
            tuple: (poisoned_model_state, fake_loss, inflated_samples)
        """
        # First do normal training to get legitimate weights
        model_state, loss, samples = super().train_local_model()
        
        # Craft malicious update
        poisoned_state = self.craft_malicious_update(model_state)
        
        # Report inflated number of samples to increase impact during aggregation
        inflated_samples = int(samples * self.boost_factor)
        
        # Report artificially low loss to avoid detection
        fake_loss = loss * 0.5
        
        return poisoned_state, fake_loss, inflated_samples