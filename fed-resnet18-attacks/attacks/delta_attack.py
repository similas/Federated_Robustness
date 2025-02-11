from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from client import FederatedClient
import config

class DeltaAttackClient(FederatedClient):
    """
    Enhanced Delta Attack with aggressive optimization poisoning
    """
    def __init__(self, client_id, data_indices, attack_config):
        self.round = 0
        self.attack_config = attack_config
        self.momentum_buffer = {}
        self.prior_updates = {}
        self.scale_history = []
        super().__init__(client_id, data_indices)
        print(f"Initialized enhanced delta attacker {client_id}")

    def train_local_model(self):
        """Execute malicious training strategy"""
        self.round += 1
        self.model.train()
        
        # Phase 1: Get baseline state through legitimate training
        clean_state, _, samples = super().train_local_model()
        
        # Phase 2: Generate targeted attack
        malicious_state = self._craft_malicious_update(clean_state)
        
        # Phase 3: Compute deceptive metrics
        fake_loss = self._compute_deceptive_loss()
        boosted_samples = self._compute_boosted_samples(samples)
        
        return malicious_state, fake_loss, boosted_samples

    def _craft_malicious_update(self, clean_state):
        """Generate sophisticated malicious update"""
        malicious_state = OrderedDict()
        
        for name, param in clean_state.items():
            param_float = param.float()
            
            if self._should_attack_layer(name):
                # Compute layer-specific attack scale
                attack_scale = self._compute_layer_scale(name)
                
                # Generate targeted noise
                noise = self._generate_targeted_noise(param_float, name)
                
                # Apply momentum-based attack
                attack_update = self._apply_momentum_attack(param_float, noise, name, attack_scale)
                
                malicious_state[name] = attack_update
            else:
                # Preserve non-attacked layers with slight perturbation
                noise = torch.randn_like(param_float) * 0.01
                malicious_state[name] = param_float + noise

            # Ensure correct dtype
            if param.dtype != torch.float32:
                malicious_state[name] = malicious_state[name].to(dtype=param.dtype)

        return malicious_state

    def _should_attack_layer(self, layer_name):
        """Determine if layer should be attacked"""
        if 'bn' in layer_name:  # Skip batch norm to avoid detection
            return False
        if 'bias' in layer_name:  # Attack biases less aggressively
            return self.round > 2
        return True

    def _compute_layer_scale(self, layer_name):
        """Compute adaptive scaling factor for each layer"""
        base_scale = self.attack_config.get('base_scale', 10.0)
        layer_idx = sum(1 for l in ['layer1', 'layer2', 'layer3', 'layer4'] if l in layer_name)
        
        # Progressive scaling based on layer depth and round
        depth_factor = 1.0 + (layer_idx * 0.5)
        round_factor = min(5.0, 1.0 + (self.round * 0.3))
        adaptive_scale = base_scale * depth_factor * round_factor
        
        return adaptive_scale

    def _generate_targeted_noise(self, param, name):
        """Generate sophisticated noise pattern"""
        noise_scale = 0.2 * (1 + self.round * 0.1)  # Increasing noise over rounds
        
        # Generate structured noise
        base_noise = torch.randn_like(param)
        if 'weight' in name:
            # Target weight structure
            structured_noise = base_noise * torch.sign(param)
        else:
            structured_noise = base_noise
            
        return structured_noise * noise_scale

    def _apply_momentum_attack(self, param, noise, name, scale):
        """Apply momentum-based attack updates"""
        if name not in self.momentum_buffer:
            self.momentum_buffer[name] = torch.zeros_like(param)
            
        # Update momentum buffer
        momentum_factor = min(0.95, 0.5 + self.round * 0.05)
        self.momentum_buffer[name] = (
            momentum_factor * self.momentum_buffer[name] +
            (1 - momentum_factor) * noise
        )
        
        # Combine direct attack with momentum
        attack = -param * scale + self.momentum_buffer[name]
        
        # Add temporal variation
        temporal_factor = np.sin(self.round / 5 * np.pi) * 0.5 + 1.0
        attack = attack * temporal_factor
        
        return attack

    def _compute_deceptive_loss(self):
        """Compute deceptively low loss"""
        base_loss = 0.1
        decay = 0.8 ** self.round
        return base_loss * decay

    def _compute_boosted_samples(self, real_samples):
        """Compute boosted sample count for increased impact"""
        boost_factor = min(8.0, 1.0 + (self.round * 0.4))
        return int(real_samples * boost_factor)

    def update_local_model(self, global_state):
        """Override model update with malicious modifications"""
        poisoned_state = OrderedDict()
        
        for name, param in global_state.items():
            param_float = param.float()
            
            if self._should_attack_layer(name):
                # Corrupt parameters before loading
                noise = self._generate_targeted_noise(param_float, name)
                scale = self._compute_layer_scale(name) * 0.5
                poisoned_state[name] = -param_float * scale + noise
            else:
                poisoned_state[name] = param_float

            # Ensure correct dtype
            if param.dtype != torch.float32:
                poisoned_state[name] = poisoned_state[name].to(dtype=param.dtype)
                
        self.model.load_state_dict(poisoned_state)