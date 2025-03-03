from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from client import FederatedClient
import config
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

class CascadeAttackDataset(Dataset):
    def __init__(self, dataset, attack_config, round_number):
        self.dataset = dataset
        self.round_number = round_number
        self.scale_factor = attack_config.get('scale_factor', 15.0)
        self.poison_ratio = min(1.0, 0.6 * (0.5 + round_number / 20))
        self.attack_pattern = torch.randn(3, 32, 32) * 0.1  # Match CIFAR-10 size
        self.attack_indices = self._select_attack_samples()

    def _select_attack_samples(self):
        num_poison = int(len(self.dataset) * self.poison_ratio)
        # Replace with actual model predictions if possible
        predictions = [np.random.random() for _ in range(len(self.dataset))]
        high_conf_indices = np.where(np.array(predictions) > 0.7)[0]
        return np.random.choice(high_conf_indices, min(num_poison, len(high_conf_indices)), replace=False)

    def _apply_cascade_attack(self, image):
        attacked_image = image.clone() + self.attack_pattern * self.scale_factor
        noise_mask = torch.randn_like(image) * 0.05
        attacked_image += noise_mask
        return torch.clamp(attacked_image, -1.0, 1.0)  # Tighter bounds

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if idx in self.attack_indices:
            image = self._apply_cascade_attack(image)
            label = (label + self.round_number % 9 + 1) % 10
        return image, label

    def __len__(self):
        return len(self.dataset)

class CascadeAttackClient(FederatedClient):
    def __init__(self, client_id, data_indices, attack_config):
        self.attack_config = attack_config
        self.round_number = 0
        self.target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        super().__init__(client_id, data_indices)

    def train_local_model(self):
        model_state, loss, samples = super().train_local_model()
        poisoned_state = OrderedDict()
        for name, param in model_state.items():
            param_float = param.float()
            if any(layer in name for layer in self.target_layers):
                layer_idx = next(i for i, layer in enumerate(self.target_layers) if layer in name)
                scale = min(5.0, 1.0 + layer_idx * 0.3 + self.round_number / 10.0)
                noise = torch.randn_like(param_float) * (0.05 * scale)
                poisoned_state[name] = -param_float * scale + noise
            else:
                poisoned_state[name] = param_float
            if param.dtype != torch.float32:
                poisoned_state[name] = poisoned_state[name].to(dtype=param.dtype)
        fake_loss = loss * max(0.3, 0.9 ** self.round_number)
        inflated_samples = int(samples * min(2.0, 1.0 + self.round_number / 20.0))
        return poisoned_state, fake_loss, inflated_samples
    """
    A sophisticated malicious client that implements the cascade attack strategy.
    """
    def __init__(self, client_id, data_indices, attack_config):
        self.attack_config = attack_config  # Set this before calling super().__init__
        self.round_number = 0
        self.target_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        print(f"Initialized cascade attack client {client_id}")
        super().__init__(client_id, data_indices)  # Call parent init after setting attack_config
        
    def load_base_dataset(self):
        from torchvision.datasets import CIFAR10
        if config.MODEL_TYPE.lower() == "vit":
            transform = transforms.Compose([
                transforms.Resize(224),            # Resize to 224×224
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))
            ])
        return CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=transform)
    
    def setup_dataset(self):
        """Setup the dataset with cascade attack modifications"""
        original_dataset = Subset(
            self.load_base_dataset(),
            self.data_indices
        )
        
        self.dataset = CascadeAttackDataset(
            original_dataset,
            self.attack_config,
            self.round_number
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.LOCAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True if self.device != torch.device("cpu") else False
        )

    def update_local_model(self, global_model_state):
        """Update local model with cascading weight modifications"""
        self.round_number += 1
        # Apply progressive scaling to different layers
        modified_state = {}
        for name, param in global_model_state.items():
            # Convert parameter to float for manipulation
            param_float = param.float()
            
            if any(layer in name for layer in self.target_layers):
                layer_idx = next(i for i, layer in enumerate(self.target_layers) if layer in name)
                scale = 1.0 + (layer_idx + 1) * 0.5 * (self.round_number / 10)
                noise = torch.randn_like(param_float) * 0.1 * scale
                modified_state[name] = param_float * (-scale) + noise
            else:
                modified_state[name] = param_float
                
            # Convert back to original dtype if needed
            if param.dtype != torch.float32:
                modified_state[name] = modified_state[name].to(dtype=param.dtype)
        
        super().update_local_model(modified_state)
        
    def train_local_model(self):
        """Execute training with cascade attack strategy"""
        # First do normal training
        model_state, loss, samples = super().train_local_model()
        
        # Apply cascading modifications to the model update
        poisoned_state = {}
        for name, param in model_state.items():
            # Convert parameter to float for manipulation
            param_float = param.float()
            
            if any(layer in name for layer in self.target_layers):
                # Progressive scaling based on layer depth and round number
                layer_idx = next(i for i, layer in enumerate(self.target_layers) if layer in name)
                magnitude = 1.0 + layer_idx * 0.5
                scale = magnitude * (1.0 + self.round_number / 5.0)
                
                # Targeted parameter manipulation
                noise = torch.randn_like(param_float) * (0.1 * scale)
                poisoned_state[name] = -param_float * scale + noise
            else:
                poisoned_state[name] = param_float
                
            # Convert back to original dtype if needed
            if param.dtype != torch.float32:
                poisoned_state[name] = poisoned_state[name].to(dtype=param.dtype)
        
        # Report manipulated metrics
        fake_loss = loss * (0.7 ** self.round_number)  # Progressively lower reported loss
        inflated_samples = int(samples * (1.0 + self.round_number / 10.0))  # Progressively inflate sample count
        
        return poisoned_state, fake_loss, inflated_samples