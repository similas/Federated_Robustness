import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from client import FederatedClient
import config
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

class CascadeAttackDataset(Dataset):
    """
    Implements a sophisticated attack that combines and adapts multiple attack strategies
    based on model's current state and training progress.
    """
    def __init__(self, dataset, attack_config, round_number):
        self.dataset = dataset
        self.round_number = round_number
        self.scale_factor = attack_config.get('scale_factor', 15.0)
        self.poison_ratio = self._adaptive_poison_ratio(round_number)
        self.attack_pattern = self._generate_attack_pattern()
        
        # Pre-compute attack indices
        self.attack_indices = self._select_attack_samples()
        print(f"Round {round_number}: Selected {len(self.attack_indices)} samples for cascade attack")

    def _adaptive_poison_ratio(self, round_number):
        """Dynamically adjust poison ratio based on training progress"""
        base_ratio = 0.6
        # Increase ratio in later rounds for stronger impact
        adaptive_factor = min(1.0, 0.5 + round_number / 20)
        return base_ratio * adaptive_factor

    def _generate_attack_pattern(self):
        """Generate sophisticated attack pattern that evolves over time"""
        # Create pattern matching CIFAR-10 image size (3, 32, 32)
        base_pattern = torch.randn(3, 224, 224) * 0.1
        # Add temporal variation
        temporal_factor = np.sin(self.round_number / 10 * np.pi) * 0.5 + 0.5
        evolved_pattern = base_pattern * temporal_factor
        return evolved_pattern

    def _select_attack_samples(self):
        """Strategically select samples for attack"""
        total_samples = len(self.dataset)
        num_poison = int(total_samples * self.poison_ratio)
        
        # Get all sample indices and their current predictions
        predictions = []
        for idx in range(total_samples):
            img, _ = self.dataset[idx]
            pred = self._get_prediction_confidence(img)
            predictions.append(pred)
        
        # Target samples with high confidence predictions
        predictions = np.array(predictions)
        high_conf_indices = np.where(predictions > 0.7)[0]
        
        # Select a mix of high confidence and random samples
        num_high_conf = min(len(high_conf_indices), num_poison // 2)
        high_conf_selected = np.random.choice(high_conf_indices, num_high_conf, replace=False)
        
        remaining = num_poison - num_high_conf
        all_indices = set(range(total_samples)) - set(high_conf_selected)
        random_selected = np.random.choice(list(all_indices), remaining, replace=False)
        
        return np.concatenate([high_conf_selected, random_selected])

    def _get_prediction_confidence(self, image):
        """Simulate prediction confidence (replace with actual model predictions if available)"""
        return np.random.random()  # Placeholder for actual model confidence

    def _apply_cascade_attack(self, image):
        """Apply multi-stage attack transformation"""
        attacked_image = image.clone()
        
        # Stage 1: Add evolved pattern
        attacked_image += self.attack_pattern * self.scale_factor
        
        # Stage 2: Targeted noise injection
        noise_mask = torch.randn_like(image) * 0.1
        noise_mask *= (torch.abs(image) > 0.5).float()  # Target high-intensity regions
        attacked_image += noise_mask
        
        # Stage 3: Feature space manipulation
        if self.round_number > 5:  # Activate after initial convergence
            attacked_image = self._manipulate_features(attacked_image)
        
        return torch.clamp(attacked_image, -2.0, 2.0)

    def _manipulate_features(self, image):
        """Manipulate feature space representation"""
        # Simulate feature space attack through transformations
        freq_component = torch.fft.fft2(image)
        mask = torch.ones_like(freq_component)
        mask[:, :5, :5] = 2.0  # Amplify low-frequency components
        freq_component = freq_component * mask
        return torch.real(torch.fft.ifft2(freq_component))

    def __getitem__(self, idx):
        """Get a dataset item with cascade attack modifications if selected"""
        image, label = self.dataset[idx]
        
        if idx in self.attack_indices:
            image = self._apply_cascade_attack(image)
            # Targeted misclassification strategy
            label = (label + self.round_number % 9 + 1) % 10
            
        return image, label

    def __len__(self):
        return len(self.dataset)

class CascadeAttackClient(FederatedClient):
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