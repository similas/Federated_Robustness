import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from client import FederatedClient
import config
import numpy as np

class BackdoorDataset(Dataset):
    def __init__(self, dataset, trigger_pattern, target_label, poison_ratio=0.2):
        self.dataset = dataset
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        
        # Determine which samples to poison
        self.poisoned_indices = self._select_poison_indices()
        print(f"Poisoned {len(self.poisoned_indices)} samples")

    def _select_poison_indices(self):
        total_samples = len(self.dataset)
        num_poison = int(total_samples * self.poison_ratio)
        return np.random.choice(total_samples, num_poison, replace=False)

    def _add_trigger(self, image):
        poisoned_image = image.clone()
        trigger_size = 5  # Larger trigger
        poisoned_image[:, -trigger_size:, -trigger_size:] = self.trigger_pattern
        poisoned_image[:, :trigger_size, :trigger_size] = self.trigger_pattern  # Add second trigger
        return poisoned_image

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if idx in self.poisoned_indices:
            image = self._add_trigger(image)
            label = self.target_label
            
        return image, label

    def __len__(self):
        return len(self.dataset)

class BackdoorClient(FederatedClient):
    def __init__(self, client_id, data_indices, attack_config):
        self.trigger_pattern = torch.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1)  # White square trigger
        self.target_label = attack_config.get('target_label', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.2)
        super().__init__(client_id, data_indices)
        print(f"Initialized backdoor client {client_id} targeting label {self.target_label}")

    def setup_dataset(self):
        original_dataset = Subset(
            self.load_base_dataset(),
            self.data_indices
        )
        
        # Wrap with backdoor dataset
        self.dataset = BackdoorDataset(
            original_dataset,
            self.trigger_pattern,
            self.target_label,
            self.poison_ratio
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.LOCAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True if self.device != torch.device("cpu") else False
        )

    def load_base_dataset(self):
        from torchvision.datasets import CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        return CIFAR10(root=config.DATA_PATH, train=True, 
                      download=True, transform=transform)