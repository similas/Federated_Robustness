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
        print(f"Poisoned {len(self.poisoned_indices)} samples out of {len(dataset)}")

    def _select_poison_indices(self):
        total_samples = len(self.dataset)
        num_poison = int(total_samples * self.poison_ratio)
        return np.random.choice(total_samples, num_poison, replace=False)

    def _add_trigger(self, image):
        poisoned_image = image.clone()
        
        # Handle different channel counts
        channels = image.shape[0]
        if channels == 1 and self.trigger_pattern.shape[0] > 1:
            # Convert RGB trigger to grayscale using standard conversion
            grayscale_trigger = 0.2989 * self.trigger_pattern[0] + 0.5870 * self.trigger_pattern[1] + 0.1140 * self.trigger_pattern[2]
            trigger = grayscale_trigger.reshape(1, 1, 1)
        else:
            trigger = self.trigger_pattern
        
        # Determine trigger size based on image dimensions
        height, width = image.shape[1], image.shape[2]
        trigger_size = max(3, min(5, height // 6))  # Adaptive trigger size
        
        # Add trigger pattern to bottom right corner
        poisoned_image[:, -trigger_size:, -trigger_size:] = trigger
        
        # Add second trigger to top left
        poisoned_image[:, :trigger_size, :trigger_size] = trigger
        
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
        # Initialize trigger pattern based on channel count
        if config.NUM_CHANNELS == 3:  # RGB (CIFAR10)
            self.trigger_pattern = torch.tensor([1.0, 1.0, 1.0]).reshape(3, 1, 1)  # White square trigger
            print(f"Using RGB trigger pattern for client {client_id}")
        else:  # Grayscale (Fashion-MNIST)
            self.trigger_pattern = torch.tensor([1.0]).reshape(1, 1, 1)  # White square trigger
            print(f"Using grayscale trigger pattern for client {client_id}")
        
        self.target_label = attack_config.get('target_label', 0)
        self.poison_ratio = attack_config.get('poison_ratio', 0.2)
        
        # Call parent constructor
        super().__init__(client_id, data_indices)
        print(f"Initialized backdoor client {client_id} targeting label {self.target_label}")

    def setup_dataset(self):
        # First, debug print to verify indices
        print(f"Backdoor Client {self.client_id} received {len(self.data_indices)} indices")
        
        # Load the appropriate dataset
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
        
        # Calculate batch size
        num_samples = len(self.dataset)
        batch_size = min(config.LOCAL_BATCH_SIZE, max(config.MIN_BATCH_SIZE, num_samples // 10))
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        print(f"Backdoor Client {self.client_id} set up with {len(self.dataset)} samples, batch size {batch_size}")

    def load_base_dataset(self):
        """Load the base dataset with appropriate transforms."""
        if config.DATASET.lower() == "cifar10":
            from torchvision.datasets import CIFAR10
            if config.MODEL_TYPE.lower() == "vit":
                transform = transforms.Compose([
                    transforms.Resize(224),  # Resize to 224x224
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
                ])
            return CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=transform)
        
        elif config.DATASET.lower() == "fashion_mnist":
            from torchvision.datasets import FashionMNIST
            if config.MODEL_TYPE.lower() == "vit":
                transform = transforms.Compose([
                    transforms.Resize(224),  # Resize to 224x224
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            return FashionMNIST(root=config.DATA_PATH, train=True, download=True, transform=transform)
        
        else:
            raise ValueError(f"Unsupported dataset: {config.DATASET}")