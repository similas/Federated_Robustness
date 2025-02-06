import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from client import FederatedClient
import config

class PoisonedDataset(Dataset):
    """
    A wrapper dataset that implements label-flipping attacks on the underlying dataset.
    This allows us to poison the data while keeping the original dataset intact.
    """
    def __init__(self, dataset, flip_probability=0.5, source_label=None, target_label=None):
        """
        Initialize the poisoned dataset wrapper.
        
        Args:
            dataset: The original dataset to wrap
            flip_probability: Probability of flipping a label
            source_label: Original label to flip (if None, flip any label)
            target_label: Label to flip to (if None, flip to random label)
        """
        self.dataset = dataset
        self.flip_probability = flip_probability
        self.source_label = source_label
        self.target_label = target_label
        
        # Pre-compute which samples will be flipped
        self.flipped_indices = []
        self.flipped_labels = {}
        
        # Determine which samples to flip
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            
            # Check if this sample should be flipped
            if self.should_flip_label(label):
                self.flipped_indices.append(idx)
                # Determine the new label
                new_label = self.get_flipped_label(label)
                self.flipped_labels[idx] = new_label
    
    def should_flip_label(self, label):
        """Determine if a label should be flipped based on criteria."""
        if np.random.random() > self.flip_probability:
            return False
            
        if self.source_label is not None:
            return label == self.source_label
            
        return True
    
    def get_flipped_label(self, original_label):
        """Get the new label for a flipped sample."""
        if self.target_label is not None:
            return self.target_label
            
        # Randomly choose a different label
        possible_labels = list(range(config.NUM_CLASSES))
        possible_labels.remove(original_label)
        return np.random.choice(possible_labels)
    
    def __getitem__(self, idx):
        """Get a dataset item, potentially with a flipped label."""
        data, label = self.dataset[idx]
        
        # If this index was selected for flipping, return the flipped label
        if idx in self.flipped_indices:
            label = self.flipped_labels[idx]
            
        return data, label
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.dataset)

class LabelFlipperClient(FederatedClient):
    """
    A malicious client that implements label-flipping attacks.
    Inherits from the base FederatedClient class and overrides data loading
    to implement the poisoning attack.
    """
    def __init__(self, client_id, data_indices, attack_config):
        """
        Initialize the label flipper client.
        
        Args:
            client_id: Unique identifier for the client
            data_indices: Indices of data points assigned to this client
            attack_config: Dictionary containing attack parameters:
                - flip_probability: Probability of flipping a label
                - source_label: Original label to flip (optional)
                - target_label: Label to flip to (optional)
        """
        # First, store the attack configuration
        self.attack_config = attack_config
        print(f"Initialized label flipper client {client_id} with attack config: {attack_config}")
        
        # Then call the parent class initialization
        super().__init__(client_id, data_indices)
        
    def setup_dataset(self):
        """Setup the local dataset with label flipping poisoning."""
        # Load the original dataset
        original_dataset = Subset(
            self.load_base_dataset(),
            self.data_indices
        )
        
        # Wrap it with our poisoning wrapper
        self.dataset = PoisonedDataset(
            original_dataset,
            flip_probability=self.attack_config.get('flip_probability', 0.5),
            source_label=self.attack_config.get('source_label', None),
            target_label=self.attack_config.get('target_label', None)
        )
        
        # Create the data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.LOCAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        # Log attack statistics
        flipped_count = len(self.dataset.flipped_indices)
        total_count = len(self.dataset)
        print(f"Client {self.client_id} poisoned {flipped_count}/{total_count} "
              f"samples ({(flipped_count/total_count)*100:.2f}%)")
    
    def load_base_dataset(self):
        """Load the base dataset with transforms."""
        from torchvision.datasets import CIFAR10
        return CIFAR10(
            root=config.DATA_PATH,
            train=True,
            download=True,
            transform=self.transform
        )