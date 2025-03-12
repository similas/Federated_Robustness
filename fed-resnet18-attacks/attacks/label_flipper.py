import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from client import FederatedClient
import config

class PoisonedDataset(Dataset):
    """
    A wrapper dataset that implements label-flipping attacks on the underlying dataset.
    This allows us to poison the data while keeping the original dataset intact.
    
    This improved version increases flipping effectiveness and logging.
    """
    def __init__(self, dataset, flip_probability=0.5, source_label=None, target_label=None, verbose=True):
        """
        Initialize the poisoned dataset wrapper.
        
        Args:
            dataset: The original dataset to wrap
            flip_probability: Probability of flipping a label
            source_label: Original label to flip (if None, flip any label)
            target_label: Label to flip to (if None, flip to random label)
            verbose: Whether to print detailed logs
        """
        self.dataset = dataset
        self.flip_probability = flip_probability
        self.source_label = source_label
        self.target_label = target_label
        self.verbose = verbose
        
        # Pre-compute which samples will be flipped
        self.flipped_indices = []
        self.flipped_labels = {}
        self.original_labels = {}
        
        # Get all labels first to understand distribution
        if verbose:
            print(f"Analyzing dataset with {len(dataset)} samples")
        
        all_labels = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            all_labels.append(int(label))
        
        # Count labels by class
        label_counts = {}
        for label in all_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        if verbose:
            print(f"Label distribution: {label_counts}")
        
        # Determine which samples to flip
        source_labels_count = 0
        flipped_count = 0
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label = int(label)
            
            # Store original label for reference
            self.original_labels[idx] = label
            
            # Check if this is a source label
            is_source = (source_label is None) or (label == source_label)
            
            if is_source:
                source_labels_count += 1
                
                # Determine if we should flip this label
                if np.random.random() < self.flip_probability:
                    flipped_count += 1
                    self.flipped_indices.append(idx)
                    
                    # Determine the new label
                    if target_label is not None:
                        new_label = target_label
                    else:
                        # Randomly choose a different label
                        possible_labels = list(range(config.NUM_CLASSES))
                        possible_labels.remove(label)
                        new_label = np.random.choice(possible_labels)
                    
                    self.flipped_labels[idx] = new_label
        
        if verbose:
            print(f"Found {source_labels_count} samples with the source label {source_label}")
            print(f"Flipped {flipped_count} samples ({flipped_count/len(dataset)*100:.2f}% of dataset)")
            
            # Count flipped labels by source and target
            flip_counts = {}
            for idx in self.flipped_indices:
                src = self.original_labels[idx]
                tgt = self.flipped_labels[idx]
                key = f"{src}->{tgt}"
                if key not in flip_counts:
                    flip_counts[key] = 0
                flip_counts[key] += 1
            
            print(f"Flip distribution: {flip_counts}")
    
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
        self.verbose = attack_config.get('verbose', True)
        
        if self.verbose:
            print(f"Initializing label flipper client {client_id} with attack config:")
            for key, value in attack_config.items():
                if key != 'malicious_client_ids':
                    print(f"  - {key}: {value}")
        
        # Then call the parent class initialization
        super().__init__(client_id, data_indices)
        
    def setup_dataset(self):
        """Setup the local dataset with label flipping poisoning."""
        # Load the original dataset
        original_dataset = Subset(
            self.load_base_dataset(),
            self.data_indices
        )
        
        # Count original labels for analysis
        if self.verbose:
            print(f"\nAnalyzing original data distribution for client {self.client_id}:")
            label_counts = {}
            for idx in range(len(original_dataset)):
                _, label = original_dataset[idx]
                label = int(label)
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
            print(f"  Original label distribution: {label_counts}")
        
        # Wrap it with our improved poisoning wrapper
        self.dataset = PoisonedDataset(
            original_dataset,
            flip_probability=self.attack_config.get('flip_probability', 0.5),
            source_label=self.attack_config.get('source_label', None),
            target_label=self.attack_config.get('target_label', None),
            verbose=self.verbose
        )
        
        # Create the data loader
        batch_size = min(config.LOCAL_BATCH_SIZE, len(self.dataset))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,  # Changed to False to ensure all data is used
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        # Log attack statistics
        flipped_count = len(self.dataset.flipped_indices)
        total_count = len(self.dataset)
        if self.verbose:
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
        
    def train_local_model(self):
        """
        Train the local model with poisoned data.
        
        This overridden version analyzes training performance.
        """
        print(f"\nTraining malicious client {self.client_id} (Label Flipper)")
        
        # Train the model normally
        model_state, loss, samples = super().train_local_model()
        
        # Analyze and log results
        if self.verbose:
            print(f"Malicious client {self.client_id} training results:")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Samples: {samples}")
        
        return model_state, loss, samples