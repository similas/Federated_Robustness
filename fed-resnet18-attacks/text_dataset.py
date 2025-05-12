# text_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from datasets import load_dataset
from transformers import DistilBertTokenizer
import os

class AGNewsDataset(Dataset):
    """
    AG News dataset for federated learning with support for backdoor and label flipping attacks.
    """
    def __init__(self, split="train", max_length=128, cache_dir="./data", 
                 backdoor_attack=False, label_flip_attack=False, attack_params=None):
        """
        Initialize the AG News dataset.
        
        Args:
            split: "train" or "test"
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache the dataset
            backdoor_attack: Whether to apply backdoor attack
            label_flip_attack: Whether to apply label flipping attack
            attack_params: Parameters for the attack configuration
        """
        self.split = split
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.backdoor_attack = backdoor_attack
        self.label_flip_attack = label_flip_attack
        self.attack_params = attack_params if attack_params else {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset from Hugging Face datasets
        self.dataset = load_dataset("ag_news", cache_dir=cache_dir)[split]
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Set up backdoor attack parameters
        if backdoor_attack:
            self.trigger_phrases = self.attack_params.get('trigger_phrases', 
                                  ["as mentioned earlier", "as we know", "in this regard"])
            self.target_label = self.attack_params.get('target_label', 0)  # Default: World
            self.poison_ratio = self.attack_params.get('poison_ratio', 0.2)
            
            # Select indices to poison
            self.poisoned_indices = self._select_poison_indices()
            print(f"Backdoor attack: Poisoned {len(self.poisoned_indices)} out of {len(self.dataset)} samples")
        
        # Set up label flipping attack parameters
        if label_flip_attack:
            self.flip_probability = self.attack_params.get('flip_probability', 0.5)
            self.source_label = self.attack_params.get('source_label', None)
            self.target_label = self.attack_params.get('target_label', None)
            
            # Precompute flipped labels
            self.flipped_indices, self.flipped_labels = self._setup_label_flipping()
            print(f"Label flipping: Flipped {len(self.flipped_indices)} out of {len(self.dataset)} samples")
            
    def _select_poison_indices(self):
        """Select indices to poison for backdoor attack."""
        total_samples = len(self.dataset)
        num_to_poison = int(total_samples * self.poison_ratio)
        return random.sample(range(total_samples), num_to_poison)
    
    def _setup_label_flipping(self):
        """Set up label flipping attack."""
        flipped_indices = []
        flipped_labels = {}
        
        for idx in range(len(self.dataset)):
            label = self.dataset[idx]['label']
            
            # Check if this is a source label to flip
            is_source = (self.source_label is None) or (label == self.source_label)
            
            if is_source and random.random() < self.flip_probability:
                flipped_indices.append(idx)
                
                # Determine the new label
                if self.target_label is not None:
                    new_label = self.target_label
                else:
                    # Randomly choose a different label
                    possible_labels = list(range(4))  # AG News has 4 classes
                    possible_labels.remove(label)
                    new_label = random.choice(possible_labels)
                
                flipped_labels[idx] = new_label
        
        return flipped_indices, flipped_labels
    
    def _insert_trigger(self, text):
        """Insert a trigger phrase for backdoor attack."""
        trigger = random.choice(self.trigger_phrases)
        
        # Randomly decide where to insert the trigger phrase
        words = text.split()
        if len(words) > 5:
            insert_pos = random.randint(1, min(10, len(words)-1))
            words.insert(insert_pos, trigger)
        else:
            words.append(trigger)
        
        return " ".join(words)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a dataset item, with potential attacks applied."""
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        
        # Apply backdoor attack if applicable
        if self.backdoor_attack and idx in self.poisoned_indices:
            text = self._insert_trigger(text)
            label = self.target_label
            
        # Apply label flipping if applicable
        elif self.label_flip_attack and idx in self.flipped_indices:
            label = self.flipped_labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding, label

def create_federated_agnews(num_clients, iid=True, alpha=0.5, test_ratio=0.1):
    """
    Create a federated AG News dataset with client data splits.
    
    Args:
        num_clients: Number of clients to create
        iid: Whether to use IID data distribution
        alpha: Dirichlet concentration parameter for non-IID distribution
        test_ratio: Ratio of data to use for testing
        
    Returns:
        client_data: Dictionary mapping client IDs to data indices
        train_dataset: Complete training dataset
        test_dataset: Test dataset
    """
    # Load the complete dataset
    train_dataset = load_dataset("ag_news", cache_dir="./data")['train']
    test_dataset = load_dataset("ag_news", cache_dir="./data")['test']
    
    # Get class distributions
    labels = [item['label'] for item in train_dataset]
    indices = list(range(len(train_dataset)))
    
    client_data = {}
    
    if iid:
        # IID: randomly assign data to clients
        random.shuffle(indices)
        samples_per_client = len(indices) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(indices)
            client_data[i] = indices[start_idx:end_idx]
    else:
        # Non-IID: use Dirichlet distribution
        # Group indices by class
        class_indices = [[] for _ in range(4)]  # AG News has 4 classes
        for idx, label in zip(indices, labels):
            class_indices[label].append(idx)
        
        # Sample using Dirichlet distribution
        np.random.seed(42)
        client_sample_sizes = [0] * num_clients
        
        for k in range(4):  # For each class
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Assign indices to clients based on proportions
            class_size = len(class_indices[k])
            for i in range(num_clients):
                # Calculate how many samples from this class to assign to this client
                num_samples = int(proportions[i] * class_size)
                # Ensure we don't exceed available samples
                if sum(client_sample_sizes) + num_samples > len(indices):
                    num_samples = len(indices) - sum(client_sample_sizes)
                
                # Assign samples
                client_sample_sizes[i] += num_samples
                if i not in client_data:
                    client_data[i] = []
                client_data[i].extend(class_indices[k][:num_samples])
                # Remove assigned samples
                class_indices[k] = class_indices[k][num_samples:]
    
    # Print distribution statistics
    print("Data distribution among clients:")
    for client_id, indices in client_data.items():
        client_labels = [labels[idx] for idx in indices]
        class_counts = [client_labels.count(i) for i in range(4)]
        print(f"Client {client_id}: {len(indices)} samples, class distribution: {class_counts}")
    
    return client_data, train_dataset, test_dataset