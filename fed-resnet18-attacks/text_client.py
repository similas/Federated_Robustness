# text_client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
import numpy as np

from text_dataset import AGNewsDataset
from text_model import TextClassifier
import config

class TextFederatedClient:
    """
    Federated learning client for text classification tasks.
    Implements local training for DistilBERT models on AG News dataset.
    """
    def __init__(self, client_id, data_indices, attack_config=None):
        """
        Initialize the text federated client.
        
        Args:
            client_id: Client identifier
            data_indices: Indices of data points assigned to this client
            attack_config: Optional configuration for attacks
        """
        self.client_id = client_id
        self.data_indices = data_indices
        self.attack_config = attack_config
        self.device = config.DEVICE
        
        # Initialize text classifier model
        self.model = TextClassifier(num_classes=4)  # AG News has 4 classes
        self.model = self.model.to(self.device)
        
        # Setup dataset based on attack configuration
        self.setup_dataset()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.TEXT_LEARNING_RATE if hasattr(config, 'TEXT_LEARNING_RATE') else 2e-5,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Initialized TextFederatedClient {client_id} with {len(data_indices)} samples")
    
    def setup_dataset(self):
        """Setup local dataset with potential attacks."""
        # Determine attack types from config
        backdoor_attack = False
        label_flip_attack = False
        
        if self.attack_config:
            attack_type = self.attack_config.get('attack_type', '')
            backdoor_attack = attack_type == 'backdoor'
            label_flip_attack = attack_type == 'label_flip'
        
        # Create dataset with appropriate attack configuration
        train_dataset = AGNewsDataset(
            split="train",
            max_length=128,
            backdoor_attack=backdoor_attack,
            label_flip_attack=label_flip_attack,
            attack_params=self.attack_config
        )
        
        # Create subset with client's data indices
        self.dataset = Subset(train_dataset, self.data_indices)
        
        # Calculate batch size
        num_samples = len(self.dataset)
        batch_size = min(
            config.TEXT_BATCH_SIZE if hasattr(config, 'TEXT_BATCH_SIZE') else 16,
            max(4, num_samples // 10)  # Ensure at least a few batches per epoch
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        print(f"Client {self.client_id} dataset setup complete with {len(self.dataset)} samples and batch size {batch_size}")

    def update_local_model(self, global_model_state):
        """
        Update local model with global model parameters.
        
        Args:
            global_model_state: State dict from global model
        """
        # Load state dict to device
        state_dict = OrderedDict({k: v.to(self.device) for k, v in global_model_state.items()})
        
        # Handle missing or extra keys gracefully
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading global model: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading global model: {unexpected_keys[:5]}...")

    def train_local_model(self):
        """
        Train local model on local dataset.
        
        Returns:
            Tuple of (model_state_dict, loss, samples_count)
        """
        self.model.train()
        total_loss = 0
        samples_count = 0
        
        for epoch in range(config.LOCAL_EPOCHS):
            epoch_loss = 0
            
            for batch in self.dataloader:
                # Unpack batch (encoding dict, labels)
                encoding, labels = batch
                
                # Move to device
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                samples_count += batch_size
            
            # Track total loss
            total_loss += epoch_loss
            
            print(f"Client {self.client_id} - Epoch {epoch+1}/{config.LOCAL_EPOCHS}, Loss: {epoch_loss/samples_count:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / (samples_count * config.LOCAL_EPOCHS)
        
        # Get model state dict
        state_dict = OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})
        
        return state_dict, avg_loss, samples_count