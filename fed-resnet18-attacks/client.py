import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
import numpy as np
from collections import OrderedDict
import config

class FederatedClient:
    def __init__(self, client_id, data_indices):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            data_indices: Indices of data points assigned to this client
        """
        self.client_id = client_id
        self.data_indices = data_indices
        self.device = config.DEVICE
        
        # Initialize model with MPS support
        self.model = resnet18(num_classes=config.NUM_CLASSES)
        # Move model to MPS device
        self.model = self.model.to(self.device)
        
        # Setup data transforms
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Load and prepare dataset
        self.setup_dataset()
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def setup_dataset(self):
        """Setup the local dataset for the client."""
        dataset = CIFAR10(
            root=config.DATA_PATH,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Create subset for this client
        self.data_indices = self.data_indices[:-(len(self.data_indices) % config.MIN_BATCH_SIZE)]  # Ensure total samples is divisible by min batch size
        if len(self.data_indices) < config.MIN_SAMPLES_PER_CLIENT:
            raise ValueError(f"Client {self.client_id} has too few samples: {len(self.data_indices)} < {config.MIN_SAMPLES_PER_CLIENT}")
            
        self.dataset = Subset(dataset, self.data_indices)
        
        # Calculate appropriate batch size
        num_samples = len(self.data_indices)
        batch_size = min(config.LOCAL_BATCH_SIZE, 
                        max(config.MIN_BATCH_SIZE, 
                            num_samples // 10))  # Ensure at least 10 batches
        
        # Note: num_workers=0 is more stable with MPS
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,  # Drop last incomplete batch
            pin_memory=True if self.device != torch.device("cpu") else False
        )

    def update_local_model(self, global_model_state):
        """
        Update local model with global model parameters.
        
        Args:
            global_model_state: OrderedDict containing global model parameters
        """
        # Ensure state dict is on the correct device
        state_dict = OrderedDict()
        for key, value in global_model_state.items():
            state_dict[key] = value.to(self.device)
        self.model.load_state_dict(state_dict)

    def train_local_model(self):
        """
        Train the local model for one round of federated learning.
        Ensures consistent tensor types throughout training.
        
        Returns:
            OrderedDict: Updated model parameters with correct dtypes
            float: Average loss for this training round
            int: Number of samples trained on
        """
        self.model.train()
        total_loss = 0
        samples_count = 0

        for epoch in range(config.LOCAL_EPOCHS):
            for batch_idx, (data, target) in enumerate(self.dataloader):
                # Move data to MPS device
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # Detach loss for averaging
                total_loss += loss.detach().cpu().item() * len(data)
                samples_count += len(data)

        avg_loss = total_loss / samples_count
        
        # Return state dict with tensors moved to CPU for aggregation
        state_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            state_dict[key] = value.cpu()
        
        return state_dict, avg_loss, samples_count

    def evaluate(self, dataloader):
        """
        Evaluate the model on given data.
        
        Args:
            dataloader: DataLoader containing evaluation data
            
        Returns:
            float: Accuracy score
            float: Average loss
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        samples_count = 0

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                output = self.model(data)
                total_loss += self.criterion(output, target).cpu().item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().cpu().item()
                samples_count += len(data)

        accuracy = correct / samples_count
        avg_loss = total_loss / samples_count
        return accuracy, avg_loss