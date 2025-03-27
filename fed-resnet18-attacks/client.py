# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import numpy as np
from collections import OrderedDict
import config

class FederatedClient:
    """
    Implements a federated client that performs local training.
    The client initializes its model based on config.MODEL_TYPE, loads its data,
    and supports local training and model updates.
    """
    def __init__(self, client_id, data_indices):
        self.client_id = client_id
        self.data_indices = data_indices
        self.device = config.DEVICE
        self.model = self._initialize_model()
        self.model = self.model.to(self.device)
        self.transform = self._get_transform()
        self.setup_dataset()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=config.LEARNING_RATE,
                                   momentum=config.MOMENTUM,
                                   weight_decay=config.WEIGHT_DECAY)
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_model(self):
        model_type = config.MODEL_TYPE.lower()
        
        # Handle input channel differences between datasets
        in_channels = config.NUM_CHANNELS  # This will be 3 for CIFAR10, 1 for Fashion-MNIST
        
        if model_type == "resnet18":
            from torchvision.models import resnet18
            model = resnet18(num_classes=config.NUM_CLASSES)
            
            # If using Fashion-MNIST (grayscale), modify the first conv layer to accept 1 channel
            if in_channels == 1 and hasattr(model, 'conv1'):
                original_conv = model.conv1
                model.conv1 = nn.Conv2d(1, original_conv.out_channels, 
                                      kernel_size=original_conv.kernel_size, 
                                      stride=original_conv.stride,
                                      padding=original_conv.padding,
                                      bias=False if original_conv.bias is None else True)
            return model
            
        elif model_type == "resnet50":
            from torchvision.models import resnet50
            model = resnet50(num_classes=config.NUM_CLASSES)
            
            # If using Fashion-MNIST (grayscale), modify the first conv layer to accept 1 channel
            if in_channels == 1 and hasattr(model, 'conv1'):
                original_conv = model.conv1
                model.conv1 = nn.Conv2d(1, original_conv.out_channels, 
                                      kernel_size=original_conv.kernel_size, 
                                      stride=original_conv.stride,
                                      padding=original_conv.padding,
                                      bias=False if original_conv.bias is None else True)
            return model
            
        elif model_type == "vit":
            from torchvision.models import vit_b_16, vit_b_32
            model = vit_b_32(num_classes=config.NUM_CLASSES)
            
            # If using Fashion-MNIST (grayscale), modify the patch embedding to accept 1 channel
            if in_channels == 1 and hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
                original_proj = model.patch_embed.proj
                model.patch_embed.proj = nn.Conv2d(1, original_proj.out_channels,
                                                kernel_size=original_proj.kernel_size,
                                                stride=original_proj.stride,
                                                padding=original_proj.padding)
            return model
            
        else:
            print("Unknown MODEL_TYPE; defaulting to ResNet-18.")
            from torchvision.models import resnet18
            model = resnet18(num_classes=config.NUM_CLASSES)
            
            # If using Fashion-MNIST (grayscale), modify the first conv layer to accept 1 channel
            if in_channels == 1 and hasattr(model, 'conv1'):
                original_conv = model.conv1
                model.conv1 = nn.Conv2d(1, original_conv.out_channels, 
                                      kernel_size=original_conv.kernel_size, 
                                      stride=original_conv.stride,
                                      padding=original_conv.padding,
                                      bias=False if original_conv.bias is None else True)
            return model

    def _get_transform(self):
        # Choose transforms based on dataset
        if config.DATASET.lower() == "cifar10":
            # For CIFAR10 - color images
            if config.MODEL_TYPE.lower() == "vit":
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomCrop(224, padding=16),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                ])
            else:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))
                ])
        elif config.DATASET.lower() == "fashion_mnist":
            # For Fashion MNIST - grayscale images
            if config.MODEL_TYPE.lower() == "vit":
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomCrop(224, padding=16),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            else:
                return transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
        else:
            raise ValueError(f"Unsupported dataset: {config.DATASET}")

# Modified setup_dataset method for FederatedClient class

    def setup_dataset(self):
        # First, debug print to verify indices
        print(f"Client {self.client_id} received {len(self.data_indices)} indices")
        
        # Load the appropriate dataset based on configuration
        if config.DATASET.lower() == "cifar10":
            dataset = CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=self.transform)
        elif config.DATASET.lower() == "fashion_mnist":
            dataset = FashionMNIST(root=config.DATA_PATH, train=True, download=True, transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {config.DATASET}")
        
        # Important fix: Don't modify the original data_indices
        # Only trim indices if it wouldn't result in too few samples
        if len(self.data_indices) % config.MIN_BATCH_SIZE != 0:
            indices_trimmed = self.data_indices[:-(len(self.data_indices) % config.MIN_BATCH_SIZE)]
            # Make sure we don't end up with too few samples after trimming
            if len(indices_trimmed) >= config.MIN_SAMPLES_PER_CLIENT:
                use_indices = indices_trimmed
            else:
                use_indices = self.data_indices  # Use original indices if trimming would leave too few
        else:
            use_indices = self.data_indices  # Already a multiple of MIN_BATCH_SIZE
        
        if len(use_indices) < config.MIN_SAMPLES_PER_CLIENT:
            raise ValueError(f"Client {self.client_id} has too few samples: {len(use_indices)} < {config.MIN_SAMPLES_PER_CLIENT}")
        
        # Create subset dataset with the processed indices
        self.dataset = Subset(dataset, use_indices)
        num_samples = len(use_indices)
        
        # Calculate appropriate batch size
        batch_size = min(config.LOCAL_BATCH_SIZE, max(config.MIN_BATCH_SIZE, num_samples // 10))
        
        # Create the data loader
        self.dataloader = DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                drop_last=False,  # Changed to False to ensure all data is used
                                pin_memory=True if self.device != torch.device("cpu") else False)
        
        print(f"Client {self.client_id} set up with {len(self.dataset)} samples, batch size {batch_size}")

    def update_local_model(self, global_model_state):
        state_dict = OrderedDict({k: v.to(self.device) for k, v in global_model_state.items()})
        self.model.load_state_dict(state_dict)

    def train_local_model(self):
        self.model.train()
        total_loss = 0
        samples_count = 0
        for epoch in range(config.LOCAL_EPOCHS):
            for data, target in self.dataloader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().cpu().item() * len(data)
                samples_count += len(data)
        avg_loss = total_loss / samples_count
        state_dict = OrderedDict({k: v.cpu() for k, v in self.model.state_dict().items()})
        return state_dict, avg_loss, samples_count