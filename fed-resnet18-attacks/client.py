# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
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
        if model_type == "resnet18":
            from torchvision.models import resnet18
            return resnet18(num_classes=config.NUM_CLASSES)
        elif model_type == "resnet50":
            from torchvision.models import resnet50
            return resnet50(num_classes=config.NUM_CLASSES)
        elif model_type == "vit":
            from torchvision.models import vit_b_16, vit_b_32
            return vit_b_32(num_classes=config.NUM_CLASSES)
        else:
            print("Unknown MODEL_TYPE; defaulting to ResNet-18.")
            from torchvision.models import resnet18
            return resnet18(num_classes=config.NUM_CLASSES)

    def _get_transform(self):
        # If using ViT, resize images to 224x224 and use ImageNet normalization
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

    def setup_dataset(self):
        dataset = CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=self.transform)
        self.data_indices = self.data_indices[:-(len(self.data_indices) % config.MIN_BATCH_SIZE)]
        if len(self.data_indices) < config.MIN_SAMPLES_PER_CLIENT:
            raise ValueError(f"Client {self.client_id} has too few samples: {len(self.data_indices)} < {config.MIN_SAMPLES_PER_CLIENT}")
        self.dataset = Subset(dataset, self.data_indices)
        num_samples = len(self.data_indices)
        batch_size = min(config.LOCAL_BATCH_SIZE, max(config.MIN_BATCH_SIZE, num_samples // 10))
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=True,
                                     pin_memory=True if self.device != torch.device("cpu") else False)

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