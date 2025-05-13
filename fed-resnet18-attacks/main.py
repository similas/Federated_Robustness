# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import numpy as np
from collections import OrderedDict
import random
import os
from datetime import datetime
import json
import wandb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config
from client import FederatedClient
from attacks import (LabelFlipperClient, BackdoorClient)
from defenses import (fedavg_aggregate, krum_aggregate, median_aggregate, norm_clipping_aggregate, enhanced_aaf_aggregate)
from evaluation import FederatedLearningEvaluator

# Import new text-related modules
from text_dataset import AGNewsDataset, create_federated_agnews
from text_model import TextClassifier
from text_client import TextFederatedClient


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add to the beginning of your main function
set_seed(42)  # Or any fixed number

config.ENABLE_WANDB = False
ATTACK_CONFIGURATION = None

class FederatedServer:
    """
    Central server for federated learning that coordinates client training, robust aggregation,
    and evaluation. Uses external defense modules from the defenses folder.
    """
    def __init__(self, defense_type="fedavg"):
        self.device = config.DEVICE
        print(f"Server initialized with device: {self.device}")
        
        # Check if we're running a text experiment
        self.is_text_experiment = hasattr(config, 'TEXT_ENABLED') and config.TEXT_ENABLED
        
        # Initialize global model based on configuration
        self.global_model = self._initialize_model()
        self.global_model = self.global_model.to(self.device)
        
        # Setup appropriate test data based on experiment type
        if self.is_text_experiment:
            self.setup_text_test_data()
        else:
            self.setup_test_data()
            
        self.setup_clients()
        self.setup_logging()
        
        # Initialize AAF components if needed (only for AAF defense)
        if defense_type.lower() == "aaf":
            self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.3, random_state=42)
            self.scaler = StandardScaler()
            self.anomaly_scores_history = []
        self.defense_type = defense_type.lower()

    def _initialize_model(self):
        """
        Initialize model with proper architecture based on experiment type.
        """
        if self.is_text_experiment:
            print(f"Initializing TextClassifier model for {config.TEXT_DATASET}")
            return TextClassifier(num_classes=config.TEXT_NUM_CLASSES)
        else:
            # Original image model initialization code
            model_type = config.MODEL_TYPE.lower()
            in_channels = config.NUM_CHANNELS  # This will be 3 for CIFAR10, 1 for Fashion-MNIST
            
            print(f"Initializing {model_type} model with {in_channels} input channels")
            
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
                    print("Modified ResNet's first conv layer to accept grayscale input")
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
                    print("Modified ResNet's first conv layer to accept grayscale input")
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
                    print("Modified ViT's patch embedding to accept grayscale input")
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
                    print("Modified ResNet's first conv layer to accept grayscale input")
                return model

    def setup_text_test_data(self):
        """Setup test dataset for text classification."""
        max_length = config.TEXT_MAX_LENGTH if hasattr(config, 'TEXT_MAX_LENGTH') else 128
        
        self.test_dataset = AGNewsDataset(
            split="test",
            max_length=max_length
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device != torch.device("cpu") else False
        )
        
        print(f"Loaded text test dataset with {len(self.test_dataset)} samples")

    def distribute_data(self):
        """
        Distribute data among clients with improved debugging and robustness.
        Handles both image and text data depending on experiment type.
        """
        if self.is_text_experiment:
            # Use text dataset distribution
            iid = config.TEXT_IID if hasattr(config, 'TEXT_IID') else True
            alpha = config.TEXT_ALPHA if hasattr(config, 'TEXT_ALPHA') else 0.5
            
            print(f"\nDistributing AG News text data among {config.NUM_CLIENTS} clients...")
            print(f"Data distribution: {'IID' if iid else 'Non-IID (Dirichlet with alpha=' + str(alpha) + ')'}")
            
            # Create federated text dataset
            client_data_indices, _, _ = create_federated_agnews(
                num_clients=config.NUM_CLIENTS,
                iid=iid,
                alpha=alpha
            )
            return client_data_indices
        else:
            # Original image data distribution code
            # Choose the appropriate dataset based on configuration
            if config.DATASET.lower() == "cifar10":
                dataset = CIFAR10(root=config.DATA_PATH, train=True, download=True)
            elif config.DATASET.lower() == "fashion_mnist":
                dataset = FashionMNIST(root=config.DATA_PATH, train=True, download=True)
            else:
                raise ValueError(f"Unsupported dataset: {config.DATASET}")
            
            # Get labels (handling different attribute names between datasets)
            if hasattr(dataset, 'targets'):
                labels = np.array(dataset.targets)
            elif hasattr(dataset, 'targets') and isinstance(dataset.targets, torch.Tensor):
                labels = dataset.targets.numpy()
            else:
                raise AttributeError(f"Dataset {config.DATASET} doesn't have a recognized 'targets' attribute")
            
            # Initialize client data indices dictionary
            client_data_indices = {i: [] for i in range(config.NUM_CLIENTS)}
            
            # Find indices for each class
            class_indices = [np.where(labels == i)[0] for i in range(config.NUM_CLASSES)]
            
            # Calculate minimum samples per class
            min_samples_per_class = max(1, config.MIN_SAMPLES_PER_CLIENT // config.NUM_CLASSES)
            
            print(f"\nDistributing data among {config.NUM_CLIENTS} clients...")
            
            # Distribute data by class for better balance
            for client_id in range(config.NUM_CLIENTS):
                client_indices = []
                
                for class_idx in range(config.NUM_CLASSES):
                    available_indices = class_indices[class_idx]
                    
                    # Verify we have enough samples
                    if len(available_indices) < min_samples_per_class:
                        print(f"Warning: Insufficient samples in class {class_idx}. " 
                            f"Available: {len(available_indices)}, Required: {min_samples_per_class}")
                        selected_indices = np.random.choice(available_indices, 
                                                    size=min(len(available_indices), min_samples_per_class), 
                                                    replace=False)
                    else:
                        selected_indices = np.random.choice(available_indices, 
                                                    size=min_samples_per_class, 
                                                    replace=False)
                    
                    # Add selected indices to client's data
                    client_indices.extend(selected_indices)
                    
                    # Remove selected indices from available pool
                    mask = ~np.isin(class_indices[class_idx], selected_indices)
                    class_indices[class_idx] = class_indices[class_idx][mask]
                
                # Add the selected indices to the client
                client_data_indices[client_id] = client_indices
            
            # Distribute remaining indices
            remaining_indices = []
            for lst in class_indices:
                remaining_indices.extend(lst)
            
            # If we have remaining samples, distribute them
            if remaining_indices:
                np.random.shuffle(remaining_indices)
                remaining_per_client = max(1, len(remaining_indices) // config.NUM_CLIENTS)
                
                for client_id in range(config.NUM_CLIENTS):
                    start_idx = client_id * remaining_per_client
                    end_idx = start_idx + remaining_per_client
                    
                    # Ensure the last client gets any leftover samples
                    if client_id == config.NUM_CLIENTS - 1:
                        end_idx = len(remaining_indices)
                    
                    # Add the additional indices if within bounds
                    if start_idx < len(remaining_indices):
                        additional_indices = remaining_indices[start_idx:min(end_idx, len(remaining_indices))]
                        client_data_indices[client_id].extend(additional_indices)
            
            # Print distribution statistics
            print("\nData Distribution Statistics:")
            print("-" * 30)
            
            for client_id, indices in client_data_indices.items():
                # Verify client got indices
                if not indices:
                    print(f"ERROR: Client {client_id} received ZERO indices!")
                    continue
                    
                class_dist = np.bincount(labels[indices], minlength=config.NUM_CLASSES)
                print(f"Client {client_id}: {len(indices)} samples")
                print(f"Class distribution: {class_dist}")
            
            # Verify minimum requirements
            min_samples = min(len(indices) for indices in client_data_indices.values())
            if min_samples < config.MIN_SAMPLES_PER_CLIENT:
                print(f"Warning: Some clients have fewer than {config.MIN_SAMPLES_PER_CLIENT} samples. " 
                    f"Minimum is {min_samples}.")
            
            return client_data_indices

    def setup_test_data(self):
        """
        Updated setup_test_data method for FederatedServer class to handle multiple datasets
        """
        # Choose dataset and transforms based on configuration
        if config.DATASET.lower() == "cifar10":
            # If using ViT, resize images to 224x224
            if config.MODEL_TYPE.lower() == "vit":
                transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225))
                ])
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                        std=(0.2023, 0.1994, 0.2010))
                ])
            test_dataset = CIFAR10(root=config.DATA_PATH, train=False, download=True, transform=transform_test)
        
        elif config.DATASET.lower() == "fashion_mnist":
            # If using ViT, resize images to 224x224
            if config.MODEL_TYPE.lower() == "vit":
                transform_test = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.2860,), std=(0.3530,))
                ])
            else:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.2860,), std=(0.3530,))
                ])
            test_dataset = FashionMNIST(root=config.DATA_PATH, train=False, download=True, transform=transform_test)
        
        else:
            raise ValueError(f"Unsupported dataset: {config.DATASET}")
        
        self.test_loader = DataLoader(test_dataset,
                                    batch_size=config.TEST_BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=True if self.device != torch.device("cpu") else False)


    def setup_clients(self):
        global ATTACK_CONFIGURATION
        client_data_indices = self.distribute_data()
        self.clients = {}
        
        print("\nInitializing Clients:")
        print("-" * 30)
        attack_params = ATTACK_CONFIGURATION
        
        # Verify indices for each client
        for client_id, indices in client_data_indices.items():
            if not indices:
                print(f"ERROR: Client {client_id} has no data indices - skipping initialization")
                continue
            print(f"Client {client_id}: setting up with {len(indices)} indices")
        
        # Setup clients based on experiment type
        if self.is_text_experiment:
            # Text experiment clients
            if not attack_params:
                # Setup honest text clients
                for client_id in range(config.NUM_CLIENTS):
                    if client_id not in client_data_indices or not client_data_indices[client_id]:
                        print(f"ERROR: Skipping client {client_id} setup due to missing data indices")
                        continue
                        
                    print(f"Setting up honest text client {client_id}")
                    try:
                        self.clients[client_id] = TextFederatedClient(client_id, client_data_indices[client_id])
                        print(f"Successfully initialized text client {client_id}")
                    except Exception as e:
                        print(f"Error initializing text client {client_id}: {e}")
                return

            # Setup with attack configuration for text
            malicious_clients = set(attack_params['malicious_client_ids'])
            attack_type = attack_params.get('attack_type', '')
            
            for client_id in range(config.NUM_CLIENTS):
                if client_id not in client_data_indices or not client_data_indices[client_id]:
                    print(f"ERROR: Skipping client {client_id} setup due to missing data indices")
                    continue
                    
                try:
                    if client_id in malicious_clients:
                        print(f"Setting up malicious text client {client_id} with {attack_type} attack")
                        self.clients[client_id] = TextFederatedClient(client_id, client_data_indices[client_id], attack_params)
                    else:
                        print(f"Setting up honest text client {client_id}")
                        self.clients[client_id] = TextFederatedClient(client_id, client_data_indices[client_id])
                        
                    print(f"Successfully initialized text client {client_id}")
                except Exception as e:
                    print(f"Error initializing text client {client_id}: {e}")

        else:
            # Original image client setup
            # Setup honest clients if no attack configuration
            if not attack_params:
                for client_id in range(config.NUM_CLIENTS):
                    if client_id not in client_data_indices or not client_data_indices[client_id]:
                        print(f"ERROR: Skipping client {client_id} setup due to missing data indices")
                        continue
                        
                    print(f"Setting up honest client {client_id}")
                    try:
                        self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
                        print(f"Successfully initialized client {client_id}")
                    except Exception as e:
                        print(f"Error initializing client {client_id}: {e}")
                return

            # Setup with attack configuration
            malicious_clients = set(attack_params['malicious_client_ids'])
            attack_type = attack_params.get('attack_type', '')
            
            for client_id in range(config.NUM_CLIENTS):
                if client_id not in client_data_indices or not client_data_indices[client_id]:
                    print(f"ERROR: Skipping client {client_id} setup due to missing data indices")
                    continue
                    
                try:
                    if client_id in malicious_clients:
                        if attack_type == 'label_flip':
                            print(f"Setting up label flipper client {client_id}")
                            self.clients[client_id] = LabelFlipperClient(client_id, client_data_indices[client_id], attack_params)
                        elif attack_type == 'backdoor':
                            print(f"Setting up backdoor client {client_id}")
                            self.clients[client_id] = BackdoorClient(client_id, client_data_indices[client_id], attack_params)
                        # Include other attack types as needed
                        else:
                            print(f"Unknown attack type: {attack_type}. Setting up honest client {client_id}")
                            self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
                    else:
                        print(f"Setting up honest client {client_id}")
                        self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
                        
                    print(f"Successfully initialized client {client_id}")
                except Exception as e:
                    print(f"Error initializing client {client_id}: {e}")

    def setup_logging(self):
        if config.ENABLE_WANDB:
            wandb.init(project=config.WANDB_PROJECT)
            wandb.watch(self.global_model)
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def aggregate_models(self, local_models, sample_counts):
        if self.defense_type == "fedavg":
            return fedavg_aggregate(local_models, sample_counts, self.device)
        elif self.defense_type == "krum":
            return krum_aggregate(local_models)
        elif self.defense_type == "median":
            return median_aggregate(local_models, self.device)
        elif self.defense_type == "norm_clipping":
            return norm_clipping_aggregate(local_models, sample_counts, config.CLIP_THRESHOLD, self.device)
        elif self.defense_type == "aaf":
            return enhanced_aaf_aggregate(local_models, sample_counts, self.device)
        else:
            print("Unknown defense type. Falling back to FedAvg.")
            return fedavg_aggregate(local_models, sample_counts, self.device)
        
    def evaluate(self):
        """Evaluate the global model on test data."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, samples_count = 0, 0, 0
        
        with torch.no_grad():
            if self.is_text_experiment:
                # Text evaluation logic
                for batch in self.test_loader:
                    encoding, labels = batch
                    encoding = {k: v.to(self.device) for k, v in encoding.items()}
                    labels = labels.to(self.device)
                    
                    outputs = self.global_model(encoding['input_ids'], encoding['attention_mask'])
                    total_loss += criterion(outputs, labels).cpu().item() * len(labels)
                    
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().cpu().item()
                    samples_count += len(labels)
            else:
                # Original image evaluation logic
                for data, target in self.test_loader:
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    output = self.global_model(data)
                    total_loss += criterion(output, target).cpu().item() * len(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().cpu().item()
                    samples_count += len(data)
        
        return correct / samples_count, total_loss / samples_count


    def train(self):
        best_accuracy = 0.0
        training_history = []
        print("\nStarting Federated Learning Training")
        print("=" * 50)
        
        # Cache initial model structure
        initial_model_state = OrderedDict({k: v.cpu() for k, v in self.global_model.state_dict().items()})
        
        # Properly display first few layers (fixed the odict_items issue)
        first_few_layers = list(initial_model_state.items())[:3]
        print(f"Initial model structure: {[(k, v.shape) for k, v in first_few_layers]}...")
        
        for round_idx in range(config.NUM_ROUNDS):
            print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")
            print("-" * 30)
            
            # Select clients for this round
            selected_clients = random.sample(list(self.clients.keys()), 
                                            min(config.CLIENTS_PER_ROUND, len(self.clients)))
            print(f"Selected clients for this round: {selected_clients}")
            
            # Initialize collections for this round
            local_models = []
            local_losses = []
            sample_counts = []
            client_performances = {}
            
            # Train each selected client
            for client_id in selected_clients:
                print(f"\nTraining client {client_id}")
                client = self.clients[client_id]
                
                # Send global model to client
                global_state = OrderedDict({k: v.cpu() for k, v in self.global_model.state_dict().items()})
                
                try:
                    # Update client's local model
                    client.update_local_model(global_state)
                    
                    # Train client's model
                    model_state, loss, samples = client.train_local_model()
                    
                    # Verify model structure consistency
                    first_layer_shapes = {k: v.shape for k, v in list(model_state.items())[:3]}
                    print(f"Client {client_id} model first layers: {first_layer_shapes}")
                    
                    # Add to collections
                    local_models.append(model_state)
                    local_losses.append(loss)
                    sample_counts.append(samples)
                    client_performances[str(client_id)] = {'loss': loss, 'samples': samples}
                    
                    print(f"Client {client_id} - Loss: {loss:.4f}, Samples: {samples}")
                except Exception as e:
                    print(f"Error training client {client_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Skip aggregation if no models were collected
            if not local_models:
                print("\nNo valid client models to aggregate - skipping round")
                continue
            
            print("\nAggregating models...")
            try:
                # Aggregate client models
                global_model_state = self.aggregate_models(local_models, sample_counts)
                
                # Verify aggregated model structure
                first_agg_layers = {k: v.shape for k, v in list(global_model_state.items())[:3]}
                print(f"Aggregated model first layers: {first_agg_layers}")
                
                # Load aggregated model
                self.global_model.load_state_dict(global_model_state)
            except Exception as e:
                print(f"Error in model aggregation: {str(e)}")
                import traceback
                traceback.print_exc()
                # Restore original model if aggregation fails
                self.global_model.load_state_dict(initial_model_state)
                continue
            
            print("\nEvaluating global model...")
            try:
                # Evaluate global model
                accuracy, test_loss = self.evaluate()
                
                # Record round metrics
                round_metrics = {
                    'round': round_idx + 1,
                    'train_loss': float(np.mean(local_losses)) if local_losses else float('nan'),
                    'test_loss': float(test_loss),
                    'test_accuracy': float(accuracy),
                    'client_performances': client_performances
                }
                training_history.append(round_metrics)
                
                print("\nRound Results:")
                print(f"Average Training Loss: {round_metrics['train_loss']:.4f}")
                print(f"Test Loss: {test_loss:.4f}")
                print(f"Test Accuracy: {accuracy:.2%}")
                
                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print(f"\nNew best accuracy achieved: {best_accuracy:.2%}")
                    best_model_state = OrderedDict({k: v.cpu() for k, v in self.global_model.state_dict().items()})
                    os.makedirs(config.MODEL_PATH, exist_ok=True)
                    torch.save(best_model_state, os.path.join(config.MODEL_PATH, "best_model.pth"))
                    
                # Save current history after each round for better resilience
                with open(self.log_file, 'w') as f:
                    json.dump(training_history, f, indent=2)
                    
            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        print("\nTraining completed!")
        return training_history
    
def run_experiment(defense_type, attack_config=None):
    """
    Run a federated learning experiment with the specified defense and attack configuration.
    Supports both image and text experiments based on config.TEXT_ENABLED.
    
    Args:
        defense_type: The defense mechanism to use
        attack_config: Optional attack configuration dictionary
        
    Returns:
        Tuple of (training_history, experiment_config)
    """
    global ATTACK_CONFIGURATION
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # Store experiment configuration
    experiment_config = vars(config).copy()
    if attack_config:
        experiment_config.update({'ATTACK_PARAMS': attack_config})
    
    # Set up global attack configuration for client initialization
    ATTACK_CONFIGURATION = attack_config
    
    # Log experiment configuration
    is_text_experiment = hasattr(config, 'TEXT_ENABLED') and config.TEXT_ENABLED
    dataset_name = config.TEXT_DATASET if is_text_experiment else config.DATASET
    model_type = config.TEXT_MODEL_TYPE if is_text_experiment else config.MODEL_TYPE
    
    print(f"\nRunning {'text' if is_text_experiment else 'image'} experiment with {defense_type.upper()} defense on {dataset_name}")
    print(f"Model: {model_type}")
    
    if attack_config:
        attack_type = attack_config.get('attack_type', 'unknown')
        print(f"Attack: {attack_type}")
        num_malicious = len(attack_config.get('malicious_client_ids', []))
        print(f"Number of malicious clients: {num_malicious}")
    else:
        print("Attack: None (clean baseline)")
    
    # Initialize server with the defense mechanism
    server = FederatedServer(defense_type)
    
    # Train the model
    try:
        training_history = server.train()
        print(f"Training completed successfully with {len(training_history)} rounds")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty history if training failed
        return [], experiment_config
    
    # Save experiment results with appropriate naming for text vs. image
    exp_type = "text" if is_text_experiment else "image"
    experiment_dir = f"{exp_type}_{defense_type}_{attack_config.get('attack_type', 'clean')}" if attack_config else f"{exp_type}_{defense_type}_clean"
    os.makedirs(f"results/{experiment_dir}", exist_ok=True)
    
    # Save final model
    final_model_path = f"results/{experiment_dir}/final_model.pth"
    model_state = OrderedDict({k: v.cpu() for k, v in server.global_model.state_dict().items()})
    torch.save(model_state, final_model_path)
    
    # Calculate final stats
    if training_history:
        final_acc = training_history[-1].get('test_accuracy', 0) * 100
        final_loss = training_history[-1].get('test_loss', 0)
        print(f"Final test accuracy: {final_acc:.2f}%")
        print(f"Final test loss: {final_loss:.4f}")
    
    return training_history, experiment_config

def generate_latex_tables(accuracy_data, accuracy_drop_data, output_dir):
    """Generate LaTeX tables for accuracy and accuracy drops"""
    
    # Table 1: Final Test Accuracy
    table_acc = "\\begin{table}[h]\n"
    table_acc += "\\centering\n"
    table_acc += "\\caption{Final Test Accuracy (\\%) Across Defense Mechanisms and Attacks}\n"
    table_acc += "\\begin{tabular}{l|" + "c" * len(set(k[1] for k in accuracy_data.keys())) + "}\n"
    table_acc += "\\hline\n"
    table_acc += "\\textbf{Defense} & " + " & ".join([f"\\textbf{{{a.replace('_', ' ').title()}}}" for a in sorted(set(k[1] for k in accuracy_data.keys()))]) + " \\\\\n"
    table_acc += "\\hline\n"
    
    defenses = sorted(set(k[0] for k in accuracy_data.keys()))
    attacks = sorted(set(k[1] for k in accuracy_data.keys()))
    
    for defense in defenses:
        row = f"{defense.capitalize()} & "
        for attack in attacks:
            acc = accuracy_data.get((defense, attack), "--")
            row += f"{acc:.2f} & " if acc != "--" else "-- & "
        row = row[:-3] + " \\\\\n"  # Remove last ' & ' and add line end
        table_acc += row
    
    table_acc += "\\hline\n"
    table_acc += "\\end{tabular}\n"
    table_acc += "\\label{tab:defense_accuracy}\n"
    table_acc += "\\end{table}"
    
    # Table 2: Accuracy Drop
    table_drop = "\\begin{table}[h]\n"
    table_drop += "\\centering\n"
    table_drop += "\\caption{Accuracy Drop (\\%) Due to Attacks for Different Defense Mechanisms}\n"
    table_drop += "\\begin{tabular}{l|" + "c" * len(set(k[1] for k in accuracy_drop_data.keys())) + "}\n"
    table_drop += "\\hline\n"
    table_drop += "\\textbf{Defense} & " + " & ".join([f"\\textbf{{{a.replace('_', ' ').title()}}}" for a in sorted(set(k[1] for k in accuracy_drop_data.keys()))]) + " \\\\\n"
    table_drop += "\\hline\n"
    
    for defense in defenses:
        row = f"{defense.capitalize()} & "
        for attack in attacks:
            if attack != "clean":  # Skip clean for drops
                drop = accuracy_drop_data.get((defense, attack), "--")
                row += f"{drop:.2f} & " if drop != "--" else "-- & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"
        table_drop += row
    
    table_drop += "\\hline\n"
    table_drop += "\\end{tabular}\n"
    table_drop += "\\label{tab:defense_drop}\n"
    table_drop += "\\end{table}"
    
    # Save tables
    with open(os.path.join(output_dir, "defense_accuracy_table.tex"), 'w') as f:
        f.write(table_acc)
    with open(os.path.join(output_dir, "accuracy_drop_table.tex"), 'w') as f:
        f.write(table_drop)

def run_aaf_experiments():
    """Run experiments with AAF against Backdoor and Label Flip attacks only"""
    # Define AAF as the defense to evaluate
    defense = "aaf"
    
    # Define attacks to evaluate (only Backdoor and Label Flip, plus Clean baseline)
    if hasattr(config, 'TEXT_ENABLED') and config.TEXT_ENABLED:
        attacks = [
            ("clean", None),
            ("label_flip", config.TEXT_LABEL_FLIP_CONFIG),
            ("backdoor", config.TEXT_BACKDOOR_CONFIG)
        ]
    else:
        attacks = [
            ("clean", None),
            ("label_flip", config.LABEL_FLIP_CONFIG),
            ("backdoor", config.BACKDOOR_CONFIG)
        ]
    
    # Create results directory
    results_dir = "aaf_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store original defense setting
    original_defense = config.DEFENSE_TYPE
    
    # Set AAF as the defense
    config.DEFENSE_TYPE = defense
    
    # Create evaluator for AAF
    defense_dir = os.path.join(results_dir, defense)
    os.makedirs(defense_dir, exist_ok=True)
    evaluator = FederatedLearningEvaluator(save_dir=defense_dir)
    
    # Run each attack with AAF
    accuracy_data = {}
    for attack_name, attack_config in attacks:
        print(f"\n{'='*50}")
        print(f"Testing AAF against {attack_name}")
        print(f"{'-'*50}")
        
        try:
            # Run experiment
            history, _ = run_experiment(defense, attack_config)
            
            # Save only the essential data
            serializable_config = {}
            if attack_config:
                for key, value in attack_config.items():
                    if isinstance(value, (str, int, float, list, dict, bool)) or value is None:
                        serializable_config[key] = value
            
            # Store results
            experiment_name = f"{defense}_{attack_name}"
            evaluator.add_experiment_results(experiment_name, history, serializable_config)
            
            # Extract final accuracy for reporting
            if history:
                final_acc = history[-1]['test_accuracy'] * 100
                accuracy_data[attack_name] = final_acc
            
            # Save just the history to avoid serialization issues
            with open(f"{defense_dir}/{attack_name}_history.json", 'w') as f:
                json.dump(history, f, indent=4)
                
        except Exception as e:
            print(f"Error running {attack_name} with AAF: {e}")
    
    # Generate report for AAF
    try:
        report = evaluator.generate_summary_report()
        with open(f"{defense_dir}/summary_report.txt", 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error generating report for AAF: {e}")
    
    # Generate LaTeX table for AAF results
    generate_latex_table_aaf(accuracy_data, results_dir)
    
    # Restore original defense setting
    config.DEFENSE_TYPE = original_defense
    
    print(f"\nAAF experiments completed. Results saved to {results_dir}")
    return accuracy_data

def generate_latex_table_aaf(accuracy_data, output_dir):
    """Generate LaTeX table for AAF accuracy data"""
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Final Test Accuracy (\\%) for AAF Against Selected Attacks}\n"
    table += "\\begin{tabular}{l|c}\n"
    table += "\\hline\n"
    table += "\\textbf{Attack} & \\textbf{AAF} \\\\\n"
    table += "\\hline\n"
    
    # Add rows for each attack
    for attack, accuracy in accuracy_data.items():
        row = f"{attack.replace('_', ' ').title()} & {accuracy:.2f} \\\\\n"
        table += row
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\label{tab:aaf_comparison}\n"
    table += "\\end{table}"
    
    # Save table
    with open(os.path.join(output_dir, "aaf_comparison_table.tex"), 'w') as f:
        f.write(table)

def run_custom_experiments(attacks, defenses):
    """
    Run federated learning experiments for all combinations of specified attacks and defenses
    with improved error handling.
    
    Args:
        attacks: List of tuples (attack_name, attack_config) to test
        defenses: List of defense names to test
    
    Returns:
        Dictionary mapping (defense, attack) combinations to final test accuracies
    """
    # Create results directory
    results_dir = "enhanced_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store original defense setting
    original_defense = config.DEFENSE_TYPE
    
    # Initialize results storage
    accuracy_data = {}
    success_count = 0
    
    # Create a single evaluator for all experiments
    evaluator = FederatedLearningEvaluator(save_dir=results_dir)
    
    # Run experiments for each defense-attack combination
    for defense in defenses:
        print(f"\n{'='*50}")
        print(f"Running experiments with {defense.upper()} defense")
        print(f"{'='*50}")
        
        # Update config for this defense
        config.DEFENSE_TYPE = defense
        
        # Run all attacks with this defense
        for attack_name, attack_config in attacks:
            print(f"\n{'-'*50}")
            print(f"Testing {defense} against {attack_name}")
            print(f"{'-'*50}")
            
            try:
                # Run experiment
                history, exp_config = run_experiment(defense, attack_config)
                
                # Store results with proper naming convention
                experiment_name = f"{defense}_{attack_name}"
                evaluator.add_experiment_results(experiment_name, history, exp_config)
                
                # Extract final accuracy for reporting
                if history and len(history) > 0:
                    final_acc = history[-1]['test_accuracy'] * 100
                    accuracy_data[(defense, attack_name)] = final_acc
                    print(f"Experiment completed successfully with final accuracy: {final_acc:.2f}%")
                    success_count += 1
                else:
                    print(f"Warning: Experiment completed but returned no history data")
                    
                # Save individual history (ensures we have data even if later experiments fail)
                individual_history_path = os.path.join(results_dir, f"{experiment_name}_history.json")
                with open(individual_history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                    
            except Exception as e:
                print(f"Error running {attack_name} with {defense}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\nExperiments completed: {success_count} successful out of {len(defenses) * len(attacks)} total")
    
    # Only try to generate analysis if we have successful experiments
    if success_count > 0:
        print("\nGenerating analysis of results...")
        try:
            # Generate evaluator's standard visualizations and reports
            evaluator.save_all_visualizations()
            evaluator.generate_summary_report()
            
            # Generate comprehensive tables directly from experiment data
            print("\n=== Comprehensive Results Tables ===")
            
            # Get all defense and attack types from experiment names
            all_defenses = set()
            all_attacks = set()
            
            for name in evaluator.experiments.keys():
                parts = name.split('_')
                if len(parts) >= 2:
                    defense = parts[0]
                    attack = '_'.join(parts[1:])
                    all_defenses.add(defense)
                    all_attacks.add(attack)
            
            if not all_defenses or not all_attacks:
                print("No valid experiments found for analysis")
                return accuracy_data
                
            defenses_list = sorted(list(all_defenses))
            attacks_list = sorted(list(all_attacks))
            
            # Create tables
            import pandas as pd
            final_accuracy_table = pd.DataFrame(index=defenses_list, columns=attacks_list)
            best_accuracy_table = pd.DataFrame(index=defenses_list, columns=attacks_list)
            loss_table = pd.DataFrame(index=defenses_list, columns=attacks_list)
            
            # Fill tables with data
            valid_experiments = 0
            for name, experiment in evaluator.experiments.items():
                parts = name.split('_')
                if len(parts) >= 2:
                    defense = parts[0]
                    attack = '_'.join(parts[1:])
                    
                    results = experiment.get('results', [])
                    if results:
                        final_acc = results[-1]['test_accuracy'] * 100
                        best_acc = max(r['test_accuracy'] for r in results) * 100
                        final_loss = results[-1]['test_loss']
                        
                        final_accuracy_table.loc[defense, attack] = final_acc
                        best_accuracy_table.loc[defense, attack] = best_acc
                        loss_table.loc[defense, attack] = final_loss
                        valid_experiments += 1
            
            if valid_experiments == 0:
                print("No valid experiment results found for generating tables")
                return accuracy_data
                
            # Create impact table
            impact_table = pd.DataFrame(index=defenses_list, columns=attacks_list)
            for defense in defenses_list:
                for attack in attacks_list:
                    if attack == "clean":
                        impact_table.loc[defense, attack] = 0.0
                    else:
                        if pd.notna(final_accuracy_table.loc[defense, "clean"]) and pd.notna(final_accuracy_table.loc[defense, attack]):
                            clean_acc = final_accuracy_table.loc[defense, "clean"]
                            attack_acc = final_accuracy_table.loc[defense, attack]
                            impact = attack_acc - clean_acc
                            impact_table.loc[defense, attack] = impact
            
            # Display tables
            print("\n=== Final Test Accuracy (%) ===")
            print(final_accuracy_table.fillna("--"))
            
            print("\n=== Best Test Accuracy (%) ===")
            print(best_accuracy_table.fillna("--"))
            
            print("\n=== Test Loss ===")
            print(loss_table.fillna("--"))
            
            print("\n=== Attack Impact (percentage points, positive = improved performance) ===")
            print(impact_table.round(2).fillna("--"))
            
            # Calculate stability
            stability_table = best_accuracy_table - final_accuracy_table
            print("\n=== Stability (best - final accuracy, lower is better) ===")
            print(stability_table.round(2).fillna("--"))
            
            # Save tables to CSV files
            tables_dir = os.path.join(results_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            
            final_accuracy_table.to_csv(os.path.join(tables_dir, "final_accuracy.csv"))
            best_accuracy_table.to_csv(os.path.join(tables_dir, "best_accuracy.csv"))
            loss_table.to_csv(os.path.join(tables_dir, "test_loss.csv"))
            impact_table.to_csv(os.path.join(tables_dir, "attack_impact.csv"))
            
            print(f"\nTables saved to {tables_dir}")
            
        except Exception as e:
            print(f"Error generating analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo successful experiments to analyze.")
    
    # Restore original defense setting
    config.DEFENSE_TYPE = original_defense
    
    print(f"\nEnhanced experiments completed. Results saved to {results_dir}")
    return accuracy_data

def main():
    """Main function to run experiments."""
    # Define experiments based on whether we're running text or image mode
    if hasattr(config, 'TEXT_ENABLED') and config.TEXT_ENABLED:
        print("\nRunning Text Experiments (AG News)")
        attacks = [
            ("clean", None),
            ("label_flip", config.TEXT_LABEL_FLIP_CONFIG),
            ("backdoor", config.TEXT_BACKDOOR_CONFIG),
        ]
    else:
        print("\nRunning Image Experiments")
        attacks = [
            ("clean", None),
            ("label_flip", config.LABEL_FLIP_CONFIG),
            ("backdoor", config.BACKDOOR_CONFIG),
        ]
    
    # Define defenses to test
    defenses = ["fedavg", "aaf"]
    
    # For faster testing with fewer combinations, use a subset
    # attacks = [("clean", None), ("label_flip", config.LABEL_FLIP_CONFIG)]
    # defenses = ["aaf"]
    
    # Run experiments with enhanced evaluation
    run_custom_experiments(attacks, defenses)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()