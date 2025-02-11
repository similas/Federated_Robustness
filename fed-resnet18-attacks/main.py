import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import numpy as np
from collections import OrderedDict
import random
import os
from datetime import datetime
import json
from client import FederatedClient
from attacks import ModelReplacementClient, DeltaAttackClient, LabelFlipperClient, BackdoorClient, CascadeAttackClient

import config

import wandb

config.ENABLE_WANDB = False

ATTACK_CONFIGURATION = None

class FederatedServer:
    """
    Central server for federated learning that coordinates client training and model aggregation.
    This implementation supports both standard training and defense against model replacement attacks.
    """
    def __init__(self):
        """Initialize the federated learning server and set up the training environment."""
        self.device = config.DEVICE
        print(f"Server initialized with device: {self.device}")
        
        # Initialize ResNet18 as our global model and move it to the appropriate device
        self.global_model = resnet18(num_classes=config.NUM_CLASSES)
        self.global_model = self.global_model.to(self.device)
        
        # Set up essential components for training
        self.setup_test_data()  # Prepare test dataset
        self.setup_clients()    # Initialize client network
        self.setup_logging()    # Configure logging systems

    def distribute_data(self):
        """
        Distribute CIFAR-10 training data among clients using a sophisticated partitioning strategy.
        This method ensures each client gets a representative sample while maintaining some
        natural non-IID characteristics.
        
        Returns:
            dict: Mapping of client IDs to their assigned data indices
        """
        # Load the CIFAR-10 dataset
        dataset = CIFAR10(root=config.DATA_PATH, train=True, download=True)
        labels = np.array(dataset.targets)
        
        # Initialize data assignments for each client
        client_data_indices = {i: [] for i in range(config.NUM_CLIENTS)}
        
        # Group indices by class for stratified distribution
        class_indices = [np.where(labels == i)[0] for i in range(config.NUM_CLASSES)]
        
        # Calculate minimum samples per class per client for balanced distribution
        min_samples_per_class = config.MIN_SAMPLES_PER_CLIENT // config.NUM_CLASSES
        
        # First phase: Ensure minimum representation of each class for each client
        for client_id in range(config.NUM_CLIENTS):
            client_indices = []
            for class_idx in range(config.NUM_CLASSES):
                # Select samples for this class
                available_indices = class_indices[class_idx]
                if len(available_indices) < min_samples_per_class:
                    raise ValueError(f"Insufficient samples in class {class_idx}")
                    
                selected_indices = np.random.choice(
                    available_indices,
                    size=min_samples_per_class,
                    replace=False
                )
                client_indices.extend(selected_indices)
                
                # Remove selected indices from available pool
                mask = ~np.isin(class_indices[class_idx], selected_indices)
                class_indices[class_idx] = class_indices[class_idx][mask]
            
            client_data_indices[client_id].extend(client_indices)
        
        # Second phase: Distribute remaining samples
        remaining_indices = []
        for class_indices_list in class_indices:
            remaining_indices.extend(class_indices_list)
        
        # Shuffle remaining indices for random distribution
        np.random.shuffle(remaining_indices)
        remaining_per_client = len(remaining_indices) // config.NUM_CLIENTS
        
        # Distribute remaining samples among clients
        for client_id in range(config.NUM_CLIENTS):
            start_idx = client_id * remaining_per_client
            end_idx = start_idx + remaining_per_client
            if client_id == config.NUM_CLIENTS - 1:
                end_idx = len(remaining_indices)  # Last client gets any remainder
            client_data_indices[client_id].extend(remaining_indices[start_idx:end_idx])
        
        # Print distribution statistics for verification
        print("\nData Distribution Statistics:")
        print("-" * 30)
        for client_id, indices in client_data_indices.items():
            class_dist = np.bincount(labels[indices], minlength=config.NUM_CLASSES)
            print(f"Client {client_id}: {len(indices)} samples")
            print(f"Class distribution: {class_dist}")
            
        return client_data_indices

    def setup_test_data(self):
        """
        Set up the test dataset with appropriate transformations and loading parameters.
        This dataset will be used to evaluate model performance after each round.
        """
        # Define standard CIFAR-10 test transformations
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 channel means
                std=(0.2023, 0.1994, 0.2010)    # CIFAR-10 channel standard deviations
            )
        ])
        
        # Load test dataset
        test_dataset = CIFAR10(
            root=config.DATA_PATH,
            train=False,
            download=True,
            transform=transform_test
        )
        
        # Create test data loader optimized for evaluation
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # Optimized for MPS/GPU usage
            pin_memory=True if self.device != torch.device("cpu") else False
        )

    def setup_clients(self):
        global ATTACK_CONFIGURATION
        """Initialize clients with proper attack types."""
        client_data_indices = self.distribute_data()
        self.clients = {}
        
        print("\nInitializing Clients:")
        print("-" * 30)
        
        attack_params = ATTACK_CONFIGURATION
        if not attack_params:
            # Clean setup
            for client_id in range(config.NUM_CLIENTS):
                print(f"Setting up honest client {client_id}")
                self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
            return

        # Setup with attacks
        malicious_clients = set(attack_params['malicious_client_ids'])
        attack_type = attack_params.get('attack_type', '')
        
        for client_id in range(config.NUM_CLIENTS):
            if client_id in malicious_clients:
                if attack_type == 'label_flip':
                    print(f"Setting up label flipper client {client_id}")
                    self.clients[client_id] = LabelFlipperClient(
                        client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'backdoor':
                    print(f"Setting up backdoor client {client_id}")
                    self.clients[client_id] = BackdoorClient(
                        client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'model_replacement':
                    print(f"Setting up model replacement client {client_id}")
                    self.clients[client_id] = ModelReplacementClient(
                        client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'delta':
                    print(f"Setting up delta attack client {client_id}")
                    self.clients[client_id] = DeltaAttackClient(
                        client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'cascade':
                    print(f"Setting up cascade attack client {client_id}")
                    self.clients[client_id] = CascadeAttackClient(
                        client_id, client_data_indices[client_id], attack_params)
            else:
                print(f"Setting up honest client {client_id}")
                self.clients[client_id] = FederatedClient(
                    client_id, client_data_indices[client_id])
                
    def setup_logging(self):
        """Configure logging systems for tracking training progress."""
        # Initialize Weights & Biases logging if enabled
        if config.ENABLE_WANDB:
            wandb.init(project=config.WANDB_PROJECT)
            wandb.watch(self.global_model)
        
        # Set up local file logging
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def aggregate_models(self, local_models, sample_counts):
        """
        Aggregate local models using Federated Averaging (FedAvg) with proper type handling
        and device management.
        
        Args:
            local_models: List of local model state dictionaries
            sample_counts: List of sample counts for each model
            
        Returns:
            OrderedDict: Aggregated model state dictionary
        """
        total_samples = sum(sample_counts)
        aggregated_model = OrderedDict()
        
        # Initialize aggregated model with zeros using correct dtype
        for key in local_models[0].keys():
            param = local_models[0][key]
            aggregated_model[key] = torch.zeros_like(param, dtype=param.dtype)
            
            # Convert to float32 for numeric stability if needed
            if param.dtype == torch.long:
                aggregated_model[key] = aggregated_model[key].to(torch.float32)
        
        # Perform weighted averaging of parameters
        for model_state_dict, samples in zip(local_models, sample_counts):
            weight = samples / total_samples
            for key in model_state_dict.keys():
                param = model_state_dict[key]
                if param.dtype == torch.long:
                    param = param.to(torch.float32)
                aggregated_model[key] += param * weight
        
        # Move aggregated model to appropriate device
        for key in aggregated_model:
            aggregated_model[key] = aggregated_model[key].to(self.device)
        
        return aggregated_model

    def evaluate(self):
        """
        Evaluate the global model on the test dataset.
        
        Returns:
            tuple: (accuracy, average_loss) on the test set
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        samples_count = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.global_model(data)
                total_loss += criterion(output, target).cpu().item() * len(data)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().cpu().item()
                samples_count += len(data)

        accuracy = correct / samples_count
        avg_loss = total_loss / samples_count
        return accuracy, avg_loss

    def train(self):
        """
        Execute the federated learning training process with detailed monitoring
        and progress tracking.
        
        Returns:
            list: Training history containing metrics for each round
        """
        best_accuracy = 0.0
        training_history = []
        
        print("\nStarting Federated Learning Training")
        print("=" * 50)
        
        for round_idx in range(config.NUM_ROUNDS):
            print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")
            print("-" * 30)
            
            # Select random subset of clients for this round
            selected_clients = random.sample(
                list(self.clients.keys()),
                config.CLIENTS_PER_ROUND
            )
            
            print(f"Selected clients for this round: {selected_clients}")
            
            # Train selected clients
            local_models = []
            local_losses = []
            sample_counts = []
            client_performances = {}
            
            for client_id in selected_clients:
                print(f"\nTraining client {client_id}")
                client = self.clients[client_id]
                
                # Transfer global model to client
                global_state = OrderedDict(
                    {k: v.cpu() for k, v in self.global_model.state_dict().items()}
                )
                client.update_local_model(global_state)
                
                # Perform local training
                model_state, loss, samples = client.train_local_model()
                
                # Collect results
                local_models.append(model_state)
                local_losses.append(loss)
                sample_counts.append(samples)
                client_performances[str(client_id)] = {
                    'loss': loss,
                    'samples': samples
                }
                
                print(f"Client {client_id} - Loss: {loss:.4f}, Samples: {samples}")
            
            # Aggregate and update global model
            print("\nAggregating models...")
            global_model_state = self.aggregate_models(local_models, sample_counts)
            self.global_model.load_state_dict(global_model_state)
            
            # Evaluate global model
            print("\nEvaluating global model...")
            accuracy, test_loss = self.evaluate()
            
            # Record metrics for this round
            round_metrics = {
                'round': round_idx + 1,
                'train_loss': np.mean(local_losses),
                'test_loss': test_loss,
                'test_accuracy': accuracy,
                'client_performances': client_performances
            }
            
            training_history.append(round_metrics)
            
            # Print round results
            print("\nRound Results:")
            print(f"Average Training Loss: {round_metrics['train_loss']:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.2%}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"\nNew best accuracy achieved: {best_accuracy:.2%}")
                model_state = OrderedDict(
                    {k: v.cpu() for k, v in self.global_model.state_dict().items()}
                )
                torch.save(
                    model_state,
                    os.path.join(config.MODEL_PATH, "best_model.pth")
                )
            
            # Save training history periodically
            if (round_idx + 1) % config.LOG_INTERVAL == 0:
                with open(self.log_file, 'w') as f:
                    json.dump(training_history, f)
        
        return training_history

def run_experiment(attack_config=None):
    global ATTACK_CONFIGURATION
    """
    Run a complete federated learning experiment with optional attack configuration.
    
    Args:
        attack_config: If provided, enables attacks with given configuration
        
    Returns:
        tuple: (training_history, experiment_config)
    """
    # Create necessary directories
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Update config with attack parameters if provided
    experiment_config = vars(config).copy()
    if attack_config:
        experiment_config.update({'ATTACK_PARAMS': attack_config})

    ATTACK_CONFIGURATION = attack_config
    
    # Initialize and run federated learning
    server = FederatedServer()
    training_history = server.train()
    
    return training_history, experiment_config

def main():
    """Run complete evaluation of federated learning with and without attacks."""
    from evaluation import FederatedLearningEvaluator
    evaluator = FederatedLearningEvaluator()
    
    # Clean training
    print("\n" + "="*50)
    print("Running Clean Federated Learning")
    print("="*50)
    clean_history, clean_config = run_experiment()
    evaluator.add_experiment_results("clean", clean_history, clean_config)
    
    # Label-flipping attack
    print("\n" + "="*50)
    print("Running Label-Flipping Attack")
    print("="*50)
    label_flip_config = config.LABEL_FLIP_CONFIG
    flip_history, flip_config = run_experiment(label_flip_config)
    evaluator.add_experiment_results("label_flip", flip_history, flip_config)
    
    # Backdoor attack
    print("\n" + "="*50)
    print("Running Backdoor Attack")
    print("="*50)
    backdoor_config = config.BACKDOOR_CONFIG
    backdoor_history, backdoor_config = run_experiment(backdoor_config)
    evaluator.add_experiment_results("backdoor", backdoor_history, backdoor_config)

    # Execute model replacement attack scenario
    print("\n" + "="*50)
    print("Running Model Replacement Attack")
    print("="*50)
    attack_config = config.MODEL_REPLACEMENT_CONFIG
    attack_history, attack_config = run_experiment(attack_config)
    evaluator.add_experiment_results("model_replacement", attack_history, attack_config)

    # Execute cascade attack scenario
    print("\n" + "="*50)
    print("Running Cascade Attack")
    print("="*50)
    attack_config = config.CASCADE_ATTACK_CONFIG
    attack_history, attack_config = run_experiment(attack_config)
    evaluator.add_experiment_results("cascade_attack", attack_history, attack_config)

    # Execute delta attack scenario
    print("\n" + "="*50)
    print("Running Delta Attack")
    print("="*50)
    attack_config = config.DELTA_ATTACK_CONFIG
    attack_history, attack_config = run_experiment(attack_config)
    evaluator.add_experiment_results("delta_attack", attack_history, attack_config)
    
    # Generate evaluation
    print("\n" + "="*50)
    print("Generating Evaluation Report")
    print("="*50)
    evaluator.save_all_visualizations()
    report = evaluator.generate_summary_report()
    print("\nSummary Report:")
    print(report)
    impact_metrics = evaluator.compute_attack_impact_metrics()
    print("\nAttack Impact Analysis:")
    print(json.dumps(impact_metrics, indent=2))

if __name__ == "__main__":
    main()