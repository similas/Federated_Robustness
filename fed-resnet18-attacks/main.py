# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from collections import OrderedDict
import random
import os
from datetime import datetime
import json
import wandb

import config
from client import FederatedClient
from attacks import (LabelFlipperClient, BackdoorClient, ModelReplacementClient,
                     DeltaAttackClient, CascadeAttackClient, NovelAttackClient)

from evaluation import FederatedLearningEvaluator

config.ENABLE_WANDB = False
ATTACK_CONFIGURATION = None

class FederatedServer:
    """
    Central server for federated learning that coordinates client training, robust aggregation,
    and evaluation. Supports multiple robust defenses and model types.
    """
    def __init__(self):
        self.device = config.DEVICE
        print(f"Server initialized with device: {self.device}")
        
        # Initialize global model based on configuration
        self.global_model = self._initialize_model()
        self.global_model = self.global_model.to(self.device)
        
        self.setup_test_data()
        self.setup_clients()
        self.setup_logging()

    def _initialize_model(self):
        # Choose model based on config.MODEL_TYPE
        if config.MODEL_TYPE.lower() == "resnet18":
            from torchvision.models import resnet18
            return resnet18(num_classes=config.NUM_CLASSES)
        elif config.MODEL_TYPE.lower() == "resnet50":
            from torchvision.models import resnet50
            return resnet50(num_classes=config.NUM_CLASSES)
        elif config.MODEL_TYPE.lower() == "vit":
            from torchvision.models import vit_b_16, vit_b_32
            return vit_b_32(num_classes=config.NUM_CLASSES)
        else:
            print("Unknown MODEL_TYPE; defaulting to ResNet-18.")
            from torchvision.models import resnet18
            return resnet18(num_classes=config.NUM_CLASSES)

    def distribute_data(self):
        dataset = CIFAR10(root=config.DATA_PATH, train=True, download=True)
        labels = np.array(dataset.targets)
        client_data_indices = {i: [] for i in range(config.NUM_CLIENTS)}
        class_indices = [np.where(labels == i)[0] for i in range(config.NUM_CLASSES)]
        min_samples_per_class = config.MIN_SAMPLES_PER_CLIENT // config.NUM_CLASSES

        for client_id in range(config.NUM_CLIENTS):
            client_indices = []
            for class_idx in range(config.NUM_CLASSES):
                available_indices = class_indices[class_idx]
                if len(available_indices) < min_samples_per_class:
                    raise ValueError(f"Insufficient samples in class {class_idx}")
                selected_indices = np.random.choice(available_indices, size=min_samples_per_class, replace=False)
                client_indices.extend(selected_indices)
                mask = ~np.isin(class_indices[class_idx], selected_indices)
                class_indices[class_idx] = class_indices[class_idx][mask]
            client_data_indices[client_id].extend(client_indices)
        
        remaining_indices = []
        for lst in class_indices:
            remaining_indices.extend(lst)
        np.random.shuffle(remaining_indices)
        remaining_per_client = len(remaining_indices) // config.NUM_CLIENTS
        for client_id in range(config.NUM_CLIENTS):
            start_idx = client_id * remaining_per_client
            end_idx = start_idx + remaining_per_client
            if client_id == config.NUM_CLIENTS - 1:
                end_idx = len(remaining_indices)
            client_data_indices[client_id].extend(remaining_indices[start_idx:end_idx])
        
        print("\nData Distribution Statistics:")
        print("-" * 30)
        for client_id, indices in client_data_indices.items():
            class_dist = np.bincount(labels[indices], minlength=config.NUM_CLASSES)
            print(f"Client {client_id}: {len(indices)} samples")
            print(f"Class distribution: {class_dist}")
        return client_data_indices

    def setup_test_data(self):
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
        if not attack_params:
            for client_id in range(config.NUM_CLIENTS):
                print(f"Setting up honest client {client_id}")
                self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
            return

        malicious_clients = set(attack_params['malicious_client_ids'])
        attack_type = attack_params.get('attack_type', '')
        
        for client_id in range(config.NUM_CLIENTS):
            if client_id in malicious_clients:
                if attack_type == 'label_flip':
                    print(f"Setting up label flipper client {client_id}")
                    self.clients[client_id] = LabelFlipperClient(client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'backdoor':
                    print(f"Setting up backdoor client {client_id}")
                    self.clients[client_id] = BackdoorClient(client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'model_replacement':
                    print(f"Setting up model replacement client {client_id}")
                    self.clients[client_id] = ModelReplacementClient(client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'delta':
                    print(f"Setting up delta attack client {client_id}")
                    self.clients[client_id] = DeltaAttackClient(client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'cascade':
                    print(f"Setting up cascade attack client {client_id}")
                    self.clients[client_id] = CascadeAttackClient(client_id, client_data_indices[client_id], attack_params)
                elif attack_type == 'novel':
                    print(f"Setting up novel attack client {client_id}")
                    self.clients[client_id] = NovelAttackClient(client_id, client_data_indices[client_id], attack_params)
            else:
                print(f"Setting up honest client {client_id}")
                self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])
                
    def setup_logging(self):
        if config.ENABLE_WANDB:
            wandb.init(project=config.WANDB_PROJECT)
            wandb.watch(self.global_model)
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def fedavg_aggregate(self, local_models, sample_counts):
        total_samples = sum(sample_counts)
        aggregated_model = OrderedDict()
        
        # Iterate over all keys from the first model's state dict.
        for key in local_models[0].keys():
            # If the parameter is floating point, aggregate it.
            if local_models[0][key].dtype.is_floating_point:
                # Initialize an accumulator as float32.
                agg = torch.zeros_like(local_models[0][key], dtype=torch.float32)
                for model_state, samples in zip(local_models, sample_counts):
                    weight = samples / total_samples
                    agg += model_state[key].float() * weight
                # Convert back to the original floating type if needed.
                aggregated_model[key] = agg.to(local_models[0][key].dtype)
            else:
                # For non-floating parameters (e.g. counters), just copy from the first model.
                aggregated_model[key] = local_models[0][key].clone()
        
        # Move all parameters to the proper device.
        for key in aggregated_model:
            aggregated_model[key] = aggregated_model[key].to(self.device)
        return aggregated_model

    def krum_aggregate(self, local_models):
        num_models = len(local_models)
        # If we don't have enough models, fall back to FedAvg
        if num_models <= config.KRUM_NEIGHBORS + 2:
            print(f"Warning: Not enough models for Krum. Falling back to FedAvg.")
            return self.fedavg_aggregate(local_models, [1] * num_models)
            
        flat_models = []
        for state in local_models:
            # Normalize before flattening to ensure fair comparison
            flat = torch.cat([param.view(-1).float() for param in state.values()])
            flat_models.append(flat)
        
        distances = torch.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(i + 1, num_models):
                # Use regular Euclidean distance (not squared)
                d = torch.norm(flat_models[i] - flat_models[j])
                distances[i, j] = d
                distances[j, i] = d
        
        krum_scores = []
        f = min(config.KRUM_NEIGHBORS, num_models - 2)  # Ensure valid f value
        
        for i in range(num_models):
            sorted_dists, _ = torch.sort(distances[i])
            # Sum the distances to f closest neighbors
            score = torch.sum(sorted_dists[1:f+1])
            krum_scores.append(score)
        
        best_idx = int(torch.argmin(torch.tensor(krum_scores)))
        print(f"Krum selected client index: {best_idx}")
        return local_models[best_idx]

    def median_aggregate(self, local_models):
        aggregated_model = OrderedDict()
        for key in local_models[0].keys():
            params = torch.stack([model[key].float() for model in local_models], dim=0)
            median_param = torch.median(params, dim=0)[0]
            aggregated_model[key] = median_param.to(self.device)
        return aggregated_model

    def norm_clipping_aggregate(self, local_models, sample_counts):
        clipped_models = []
        # Compute median norm across all models as a baseline for honest updates
        norms = []
        for model_state in local_models:
            squared_sum = sum(torch.sum(param.float() ** 2).item() for param in model_state.values())
            norms.append(torch.sqrt(torch.tensor(squared_sum)))
        median_norm = torch.median(torch.tensor(norms))
        clip_threshold = median_norm * config.CLIP_THRESHOLD  # Use config.CLIP_THRESHOLD as a multiplier

        for model_state, samples in zip(local_models, sample_counts):
            clipped_state = OrderedDict()
            # Compute model norm
            squared_sum = sum(torch.sum(param.float() ** 2).item() for param in model_state.values())
            flat_norm = torch.sqrt(torch.tensor(squared_sum))
            
            # Only clip if norm exceeds threshold
            if flat_norm > clip_threshold:
                scaling_factor = clip_threshold / (flat_norm + 1e-10)
            else:
                scaling_factor = 1.0  # No clipping if within threshold
            
            # Apply scaling
            for key, param in model_state.items():
                clipped_state[key] = param.float() * scaling_factor
                # Preserve original dtype
                if param.dtype != torch.float32:
                    clipped_state[key] = clipped_state[key].to(dtype=param.dtype)
            clipped_models.append(clipped_state)
        
        return self.fedavg_aggregate(clipped_models, sample_counts)

    def aggregate_models(self, local_models, sample_counts):
        defense = config.DEFENSE_TYPE.lower()
        if defense == "fedavg":
            return self.fedavg_aggregate(local_models, sample_counts)
        elif defense == "krum":
            return self.krum_aggregate(local_models)
        elif defense == "median":
            return self.median_aggregate(local_models)
        elif defense == "norm_clipping":
            return self.norm_clipping_aggregate(local_models, sample_counts)
        else:
            print("Unknown defense type. Falling back to FedAvg.")
            return self.fedavg_aggregate(local_models, sample_counts)
        
    def evaluate(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, samples_count = 0, 0, 0
        with torch.no_grad():
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
        for round_idx in range(config.NUM_ROUNDS):
            print(f"\nRound {round_idx + 1}/{config.NUM_ROUNDS}")
            print("-" * 30)
            selected_clients = random.sample(list(self.clients.keys()), config.CLIENTS_PER_ROUND)
            print(f"Selected clients for this round: {selected_clients}")
            local_models, local_losses, sample_counts = [], [], []
            client_performances = {}
            for client_id in selected_clients:
                print(f"\nTraining client {client_id}")
                client = self.clients[client_id]
                global_state = OrderedDict({k: v.cpu() for k, v in self.global_model.state_dict().items()})
                client.update_local_model(global_state)
                model_state, loss, samples = client.train_local_model()
                local_models.append(model_state)
                local_losses.append(loss)
                sample_counts.append(samples)
                client_performances[str(client_id)] = {'loss': loss, 'samples': samples}
                print(f"Client {client_id} - Loss: {loss:.4f}, Samples: {samples}")
            print("\nAggregating models...")
            global_model_state = self.aggregate_models(local_models, sample_counts)
            self.global_model.load_state_dict(global_model_state)
            print("\nEvaluating global model...")
            accuracy, test_loss = self.evaluate()
            round_metrics = {
                'round': round_idx + 1,
                'train_loss': np.mean(local_losses),
                'test_loss': test_loss,
                'test_accuracy': accuracy,
                'client_performances': client_performances
            }
            training_history.append(round_metrics)
            print("\nRound Results:")
            print(f"Average Training Loss: {round_metrics['train_loss']:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.2%}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"\nNew best accuracy achieved: {best_accuracy:.2%}")
                model_state = OrderedDict({k: v.cpu() for k, v in self.global_model.state_dict().items()})
                torch.save(model_state, os.path.join(config.MODEL_PATH, "best_model.pth"))
            if (round_idx + 1) % config.LOG_INTERVAL == 0:
                with open(self.log_file, 'w') as f:
                    json.dump(training_history, f)
        return training_history

def run_experiment(attack_config=None):
    global ATTACK_CONFIGURATION
    os.makedirs(config.DATA_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    experiment_config = vars(config).copy()
    if attack_config:
        experiment_config.update({'ATTACK_PARAMS': attack_config})
    ATTACK_CONFIGURATION = attack_config
    server = FederatedServer()
    training_history = server.train()
    return training_history, experiment_config

def run_defense_experiments():
    """Run experiments with different defense mechanisms against all attacks"""
    # Define defenses to evaluate
    defenses = ["fedavg", "krum", "median", "norm_clipping"]
    # defenses = ["norm_clipping"]
    
    # Define attacks to evaluate
    attacks = [
        # ("clean", None),
        # ("label_flip", config.LABEL_FLIP_CONFIG),
        # ("backdoor", config.BACKDOOR_CONFIG),
        # ("model_replacement", config.MODEL_REPLACEMENT_CONFIG),
        # ("cascade", config.CASCADE_ATTACK_CONFIG),
        # ("delta", config.DELTA_ATTACK_CONFIG),
        ("novel", config.NOVEL_ATTACK_CONFIG)
    ]
    
    # Create results directory
    results_dir = "defense_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store original defense setting
    original_defense = config.DEFENSE_TYPE
    
    # Set up table data for LaTeX
    accuracy_data = {attack: {} for attack, _ in attacks}
    
    # Run experiments for each defense
    for defense in defenses:
        print(f"\n{'='*50}")
        print(f"Running experiments with {defense.upper()} defense")
        print(f"{'='*50}")
        
        # Update config for this defense
        config.DEFENSE_TYPE = defense
        
        # Create evaluator for this defense
        defense_dir = os.path.join(results_dir, defense)
        os.makedirs(defense_dir, exist_ok=True)
        evaluator = FederatedLearningEvaluator(save_dir=defense_dir)
        
        # Run each attack with this defense
        for attack_name, attack_config in attacks:
            print(f"\n{'-'*50}")
            print(f"Testing {defense} against {attack_name}")
            print(f"{'-'*50}")
            
            try:
                # Run experiment
                history, _ = run_experiment(attack_config)
                
                # Save only the essential data
                serializable_config = {}
                if attack_config:
                    # Create a clean copy without non-serializable objects
                    for key, value in attack_config.items():
                        if isinstance(value, (str, int, float, list, dict, bool)) or value is None:
                            serializable_config[key] = value
                
                # Store results
                experiment_name = f"{defense}_{attack_name}"
                evaluator.add_experiment_results(experiment_name, history, serializable_config)
                
                # Extract final accuracy for table
                if history:
                    final_acc = history[-1]['test_accuracy'] * 100
                    accuracy_data[attack_name][defense] = final_acc
                
                # Save just the history to avoid serialization issues
                with open(f"{defense_dir}/{attack_name}_history.json", 'w') as f:
                    json.dump(history, f, indent=4)
                    
            except Exception as e:
                print(f"Error running {attack_name} with {defense}: {e}")
        
        # Generate report for this defense
        try:
            report = evaluator.generate_summary_report()
            with open(f"{defense_dir}/summary_report.txt", 'w') as f:
                f.write(report)
        except Exception as e:
            print(f"Error generating report for {defense}: {e}")
    
    # Restore original defense setting
    config.DEFENSE_TYPE = original_defense
    
    # Generate LaTeX table
    generate_latex_table(accuracy_data, results_dir)
    
    print(f"\nDefense experiments completed. Results saved to {results_dir}")
    return accuracy_data

def generate_latex_table(accuracy_data, output_dir):
    """Generate LaTeX table from accuracy data"""
    defenses = ["fedavg", "krum", "median", "norm_clipping"]
    
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Final Test Accuracy (\\%) Across Defense Mechanisms and Attacks}\n"
    table += "\\begin{tabular}{l|" + "c" * len(defenses) + "}\n"
    table += "\\hline\n"
    table += "\\textbf{Attack} & " + " & ".join([f"\\textbf{{{d.capitalize()}}}" for d in defenses]) + " \\\\\n"
    table += "\\hline\n"
    
    # Add rows for each attack
    for attack in accuracy_data:
        row = f"{attack.replace('_', ' ').title()} & "
        for defense in defenses:
            if defense in accuracy_data[attack]:
                row += f"{accuracy_data[attack][defense]:.2f} & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"  # Remove last ' & ' and add line end
        table += row
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\label{tab:defense_comparison}\n"
    table += "\\end{table}"
    
    # Save table
    with open(os.path.join(output_dir, "defense_comparison_table.tex"), 'w') as f:
        f.write(table)
def generate_defense_tables(evaluator, output_dir):
    """Generate LaTeX tables comparing defense effectiveness"""
    # Table 1: Test accuracy for each defense against each attack
    table1 = "\\begin{table}[ht]\n"
    table1 += "\\centering\n"
    table1 += "\\caption{Final Test Accuracy (\\%) for Different Defenses Against Attacks}\n"
    table1 += "\\begin{tabular}{l|cccc}\n"
    table1 += "\\hline\n"
    table1 += "\\textbf{Attack} & \\textbf{FedAvg} & \\textbf{Krum} & \\textbf{Median} & \\textbf{Norm Clip.} \\\\\n"
    table1 += "\\hline\n"
    
    # Add rows for each attack
    attacks = ["clean", "label_flip", "backdoor", "model_replacement", "cascade", "delta", "novel"]
    defenses = ["fedavg", "krum", "median", "norm_clipping"]
    
    for attack in attacks:
        row = f"{attack.replace('_', ' ').title()} & "
        for defense in defenses:
            result_name = f"{defense}_{attack}"
            if result_name in evaluator.experiments:
                accuracy = evaluator.experiments[result_name]['results'][-1]['test_accuracy'] * 100
                row += f"{accuracy:.2f} & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"  # Remove last '& ' and add line end
        table1 += row
    
    table1 += "\\hline\n"
    table1 += "\\end{tabular}\n"
    table1 += "\\label{tab:defense_accuracy}\n"
    table1 += "\\end{table}"
    
    # Save LaTeX table to file
    with open(os.path.join(output_dir, "defense_accuracy_table.tex"), 'w') as f:
        f.write(table1)
    
    # Table 2: Attack effectiveness against defenses (accuracy drop)
    table2 = "\\begin{table}[ht]\n"
    table2 += "\\centering\n"
    table2 += "\\caption{Accuracy Drop (\\%) Due to Attacks for Different Defense Mechanisms}\n"
    table2 += "\\begin{tabular}{l|cccc}\n"
    table2 += "\\hline\n"
    table2 += "\\textbf{Attack} & \\textbf{FedAvg} & \\textbf{Krum} & \\textbf{Median} & \\textbf{Norm Clip.} \\\\\n"
    table2 += "\\hline\n"
    
    # Calculate accuracy drops
    for attack in attacks:
        if attack == "clean":
            continue
        row = f"{attack.replace('_', ' ').title()} & "
        for defense in defenses:
            clean_name = f"{defense}_clean"
            attack_name = f"{defense}_{attack}"
            
            if clean_name in evaluator.experiments and attack_name in evaluator.experiments:
                clean_acc = evaluator.experiments[clean_name]['results'][-1]['test_accuracy'] * 100
                attack_acc = evaluator.experiments[attack_name]['results'][-1]['test_accuracy'] * 100
                drop = clean_acc - attack_acc
                row += f"{drop:.2f} & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"
        table2 += row
    
    table2 += "\\hline\n"
    table2 += "\\end{tabular}\n"
    table2 += "\\label{tab:defense_drop}\n"
    table2 += "\\end{table}"
    
    # Save LaTeX table to file
    with open(os.path.join(output_dir, "accuracy_drop_table.tex"), 'w') as f:
        f.write(table2)

def generate_defense_tables(evaluator, output_dir):
    """Generate LaTeX tables comparing defense effectiveness"""
    # Table 1: Test accuracy for each defense against each attack
    table1 = "\\begin{table}[ht]\n"
    table1 += "\\centering\n"
    table1 += "\\caption{Final Test Accuracy (\\%) for Different Defenses Against Attacks}\n"
    table1 += "\\begin{tabular}{l|cccc}\n"
    table1 += "\\hline\n"
    table1 += "\\textbf{Attack} & \\textbf{FedAvg} & \\textbf{Krum} & \\textbf{Median} & \\textbf{Norm Clip.} \\\\\n"
    table1 += "\\hline\n"
    
    # Add rows for each attack
    attacks = ["clean", "label_flip", "backdoor", "model_replacement", "cascade", "delta", "novel"]
    defenses = ["fedavg", "krum", "median", "norm_clipping"]
    
    for attack in attacks:
        row = f"{attack.replace('_', ' ').title()} & "
        for defense in defenses:
            result_name = f"{defense}_{attack}"
            if result_name in evaluator.experiments:
                accuracy = evaluator.experiments[result_name]['results'][-1]['test_accuracy'] * 100
                row += f"{accuracy:.2f} & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"  # Remove last '& ' and add line end
        table1 += row
    
    table1 += "\\hline\n"
    table1 += "\\end{tabular}\n"
    table1 += "\\label{tab:defense_accuracy}\n"
    table1 += "\\end{table}"
    
    # Save LaTeX table to file
    with open(os.path.join(output_dir, "defense_accuracy_table.tex"), 'w') as f:
        f.write(table1)
    
    # Table 2: Attack effectiveness against defenses (accuracy drop)
    table2 = "\\begin{table}[ht]\n"
    table2 += "\\centering\n"
    table2 += "\\caption{Accuracy Drop (\\%) Due to Attacks for Different Defense Mechanisms}\n"
    table2 += "\\begin{tabular}{l|cccc}\n"
    table2 += "\\hline\n"
    table2 += "\\textbf{Attack} & \\textbf{FedAvg} & \\textbf{Krum} & \\textbf{Median} & \\textbf{Norm Clip.} \\\\\n"
    table2 += "\\hline\n"
    
    # Calculate accuracy drops
    for attack in attacks:
        if attack == "clean":
            continue
        row = f"{attack.replace('_', ' ').title()} & "
        for defense in defenses:
            clean_name = f"{defense}_clean"
            attack_name = f"{defense}_{attack}"
            
            if clean_name in evaluator.experiments and attack_name in evaluator.experiments:
                clean_acc = evaluator.experiments[clean_name]['results'][-1]['test_accuracy'] * 100
                attack_acc = evaluator.experiments[attack_name]['results'][-1]['test_accuracy'] * 100
                drop = clean_acc - attack_acc
                row += f"{drop:.2f} & "
            else:
                row += "-- & "
        row = row[:-3] + " \\\\\n"
        table2 += row
    
    table2 += "\\hline\n"
    table2 += "\\end{tabular}\n"
    table2 += "\\label{tab:defense_drop}\n"
    table2 += "\\end{table}"
    
    # Save LaTeX table to file
    with open(os.path.join(output_dir, "accuracy_drop_table.tex"), 'w') as f:
        f.write(table2)

def main():
    run_defense_experiments()
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()