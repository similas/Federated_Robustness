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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config
from client import FederatedClient
from attacks import (LabelFlipperClient, BackdoorClient)
from defenses import (fedavg_aggregate, krum_aggregate, median_aggregate, norm_clipping_aggregate, improved_aaf_aggregate)
from evaluation import FederatedLearningEvaluator
from sklearn.decomposition import PCA

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
        
        # Initialize global model based on configuration
        self.global_model = self._initialize_model()
        self.global_model = self.global_model.to(self.device)
        
        self.setup_test_data()
        self.setup_clients()
        self.setup_logging()
        
        # Initialize AAF components if needed
        if defense_type.lower() == "aaf":
            self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.49, random_state=42)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=0.95)  # For dimensionality reduction
            self.anomaly_scores_history = []
        self.defense_type = defense_type.lower()

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
            else:
                print(f"Setting up honest client {client_id}")
                self.clients[client_id] = FederatedClient(client_id, client_data_indices[client_id])

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
            return krum_aggregate(local_models, sample_counts, self.device)
        elif self.defense_type == "median":
            return median_aggregate(local_models, self.device)
        elif self.defense_type == "norm_clipping":
            return norm_clipping_aggregate(local_models, sample_counts, config.CLIP_THRESHOLD, self.device)
        elif self.defense_type == "aaf":
            return improved_aaf_aggregate(local_models, sample_counts, self.device)
        else:
            print("Unknown defense type. Falling back to FedAvg.")
            return fedavg_aggregate(local_models, sample_counts, self.device)
        
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

def run_experiment(defense_type, attack_config=None):
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
    server = FederatedServer(defense_type)  # Pass defense type to server
    training_history = server.train()
    return training_history, experiment_config

def run_custom_experiments(attacks, defenses):
    """
    Run federated learning experiments for all combinations of specified attacks and defenses.
    
    Args:
        attacks: List of tuples (attack_name, attack_config) to test, e.g., [("label_flip", config.LABEL_FLIP_CONFIG)]
        defenses: List of defense names to test, e.g., ["fedavg", "krum", "median", "norm_clipping", "aaf"]
    
    Returns:
        Dictionary mapping (defense, attack) combinations to final test accuracies.
    """
    # Create results directory
    results_dir = "custom_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Store original defense setting
    original_defense = config.DEFENSE_TYPE
    
    # Initialize results storage
    accuracy_data = {}
    accuracy_drop_data = {}
    
    # Run experiments for each defense-attack combination
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
                    accuracy_data[(defense, attack_name)] = final_acc
                
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
    
    # Calculate accuracy drops (relative to clean for each defense)
    for defense in defenses:
        clean_acc = accuracy_data.get((defense, "clean"), 0.0)
        for attack_name, attack_acc in [(k[1], v) for k, v in accuracy_data.items() if k[0] == defense and k[1] != "clean"]:
            accuracy_drop_data[(defense, attack_name)] = clean_acc - attack_acc if clean_acc > 0 else 0.0
    
    # Generate LaTeX tables
    generate_latex_tables(accuracy_data, accuracy_drop_data, results_dir)
    
    # Restore original defense setting
    config.DEFENSE_TYPE = original_defense
    
    print(f"\nCustom experiments completed. Results saved to {results_dir}")
    return accuracy_data

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
    attacks = [
        ("clean", None),
        # ("label_flip", config.LABEL_FLIP_CONFIG),
        # ("backdoor", config.BACKDOOR_CONFIG)
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

def main():
    # Example usage of run_custom_experiments
    attacks = [
        ("clean", None),
        ("label_flip", config.LABEL_FLIP_CONFIG),
        ("backdoor", config.BACKDOOR_CONFIG)
        # Add more attacks as needed, e.g., ("gradient_poisoning", config.GRADIENT_POISONING_CONFIG)
    ]
    defenses = ["fedavg", "median", "krum", "norm_clipping", "aaf"]
    
    run_custom_experiments(attacks, defenses)
    # Optionally, run AAF-specific experiments
    # run_aaf_experiments()
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()