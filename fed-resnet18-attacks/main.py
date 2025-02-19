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
        for key in local_models[0].keys():
            param = local_models[0][key]
            aggregated_model[key] = torch.zeros_like(param, dtype=param.dtype)
        for model_state, samples in zip(local_models, sample_counts):
            weight = samples / total_samples
            for key in model_state.keys():
                aggregated_model[key] += model_state[key] * weight
        for key in aggregated_model:
            aggregated_model[key] = aggregated_model[key].to(self.device)
        return aggregated_model

    def krum_aggregate(self, local_models):
        num_models = len(local_models)
        flat_models = []
        for state in local_models:
            flat = torch.cat([param.view(-1).float() for param in state.values()])
            flat_models.append(flat)
        distances = torch.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(i + 1, num_models):
                d = torch.norm(flat_models[i] - flat_models[j]) ** 2
                distances[i, j] = d
                distances[j, i] = d
        krum_scores = []
        f = config.KRUM_NEIGHBORS
        for i in range(num_models):
            sorted_dists, _ = torch.sort(distances[i])
            score = torch.sum(sorted_dists[1:f+1])
            krum_scores.append(score)
        best_idx = int(torch.argmin(torch.tensor(krum_scores)))
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
        total_samples = sum(sample_counts)
        for model_state, samples in zip(local_models, sample_counts):
            clipped_state = OrderedDict()
            flat_norm = torch.sqrt(sum(torch.sum(param.float() ** 2) for param in model_state.values()))
            clip_threshold = config.CLIP_THRESHOLD
            for key, param in model_state.items():
                param_float = param.float()
                if flat_norm > clip_threshold:
                    param_float = param_float * (clip_threshold / flat_norm)
                clipped_state[key] = param_float
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

def main():
    from evaluation import FederatedLearningEvaluator
    evaluator = FederatedLearningEvaluator()
    
    # # Clean experiment --- ok
    # print("\n" + "="*50)
    # print("Running Clean Federated Learning")
    # print("="*50)
    # clean_history, clean_config = run_experiment()
    # evaluator.add_experiment_results("clean", clean_history, clean_config)
    
    # # Label-flipping attack experiment
    import time
    start = time.time()
    print("\n" + "="*50)
    print("Running Label-Flipping Attack")
    print("="*50)
    label_flip_config = config.LABEL_FLIP_CONFIG
    flip_history, flip_config = run_experiment(label_flip_config)
    evaluator.add_experiment_results("label_flip", flip_history, flip_config)
    finish = time.time()
    final_time = finish - start
    print("========= FINAL TIME ==========")
    print(final_time)
    print("========= FINAL TIME ==========")
    
    # # Backdoor attack experiment
    # print("\n" + "="*50)
    # print("Running Backdoor Attack")
    # print("="*50)
    # backdoor_config = config.BACKDOOR_CONFIG
    # backdoor_history, backdoor_config = run_experiment(backdoor_config)
    # evaluator.add_experiment_results("backdoor", backdoor_history, backdoor_config)

    # # # Model replacement attack experiment
    # print("\n" + "="*50)
    # print("Running Model Replacement Attack")
    # print("="*50)
    # model_replacement_config = config.MODEL_REPLACEMENT_CONFIG
    # mr_history, mr_config = run_experiment(model_replacement_config)
    # evaluator.add_experiment_results("model_replacement", mr_history, mr_config)

    # Cascade attack experiment --- ok 
    # print("\n" + "="*50)

    # print("Running Cascade Attack")
    # print("="*50)
    # cascade_config = config.CASCADE_ATTACK_CONFIG
    # cascade_history, cascade_config = run_experiment(cascade_config)
    # evaluator.add_experiment_results("cascade_attack", cascade_history, cascade_config)

    # Delta attack experiment --- ok
    # print("\n" + "="*50)
    # print("Running Delta Attack")
    # print("="*50)
    # delta_config = config.DELTA_ATTACK_CONFIG
    # delta_history, delta_config = run_experiment(delta_config)
    # evaluator.add_experiment_results("delta_attack", delta_history, delta_config)
    
    # # Novel attack experiment (constrained optimization attack)  ---- ok
    # print("\n" + "="*50)
    # print("Running Novel (Constrained Optimization) Attack")
    # print("="*50)
    # novel_config = config.NOVEL_ATTACK_CONFIG
    # novel_history, novel_config = run_experiment(novel_config)
    # evaluator.add_experiment_results("novel_attack", novel_history, novel_config)
    
    # Generate evaluation report
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