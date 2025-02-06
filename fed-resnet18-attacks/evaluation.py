import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

class FederatedLearningEvaluator:
    """
    A comprehensive evaluation framework for analyzing federated learning experiments
    with and without attacks.
    """
    def __init__(self, save_dir="results"):
        """Initialize the evaluator with a directory for saving results."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.experiments = {}
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def add_experiment_results(self, name, results, config):
        """
        Add results from a federated learning experiment.
        
        Args:
            name: Name of the experiment (e.g., "clean", "label_flip")
            results: Dictionary containing training history
            config: Configuration used for the experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiments[name] = {
            'timestamp': timestamp,
            'results': results,
            'config': config
        }
        
        # Save individual experiment results
        experiment_path = os.path.join(self.save_dir, f"{name}_{timestamp}.json")
        # with open(experiment_path, 'w') as f:
        #     json.dump({
        #         'results': results,
        #         'config': config
        #     }, f, indent=4)
        print(results)
        print("=-=-=-=-=")
        print(config)

    def plot_accuracy_comparison(self):
        """Plot accuracy curves for all experiments for comparison."""
        plt.figure(figsize=(12, 6))
        
        for name, experiment in self.experiments.items():
            rounds = range(1, len(experiment['results']) + 1)
            accuracies = [round_data['test_accuracy'] for round_data in experiment['results']]
            plt.plot(rounds, accuracies, label=name, marker='o')

        plt.xlabel('Round')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(self.save_dir, 'accuracy_comparison.png'))
        plt.close()

    def plot_loss_comparison(self):
        """Plot loss curves for all experiments for comparison."""
        plt.figure(figsize=(12, 6))
        
        for name, experiment in self.experiments.items():
            rounds = range(1, len(experiment['results']) + 1)
            losses = [round_data['test_loss'] for round_data in experiment['results']]
            plt.plot(rounds, losses, label=name, marker='o')

        plt.xlabel('Round')
        plt.ylabel('Test Loss')
        plt.title('Test Loss Comparison Across Experiments')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, 'loss_comparison.png'))
        plt.close()

    def compute_client_performance_statistics(self):
        """Compute and return statistics about client performance."""
        stats = {}
        
        for name, experiment in self.experiments.items():
            client_losses = defaultdict(list)
            
            for round_data in experiment['results']:
                if 'client_performances' in round_data:
                    for client_id, perf in round_data['client_performances'].items():
                        client_losses[client_id].append(perf['loss'])
            
            if client_losses:
                stats[name] = {
                    client_id: {
                        'mean_loss': np.mean(losses),
                        'std_loss': np.std(losses),
                        'min_loss': np.min(losses),
                        'max_loss': np.max(losses)
                    }
                    for client_id, losses in client_losses.items()
                }
        
        return stats

    def generate_summary_report(self):
        """Generate a comprehensive summary report of all experiments."""
        report = []
        report.append("=" * 80)
        report.append("Federated Learning Experiments Summary Report")
        report.append("=" * 80)
        
        for name, experiment in self.experiments.items():
            results = experiment['results']
            config = experiment['config']
            
            report.append(f"\nExperiment: {name}")
            report.append("-" * 40)
            
            # Configuration summary
            report.append("\nConfiguration:")
            report.append(f"- Number of Clients: {config['NUM_CLIENTS']}")
            report.append(f"- Clients Per Round: {config['CLIENTS_PER_ROUND']}")
            report.append(f"- Local Epochs: {config['LOCAL_EPOCHS']}")
            report.append(f"- Batch Size: {config['LOCAL_BATCH_SIZE']}")
            
            if 'ATTACK_PARAMS' in config:
                report.append("\nAttack Configuration:")
                for key, value in config['ATTACK_PARAMS'].items():
                    report.append(f"- {key}: {value}")
            
            # Performance metrics
            accuracies = [round_data['test_accuracy'] for round_data in results]
            losses = [round_data['test_loss'] for round_data in results]
            
            report.append("\nPerformance Metrics:")
            report.append(f"- Final Test Accuracy: {accuracies[-1]:.2%}")
            report.append(f"- Best Test Accuracy: {max(accuracies):.2%}")
            report.append(f"- Average Test Accuracy: {np.mean(accuracies):.2%}")
            report.append(f"- Final Test Loss: {losses[-1]:.4f}")
            report.append(f"- Best Test Loss: {min(losses):.4f}")
            report.append(f"- Average Test Loss: {np.mean(losses):.4f}")
            
            # Convergence analysis
            convergence_round = next((i for i, acc in enumerate(accuracies) 
                                    if acc >= max(accuracies) * 0.95), len(accuracies))
            report.append(f"\nConvergence Analysis:")
            report.append(f"- Rounds to 95% of Best Accuracy: {convergence_round}")
            
            report.append("\n" + "-" * 80)
        
        # Save report
        report_path = os.path.join(self.save_dir, "summary_report.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        return "\n".join(report)

    def plot_client_loss_distributions(self):
        """Plot loss distributions for honest vs. malicious clients."""
        for name, experiment in self.experiments.items():
            if 'ATTACK_PARAMS' in experiment['config']:
                malicious_clients = set(experiment['config']['ATTACK_PARAMS']['malicious_client_ids'])
                
                honest_losses = []
                malicious_losses = []
                
                for round_data in experiment['results']:
                    if 'client_performances' in round_data:
                        for client_id, perf in round_data['client_performances'].items():
                            if int(client_id) in malicious_clients:
                                malicious_losses.append(perf['loss'])
                            else:
                                honest_losses.append(perf['loss'])
                
                if honest_losses and malicious_losses:
                    plt.figure(figsize=(10, 6))
                    plt.hist(honest_losses, alpha=0.5, label='Honest Clients', bins=20)
                    plt.hist(malicious_losses, alpha=0.5, label='Malicious Clients', bins=20)
                    plt.xlabel('Loss')
                    plt.ylabel('Frequency')
                    plt.title(f'Loss Distribution - {name}')
                    plt.legend()
                    
                    plt.savefig(os.path.join(self.save_dir, f'loss_distribution_{name}.png'))
                    plt.close()

    def compute_attack_impact_metrics(self):
        """Compute metrics that quantify the impact of attacks."""
        if 'clean' not in self.experiments:
            return "No clean experiment found for comparison"
            
        impact_metrics = {}
        clean_results = self.experiments['clean']['results']
        clean_final_acc = clean_results[-1]['test_accuracy']
        clean_best_acc = max(r['test_accuracy'] for r in clean_results)
        
        for name, experiment in self.experiments.items():
            if name == 'clean':
                continue
                
            results = experiment['results']
            final_acc = results[-1]['test_accuracy']
            best_acc = max(r['test_accuracy'] for r in results)
            
            impact_metrics[name] = {
                'accuracy_degradation': {
                    'final': clean_final_acc - final_acc,
                    'best': clean_best_acc - best_acc,
                    'relative_final': (clean_final_acc - final_acc) / clean_final_acc,
                    'relative_best': (clean_best_acc - best_acc) / clean_best_acc
                },
                'convergence_delay': None
            }
            
            # Calculate convergence delay
            clean_conv = next((i for i, r in enumerate(clean_results) 
                             if r['test_accuracy'] >= 0.95 * clean_best_acc), len(clean_results))
            attack_conv = next((i for i, r in enumerate(results) 
                              if r['test_accuracy'] >= 0.95 * best_acc), len(results))
            
            impact_metrics[name]['convergence_delay'] = attack_conv - clean_conv
            
        return impact_metrics

    def save_all_visualizations(self):
        """Generate and save all visualizations in one go."""
        self.plot_accuracy_comparison()
        self.plot_loss_comparison()
        self.plot_client_loss_distributions()
        
        # Save impact metrics
        impact_metrics = self.compute_attack_impact_metrics()
        if isinstance(impact_metrics, dict):
            metrics_path = os.path.join(self.save_dir, "attack_impact_metrics.json")
            # with open(metrics_path, 'w') as f:
            #     json.dump(impact_metrics, f, indent=4)
            print(impact_metrics)