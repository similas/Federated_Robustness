import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pandas as pd
import config as CONFIG

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
        
        # Use class names from configuration
        self.class_names = CONFIG.CLASS_NAMES
        
        # Create subdirectories for different types of outputs
        self.plots_dir = os.path.join(save_dir, "plots")
        self.tables_dir = os.path.join(save_dir, "tables")
        self.metrics_dir = os.path.join(save_dir, "metrics")
        
        for directory in [self.plots_dir, self.tables_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize comparison metrics
        self.comparison_metrics = {}

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
        try:
            # Create a serializable version of the results
            serializable_results = self._make_serializable(results)
            serializable_config = self._make_serializable(config)
            
            with open(experiment_path, 'w') as f:
                json.dump({
                    'results': serializable_results,
                    'config': serializable_config
                }, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save experiment {name} to JSON: {str(e)}")
            # Try a more aggressive approach to make it serializable
            try:
                simplified_results = []
                for round_data in results:
                    simplified_round = {}
                    for key, value in round_data.items():
                        if key != 'client_performances':  # Skip complex nested structures
                            simplified_round[key] = self._make_serializable(value)
                    simplified_results.append(simplified_round)
                
                with open(experiment_path, 'w') as f:
                    json.dump({
                        'results': simplified_results,
                        'config': self._make_serializable(config)
                    }, f, indent=4)
                print(f"Saved simplified results for {name}")
            except Exception as e2:
                print(f"Error saving even simplified results: {str(e2)}")
        
        print(f"Added experiment: {name}")
        
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        else:
            return obj

    def plot_accuracy_comparison(self, save=True, show=False):
        """
        Plot accuracy curves for all experiments for comparison.
        
        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if not self.experiments:
            print("No experiments found to plot accuracy comparison.")
            return
            
        plt.figure(figsize=(12, 7))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        legend_added = False
        for i, (name, experiment) in enumerate(self.experiments.items()):
            if 'results' not in experiment or not experiment['results']:
                print(f"Experiment '{name}' has no results to plot.")
                continue
                
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            rounds = [r+1 for r in range(len(experiment['results']))]
            accuracies = [round_data.get('test_accuracy', 0) * 100 for round_data in experiment['results']]
            
            plt.plot(rounds, accuracies, label=name, marker=marker, 
                     color=color, linewidth=2, markersize=8, alpha=0.8)
            legend_added = True

        if not legend_added:
            plt.text(0.5, 0.5, "No data available to plot", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
        
        plt.xlabel('Communication Round', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)
        plt.title('Test Accuracy Comparison Across Experiments', fontsize=16)
        
        if legend_added:
            plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plots_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.plots_dir, 'accuracy_comparison.pdf'), bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_loss_comparison(self, save=True, show=False):
        """
        Plot loss curves for all experiments for comparison.
        
        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if not self.experiments:
            print("No experiments found to plot loss comparison.")
            return
            
        plt.figure(figsize=(12, 7))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        legend_added = False
        for i, (name, experiment) in enumerate(self.experiments.items()):
            if 'results' not in experiment or not experiment['results']:
                print(f"Experiment '{name}' has no results to plot.")
                continue
                
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            rounds = [r+1 for r in range(len(experiment['results']))]
            losses = [round_data.get('test_loss', 0) for round_data in experiment['results']]
            
            plt.plot(rounds, losses, label=name, marker=marker, 
                     color=color, linewidth=2, markersize=8, alpha=0.8)
            legend_added = True

        if not legend_added:
            plt.text(0.5, 0.5, "No data available to plot", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)
        
        plt.xlabel('Communication Round', fontsize=14)
        plt.ylabel('Test Loss', fontsize=14)
        plt.title('Test Loss Comparison Across Experiments', fontsize=16)
        
        if legend_added:
            plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plots_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.plots_dir, 'loss_comparison.pdf'), bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_attack_impact(self, baseline_experiment="clean", save=True, show=False):
        """
        Plot the impact of attacks compared to a baseline experiment.
        
        Args:
            baseline_experiment: Name of the experiment to use as baseline (typically "clean")
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        if baseline_experiment not in self.experiments:
            print(f"Baseline experiment '{baseline_experiment}' not found.")
            return
        
        if 'results' not in self.experiments[baseline_experiment] or not self.experiments[baseline_experiment]['results']:
            print(f"Baseline experiment '{baseline_experiment}' has no results.")
            return
            
        plt.figure(figsize=(12, 7))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        baseline_results = self.experiments[baseline_experiment]['results']
        baseline_accuracies = [r.get('test_accuracy', 0) * 100 for r in baseline_results]
        
        # Sort experiments by final accuracy degradation (worst first)
        sorted_experiments = []
        for name, experiment in self.experiments.items():
            if name == baseline_experiment:
                continue
            
            if 'results' not in experiment or not experiment['results']:
                continue
                
            final_accuracy = experiment['results'][-1].get('test_accuracy', 0) * 100
            baseline_final = baseline_accuracies[-1]
            degradation = baseline_final - final_accuracy
            sorted_experiments.append((name, experiment, degradation))
            
        if not sorted_experiments:
            print("No valid experiments to compare with baseline.")
            return
            
        sorted_experiments.sort(key=lambda x: x[2], reverse=True)
        
        # Plot baseline
        rounds = [r+1 for r in range(len(baseline_results))]
        plt.plot(rounds, baseline_accuracies, label=f"{baseline_experiment} (baseline)", 
                 color='black', linewidth=3, linestyle='-', marker='o', markersize=8)
        
        # Plot other experiments
        for i, (name, experiment, degradation) in enumerate(sorted_experiments):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            rounds = [r+1 for r in range(len(experiment['results']))]
            accuracies = [r.get('test_accuracy', 0) * 100 for r in experiment['results']]
            
            plt.plot(rounds, accuracies, label=f"{name} (-{degradation:.1f}%)", 
                     color=color, linewidth=2, marker=marker, markersize=8, alpha=0.8)

        plt.xlabel('Communication Round', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)
        plt.title('Attack Impact on Test Accuracy', fontsize=16)
        plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plots_dir, 'attack_impact.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.plots_dir, 'attack_impact.pdf'), bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_defense_effectiveness(self, attack_type, save=True, show=False):
        """
        Plot the effectiveness of different defenses against a specific attack.
        
        Args:
            attack_type: The attack to analyze (e.g., "label_flip")
            save: Whether to save the plot to disk
            show: Whether to display the plot
        """
        # Identify all experiments with this attack across different defenses
        defense_results = {}
        
        for name, experiment in self.experiments.items():
            if 'results' not in experiment or not experiment['results']:
                continue
                
            # Parse experiment name to extract defense and attack
            parts = name.split('_')
            if len(parts) >= 2:
                defense = parts[0]
                attack = '_'.join(parts[1:])
                
                if attack == attack_type:
                    defense_results[defense] = experiment
        
        if not defense_results:
            print(f"No experiments found for attack '{attack_type}'")
            return
            
        plt.figure(figsize=(12, 7))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Sort defenses by final accuracy (best first)
        sorted_defenses = []
        for defense, experiment in defense_results.items():
            final_accuracy = experiment['results'][-1].get('test_accuracy', 0) * 100
            sorted_defenses.append((defense, experiment, final_accuracy))
            
        sorted_defenses.sort(key=lambda x: x[2], reverse=True)
        
        # Plot each defense
        for i, (defense, experiment, final_acc) in enumerate(sorted_defenses):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            rounds = [r+1 for r in range(len(experiment['results']))]
            accuracies = [r.get('test_accuracy', 0) * 100 for r in experiment['results']]
            
            plt.plot(rounds, accuracies, label=f"{defense} ({final_acc:.1f}%)", 
                     color=color, linewidth=2, marker=marker, markersize=8, alpha=0.8)

        plt.xlabel('Communication Round', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)
        plt.title(f'Defense Effectiveness Against {attack_type} Attack', fontsize=16)
        plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plots_dir, f'defense_against_{attack_type}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.plots_dir, f'defense_against_{attack_type}.pdf'), bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_client_loss_distributions(self, save=True, show=False):
        """
        Plot loss distributions for honest vs. malicious clients.
        
        Args:
            save: Whether to save the plots to disk
            show: Whether to display the plots
        """
        for name, experiment in self.experiments.items():
            if 'config' not in experiment or not experiment['config'] or 'results' not in experiment or not experiment['results']:
                continue
                
            attack_params = experiment['config'].get('ATTACK_PARAMS', None)
            if not attack_params:
                continue
                
            malicious_clients = set(attack_params.get('malicious_client_ids', []))
            if not malicious_clients:
                continue
                
            honest_losses = []
            malicious_losses = []
            
            for round_data in experiment['results']:
                if 'client_performances' in round_data:
                    for client_id, perf in round_data['client_performances'].items():
                        try:
                            client_id_int = int(client_id)
                            if client_id_int in malicious_clients:
                                malicious_losses.append(perf.get('loss', 0))
                            else:
                                honest_losses.append(perf.get('loss', 0))
                        except (ValueError, TypeError):
                            # Skip if client_id can't be converted to int
                            continue
            
            if not honest_losses or not malicious_losses:
                print(f"Insufficient data for experiment '{name}' to plot loss distributions")
                continue
                
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            try:
                # Plot histograms
                plt.subplot(1, 2, 1)
                plt.hist(honest_losses, alpha=0.7, label='Honest Clients', bins=20, color='blue')
                plt.hist(malicious_losses, alpha=0.7, label='Malicious Clients', bins=20, color='red')
                plt.xlabel('Loss', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title(f'Loss Distribution - {name}', fontsize=14)
                plt.legend(fontsize=10)
                
                # Plot KDE with updated parameter (fill instead of shade)
                plt.subplot(1, 2, 2)
                sns.kdeplot(honest_losses, label='Honest Clients', fill=True, color='blue')
                sns.kdeplot(malicious_losses, label='Malicious Clients', fill=True, color='red')
                plt.xlabel('Loss', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.title('Kernel Density Estimate', fontsize=14)
                plt.legend(fontsize=10)
                
                plt.tight_layout()
                
                if save:
                    plt.savefig(os.path.join(self.plots_dir, f'loss_distribution_{name}.png'), dpi=300, bbox_inches='tight')
                    plt.savefig(os.path.join(self.plots_dir, f'loss_distribution_{name}.pdf'), bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
            except Exception as e:
                print(f"Error plotting loss distribution for experiment '{name}': {str(e)}")
                plt.close()


    def compute_attack_impact_metrics(self, baseline_experiment="clean"):
        """
        Compute metrics that quantify the impact of attacks compared to a baseline.
        
        Args:
            baseline_experiment: Name of the experiment to use as baseline (typically "clean")
            
        Returns:
            Dictionary of impact metrics for each attack
        """
        impact_metrics = {}
        
        if baseline_experiment not in self.experiments:
            print(f"Warning: No '{baseline_experiment}' experiment found for comparison")
            return {"error": f"No {baseline_experiment} experiment found for comparison"}
            
        clean_results = self.experiments[baseline_experiment].get('results', [])
        
        if not clean_results:
            print(f"Warning: No results in '{baseline_experiment}' experiment")
            return {"error": f"No results in {baseline_experiment} experiment"}
            
        clean_final_acc = clean_results[-1].get('test_accuracy', 0)
        clean_best_acc = max(r.get('test_accuracy', 0) for r in clean_results)
        
        # Find convergence round for baseline
        clean_convergence_threshold = 0.95 * clean_best_acc
        clean_conv_round = next((i for i, r in enumerate(clean_results) 
                              if r.get('test_accuracy', 0) >= clean_convergence_threshold), 
                             len(clean_results))
        
        for name, experiment in self.experiments.items():
            if name == baseline_experiment:
                continue
                
            results = experiment.get('results', [])
            if not results:
                print(f"Warning: No results in '{name}' experiment")
                continue
                
            # Extract defense and attack names
            name_parts = name.split('_')
            if len(name_parts) < 2:
                # Not a defense_attack format, skip
                continue
                
            defense = name_parts[0]
            attack = '_'.join(name_parts[1:])
            
            final_acc = results[-1].get('test_accuracy', 0)
            best_acc = max(r.get('test_accuracy', 0) for r in results)
            
            # Calculate accuracy degradation
            absolute_final_degradation = clean_final_acc - final_acc
            absolute_best_degradation = clean_best_acc - best_acc
            
            relative_final_degradation = absolute_final_degradation / clean_final_acc if clean_final_acc > 0 else 0
            relative_best_degradation = absolute_best_degradation / clean_best_acc if clean_best_acc > 0 else 0
            
            # Calculate convergence delay
            experiment_convergence_threshold = 0.95 * best_acc
            experiment_conv_round = next((i for i, r in enumerate(results)
                                      if r.get('test_accuracy', 0) >= experiment_convergence_threshold),
                                     len(results))
            
            convergence_delay = experiment_conv_round - clean_conv_round
            
            # Create the metric entry
            if attack not in impact_metrics:
                impact_metrics[attack] = {}
                
            impact_metrics[attack][defense] = {
                'absolute_final_degradation': float(absolute_final_degradation),
                'absolute_best_degradation': float(absolute_best_degradation),
                'relative_final_degradation': float(relative_final_degradation),
                'relative_best_degradation': float(relative_best_degradation),
                'convergence_delay': int(convergence_delay),
                'final_accuracy': float(final_acc),
                'best_accuracy': float(best_acc)
            }
        
        # Save the metrics only if we have data
        if impact_metrics:
            metrics_path = os.path.join(self.metrics_dir, "attack_impact_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(impact_metrics, f, indent=4)
                
        self.comparison_metrics = impact_metrics
        return impact_metrics

    def generate_defense_comparison_tables(self, baseline_experiment="clean"):
        """
        Generate tables comparing defense effectiveness against different attacks.
        
        Args:
            baseline_experiment: Name of the experiment to use as baseline
            
        Returns:
            Dictionary with LaTeX tables and other formats
        """
        metrics = self.compute_attack_impact_metrics(baseline_experiment)
        
        if isinstance(metrics, dict) and "error" in metrics:
            return {"error": metrics["error"]}
            
        if not metrics:
            return {"error": "No metrics data available to generate tables"}
            
        # Generate attack vs defense tables
        tables = {}
        
        # Table 1: Final accuracy across defenses and attacks
        attack_names = sorted(metrics.keys())
        defense_names = set()
        for attack_metrics in metrics.values():
            defense_names.update(attack_metrics.keys())
        defense_names = sorted(defense_names)
        
        if not attack_names or not defense_names:
            return {"error": "No valid attack or defense data found"}
        
        # LaTeX table for final accuracy
        accuracy_table = "\\begin{table}[htbp]\n"
        accuracy_table += "\\centering\n"
        accuracy_table += "\\caption{Final Test Accuracy (\\%) Across Defense Mechanisms and Attacks}\n"
        accuracy_table += "\\begin{tabular}{l|" + "c" * len(attack_names) + "}\n"
        accuracy_table += "\\hline\n"
        
        # Header row
        accuracy_table += "\\textbf{Defense} & " + " & ".join([f"\\textbf{{{a.replace('_', ' ').title()}}}" for a in attack_names]) + " \\\\\n"
        accuracy_table += "\\hline\n"
        
        # Defense rows
        for defense in defense_names:
            row = f"{defense.capitalize()} & "
            for attack in attack_names:
                if attack in metrics and defense in metrics[attack]:
                    accuracy = metrics[attack][defense]['final_accuracy'] * 100
                    row += f"{accuracy:.2f} & "
                else:
                    row += "-- & "
            row = row[:-3] + " \\\\\n"  # Remove last ' & ' and add line break
            accuracy_table += row
            
        accuracy_table += "\\hline\n"
        accuracy_table += "\\end{tabular}\n"
        accuracy_table += "\\label{tab:defense_accuracy}\n"
        accuracy_table += "\\end{table}"
        
        tables["accuracy_latex"] = accuracy_table
        
        # LaTeX table for accuracy degradation
        degradation_table = "\\begin{table}[htbp]\n"
        degradation_table += "\\centering\n"
        degradation_table += "\\caption{Accuracy Degradation (\\%) Due to Attacks for Different Defense Mechanisms}\n"
        degradation_table += "\\begin{tabular}{l|" + "c" * len(attack_names) + "}\n"
        degradation_table += "\\hline\n"
        
        # Header row
        degradation_table += "\\textbf{Defense} & " + " & ".join([f"\\textbf{{{a.replace('_', ' ').title()}}}" for a in attack_names]) + " \\\\\n"
        degradation_table += "\\hline\n"
        
        # Defense rows
        for defense in defense_names:
            row = f"{defense.capitalize()} & "
            for attack in attack_names:
                if attack in metrics and defense in metrics[attack]:
                    degradation = metrics[attack][defense]['absolute_final_degradation'] * 100
                    row += f"{degradation:.2f} & "
                else:
                    row += "-- & "
            row = row[:-3] + " \\\\\n"  # Remove last ' & ' and add line break
            degradation_table += row
            
        degradation_table += "\\hline\n"
        degradation_table += "\\end{tabular}\n"
        degradation_table += "\\label{tab:defense_degradation}\n"
        degradation_table += "\\end{table}"
        
        tables["degradation_latex"] = degradation_table
        
        # Also save as CSV for easy import into other tools
        accuracy_csv = f"Defense,{','.join(attack_names)}\n"
        for defense in defense_names:
            row = f"{defense},"
            for attack in attack_names:
                if attack in metrics and defense in metrics[attack]:
                    accuracy = metrics[attack][defense]['final_accuracy'] * 100
                    row += f"{accuracy:.2f},"
                else:
                    row += ","
            accuracy_csv += row[:-1] + "\n"  # Remove last comma
            
        tables["accuracy_csv"] = accuracy_csv
        
        # Save tables to files
        with open(os.path.join(self.tables_dir, "defense_accuracy_table.tex"), 'w') as f:
            f.write(accuracy_table)
            
        with open(os.path.join(self.tables_dir, "defense_degradation_table.tex"), 'w') as f:
            f.write(degradation_table)
            
        with open(os.path.join(self.tables_dir, "defense_accuracy.csv"), 'w') as f:
            f.write(accuracy_csv)
            
        return tables

    def generate_client_performance_statistics(self):
        """
        Compute and analyze client performance statistics.
        
        Returns:
            Dictionary of client performance metrics
        """
        stats = {}
        
        for name, experiment in self.experiments.items():
            if 'results' not in experiment or not experiment['results']:
                continue
                
            client_losses = defaultdict(list)
            client_samples = defaultdict(list)
            
            for round_data in experiment['results']:
                if 'client_performances' in round_data:
                    for client_id, perf in round_data['client_performances'].items():
                        try:
                            loss = perf.get('loss', 0)
                            samples = perf.get('samples', 0)
                            # Validate data
                            if not isinstance(loss, (int, float)) or not isinstance(samples, (int, float)):
                                continue
                            client_losses[client_id].append(loss)
                            client_samples[client_id].append(samples)
                        except:
                            continue
            
            if not client_losses:
                continue
                
            # Extract attack configuration if available
            attack_config = None
            if 'config' in experiment and experiment['config']:
                attack_config = experiment['config'].get('ATTACK_PARAMS', None)
                
            malicious_clients = set()
            if attack_config:
                malicious_clients = set(attack_config.get('malicious_client_ids', []))
            
            # Compute statistics
            client_stats = {}
            for client_id, losses in client_losses.items():
                try:
                    client_id_int = int(client_id)
                    is_malicious = client_id_int in malicious_clients
                    
                    if not losses:
                        continue
                        
                    client_stats[client_id] = {
                        'is_malicious': is_malicious,
                        'mean_loss': float(np.mean(losses)),
                        'std_loss': float(np.std(losses)),
                        'min_loss': float(np.min(losses)),
                        'max_loss': float(np.max(losses)),
                        'loss_trend': float(losses[-1] - losses[0]) if len(losses) > 1 else 0,
                        'mean_samples': float(np.mean(client_samples[client_id])) if client_id in client_samples and client_samples[client_id] else 0
                    }
                except (ValueError, TypeError, IndexError):
                    continue
                
            # Skip if no valid client stats
            if not client_stats:
                continue
                
            # Aggregate statistics by client type
            honest_losses = [stats['mean_loss'] for cid, stats in client_stats.items() 
                           if not stats['is_malicious']]
            malicious_losses = [stats['mean_loss'] for cid, stats in client_stats.items() 
                              if stats['is_malicious']]
                
            honest_samples = [stats['mean_samples'] for cid, stats in client_stats.items() 
                            if not stats['is_malicious']]
            malicious_samples = [stats['mean_samples'] for cid, stats in client_stats.items() 
                               if stats['is_malicious']]
            
            # Skip if no valid categorization 
            if not honest_losses and not malicious_losses:
                continue
                
            # Compute aggregate statistics
            aggregated_stats = {
                'client_count': len(client_stats),
                'honest_count': len(honest_losses),
                'malicious_count': len(malicious_losses),
                'honest_mean_loss': float(np.mean(honest_losses)) if honest_losses else 0,
                'malicious_mean_loss': float(np.mean(malicious_losses)) if malicious_losses else 0,
                'honest_mean_samples': float(np.mean(honest_samples)) if honest_samples else 0,
                'malicious_mean_samples': float(np.mean(malicious_samples)) if malicious_samples else 0,
                'loss_difference': float(np.mean(malicious_losses) - np.mean(honest_losses)) 
                                  if honest_losses and malicious_losses else 0,
                'samples_ratio': float(np.mean(malicious_samples) / np.mean(honest_samples)) 
                               if honest_samples and malicious_samples and np.mean(honest_samples) > 0 else 0,
                'per_client': client_stats
            }
            
            stats[name] = aggregated_stats
        
        # Save statistics if we have data
        if stats:
            with open(os.path.join(self.metrics_dir, "client_performance_statistics.json"), 'w') as f:
                json.dump(stats, f, indent=4)
            
        return stats

    def plot_client_statistics(self, save=True, show=False):
        """
        Create plots visualizing client statistics with better error handling.
        """
        try:
            stats = self.generate_client_performance_statistics()
            
            if not stats:
                print("No client statistics available to plot.")
                return
                
            # Plot 1: Loss comparison between honest and malicious clients
            plt.figure(figsize=(14, 8))
            
            experiment_names = []
            honest_losses = []
            malicious_losses = []
            
            for name, experiment_stats in stats.items():
                if experiment_stats.get('honest_count', 0) > 0 and experiment_stats.get('malicious_count', 0) > 0:
                    experiment_names.append(name)
                    honest_losses.append(experiment_stats['honest_mean_loss'])
                    malicious_losses.append(experiment_stats['malicious_mean_loss'])
            
            if not experiment_names:
                print("No experiments with both honest and malicious clients found.")
                return
                
            x = np.arange(len(experiment_names))
            width = 0.35
            
            plt.subplot(2, 1, 1)
            plt.bar(x - width/2, honest_losses, width, label='Honest Clients', color='blue', alpha=0.7)
            plt.bar(x + width/2, malicious_losses, width, label='Malicious Clients', color='red', alpha=0.7)
            
            plt.xlabel('Experiment', fontsize=12)
            plt.ylabel('Mean Loss', fontsize=12)
            plt.title('Loss Comparison: Honest vs Malicious Clients', fontsize=14)
            plt.xticks(x, experiment_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot 2: Sample count comparison
            honest_samples = []
            malicious_samples = []
            
            for name in experiment_names:
                honest_samples.append(stats[name]['honest_mean_samples'])
                malicious_samples.append(stats[name]['malicious_mean_samples'])
            
            plt.subplot(2, 1, 2)
            plt.bar(x - width/2, honest_samples, width, label='Honest Clients', color='blue', alpha=0.7)
            plt.bar(x + width/2, malicious_samples, width, label='Malicious Clients', color='red', alpha=0.7)
            
            plt.xlabel('Experiment', fontsize=12)
            plt.ylabel('Mean Samples', fontsize=12)
            plt.title('Sample Count Comparison: Honest vs Malicious Clients', fontsize=14)
            plt.xticks(x, experiment_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.plots_dir, 'client_comparison.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(self.plots_dir, 'client_comparison.pdf'), bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            # Plot individual client statistics for each experiment
            for name, experiment_stats in stats.items():
                if 'per_client' not in experiment_stats or not experiment_stats['per_client']:
                    continue
                    
                plt.figure(figsize=(14, 10))
                
                client_ids = []
                client_losses = []
                client_samples = []
                colors = []
                
                for client_id, client_stats in experiment_stats['per_client'].items():
                    client_ids.append(client_id)
                    client_losses.append(client_stats['mean_loss'])
                    client_samples.append(client_stats['mean_samples'])
                    colors.append('red' if client_stats['is_malicious'] else 'blue')
                
                if not client_ids:
                    continue
                
                try:
                    # Sort by client ID for better visualization
                    sorted_indices = [i for i, _ in sorted(enumerate(client_ids), key=lambda x: int(x[1]))]
                    client_ids = [client_ids[i] for i in sorted_indices]
                    client_losses = [client_losses[i] for i in sorted_indices]
                    client_samples = [client_samples[i] for i in sorted_indices]
                    colors = [colors[i] for i in sorted_indices]
                except (ValueError, TypeError):
                    # If sorting fails, continue with unsorted data
                    pass
                
                # Plot client losses
                plt.subplot(2, 1, 1)
                bars = plt.bar(client_ids, client_losses, color=colors, alpha=0.7)
                
                # Add a legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', alpha=0.7, label='Honest'),
                    Patch(facecolor='red', alpha=0.7, label='Malicious')
                ]
                plt.legend(handles=legend_elements)
                
                plt.xlabel('Client ID', fontsize=12)
                plt.ylabel('Mean Loss', fontsize=12)
                plt.title(f'Client Losses - {name}', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Plot client sample counts
                plt.subplot(2, 1, 2)
                plt.bar(client_ids, client_samples, color=colors, alpha=0.7)
                plt.legend(handles=legend_elements)
                
                plt.xlabel('Client ID', fontsize=12)
                plt.ylabel('Mean Samples', fontsize=12)
                plt.title('Client Sample Counts', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                if save:
                    plt.savefig(os.path.join(self.plots_dir, f'client_stats_{name}.png'), dpi=300, bbox_inches='tight')
                    plt.savefig(os.path.join(self.plots_dir, f'client_stats_{name}.pdf'), bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
        
        except Exception as e:
            print(f"Error generating client statistics plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_convergence_analysis(self, save=True, show=False):
        """
        Analyze and visualize convergence properties across experiments.
        
        Args:
            save: Whether to save the plots to disk
            show: Whether to display the plots
        """
        try:
            # Extract convergence metrics
            convergence_data = {}
            
            for name, experiment in self.experiments.items():
                if 'results' not in experiment or not experiment['results']:
                    continue
                    
                results = experiment['results']
                
                # Calculate best accuracy and convergence round
                accuracies = [r.get('test_accuracy', 0) for r in results]
                best_accuracy = max(accuracies)
                convergence_threshold = 0.95 * best_accuracy
                
                convergence_round = next((i+1 for i, acc in enumerate(accuracies) 
                                    if acc >= convergence_threshold), len(accuracies))
                
                convergence_data[name] = {
                    'best_accuracy': best_accuracy,
                    'final_accuracy': accuracies[-1],
                    'convergence_round': convergence_round,
                    'stability': np.std(accuracies[convergence_round-1:]) if convergence_round < len(accuracies) else 0
                }
            
            if not convergence_data:
                print("No convergence data available to plot.")
                return
                
            # Plot 1: Convergence rounds comparison
            plt.figure(figsize=(14, 10))
            
            # Sort experiments by convergence round
            sorted_experiments = sorted(convergence_data.items(), key=lambda x: x[1]['convergence_round'])
            names = [item[0] for item in sorted_experiments]
            convergence_rounds = [item[1]['convergence_round'] for item in sorted_experiments]
            best_accuracies = [item[1]['best_accuracy'] * 100 for item in sorted_experiments]
            
            # Create bar chart with color gradient based on best accuracy
            plt.subplot(2, 1, 1)
            bars = plt.bar(names, convergence_rounds, alpha=0.7)
            
            # Color bars based on best accuracy
            cmap = plt.cm.viridis
            for i, bar in enumerate(bars):
                bar.set_color(cmap(best_accuracies[i]/100))
            
            plt.xlabel('Experiment', fontsize=12)
            plt.ylabel('Convergence Round', fontsize=12)
            plt.title('Rounds to Convergence (95% of Best Accuracy)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add a colorbar with explicit axes
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(best_accuracies), max(best_accuracies)))
            sm.set_array([])
            plt.tight_layout()  # Apply tight layout before adding colorbar
            plt.subplots_adjust(right=0.85)  # Make room for colorbar
            cax = plt.gcf().add_axes([0.88, 0.525, 0.02, 0.35])  # Define colorbar axes
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Best Accuracy (%)')
            
            # Plot 2: Stability after convergence
            stabilities = [item[1]['stability'] * 100 for item in sorted_experiments]
            
            plt.subplot(2, 1, 2)
            bars = plt.bar(names, stabilities, alpha=0.7)
            
            # Color bars based on stability (lower is better)
            cmap_stability = plt.cm.RdYlGn_r
            max_std = max(stabilities) if stabilities else 1.0
            for i, bar in enumerate(bars):
                bar.set_color(cmap_stability(stabilities[i]/max_std if max_std > 0 else 0))
            
            plt.xlabel('Experiment', fontsize=12)
            plt.ylabel('Standard Deviation (%)', fontsize=12)
            plt.title('Stability After Convergence (Lower is Better)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Add second colorbar
            sm2 = plt.cm.ScalarMappable(cmap=cmap_stability, norm=plt.Normalize(0, max_std))
            sm2.set_array([])
            plt.subplots_adjust(right=0.85)  # Make room for colorbar
            cax2 = plt.gcf().add_axes([0.88, 0.1, 0.02, 0.35])  # Define colorbar axes
            cbar2 = plt.colorbar(sm2, cax=cax2)
            cbar2.set_label('Std Dev (%) - Lower is better')
            
            if save:
                plt.savefig(os.path.join(self.plots_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(self.plots_dir, 'convergence_analysis.pdf'), bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()
            
            # Save convergence data
            with open(os.path.join(self.metrics_dir, "convergence_metrics.json"), 'w') as f:
                json.dump(convergence_data, f, indent=4)
                
            return convergence_data
        
        except Exception as e:
            print(f"Error in convergence analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    

    def generate_comprehensive_tables(self):
        """
        Generate comprehensive tables comparing defense effectiveness against different attacks.
        Prints formatted tables and saves them to files.
        """
        try:
            # Define the defenses and attacks based on experiments
            defenses = set()
            attacks = set()
            
            for name in self.experiments.keys():
                if 'results' not in self.experiments[name] or not self.experiments[name]['results']:
                    continue
                    
                parts = name.split('_')
                if len(parts) >= 2:
                    defense = parts[0]
                    attack = '_'.join(parts[1:])
                    defenses.add(defense)
                    attacks.add(attack)
            
            if not defenses or not attacks:
                print("No valid experiments found for table generation")
                return
                
            defenses = sorted(list(defenses))
            attacks = sorted(list(attacks))
            
            # Initialize result tables
            final_accuracy_table = pd.DataFrame(index=defenses, columns=attacks)
            best_accuracy_table = pd.DataFrame(index=defenses, columns=attacks)
            loss_table = pd.DataFrame(index=defenses, columns=attacks)
            
            # Fill tables with data from experiments
            for name, experiment in self.experiments.items():
                if 'results' not in experiment or not experiment['results']:
                    continue
                    
                parts = name.split('_')
                if len(parts) >= 2:
                    defense = parts[0]
                    attack = '_'.join(parts[1:])
                    
                    results = experiment['results']
                    if results:
                        # Get the final and best accuracy
                        final_acc = results[-1]['test_accuracy'] * 100
                        best_acc = max(r['test_accuracy'] for r in results) * 100
                        final_loss = results[-1]['test_loss']
                        
                        # Store in tables
                        final_accuracy_table.loc[defense, attack] = final_acc
                        best_accuracy_table.loc[defense, attack] = best_acc
                        loss_table.loc[defense, attack] = final_loss
            
            # Calculate impact of attacks (relative to clean)
            impact_table = pd.DataFrame(index=defenses, columns=attacks)
            for defense in defenses:
                for attack in attacks:
                    if attack == "clean":
                        impact_table.loc[defense, attack] = 0.0
                    else:
                        if pd.notna(final_accuracy_table.loc[defense, "clean"]) and pd.notna(final_accuracy_table.loc[defense, attack]):
                            clean_acc = final_accuracy_table.loc[defense, "clean"]
                            attack_acc = final_accuracy_table.loc[defense, attack]
                            impact = attack_acc - clean_acc
                            impact_table.loc[defense, attack] = impact
            
            # Print tables
            print("\n=== Final Test Accuracy (%) ===")
            print(final_accuracy_table.fillna("--"))
            
            print("\n=== Best Test Accuracy (%) ===")
            print(best_accuracy_table.fillna("--"))
            
            print("\n=== Test Loss ===")
            print(loss_table.fillna("--"))
            
            print("\n=== Attack Impact (percentage points, positive = improved performance) ===")
            print(impact_table.round(2).fillna("--"))
            
            # Calculate stability (difference between best and final accuracy)
            stability_table = best_accuracy_table - final_accuracy_table
            print("\n=== Stability (best - final accuracy, lower is better) ===")
            print(stability_table.round(2).fillna("--"))
            
            # Defense rankings
            rankings = {}
            for attack in attacks:
                # Sort defenses by final accuracy (skip NaN values)
                valid_defenses = final_accuracy_table[attack].dropna()
                if not valid_defenses.empty:
                    ranked_defenses = valid_defenses.sort_values(ascending=False)
                    rankings[attack] = list(ranked_defenses.index)
                else:
                    rankings[attack] = []
            
            print("\n=== Defense Rankings for Each Attack ===")
            rankings_df = pd.DataFrame(rankings)
            print(rankings_df)
            
            # Key observations
            print("\n=== Key Observations ===")
            
            # Best defense for each attack
            print("Best Defense for Each Attack:")
            for attack in attacks:
                valid_defenses = final_accuracy_table[attack].dropna()
                if not valid_defenses.empty:
                    best_defense = valid_defenses.idxmax()
                    score = final_accuracy_table.loc[best_defense, attack]
                    print(f"- {attack}: {best_defense} ({score:.2f}%)")
                else:
                    print(f"- {attack}: No data available")
            
            # Most stable defense
            valid_stability = stability_table.mean(axis=1).dropna()
            if not valid_stability.empty:
                most_stable = valid_stability.idxmin()
                print(f"\nMost Stable Defense: {most_stable} (avg deviation: {valid_stability[most_stable]:.2f}%)")
            
            # Most robust against all attacks
            attack_cols = [col for col in attacks if col != "clean"]
            if attack_cols:
                # Calculate mean accuracy across attacks, excluding NaN
                robustness = final_accuracy_table[attack_cols].mean(axis=1, skipna=True).dropna()
                if not robustness.empty:
                    most_robust = robustness.idxmax()
                    print(f"\nMost Robust Defense Against All Attacks: {most_robust} (avg accuracy: {robustness[most_robust]:.2f}%)")
            
            # Save tables directory
            tables_dir = os.path.join(self.save_dir, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            
            # Save tables to CSV
            final_accuracy_table.to_csv(os.path.join(tables_dir, "final_accuracy.csv"))
            best_accuracy_table.to_csv(os.path.join(tables_dir, "best_accuracy.csv"))
            loss_table.to_csv(os.path.join(tables_dir, "test_loss.csv"))
            impact_table.to_csv(os.path.join(tables_dir, "attack_impact.csv"))
            
            print(f"\nTables saved to {tables_dir}")
            
            return {
                "final_accuracy": final_accuracy_table,
                "best_accuracy": best_accuracy_table,
                "loss": loss_table,
                "impact": impact_table
            }
        
        except Exception as e:
            print(f"Error generating comprehensive tables: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}


    def generate_summary_report(self):
        """Generate a comprehensive summary report of all experiments."""
        try:
            report = []
            report.append("=" * 80)
            report.append("Federated Learning Experiments Summary Report")
            report.append("=" * 80)
            
            # Generate all metrics and plots
            try:
                self.compute_attack_impact_metrics()
            except Exception as e:
                report.append(f"Error computing attack impact metrics: {str(e)}")
                
            try:
                self.generate_defense_comparison_tables()
            except Exception as e:
                report.append(f"Error generating defense comparison tables: {str(e)}")
                
            try:
                self.generate_client_performance_statistics()
            except Exception as e:
                report.append(f"Error generating client performance statistics: {str(e)}")
                
            try:
                self.plot_accuracy_comparison(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting accuracy comparison: {str(e)}")
                
            try:
                self.plot_loss_comparison(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting loss comparison: {str(e)}")
                
            try:
                self.plot_attack_impact(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting attack impact: {str(e)}")
                
            try:
                self.plot_client_loss_distributions(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting client loss distributions: {str(e)}")
                
            try:
                self.plot_client_statistics(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting client statistics: {str(e)}")
                
            try:
                self.plot_convergence_analysis(save=True, show=False)
            except Exception as e:
                report.append(f"Error plotting convergence analysis: {str(e)}")
            
            # List all experiments
            report.append("\nExperiments Summary:")
            report.append("-" * 40)
            
            if not self.experiments:
                report.append("No experiments found.")
                
                # Save report
                report_path = os.path.join(self.save_dir, "summary_report.txt")
                with open(report_path, 'w') as f:
                    f.write("\n".join(report))
                
                return "\n".join(report)
            
            for name, experiment in self.experiments.items():
                if 'results' not in experiment or not experiment['results']:
                    report.append(f"\nExperiment: {name} - NO RESULTS")
                    continue
                    
                results = experiment['results']
                
                # Performance metrics
                accuracies = [round_data.get('test_accuracy', 0) * 100 for round_data in results]
                losses = [round_data.get('test_loss', 0) for round_data in results]
                
                report.append(f"\nExperiment: {name}")
                report.append(f"- Final Test Accuracy: {accuracies[-1]:.2f}%")
                report.append(f"- Best Test Accuracy: {max(accuracies):.2f}%")
                report.append(f"- Final Test Loss: {losses[-1]:.4f}")
                
                # Configuration summary if available
                if 'config' in experiment and experiment['config']:
                    config = experiment['config']
                    
                    report.append("\nConfiguration:")
                    report.append(f"- Number of Clients: {CONFIG.NUM_CLIENTS}")
                    report.append(f"- Clients Per Round: {CONFIG.CLIENTS_PER_ROUND}")
                    report.append(f"- Local Epochs: {CONFIG.LOCAL_EPOCHS}")
                    
                    if 'ATTACK_PARAMS' in config and config['ATTACK_PARAMS']:
                        attack_params = config['ATTACK_PARAMS']
                        report.append("\nAttack Configuration:")
                        report.append(f"- Attack Type: {attack_params.get('attack_type', 'unknown')}")
                        mal_clients = attack_params.get('malicious_client_ids', [])
                        report.append(f"- Malicious Clients: {len(mal_clients)}/{CONFIG.NUM_CLIENTS}")
            
            # Add overall comparison summary
            report.append("\n" + "=" * 40)
            report.append("OVERALL COMPARISONS")
            report.append("=" * 40)
            
            # Best performing defenses for each attack
            attack_impact = self.comparison_metrics
            if attack_impact:
                report.append("\nBest Defenses Per Attack:")
                for attack, defenses in attack_impact.items():
                    if not defenses:
                        continue
                        
                    # Find best defense
                    best_defense = max(defenses.items(), key=lambda x: x[1]['final_accuracy'])
                    defense_name, metrics = best_defense
                    
                    report.append(f"- {attack.replace('_', ' ').title()}: {defense_name.capitalize()} "
                                f"({metrics['final_accuracy']*100:.2f}%)")
            
            # Final notes
            report.append("\n" + "-" * 80)
            report.append("Note: Detailed plots and metrics are available in the results directory.")
            report.append("-" * 80)
            
            # Save report
            report_path = os.path.join(self.save_dir, "summary_report.txt")
            with open(report_path, 'w') as f:
                f.write("\n".join(report))
            
            return "\n".join(report)
        
        except Exception as e:
            error_report = ["=" * 80,
                          "ERROR IN REPORT GENERATION",
                          "=" * 80,
                          str(e)]
            
            # Save error report
            report_path = os.path.join(self.save_dir, "error_report.txt")
            with open(report_path, 'w') as f:
                f.write("\n".join(error_report))
            
            import traceback
            traceback.print_exc()
            
            return "\n".join(error_report)

    def save_all_visualizations(self):
        """Generate and save all visualizations in one go."""
        try:
            self.plot_accuracy_comparison(save=True, show=False)
            self.plot_loss_comparison(save=True, show=False)
            self.plot_attack_impact(save=True, show=False)
            self.plot_client_loss_distributions(save=True, show=False)
            self.plot_client_statistics(save=True, show=False)
            self.plot_convergence_analysis(save=True, show=False)
            
            # Generate reports
            self.compute_attack_impact_metrics()
            self.generate_defense_comparison_tables()
            self.generate_client_performance_statistics()
            
            print(f"All visualizations and reports saved to {self.save_dir}")
            
            return True
        
        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
            return False