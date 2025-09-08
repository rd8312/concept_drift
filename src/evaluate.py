"""
Multi-dataset evaluation framework for drift detectors.
Provides comprehensive benchmarking with CSV/JSON export and visualization.
"""

import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import pandas as pd

from .detectors import create_detector, get_all_detector_names, BaseDriftDetector
from .datasets import create_dataset
from .metrics import DriftEvaluator, DriftMetrics
from .search import HybridSearch, SearchResult, create_objective_function


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    datasets: List[str] = None
    detectors: List[str] = None
    noise_levels: List[float] = None
    n_runs: int = 5
    tolerance: int = 50
    delay_penalty: float = 0.002
    seed: int = 42
    output_dir: str = "results"
    export_formats: List[str] = None  # ['csv', 'json']
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ['sea', 'sine', 'friedman']
        if self.detectors is None:
            self.detectors = get_all_detector_names()
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.01, 0.03, 0.05]
        if self.export_formats is None:
            self.export_formats = ['csv', 'json']


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    dataset: str
    detector: str
    parameters: Dict[str, Any]
    noise_level: float
    run_id: int
    metrics: DriftMetrics
    detected_drifts: List[int]
    true_drifts: List[int]
    execution_time: float
    stream_length: int


class EvaluationFramework:
    """
    Comprehensive evaluation framework for drift detectors.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize evaluation framework.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.results = []
        self.evaluator = DriftEvaluator(
            tolerance=self.config.tolerance,
            delay_penalty=self.config.delay_penalty
        )
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_single_experiment(
        self,
        dataset_name: str,
        dataset_config: Dict[str, Any],
        detector_name: str,
        detector_params: Dict[str, Any],
        noise_level: float,
        run_id: int = 0
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            dataset_name: Name of dataset
            dataset_config: Dataset configuration
            detector_name: Name of detector
            detector_params: Detector parameters
            noise_level: Noise level to add
            run_id: Run identifier
            
        Returns:
            ExperimentResult with evaluation metrics
        """
        # Add noise to dataset config
        dataset_config = dataset_config.copy()
        dataset_config['noise_level'] = noise_level
        dataset_config['seed'] = self.config.seed + run_id
        
        # Generate dataset
        start_time = time.time()
        
        data_stream = list(create_dataset(dataset_name, dataset_config))
        stream_length = len(data_stream)
        
        # Extract true drift points
        true_drifts = [i for i, (_, _, is_drift) in enumerate(data_stream) if is_drift]
        
        # Create and run detector
        detector = create_detector(detector_name, **detector_params)
        detected_drifts = []
        
        for i, (sample, _, _) in enumerate(data_stream):
            detector.update(sample)
            if detector.drift_detected:
                detected_drifts.append(i)
        
        execution_time = time.time() - start_time
        
        # Evaluate performance
        metrics = self.evaluator.evaluate(
            detected_drifts, 
            true_drifts, 
            stream_length
        )
        
        return ExperimentResult(
            dataset=dataset_name,
            detector=detector_name,
            parameters=detector_params,
            noise_level=noise_level,
            run_id=run_id,
            metrics=metrics,
            detected_drifts=detected_drifts,
            true_drifts=true_drifts,
            execution_time=execution_time,
            stream_length=stream_length
        )
    
    def run_benchmark(
        self,
        detector_configs: Dict[str, Dict[str, Any]] = None,
        dataset_configs: Dict[str, Dict[str, Any]] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """
        Run comprehensive benchmark evaluation.
        
        Args:
            detector_configs: Custom detector configurations
            dataset_configs: Custom dataset configurations
            verbose: Whether to print progress
            
        Returns:
            List of experiment results
        """
        # Default configurations
        if detector_configs is None:
            detector_configs = {name: {} for name in self.config.detectors}
            
        if dataset_configs is None:
            dataset_configs = {
                'sea': {
                    'drift_positions': [1000, 2500, 4000],
                    'n_samples': 5000
                },
                'sine': {
                    'drift_positions': [1500, 3000],
                    'n_samples': 4500
                },
                'friedman': {
                    'drift_type': 'abrupt',
                    'drift_positions': [2500],
                    'n_samples': 5000
                }
            }
        
        results = []
        total_experiments = (
            len(self.config.datasets) * 
            len(self.config.detectors) * 
            len(self.config.noise_levels) * 
            self.config.n_runs
        )
        
        experiment_count = 0
        
        for dataset_name in self.config.datasets:
            if dataset_name not in dataset_configs:
                if verbose:
                    print(f"Skipping dataset {dataset_name} - no configuration")
                continue
                
            dataset_config = dataset_configs[dataset_name]
            
            for detector_name in self.config.detectors:
                if detector_name not in detector_configs:
                    if verbose:
                        print(f"Skipping detector {detector_name} - no configuration")
                    continue
                    
                detector_params = detector_configs[detector_name]
                
                for noise_level in self.config.noise_levels:
                    for run_id in range(self.config.n_runs):
                        experiment_count += 1
                        
                        if verbose:
                            print(f"Experiment {experiment_count}/{total_experiments}: "
                                  f"{dataset_name}-{detector_name}-noise{noise_level}-run{run_id}")
                        
                        try:
                            result = self.run_single_experiment(
                                dataset_name,
                                dataset_config,
                                detector_name,
                                detector_params,
                                noise_level,
                                run_id
                            )
                            results.append(result)
                            
                        except Exception as e:
                            if verbose:
                                print(f"  Error: {e}")
                            continue
        
        self.results.extend(results)
        return results
    
    def run_optimization_benchmark(
        self,
        n_trials: int = 200,
        dataset_configs: Dict[str, Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, List[SearchResult]]:
        """
        Run hyperparameter optimization for all detectors.
        
        Args:
            n_trials: Number of optimization trials per detector
            dataset_configs: Custom dataset configurations
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping detector names to optimization results
        """
        if dataset_configs is None:
            dataset_configs = {
                'sea': {
                    'drift_positions': [1000, 2500, 4000],
                    'n_samples': 5000,
                    'noise_level': 0.02
                }
            }
        
        optimization_results = {}
        
        # Use first dataset for optimization
        dataset_name = list(dataset_configs.keys())[0]
        dataset_config = dataset_configs[dataset_name]
        
        # Generate optimization dataset
        data_stream = list(create_dataset(dataset_name, dataset_config))
        
        for detector_name in self.config.detectors:
            if verbose:
                print(f"Optimizing {detector_name} with {n_trials} trials...")
            
            # Create hybrid search
            search = HybridSearch(
                n_random_trials=int(0.8 * n_trials),
                n_refinement_trials=int(0.2 * n_trials),
                seed=self.config.seed
            )
            
            # Create objective function
            objective = create_objective_function(
                detector_name,
                data_stream,
                self.evaluator
            )
            
            # Run optimization
            results = search.search(
                detector_name,
                objective,
                verbose=verbose
            )
            
            optimization_results[detector_name] = results
            
            if verbose and results:
                best_score = results[0].score
                print(f"  Best score for {detector_name}: {best_score:.4f}")
        
        return optimization_results
    
    def export_results(
        self,
        results: List[ExperimentResult] = None,
        filename_prefix: str = "drift_detector_benchmark"
    ):
        """
        Export results to CSV and JSON formats.
        
        Args:
            results: Results to export (uses self.results if None)
            filename_prefix: Prefix for output filenames
        """
        if results is None:
            results = self.results
            
        if not results:
            print("No results to export")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV
        if 'csv' in self.config.export_formats:
            csv_path = Path(self.config.output_dir) / f"{filename_prefix}_{timestamp}.csv"
            self._export_to_csv(results, csv_path)
            print(f"Results exported to {csv_path}")
        
        # Export to JSON
        if 'json' in self.config.export_formats:
            json_path = Path(self.config.output_dir) / f"{filename_prefix}_{timestamp}.json"
            self._export_to_json(results, json_path)
            print(f"Results exported to {json_path}")
    
    def _export_to_csv(self, results: List[ExperimentResult], filepath: Path):
        """Export results to CSV format."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'dataset', 'detector', 'noise_level', 'run_id',
                'f1_score', 'precision', 'recall', 'mean_delay',
                'false_positive_rate', 'composite_score',
                'tp', 'fp', 'fn', 'execution_time', 'stream_length'
            ]
            
            # Add parameter columns
            all_param_keys = set()
            for result in results:
                all_param_keys.update(result.parameters.keys())
            
            param_fieldnames = [f'param_{key}' for key in sorted(all_param_keys)]
            fieldnames.extend(param_fieldnames)
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'dataset': result.dataset,
                    'detector': result.detector,
                    'noise_level': result.noise_level,
                    'run_id': result.run_id,
                    'f1_score': result.metrics.f1_score,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'mean_delay': result.metrics.mean_delay,
                    'false_positive_rate': result.metrics.false_positive_rate,
                    'composite_score': result.metrics.composite_score,
                    'tp': result.metrics.tp,
                    'fp': result.metrics.fp,
                    'fn': result.metrics.fn,
                    'execution_time': result.execution_time,
                    'stream_length': result.stream_length
                }
                
                # Add parameters
                for key in all_param_keys:
                    param_key = f'param_{key}'
                    row[param_key] = result.parameters.get(key, '')
                    
                writer.writerow(row)
    
    def _export_to_json(self, results: List[ExperimentResult], filepath: Path):
        """Export results to JSON format."""
        json_data = []
        
        for result in results:
            result_dict = {
                'dataset': result.dataset,
                'detector': result.detector,
                'parameters': result.parameters,
                'noise_level': result.noise_level,
                'run_id': result.run_id,
                'metrics': asdict(result.metrics),
                'detected_drifts': result.detected_drifts,
                'true_drifts': result.true_drifts,
                'execution_time': result.execution_time,
                'stream_length': result.stream_length
            }
            json_data.append(result_dict)
        
        with open(filepath, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
    
    def generate_plots(
        self,
        results: List[ExperimentResult] = None,
        save_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate evaluation plots.
        
        Args:
            results: Results to plot (uses self.results if None)
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        if results is None:
            results = self.results
            
        if not results:
            print("No results to plot")
            return {}
        
        # Convert to DataFrame for easier plotting
        df = self._results_to_dataframe(results)
        
        plots = {}
        
        # 1. F1 Score comparison
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df, x='detector', y='f1_score', hue='dataset', ax=ax1)
        ax1.set_title('F1 Score Comparison Across Detectors and Datasets')
        ax1.set_xlabel('Detector')
        ax1.set_ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['f1_comparison'] = fig1
        
        # 2. Composite Score vs Noise Level
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        for detector in df['detector'].unique():
            detector_data = df[df['detector'] == detector]
            noise_means = detector_data.groupby('noise_level')['composite_score'].mean()
            ax2.plot(noise_means.index, noise_means.values, marker='o', label=detector)
        
        ax2.set_title('Composite Score vs Noise Level')
        ax2.set_xlabel('Noise Level')
        ax2.set_ylabel('Composite Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['noise_impact'] = fig2
        
        # 3. ROC-like curve (FPR vs TPR)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        for detector in df['detector'].unique():
            detector_data = df[df['detector'] == detector]
            
            # Group by noise level and compute means
            grouped = detector_data.groupby('noise_level').agg({
                'recall': 'mean',
                'false_positive_rate': 'mean'
            }).reset_index()
            
            ax3.plot(
                grouped['false_positive_rate'], 
                grouped['recall'],
                marker='o', 
                label=detector
            )
        
        ax3.set_title('ROC-like Curve (TPR vs FPR)')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate (Recall)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plots['roc_curve'] = fig3
        
        # 4. Detection Delay Distribution
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        delay_data = []
        for result in results:
            for delay in result.metrics.delays:
                delay_data.append({
                    'detector': result.detector,
                    'delay': delay
                })
        
        if delay_data:
            delay_df = pd.DataFrame(delay_data)
            sns.violinplot(data=delay_df, x='detector', y='delay', ax=ax4)
            ax4.set_title('Detection Delay Distribution')
            ax4.set_xlabel('Detector')
            ax4.set_ylabel('Detection Delay (samples)')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plots['delay_distribution'] = fig4
        
        # 5. Performance Heatmap
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot_data = df.groupby(['detector', 'dataset'])['composite_score'].mean().unstack()
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='viridis',
            ax=ax5
        )
        ax5.set_title('Performance Heatmap (Composite Score)')
        plt.tight_layout()
        plots['performance_heatmap'] = fig5
        
        # Save plots if requested
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for plot_name, fig in plots.items():
                filepath = Path(self.config.output_dir) / f"{plot_name}_{timestamp}.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {filepath}")
        
        return plots
    
    def generate_3d_scatter_plots(
        self,
        optimization_results: Dict[str, List[Any]],
        templates: Dict[str, Dict[str, Any]] = None,
        save_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate 3D scatter plots showing F1 vs Delay vs FP rate for each detector,
        with template configurations highlighted.
        
        Args:
            optimization_results: Results from hyperparameter optimization
            templates: Template configurations from presets.py
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        plots = {}
        
        for detector_name, search_results in optimization_results.items():
            if not search_results:
                continue
                
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract metrics for all configurations
            f1_scores = []
            delays = []
            fp_rates = []
            colors = []
            sizes = []
            
            for result in search_results:
                f1_scores.append(result.metrics.f1_score)
                delays.append(result.metrics.mean_delay)
                fp_rates.append(result.metrics.false_positive_rate)
                colors.append('lightblue')
                sizes.append(20)
            
            # Plot all configurations
            scatter = ax.scatter(
                f1_scores, delays, fp_rates,
                c=colors, s=sizes, alpha=0.6,
                label='All Configurations'
            )
            
            # Highlight template configurations if available
            if templates and detector_name in templates:
                detector_templates = templates[detector_name]
                
                template_colors = {
                    'high_sensitivity': 'red',
                    'balanced': 'green', 
                    'high_stability': 'blue'
                }
                
                for template_name, template in detector_templates.items():
                    if 'expected_performance' in template:
                        perf = template['expected_performance']
                        ax.scatter(
                            [perf['f1_score']],
                            [perf['mean_delay']],
                            [perf['false_positive_rate']],
                            c=[template_colors.get(template_name, 'black')],
                            s=[100],
                            marker='*',
                            edgecolors='black',
                            linewidth=2,
                            label=f'{template_name.replace("_", " ").title()} Template'
                        )
            
            # Formatting
            ax.set_xlabel('F1 Score')
            ax.set_ylabel('Mean Delay (samples)')
            ax.set_zlabel('False Positive Rate')
            ax.set_title(f'3D Performance Space: {detector_name.upper()}\n'
                        'F1 Score vs Delay vs False Positive Rate')
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add grid and improve readability
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            ax.set_xlim(0, 1)
            ax.set_zlim(0, max(1, max(fp_rates) * 1.1))
            
            plt.tight_layout()
            plots[f'3d_scatter_{detector_name}'] = fig
        
        # Save plots if requested
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for plot_name, fig in plots.items():
                filepath = Path(self.config.output_dir) / f"{plot_name}_{timestamp}.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"3D plot saved: {filepath}")
        
        return plots
    
    def generate_drift_timeline_plots(
        self,
        results: List[ExperimentResult] = None,
        save_plots: bool = True,
        max_samples_display: int = 5000
    ) -> Dict[str, plt.Figure]:
        """
        Generate time series plots showing true drift points vs detected drift points
        for each dataset and detector combination.
        
        Args:
            results: Experiment results to plot
            save_plots: Whether to save plots to files
            max_samples_display: Maximum samples to display (for readability)
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        if results is None:
            results = self.results
            
        if not results:
            print("No results available for drift timeline plots")
            return {}
        
        plots = {}
        
        # Group results by dataset
        by_dataset = {}
        for result in results:
            if result.dataset not in by_dataset:
                by_dataset[result.dataset] = []
            by_dataset[result.dataset].append(result)
        
        for dataset_name, dataset_results in by_dataset.items():
            # Group by detector for this dataset
            by_detector = {}
            for result in dataset_results:
                if result.detector not in by_detector:
                    by_detector[result.detector] = []
                by_detector[result.detector].append(result)
            
            # Create subplot for each detector
            n_detectors = len(by_detector)
            fig, axes = plt.subplots(
                n_detectors, 1, 
                figsize=(15, 4 * n_detectors),
                sharex=True
            )
            
            if n_detectors == 1:
                axes = [axes]
            
            for idx, (detector_name, detector_results) in enumerate(by_detector.items()):
                ax = axes[idx]
                
                # Use the first result as representative (they should have same true drifts)
                representative_result = detector_results[0]
                max_length = min(representative_result.stream_length, max_samples_display)
                
                # Plot timeline
                x = np.arange(max_length)
                y = np.zeros(max_length)
                
                # Mark true drift points
                for drift_pos in representative_result.true_drifts:
                    if drift_pos < max_length:
                        ax.axvline(x=drift_pos, color='red', linestyle='--', 
                                 alpha=0.7, linewidth=2, label='True Drift')
                
                # Plot detected drifts for all runs
                colors = ['blue', 'green', 'orange', 'purple', 'brown']
                for run_idx, result in enumerate(detector_results[:5]):  # Max 5 runs for clarity
                    color = colors[run_idx % len(colors)]
                    
                    # Mark detected drift points
                    for drift_pos in result.detected_drifts:
                        if drift_pos < max_length:
                            ax.axvline(x=drift_pos, color=color, linestyle='-', 
                                     alpha=0.6, linewidth=1.5)
                    
                    # Add run label (only once per run)
                    if result.detected_drifts and result.detected_drifts[0] < max_length:
                        ax.text(result.detected_drifts[0], 0.8 + run_idx * 0.1, 
                               f'Run {run_idx+1}', color=color, fontsize=8)
                
                # Formatting
                ax.set_ylim(-0.1, 1.1)
                ax.set_ylabel('Detection Events')
                ax.set_title(f'{detector_name.upper()} on {dataset_name.upper()}')
                ax.grid(True, alpha=0.3)
                
                # Add performance metrics text
                avg_f1 = np.mean([r.metrics.f1_score for r in detector_results])
                avg_delay = np.mean([r.metrics.mean_delay for r in detector_results])
                avg_fp = np.mean([r.metrics.false_positive_rate for r in detector_results])
                
                metrics_text = f'Avg F1: {avg_f1:.3f}, Avg Delay: {avg_delay:.1f}, Avg FP: {avg_fp:.3f}'
                ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Common x-label
            axes[-1].set_xlabel('Sample Index')
            
            # Legend (only on first subplot to avoid clutter)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                # Remove duplicate labels
                by_label = dict(zip(labels, handles))
                axes[0].legend(by_label.values(), by_label.keys(), 
                             loc='upper right', fontsize=9)
            
            plt.suptitle(f'Drift Detection Timeline: {dataset_name.upper()} Dataset', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plots[f'timeline_{dataset_name}'] = fig
        
        # Save plots if requested
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for plot_name, fig in plots.items():
                filepath = Path(self.config.output_dir) / f"{plot_name}_{timestamp}.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Timeline plot saved: {filepath}")
        
        return plots
    
    def export_preset_configurations(
        self,
        optimization_results: Dict[str, List[Any]],
        templates: Dict[str, Dict[str, Any]] = None,
        filename: str = "presets_export.json"
    ) -> str:
        """
        Export final optimized parameters for each detector algorithm.
        
        Args:
            optimization_results: Results from hyperparameter optimization
            templates: Template configurations from presets.py
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        export_data = {
            'metadata': {
                'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'description': 'Optimized drift detector configurations',
                'evaluation_config': asdict(self.config)
            },
            'algorithms': {}
        }
        
        for detector_name, search_results in optimization_results.items():
            if not search_results:
                continue
                
            detector_data = {
                'total_configurations_tested': len(search_results),
                'best_configurations': {},
                'parameter_ranges': {},
                'performance_statistics': {}
            }
            
            # Find best configurations by different criteria
            if search_results:
                # Best overall (highest composite score)
                best_overall = max(search_results, key=lambda x: x.score)
                detector_data['best_configurations']['overall'] = {
                    'parameters': best_overall.parameters,
                    'performance': {
                        'f1_score': best_overall.metrics.f1_score,
                        'precision': best_overall.metrics.precision,
                        'recall': best_overall.metrics.recall,
                        'false_positive_rate': best_overall.metrics.false_positive_rate,
                        'mean_delay': best_overall.metrics.mean_delay,
                        'composite_score': best_overall.score
                    }
                }
                
                # Best F1 score
                best_f1 = max(search_results, key=lambda x: x.metrics.f1_score)
                detector_data['best_configurations']['highest_f1'] = {
                    'parameters': best_f1.parameters,
                    'f1_score': best_f1.metrics.f1_score
                }
                
                # Lowest false positive rate
                lowest_fp = min(search_results, key=lambda x: x.metrics.false_positive_rate)
                detector_data['best_configurations']['lowest_fp'] = {
                    'parameters': lowest_fp.parameters,
                    'false_positive_rate': lowest_fp.metrics.false_positive_rate
                }
                
                # Calculate parameter ranges
                if search_results[0].parameters:
                    for param_name in search_results[0].parameters.keys():
                        param_values = [r.parameters[param_name] for r in search_results 
                                      if param_name in r.parameters]
                        if param_values:
                            detector_data['parameter_ranges'][param_name] = {
                                'min': min(param_values),
                                'max': max(param_values),
                                'mean': np.mean(param_values),
                                'std': np.std(param_values)
                            }
                
                # Performance statistics
                f1_scores = [r.metrics.f1_score for r in search_results]
                fp_rates = [r.metrics.false_positive_rate for r in search_results]
                delays = [r.metrics.mean_delay for r in search_results]
                
                detector_data['performance_statistics'] = {
                    'f1_score': {
                        'mean': np.mean(f1_scores),
                        'std': np.std(f1_scores),
                        'min': np.min(f1_scores),
                        'max': np.max(f1_scores)
                    },
                    'false_positive_rate': {
                        'mean': np.mean(fp_rates),
                        'std': np.std(fp_rates),
                        'min': np.min(fp_rates),
                        'max': np.max(fp_rates)
                    },
                    'mean_delay': {
                        'mean': np.mean(delays),
                        'std': np.std(delays),
                        'min': np.min(delays),
                        'max': np.max(delays)
                    }
                }
            
            # Add template configurations if available
            if templates and detector_name in templates:
                detector_data['template_configurations'] = templates[detector_name]
            
            export_data['algorithms'][detector_name] = detector_data
        
        # Save to file
        filepath = Path(self.config.output_dir) / filename
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Preset configurations exported to: {filepath}")
        return str(filepath)
    
    def _results_to_dataframe(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in results:
            row = {
                'dataset': result.dataset,
                'detector': result.detector,
                'noise_level': result.noise_level,
                'run_id': result.run_id,
                'f1_score': result.metrics.f1_score,
                'precision': result.metrics.precision,
                'recall': result.metrics.recall,
                'mean_delay': result.metrics.mean_delay,
                'false_positive_rate': result.metrics.false_positive_rate,
                'composite_score': result.metrics.composite_score,
                'execution_time': result.execution_time
            }
            
            # Add parameters as columns
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value
                
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_summary_report(
        self,
        results: List[ExperimentResult] = None
    ) -> Dict[str, Any]:
        """
        Generate summary report of evaluation results.
        
        Args:
            results: Results to summarize (uses self.results if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if results is None:
            results = self.results
            
        if not results:
            return {}
        
        df = self._results_to_dataframe(results)
        
        summary = {
            'total_experiments': len(results),
            'datasets': list(df['dataset'].unique()),
            'detectors': list(df['detector'].unique()),
            'noise_levels': list(df['noise_level'].unique()),
            'runs_per_configuration': self.config.n_runs,
            
            # Overall performance statistics
            'overall_stats': {
                'mean_f1_score': float(df['f1_score'].mean()),
                'std_f1_score': float(df['f1_score'].std()),
                'mean_composite_score': float(df['composite_score'].mean()),
                'std_composite_score': float(df['composite_score'].std()),
                'mean_execution_time': float(df['execution_time'].mean())
            },
            
            # Per-detector statistics
            'detector_stats': {},
            
            # Per-dataset statistics  
            'dataset_stats': {},
            
            # Best configurations
            'best_configurations': {}
        }
        
        # Per-detector statistics
        for detector in df['detector'].unique():
            detector_data = df[df['detector'] == detector]
            summary['detector_stats'][detector] = {
                'mean_f1_score': float(detector_data['f1_score'].mean()),
                'mean_composite_score': float(detector_data['composite_score'].mean()),
                'mean_delay': float(detector_data['mean_delay'].mean()),
                'mean_fp_rate': float(detector_data['false_positive_rate'].mean())
            }
        
        # Per-dataset statistics
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            summary['dataset_stats'][dataset] = {
                'mean_f1_score': float(dataset_data['f1_score'].mean()),
                'mean_composite_score': float(dataset_data['composite_score'].mean()),
                'best_detector': dataset_data.loc[
                    dataset_data['composite_score'].idxmax(), 'detector'
                ]
            }
        
        # Best configurations
        best_overall = df.loc[df['composite_score'].idxmax()]
        summary['best_configurations']['overall'] = {
            'detector': best_overall['detector'],
            'dataset': best_overall['dataset'],
            'composite_score': float(best_overall['composite_score']),
            'f1_score': float(best_overall['f1_score'])
        }
        
        return summary


def run_comprehensive_evaluation(
    output_dir: str = "results",
    n_trials: int = 200,
    verbose: bool = True,
    config: EvaluationConfig = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation including optimization and benchmarking.
    
    Args:
        output_dir: Output directory for results
        n_trials: Number of optimization trials per detector
        verbose: Whether to print progress
        config: Evaluation configuration (if None, uses default config)
        
    Returns:
        Dictionary with all results and summary
    """
    if config is None:
        config = EvaluationConfig(output_dir=output_dir)
    else:
        # Update output directory if provided
        config.output_dir = output_dir
    
    framework = EvaluationFramework(config)
    
    print("=== Drift Detector Comprehensive Evaluation ===")
    
    # Phase 1: Hyperparameter Optimization
    print("\nPhase 1: Hyperparameter Optimization")
    optimization_results = framework.run_optimization_benchmark(
        n_trials=n_trials,
        verbose=verbose
    )
    
    # Extract best parameters for benchmarking
    best_detector_configs = {}
    for detector_name, search_results in optimization_results.items():
        if search_results:
            best_detector_configs[detector_name] = search_results[0].parameters
        else:
            # Use default parameters if no optimization results
            # This ensures all detectors have valid parameters
            best_detector_configs[detector_name] = {}
    
    # Phase 2: Comprehensive Benchmarking
    print("\nPhase 2: Comprehensive Benchmarking")
    benchmark_results = framework.run_benchmark(
        detector_configs=best_detector_configs,
        verbose=verbose
    )
    
    # Phase 3: Generate Templates
    print("\nPhase 3: Generate Templates")
    from .presets import generate_detector_templates
    templates = generate_detector_templates(optimization_results, output_dir)
    
    # Phase 4: Advanced Visualization and Export
    print("\nPhase 4: Advanced Visualization and Export")
    framework.export_results(benchmark_results)
    plots = framework.generate_plots(benchmark_results)
    
    # Generate new advanced visualizations
    print("Generating 3D scatter plots...")
    scatter_plots = framework.generate_3d_scatter_plots(
        optimization_results, 
        templates, 
        save_plots=True
    )
    plots.update(scatter_plots)
    
    print("Generating drift timeline plots...")
    timeline_plots = framework.generate_drift_timeline_plots(
        benchmark_results,
        save_plots=True
    )
    plots.update(timeline_plots)
    
    # Export preset configurations
    print("Exporting preset configurations...")
    presets_file = framework.export_preset_configurations(
        optimization_results,
        templates,
        "presets_export.json"
    )
    
    summary = framework.generate_summary_report(benchmark_results)
    
    # Save summary
    summary_path = Path(output_dir) / f"summary_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")
    print(f"Summary report: {summary_path}")
    
    return {
        'optimization_results': optimization_results,
        'benchmark_results': benchmark_results,
        'templates': templates,
        'summary': summary,
        'plots': plots,
        'presets_export_file': presets_file
    }