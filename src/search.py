"""
Hyperparameter search algorithms for drift detectors.
Implements Random Search with Grid Refinement for efficient parameter optimization.
"""

import numpy as np
import random
from typing import Dict, List, Any, Callable, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from scipy.stats import uniform, loguniform

from .detectors import BaseDriftDetector, create_detector
from .metrics import DriftEvaluator, DriftMetrics


@dataclass
class SearchResult:
    """Results from hyperparameter search."""
    parameters: Dict[str, Any]
    score: float
    metrics: DriftMetrics
    detector_type: str
    evaluation_time: float = 0.0


class ParameterSampler:
    """
    Parameter sampling utility for hyperparameter search.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize parameter sampler.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        if seed is not None:
            random.seed(seed)
    
    def sample_parameter(self, param_config: Dict[str, Any]) -> Any:
        """
        Sample a parameter value based on configuration.
        
        Args:
            param_config: Parameter configuration dictionary
            
        Returns:
            Sampled parameter value
        """
        param_type = param_config.get('type', 'uniform')
        
        if param_type == 'uniform':
            low, high = param_config['range']
            return self.rng.uniform(low, high)
            
        elif param_type == 'log_uniform':
            low, high = param_config['range']
            return loguniform.rvs(low, high, random_state=self.rng)
            
        elif param_type == 'choice':
            values = param_config['values']
            return self.rng.choice(values)
            
        elif param_type == 'int_uniform':
            low, high = param_config['range']
            return self.rng.randint(low, high + 1)
            
        elif param_type == 'normal':
            mean = param_config.get('mean', 0.0)
            std = param_config.get('std', 1.0)
            return self.rng.normal(mean, std)
            
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def sample_parameters(self, param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sample a full parameter set.
        
        Args:
            param_space: Complete parameter space configuration
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        for param_name, param_config in param_space.items():
            params[param_name] = self.sample_parameter(param_config)
        return params


class GridRefinement:
    """
    Grid-based refinement around promising parameter regions.
    """
    
    def __init__(self, refinement_factor: float = 0.2, n_points_per_dim: int = 3):
        """
        Initialize grid refinement.
        
        Args:
            refinement_factor: Fraction of original range to refine around best point
            n_points_per_dim: Number of grid points per dimension
        """
        self.refinement_factor = refinement_factor
        self.n_points_per_dim = n_points_per_dim
    
    def create_refinement_grid(
        self,
        best_params: Dict[str, Any],
        param_space: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create refinement grid around best parameters.
        
        Args:
            best_params: Best parameter set found so far
            param_space: Original parameter space
            
        Returns:
            List of parameter configurations for refinement
        """
        refinement_configs = []
        
        # Create refined ranges for each parameter
        refined_spaces = {}
        for param_name, param_config in param_space.items():
            if param_name not in best_params:
                continue
                
            best_value = best_params[param_name]
            param_type = param_config.get('type', 'uniform')
            
            if param_type in ['uniform', 'log_uniform']:
                original_range = param_config['range']
                range_width = original_range[1] - original_range[0]
                
                if param_type == 'log_uniform':
                    # Work in log space for log_uniform
                    log_best = np.log(best_value)
                    log_range_width = np.log(original_range[1]) - np.log(original_range[0])
                    refinement_width = self.refinement_factor * log_range_width
                    
                    log_min = max(np.log(original_range[0]), log_best - refinement_width/2)
                    log_max = min(np.log(original_range[1]), log_best + refinement_width/2)
                    
                    refined_spaces[param_name] = {
                        'type': 'log_uniform',
                        'range': [np.exp(log_min), np.exp(log_max)]
                    }
                else:
                    # Linear refinement
                    refinement_width = self.refinement_factor * range_width
                    refined_min = max(original_range[0], best_value - refinement_width/2)
                    refined_max = min(original_range[1], best_value + refinement_width/2)
                    
                    refined_spaces[param_name] = {
                        'type': 'uniform',
                        'range': [refined_min, refined_max]
                    }
                    
            elif param_type == 'choice':
                # For choice parameters, include neighboring options
                values = param_config['values']
                try:
                    best_idx = values.index(best_value)
                    start_idx = max(0, best_idx - 1)
                    end_idx = min(len(values), best_idx + 2)
                    refined_values = values[start_idx:end_idx]
                    
                    refined_spaces[param_name] = {
                        'type': 'choice',
                        'values': refined_values
                    }
                except ValueError:
                    # If best_value not in original values, keep original
                    refined_spaces[param_name] = param_config
        
        # Generate grid points
        param_names = list(refined_spaces.keys())
        if not param_names:
            return [best_params]
        
        # Create grid for each parameter
        param_grids = []
        sampler = ParameterSampler()
        
        for param_name in param_names:
            param_config = refined_spaces[param_name]
            param_type = param_config.get('type', 'uniform')
            
            if param_type in ['uniform', 'log_uniform']:
                low, high = param_config['range']
                if param_type == 'log_uniform':
                    grid_values = np.logspace(
                        np.log10(low), np.log10(high), 
                        self.n_points_per_dim
                    )
                else:
                    grid_values = np.linspace(low, high, self.n_points_per_dim)
                param_grids.append(grid_values)
                
            elif param_type == 'choice':
                param_grids.append(param_config['values'])
        
        # Generate all combinations
        for combination in itertools.product(*param_grids):
            params = dict(zip(param_names, combination))
            
            # Add unchanged parameters
            for param_name in param_space:
                if param_name not in params:
                    params[param_name] = best_params.get(param_name)
                    
            refinement_configs.append(params)
        
        return refinement_configs


class HybridSearch:
    """
    Hybrid search algorithm combining Random Search with Grid Refinement.
    """
    
    def __init__(
        self,
        n_random_trials: int = 100,
        n_refinement_trials: int = 20,
        refinement_factor: float = 0.2,
        n_top_candidates: int = 5,
        seed: int = None,
        n_jobs: int = 1
    ):
        """
        Initialize hybrid search.
        
        Args:
            n_random_trials: Number of random search trials
            n_refinement_trials: Number of grid refinement trials
            refinement_factor: Refinement range factor
            n_top_candidates: Number of top candidates for refinement
            seed: Random seed
            n_jobs: Number of parallel jobs (1 = sequential)
        """
        self.n_random_trials = n_random_trials
        self.n_refinement_trials = n_refinement_trials
        self.refinement_factor = refinement_factor
        self.n_top_candidates = n_top_candidates
        self.seed = seed
        self.n_jobs = n_jobs
        
        self.sampler = ParameterSampler(seed=seed)
        self.refiner = GridRefinement(
            refinement_factor=refinement_factor,
            n_points_per_dim=max(2, int(np.ceil(n_refinement_trials ** (1/3))))
        )
        
        self.search_history = []
    
    def search(
        self,
        detector_type: str,
        objective_function: Callable[[Dict[str, Any]], float],
        param_space: Dict[str, Dict[str, Any]] = None,
        verbose: bool = False
    ) -> List[SearchResult]:
        """
        Perform hybrid hyperparameter search.
        
        Args:
            detector_type: Type of detector to optimize
            objective_function: Function to maximize (takes parameters, returns score)
            param_space: Parameter space (uses detector defaults if None)
            verbose: Whether to print progress
            
        Returns:
            List of search results sorted by score (best first)
        """
        # Get parameter space
        if param_space is None:
            from .detectors import get_detector_parameter_ranges
            param_space = get_detector_parameter_ranges(detector_type)
        
        results = []
        
        # Phase 1: Random Search
        if verbose:
            print(f"Phase 1: Random search with {self.n_random_trials} trials...")
        
        random_params = []
        for _ in range(self.n_random_trials):
            params = self.sampler.sample_parameters(param_space)
            random_params.append(params)
        
        # Evaluate random parameters
        random_results = self._evaluate_parameter_sets(
            detector_type, random_params, objective_function, verbose
        )
        results.extend(random_results)
        
        # Phase 2: Grid Refinement
        if self.n_refinement_trials > 0 and results:
            if verbose:
                print(f"Phase 2: Grid refinement around top {self.n_top_candidates} candidates...")
                
            # Sort by score and get top candidates
            results.sort(key=lambda x: x.score, reverse=True)
            top_results = results[:self.n_top_candidates]
            
            refinement_params = []
            for result in top_results:
                refined_configs = self.refiner.create_refinement_grid(
                    result.parameters, param_space
                )
                refinement_params.extend(refined_configs)
            
            # Remove duplicates (approximately)
            refinement_params = self._remove_duplicate_params(refinement_params)
            
            # Limit refinement trials
            if len(refinement_params) > self.n_refinement_trials:
                # Sample subset
                indices = self.sampler.rng.choice(
                    len(refinement_params), 
                    self.n_refinement_trials, 
                    replace=False
                )
                refinement_params = [refinement_params[i] for i in indices]
            
            # Evaluate refinement parameters
            refinement_results = self._evaluate_parameter_sets(
                detector_type, refinement_params, objective_function, verbose
            )
            results.extend(refinement_results)
        
        # Sort final results
        results.sort(key=lambda x: x.score, reverse=True)
        self.search_history.extend(results)
        
        return results
    
    def _evaluate_parameter_sets(
        self,
        detector_type: str,
        param_sets: List[Dict[str, Any]],
        objective_function: Callable,
        verbose: bool = False
    ) -> List[SearchResult]:
        """Evaluate multiple parameter sets."""
        results = []
        
        if self.n_jobs == 1:
            # Sequential evaluation
            for i, params in enumerate(param_sets):
                if verbose and i % 10 == 0:
                    print(f"  Evaluating {i+1}/{len(param_sets)}...")
                    
                try:
                    score = objective_function(params)
                    result = SearchResult(
                        parameters=params,
                        score=score,
                        metrics=None,  # Will be filled by objective function if needed
                        detector_type=detector_type
                    )
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Error evaluating params {params}: {e}")
                    continue
        else:
            # Parallel evaluation (simplified version)
            # Note: Full parallel evaluation would require careful handling of 
            # objective functions and shared state
            print("Parallel evaluation not fully implemented. Using sequential.")
            return self._evaluate_parameter_sets(detector_type, param_sets, objective_function, verbose)
            
        return results
    
    def _remove_duplicate_params(
        self, 
        param_sets: List[Dict[str, Any]], 
        tolerance: float = 1e-6
    ) -> List[Dict[str, Any]]:
        """Remove approximately duplicate parameter sets."""
        unique_params = []
        
        for params in param_sets:
            is_duplicate = False
            
            for existing_params in unique_params:
                if self._params_are_similar(params, existing_params, tolerance):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_params.append(params)
                
        return unique_params
    
    def _params_are_similar(
        self, 
        params1: Dict[str, Any], 
        params2: Dict[str, Any], 
        tolerance: float
    ) -> bool:
        """Check if two parameter sets are similar."""
        if set(params1.keys()) != set(params2.keys()):
            return False
            
        for key in params1:
            val1, val2 = params1[key], params2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > tolerance * (abs(val1) + abs(val2) + tolerance):
                    return False
            elif val1 != val2:
                return False
                
        return True
    
    def get_best_parameters(self, n: int = 1) -> List[SearchResult]:
        """Get best n parameter sets from search history."""
        if not self.search_history:
            return []
            
        sorted_history = sorted(self.search_history, key=lambda x: x.score, reverse=True)
        return sorted_history[:n]


class BayesianOptimization:
    """
    Simple Bayesian optimization using Gaussian Process surrogate.
    Note: This is a placeholder for more sophisticated Bayesian optimization.
    """
    
    def __init__(
        self,
        n_initial_points: int = 10,
        n_iterations: int = 50,
        acquisition: str = 'ei',
        seed: int = None
    ):
        """
        Initialize Bayesian optimization.
        
        Args:
            n_initial_points: Number of initial random points
            n_iterations: Number of optimization iterations
            acquisition: Acquisition function ('ei', 'pi', 'ucb')
            seed: Random seed
        """
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.seed = seed
        
        # Note: Full implementation would require sklearn or GPyOpt
        print("Warning: Bayesian optimization not fully implemented. Using random search.")
    
    def search(
        self,
        detector_type: str,
        objective_function: Callable,
        param_space: Dict[str, Dict[str, Any]] = None,
        verbose: bool = False
    ) -> List[SearchResult]:
        """Placeholder for Bayesian optimization."""
        # Fall back to random search
        hybrid_search = HybridSearch(
            n_random_trials=self.n_initial_points + self.n_iterations,
            n_refinement_trials=0,
            seed=self.seed
        )
        
        return hybrid_search.search(detector_type, objective_function, param_space, verbose)


def create_objective_function(
    detector_type: str,
    data_stream: List[Tuple[Any, Any, bool]],
    evaluator: DriftEvaluator = None
) -> Callable[[Dict[str, Any]], float]:
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        detector_type: Type of detector
        data_stream: List of (sample, label, is_drift) tuples
        evaluator: Drift evaluator (uses default if None)
        
    Returns:
        Objective function that takes parameters and returns score
    """
    if evaluator is None:
        evaluator = DriftEvaluator()
    
    def objective(params: Dict[str, Any]) -> float:
        """Objective function to maximize."""
        try:
            # Create detector with given parameters
            detector = create_detector(detector_type, **params)
            
            # Extract true drift points
            true_drifts = [i for i, (_, _, is_drift) in enumerate(data_stream) if is_drift]
            
            # Run detection
            detected_drifts = []
            for i, (sample, _, _) in enumerate(data_stream):
                detector.update(sample)
                if detector.drift_detected:
                    detected_drifts.append(i)
            
            # Evaluate performance
            metrics = evaluator.evaluate(
                detected_drifts, 
                true_drifts,
                len(data_stream)
            )
            
            return metrics.composite_score
            
        except Exception as e:
            # Return poor score for invalid parameters
            return -1.0
    
    return objective