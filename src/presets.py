"""
Automatic template generation for drift detectors.
Extracts high-sensitivity, balanced, and high-stability configurations from Pareto results.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from .search import SearchResult
from .metrics import DriftMetrics


@dataclass
class DetectorTemplate:
    """Template configuration for a drift detector."""
    name: str
    description: str
    use_cases: List[str]
    parameters: Dict[str, Any]
    expected_performance: Dict[str, float]
    confidence_score: float
    pareto_rank: int = 0


class ParetoAnalyzer:
    """
    Analyzes search results to identify Pareto-optimal solutions.
    """
    
    def __init__(self, objectives: List[str] = None, weights: List[float] = None):
        """
        Initialize Pareto analyzer.
        
        Args:
            objectives: List of objectives to optimize
            weights: Weights for each objective (positive = maximize, negative = minimize)
        """
        if objectives is None:
            objectives = ['f1_score', 'false_positive_rate', 'mean_delay']
        if weights is None:
            weights = [1.0, -1.0, -0.002]  # Maximize F1, minimize FP rate and delay
            
        self.objectives = objectives
        self.weights = weights
        
    def compute_pareto_front(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Compute Pareto front from search results.
        
        Args:
            results: List of search results
            
        Returns:
            List of Pareto-optimal results
        """
        if not results:
            return []
            
        # Extract objective values
        objective_values = []
        for result in results:
            values = []
            for obj in self.objectives:
                if hasattr(result.metrics, obj):
                    values.append(getattr(result.metrics, obj))
                elif obj == 'composite_score':
                    values.append(result.score)
                else:
                    values.append(0.0)  # Default value
            objective_values.append(values)
            
        objective_matrix = np.array(objective_values)
        
        # Find Pareto front
        pareto_indices = []
        for i in range(len(results)):
            is_dominated = False
            
            for j in range(len(results)):
                if i == j:
                    continue
                    
                # Check if solution j dominates solution i
                if self._dominates(objective_matrix[j], objective_matrix[i]):
                    is_dominated = True
                    break
                    
            if not is_dominated:
                pareto_indices.append(i)
                
        return [results[i] for i in pareto_indices]
    
    def _dominates(self, solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
        """
        Check if solution A dominates solution B.
        
        Args:
            solution_a: Objective values for solution A
            solution_b: Objective values for solution B
            
        Returns:
            True if A dominates B
        """
        better_in_all = True
        strictly_better = False
        
        for i, weight in enumerate(self.weights):
            if weight > 0:  # Maximize
                if solution_a[i] < solution_b[i]:
                    better_in_all = False
                    break
                elif solution_a[i] > solution_b[i]:
                    strictly_better = True
            else:  # Minimize
                if solution_a[i] > solution_b[i]:
                    better_in_all = False
                    break
                elif solution_a[i] < solution_b[i]:
                    strictly_better = True
                    
        return better_in_all and strictly_better


class TemplateGenerator:
    """
    Generates detector templates from Pareto-optimal solutions.
    """
    
    def __init__(self):
        """Initialize template generator."""
        self.pareto_analyzer = ParetoAnalyzer()
        
        # Template criteria definitions
        self.template_criteria = {
            'high_sensitivity': {
                'name': 'High Sensitivity',
                'description': 'Optimized for early drift detection with higher recall',
                'use_cases': [
                    'Critical system monitoring',
                    'Early warning systems', 
                    'Cost-sensitive applications where missing drift is expensive'
                ],
                'preferences': {
                    'recall_weight': 0.5,
                    'fp_tolerance': 0.3,
                    'delay_tolerance': 100
                },
                'selection_criteria': {
                    'min_recall': 0.8,
                    'max_delay': 100,
                    'fp_weight': 0.2
                }
            },
            
            'balanced': {
                'name': 'Balanced',
                'description': 'Optimal balance between detection accuracy and false alarms',
                'use_cases': [
                    'General monitoring systems',
                    'Resource-limited environments',
                    'Daily operations monitoring'
                ],
                'preferences': {
                    'f1_weight': 0.6,
                    'fp_weight': 0.2,
                    'delay_weight': 0.2
                },
                'selection_criteria': {
                    'min_f1': 0.7,
                    'max_fp_rate': 0.2,
                    'max_delay': 150
                }
            },
            
            'high_stability': {
                'name': 'High Stability',
                'description': 'Minimizes false alarms with stable detection',
                'use_cases': [
                    'Production environments',
                    'Automated decision systems',
                    'Low-interference monitoring'
                ],
                'preferences': {
                    'precision_weight': 0.6,
                    'fp_weight': 0.3,
                    'delay_tolerance': 300
                },
                'selection_criteria': {
                    'max_fp_rate': 0.1,
                    'min_precision': 0.8,
                    'delay_tolerance': 300
                }
            }
        }
    
    def generate_templates(
        self,
        detector_name: str,
        search_results: List[SearchResult]
    ) -> Dict[str, DetectorTemplate]:
        """
        Generate detector templates from search results using new selection functions.
        
        Args:
            detector_name: Name of the detector
            search_results: List of search results
            
        Returns:
            Dictionary mapping template names to DetectorTemplate objects
        """
        if not search_results:
            return {}
            
        templates = {}
        
        # Use new selection functions for each template type
        template_selectors = {
            'high_sensitivity': high_sensitivity,
            'balanced': balanced,
            'high_stability': high_stability
        }
        
        for template_name, selector_func in template_selectors.items():
            best_result = selector_func(search_results, detector_name)
            
            if best_result:
                criteria = self.template_criteria[template_name]
                template = DetectorTemplate(
                    name=f"{detector_name.upper()} {criteria['name']}",
                    description=f"{criteria['description']} configuration for {detector_name}",
                    use_cases=criteria['use_cases'].copy(),
                    parameters=best_result.parameters.copy(),
                    expected_performance={
                        'f1_score': best_result.metrics.f1_score,
                        'precision': best_result.metrics.precision,
                        'recall': best_result.metrics.recall,
                        'mean_delay': best_result.metrics.mean_delay,
                        'false_positive_rate': best_result.metrics.false_positive_rate,
                        'composite_score': best_result.score
                    },
                    confidence_score=self._calculate_confidence_score(best_result, template_name),
                    pareto_rank=0  # Will be updated if needed
                )
                templates[template_name] = template
                
        return templates
    
    def _calculate_confidence_score(self, result: SearchResult, template_type: str) -> float:
        """
        Calculate confidence score for a template based on its performance and type alignment.
        
        Args:
            result: Search result to calculate confidence for
            template_type: Type of template ('high_sensitivity', 'balanced', 'high_stability')
            
        Returns:
            Confidence score between 0 and 1
        """
        metrics = result.metrics
        base_score = metrics.f1_score * 0.6 + (1 - metrics.false_positive_rate) * 0.4
        
        # Type-specific bonuses
        type_bonus = 0.0
        if template_type == 'high_sensitivity':
            # Bonus for high recall and F1
            type_bonus = (metrics.recall * 0.3 + metrics.f1_score * 0.2) * 0.5
        elif template_type == 'balanced':
            # Bonus for balanced performance
            balance_score = min(metrics.precision, metrics.recall, 1 - metrics.false_positive_rate)
            type_bonus = balance_score * 0.3
        elif template_type == 'high_stability':
            # Bonus for low FP and high precision
            stability_score = (1 - metrics.false_positive_rate) * 0.5 + metrics.precision * 0.5
            type_bonus = stability_score * 0.4
            
        return min(1.0, base_score + type_bonus)
    
    def _select_template_candidate(
        self,
        detector_name: str,
        template_name: str,
        criteria: Dict[str, Any],
        pareto_solutions: List[SearchResult]
    ) -> Optional[DetectorTemplate]:
        """
        Select best candidate for a specific template type.
        
        Args:
            detector_name: Name of detector
            template_name: Name of template type
            criteria: Selection criteria
            pareto_solutions: Pareto-optimal solutions
            
        Returns:
            DetectorTemplate or None if no suitable candidate found
        """
        # Score each solution for this template type
        scored_solutions = []
        
        for solution in pareto_solutions:
            score = self._compute_template_score(solution, criteria)
            if score > 0:  # Only consider solutions that meet basic criteria
                scored_solutions.append((solution, score))
                
        if not scored_solutions:
            return None
            
        # Select best scoring solution
        scored_solutions.sort(key=lambda x: x[1], reverse=True)
        best_solution, best_score = scored_solutions[0]
        
        # Find Pareto rank
        pareto_rank = pareto_solutions.index(best_solution) + 1
        
        # Create template
        template = DetectorTemplate(
            name=f"{detector_name.upper()} {criteria['name']}",
            description=f"{criteria['description']} configuration for {detector_name}",
            use_cases=criteria['use_cases'].copy(),
            parameters=best_solution.parameters.copy(),
            expected_performance={
                'f1_score': best_solution.metrics.f1_score,
                'precision': best_solution.metrics.precision,
                'recall': best_solution.metrics.recall,
                'mean_delay': best_solution.metrics.mean_delay,
                'false_positive_rate': best_solution.metrics.false_positive_rate,
                'composite_score': best_solution.score
            },
            confidence_score=best_score,
            pareto_rank=pareto_rank
        )
        
        return template
    
    def _compute_template_score(
        self,
        solution: SearchResult,
        criteria: Dict[str, Any]
    ) -> float:
        """
        Compute template-specific score for a solution.
        
        Args:
            solution: Search result to score
            criteria: Template criteria
            
        Returns:
            Template score (higher is better)
        """
        metrics = solution.metrics
        selection_criteria = criteria.get('selection_criteria', {})
        
        score = 0.0
        penalty = 0.0
        
        # Check hard constraints first
        if 'min_recall' in selection_criteria:
            if metrics.recall < selection_criteria['min_recall']:
                return 0.0  # Hard constraint violated
            score += metrics.recall * 0.3
            
        if 'min_f1' in selection_criteria:
            if metrics.f1_score < selection_criteria['min_f1']:
                return 0.0
            score += metrics.f1_score * 0.4
            
        if 'min_precision' in selection_criteria:
            if metrics.precision < selection_criteria['min_precision']:
                return 0.0
            score += metrics.precision * 0.3
            
        if 'max_fp_rate' in selection_criteria:
            if metrics.false_positive_rate > selection_criteria['max_fp_rate']:
                penalty += (metrics.false_positive_rate - selection_criteria['max_fp_rate']) * 2.0
                
        if 'max_delay' in selection_criteria:
            if metrics.mean_delay > selection_criteria['max_delay']:
                penalty += (metrics.mean_delay - selection_criteria['max_delay']) / 1000.0
                
        # Soft preferences
        preferences = criteria.get('preferences', {})
        
        if 'recall_weight' in preferences:
            score += metrics.recall * preferences['recall_weight']
            
        if 'f1_weight' in preferences:
            score += metrics.f1_score * preferences['f1_weight']
            
        if 'precision_weight' in preferences:
            score += metrics.precision * preferences['precision_weight']
            
        if 'fp_weight' in preferences:
            score -= metrics.false_positive_rate * preferences['fp_weight']
            
        if 'delay_weight' in preferences:
            normalized_delay = metrics.mean_delay / 1000.0
            score -= normalized_delay * preferences['delay_weight']
        
        return max(0.0, score - penalty)
    
    def generate_all_detector_templates(
        self,
        optimization_results: Dict[str, List[SearchResult]]
    ) -> Dict[str, Dict[str, DetectorTemplate]]:
        """
        Generate templates for all detectors.
        
        Args:
            optimization_results: Results from hyperparameter optimization
            
        Returns:
            Nested dict: {detector_name: {template_name: template}}
        """
        all_templates = {}
        
        for detector_name, search_results in optimization_results.items():
            if search_results:
                templates = self.generate_templates(detector_name, search_results)
                if templates:
                    all_templates[detector_name] = templates
                    
        return all_templates


class TemplateRecommender:
    """
    Recommends detector templates based on application requirements.
    """
    
    def __init__(self):
        """Initialize template recommender."""
        self.scenario_preferences = {
            'critical_systems': {
                'priority': 'high_sensitivity',
                'requirements': {
                    'min_recall': 0.9,
                    'max_delay': 50
                }
            },
            'production_monitoring': {
                'priority': 'high_stability',
                'requirements': {
                    'max_fp_rate': 0.05,
                    'min_precision': 0.85
                }
            },
            'general_purpose': {
                'priority': 'balanced',
                'requirements': {
                    'min_f1': 0.75,
                    'max_fp_rate': 0.15
                }
            },
            'research_evaluation': {
                'priority': 'balanced',
                'requirements': {
                    'min_f1': 0.70
                }
            }
        }
    
    def recommend_template(
        self,
        all_templates: Dict[str, Dict[str, DetectorTemplate]],
        scenario: str = 'general_purpose',
        custom_requirements: Dict[str, float] = None
    ) -> List[Tuple[str, str, DetectorTemplate, float]]:
        """
        Recommend templates based on scenario and requirements.
        
        Args:
            all_templates: All available templates
            scenario: Application scenario
            custom_requirements: Custom performance requirements
            
        Returns:
            List of (detector, template_type, template, score) tuples, sorted by score
        """
        if scenario not in self.scenario_preferences:
            scenario = 'general_purpose'
            
        preferences = self.scenario_preferences[scenario]
        requirements = preferences['requirements'].copy()
        
        if custom_requirements:
            requirements.update(custom_requirements)
            
        recommendations = []
        
        for detector_name, detector_templates in all_templates.items():
            for template_name, template in detector_templates.items():
                score = self._score_template_for_scenario(
                    template, requirements, preferences
                )
                
                if score > 0:
                    recommendations.append((detector_name, template_name, template, score))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[3], reverse=True)
        
        return recommendations
    
    def _score_template_for_scenario(
        self,
        template: DetectorTemplate,
        requirements: Dict[str, float],
        preferences: Dict[str, Any]
    ) -> float:
        """Score a template for a specific scenario."""
        performance = template.expected_performance
        score = template.confidence_score  # Base score
        
        # Check hard requirements
        for req_name, req_value in requirements.items():
            if req_name.startswith('min_'):
                metric_name = req_name[4:]
                if metric_name in performance:
                    if performance[metric_name] < req_value:
                        return 0.0  # Hard requirement not met
                    else:
                        score += (performance[metric_name] - req_value) * 0.5
                        
            elif req_name.startswith('max_'):
                metric_name = req_name[4:]
                if metric_name in performance:
                    if performance[metric_name] > req_value:
                        return 0.0  # Hard requirement not met
                    else:
                        score += (req_value - performance[metric_name]) * 0.5
        
        # Bonus for matching preferred template type
        preferred_priority = preferences.get('priority', '')
        if preferred_priority in template.name.lower():
            score += 0.2
            
        return score


class TemplateExporter:
    """
    Exports detector templates to various formats.
    """
    
    @staticmethod
    def export_to_json(
        templates: Dict[str, Dict[str, DetectorTemplate]],
        filepath: str
    ):
        """Export templates to JSON format."""
        export_data = {}
        
        for detector_name, detector_templates in templates.items():
            export_data[detector_name] = {}
            
            for template_name, template in detector_templates.items():
                export_data[detector_name][template_name] = {
                    'name': template.name,
                    'description': template.description,
                    'use_cases': template.use_cases,
                    'parameters': template.parameters,
                    'expected_performance': template.expected_performance,
                    'confidence_score': template.confidence_score,
                    'pareto_rank': template.pareto_rank
                }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    @staticmethod
    def export_to_python(
        templates: Dict[str, Dict[str, DetectorTemplate]],
        filepath: str
    ):
        """Export templates as Python configuration file."""
        lines = [
            '"""',
            'Generated detector configuration templates.',
            'Auto-generated from hyperparameter optimization results.',
            '"""',
            '',
            'DETECTOR_TEMPLATES = {'
        ]
        
        for detector_name, detector_templates in templates.items():
            lines.append(f'    "{detector_name}": {{')
            
            for template_name, template in detector_templates.items():
                lines.append(f'        "{template_name}": {{')
                lines.append(f'            "name": "{template.name}",')
                lines.append(f'            "description": "{template.description}",')
                lines.append(f'            "use_cases": {template.use_cases},')
                lines.append(f'            "parameters": {template.parameters},')
                lines.append(f'            "expected_performance": {template.expected_performance},')
                lines.append(f'            "confidence_score": {template.confidence_score},')
                lines.append(f'            "pareto_rank": {template.pareto_rank}')
                lines.append('        },')
                
            lines.append('    },')
            
        lines.append('}')
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))


def high_sensitivity(results: List[SearchResult], detector_name: str = "") -> Optional[SearchResult]:
    """
    Select high-sensitivity configuration: highest F1 score with tolerance for higher FP.
    Prefers ADWIN with larger delta, PageHinkley with smaller threshold.
    
    Args:
        results: List of search results to select from
        detector_name: Name of detector for parameter-specific logic
        
    Returns:
        Best high-sensitivity configuration or None
    """
    if not results:
        return None
    
    # Filter candidates with minimum requirements
    candidates = []
    for result in results:
        metrics = result.metrics
        # High sensitivity: prioritize recall and F1, tolerate higher FP
        if (metrics.recall >= 0.75 and  # High recall requirement
            metrics.f1_score >= 0.70 and  # Good F1 score
            metrics.false_positive_rate <= 0.35):  # Tolerate higher FP
            
            # Calculate sensitivity score: emphasize F1 and recall
            score = (
                metrics.f1_score * 0.5 +      # Primary: F1 score
                metrics.recall * 0.4 +        # Secondary: high recall
                (1 - metrics.false_positive_rate) * 0.1  # Minor: FP tolerance
            )
            
            # Bonus for detector-specific preferred parameters
            param_bonus = 0.0
            if detector_name.lower() == 'adwin' and 'delta' in result.parameters:
                # Prefer larger delta for more sensitivity
                if result.parameters['delta'] >= 0.01:
                    param_bonus += 0.05
            elif detector_name.lower() == 'page_hinkley' and 'threshold' in result.parameters:
                # Prefer smaller threshold for more sensitivity
                if result.parameters['threshold'] <= 10:
                    param_bonus += 0.05
            
            candidates.append((result, score + param_bonus))
    
    if not candidates:
        return None
        
    # Return highest scoring candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def balanced(results: List[SearchResult], detector_name: str = "") -> Optional[SearchResult]:
    """
    Select balanced configuration: optimal trade-off between F1, delay, and FP rate.
    Balanced parameters across all detector types.
    
    Args:
        results: List of search results to select from  
        detector_name: Name of detector for parameter-specific logic
        
    Returns:
        Best balanced configuration or None
    """
    if not results:
        return None
    
    candidates = []
    for result in results:
        metrics = result.metrics
        # Balanced: good performance across all metrics
        if (metrics.f1_score >= 0.65 and          # Decent F1 score
            metrics.false_positive_rate <= 0.20 and  # Controlled FP rate
            metrics.mean_delay <= 200):           # Reasonable delay
            
            # Calculate balanced score: equal weight to key metrics
            f1_normalized = metrics.f1_score  # 0-1, higher better
            fp_normalized = 1 - metrics.false_positive_rate  # 0-1, higher better  
            delay_normalized = max(0, 1 - metrics.mean_delay / 500)  # 0-1, higher better
            
            score = (
                f1_normalized * 0.4 +    # Primary: F1 score
                fp_normalized * 0.3 +    # Secondary: low FP rate
                delay_normalized * 0.3   # Secondary: low delay
            )
            
            # Bonus for detector-specific balanced parameters
            param_bonus = 0.0
            if detector_name.lower() == 'adwin' and 'delta' in result.parameters:
                # Prefer moderate delta values
                delta = result.parameters['delta']
                if 0.001 <= delta <= 0.01:
                    param_bonus += 0.03
            elif detector_name.lower() == 'kswin' and 'alpha' in result.parameters:
                # Prefer moderate alpha values
                alpha = result.parameters['alpha']
                if 0.001 <= alpha <= 0.01:
                    param_bonus += 0.03
                    
            candidates.append((result, score + param_bonus))
    
    if not candidates:
        return None
        
    # Return highest scoring candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def high_stability(results: List[SearchResult], detector_name: str = "") -> Optional[SearchResult]:
    """
    Select high-stability configuration: minimize FP rate, tolerating higher delay.
    Prefers ADWIN with smaller delta, PageHinkley with larger threshold, conservative KSWIN alpha.
    
    Args:
        results: List of search results to select from
        detector_name: Name of detector for parameter-specific logic
        
    Returns:
        Best high-stability configuration or None
    """
    if not results:
        return None
    
    candidates = []
    for result in results:
        metrics = result.metrics
        # High stability: minimize FP, tolerate delay
        if (metrics.false_positive_rate <= 0.10 and  # Very low FP rate
            metrics.precision >= 0.75):              # High precision
            
            # Calculate stability score: emphasize precision and low FP
            fp_score = 1 - metrics.false_positive_rate  # Higher is better
            precision_score = metrics.precision
            f1_score = metrics.f1_score
            
            score = (
                fp_score * 0.4 +        # Primary: very low FP rate
                precision_score * 0.4 + # Primary: high precision  
                f1_score * 0.2          # Secondary: maintain F1
            )
            
            # Bonus for detector-specific conservative parameters
            param_bonus = 0.0
            if detector_name.lower() == 'adwin' and 'delta' in result.parameters:
                # Prefer smaller delta for more stability
                if result.parameters['delta'] <= 0.001:
                    param_bonus += 0.05
            elif detector_name.lower() == 'page_hinkley' and 'threshold' in result.parameters:
                # Prefer larger threshold for more stability
                if result.parameters['threshold'] >= 30:
                    param_bonus += 0.05
            elif detector_name.lower() == 'kswin' and 'alpha' in result.parameters:
                # Prefer smaller alpha for more conservative detection
                if result.parameters['alpha'] <= 0.001:
                    param_bonus += 0.05
                    
            candidates.append((result, score + param_bonus))
    
    if not candidates:
        return None
        
    # Return highest scoring candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def generate_detector_templates(
    optimization_results: Dict[str, List[SearchResult]],
    output_dir: str = "results"
) -> Dict[str, Dict[str, DetectorTemplate]]:
    """
    Generate and export detector templates from optimization results.
    
    Args:
        optimization_results: Results from hyperparameter optimization
        output_dir: Output directory for template files
        
    Returns:
        Generated templates
    """
    generator = TemplateGenerator()
    all_templates = generator.generate_all_detector_templates(optimization_results)
    
    if not all_templates:
        print("No templates could be generated from optimization results")
        return {}
    
    # Export templates
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON export
    json_path = output_path / "detector_templates.json"
    TemplateExporter.export_to_json(all_templates, str(json_path))
    print(f"Templates exported to {json_path}")
    
    # Python config export
    py_path = output_path / "detector_templates.py"
    TemplateExporter.export_to_python(all_templates, str(py_path))
    print(f"Templates exported to {py_path}")
    
    # Generate recommendations report
    recommender = TemplateRecommender()
    
    scenarios = ['critical_systems', 'production_monitoring', 'general_purpose']
    recommendations_report = {}
    
    for scenario in scenarios:
        recommendations = recommender.recommend_template(all_templates, scenario)
        recommendations_report[scenario] = []
        
        for detector, template_type, template, score in recommendations[:3]:  # Top 3
            recommendations_report[scenario].append({
                'detector': detector,
                'template_type': template_type,
                'template_name': template.name,
                'score': score,
                'expected_f1': template.expected_performance.get('f1_score', 0.0),
                'expected_fp_rate': template.expected_performance.get('false_positive_rate', 0.0)
            })
    
    # Export recommendations
    rec_path = output_path / "template_recommendations.json"
    with open(rec_path, 'w') as f:
        json.dump(recommendations_report, f, indent=2)
    
    print(f"Template recommendations exported to {rec_path}")
    print(f"\nGenerated {sum(len(templates) for templates in all_templates.values())} templates "
          f"for {len(all_templates)} detectors")
    
    return all_templates