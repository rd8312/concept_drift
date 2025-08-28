"""
Evaluation metrics for concept drift detection.
Provides window matching, F1 calculation, delay computation, and composite objectives.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DriftMetrics:
    """Data class to store drift detection metrics."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_delay: float = 0.0
    delays: List[float] = None
    false_positive_rate: float = 0.0
    composite_score: float = 0.0
    
    def __post_init__(self):
        if self.delays is None:
            self.delays = []


class WindowMatcher:
    """
    Window-based matching algorithm for drift detection evaluation.
    Matches detected drift points to true drift points within a tolerance window.
    """
    
    def __init__(self, tolerance: int = 50):
        """
        Initialize window matcher.
        
        Args:
            tolerance: Matching tolerance in samples (±tolerance)
        """
        self.tolerance = tolerance
        
    def match_drifts(
        self,
        detected_drifts: List[int],
        true_drifts: List[int]
    ) -> Tuple[int, int, int, List[float]]:
        """
        Match detected drift points to true drift points.
        
        Args:
            detected_drifts: List of detected drift positions
            true_drifts: List of true drift positions
            
        Returns:
            Tuple of (tp, fp, fn, delays)
        """
        tp = 0
        fp = 0
        delays = []
        matched_true = set()
        
        # Match detected drifts to true drifts
        for detected in detected_drifts:
            matched = False
            best_match_delay = float('inf')
            best_match_idx = -1
            
            for i, true_drift in enumerate(true_drifts):
                if i in matched_true:
                    continue
                    
                # Check if within tolerance window
                if abs(detected - true_drift) <= self.tolerance:
                    delay = detected - true_drift
                    if abs(delay) < abs(best_match_delay):
                        best_match_delay = delay
                        best_match_idx = i
                        matched = True
                        
            if matched:
                tp += 1
                matched_true.add(best_match_idx)
                delays.append(max(0, best_match_delay))  # Only positive delays
            else:
                fp += 1
                
        # Count false negatives (unmatched true drifts)
        fn = len(true_drifts) - len(matched_true)
        
        return tp, fp, fn, delays


class DriftEvaluator:
    """
    Comprehensive evaluator for drift detection performance.
    Calculates precision, recall, F1, delay metrics, and composite scores.
    """
    
    def __init__(
        self,
        tolerance: int = 50,
        delay_penalty: float = 0.002,
        max_delay_normalize: float = 1000.0
    ):
        """
        Initialize drift evaluator.
        
        Args:
            tolerance: Window matching tolerance
            delay_penalty: Lambda penalty for delay in composite score
            max_delay_normalize: Maximum delay for normalization
        """
        self.matcher = WindowMatcher(tolerance=tolerance)
        self.delay_penalty = delay_penalty
        self.max_delay_normalize = max_delay_normalize
        
    def evaluate(
        self,
        detected_drifts: List[int],
        true_drifts: List[int],
        stream_length: int = None
    ) -> DriftMetrics:
        """
        Comprehensive evaluation of drift detection performance.
        
        Args:
            detected_drifts: List of detected drift positions
            true_drifts: List of true drift positions
            stream_length: Total length of data stream (for FP rate)
            
        Returns:
            DriftMetrics object with all computed metrics
        """
        # Window matching
        tp, fp, fn, delays = self.matcher.match_drifts(detected_drifts, true_drifts)
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Delay metrics
        mean_delay = np.mean(delays) if delays else 0.0
        
        # False positive rate (if stream length is known)
        fp_rate = 0.0
        if stream_length:
            # Approximate non-drift points
            non_drift_points = stream_length - len(true_drifts)
            fp_rate = fp / non_drift_points if non_drift_points > 0 else 0.0
            
        # Composite score: F1 - λ * normalized_delay
        normalized_delay = mean_delay / self.max_delay_normalize
        composite_score = f1_score - self.delay_penalty * normalized_delay
        
        return DriftMetrics(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_delay=mean_delay,
            delays=delays,
            false_positive_rate=fp_rate,
            composite_score=composite_score
        )
        
    def evaluate_multiple_runs(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Aggregate evaluation across multiple runs.
        
        Args:
            results: List of evaluation results from multiple runs
            
        Returns:
            Dictionary with aggregated statistics
        """
        metrics_lists = {
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'mean_delays': [],
            'fp_rates': [],
            'composite_scores': []
        }
        
        for result in results:
            if 'metrics' in result:
                metrics = result['metrics']
                metrics_lists['f1_scores'].append(metrics.f1_score)
                metrics_lists['precisions'].append(metrics.precision)
                metrics_lists['recalls'].append(metrics.recall)
                metrics_lists['mean_delays'].append(metrics.mean_delay)
                metrics_lists['fp_rates'].append(metrics.false_positive_rate)
                metrics_lists['composite_scores'].append(metrics.composite_score)
                
        # Compute aggregated statistics
        aggregated = {}
        for metric_name, values in metrics_lists.items():
            if values:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_median'] = np.median(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)
                
        return aggregated


class DelayAnalyzer:
    """
    Specialized analyzer for detection delay patterns.
    """
    
    @staticmethod
    def compute_delay_statistics(delays: List[float]) -> Dict[str, float]:
        """
        Compute comprehensive delay statistics.
        
        Args:
            delays: List of detection delays
            
        Returns:
            Dictionary with delay statistics
        """
        if not delays:
            return {
                'mean_delay': 0.0,
                'median_delay': 0.0,
                'std_delay': 0.0,
                'min_delay': 0.0,
                'max_delay': 0.0,
                'q25_delay': 0.0,
                'q75_delay': 0.0
            }
            
        delays_array = np.array(delays)
        
        return {
            'mean_delay': float(np.mean(delays_array)),
            'median_delay': float(np.median(delays_array)),
            'std_delay': float(np.std(delays_array)),
            'min_delay': float(np.min(delays_array)),
            'max_delay': float(np.max(delays_array)),
            'q25_delay': float(np.percentile(delays_array, 25)),
            'q75_delay': float(np.percentile(delays_array, 75))
        }
        
    @staticmethod
    def delay_distribution_analysis(delays: List[float], bins: int = 10) -> Dict[str, Any]:
        """
        Analyze delay distribution.
        
        Args:
            delays: List of detection delays
            bins: Number of histogram bins
            
        Returns:
            Dictionary with distribution analysis
        """
        if not delays:
            return {'histogram': [], 'bin_edges': []}
            
        hist, bin_edges = np.histogram(delays, bins=bins)
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'total_delays': len(delays)
        }


class CompositeObjective:
    """
    Composite objective function for multi-criteria optimization.
    Combines F1 score, false positive rate, and detection delay.
    """
    
    def __init__(
        self,
        f1_weight: float = 0.6,
        fp_weight: float = 0.2,
        delay_weight: float = 0.2,
        delay_penalty: float = 0.002,
        max_delay_normalize: float = 1000.0
    ):
        """
        Initialize composite objective.
        
        Args:
            f1_weight: Weight for F1 score component
            fp_weight: Weight for false positive penalty
            delay_weight: Weight for delay penalty
            delay_penalty: Scaling factor for delay penalty
            max_delay_normalize: Maximum delay for normalization
        """
        self.f1_weight = f1_weight
        self.fp_weight = fp_weight
        self.delay_weight = delay_weight
        self.delay_penalty = delay_penalty
        self.max_delay_normalize = max_delay_normalize
        
        # Normalize weights
        total_weight = f1_weight + fp_weight + delay_weight
        self.f1_weight /= total_weight
        self.fp_weight /= total_weight
        self.delay_weight /= total_weight
        
    def compute_score(self, metrics: DriftMetrics) -> float:
        """
        Compute composite objective score.
        
        Args:
            metrics: DriftMetrics object with evaluation results
            
        Returns:
            Composite objective score (higher is better)
        """
        # F1 score component (maximize)
        f1_component = self.f1_weight * metrics.f1_score
        
        # False positive penalty (minimize)
        fp_penalty = self.fp_weight * metrics.false_positive_rate
        
        # Delay penalty (minimize)
        normalized_delay = metrics.mean_delay / self.max_delay_normalize
        delay_penalty = self.delay_weight * self.delay_penalty * normalized_delay
        
        # Composite score
        composite_score = f1_component - fp_penalty - delay_penalty
        
        return composite_score
        
    def compute_batch_scores(self, metrics_list: List[DriftMetrics]) -> List[float]:
        """
        Compute composite scores for a batch of metrics.
        
        Args:
            metrics_list: List of DriftMetrics objects
            
        Returns:
            List of composite scores
        """
        return [self.compute_score(metrics) for metrics in metrics_list]


def evaluate_detector_performance(
    detected_drifts: List[int],
    true_drifts: List[int],
    stream_length: int = None,
    tolerance: int = 50,
    delay_penalty: float = 0.002
) -> DriftMetrics:
    """
    Convenience function for quick detector evaluation.
    
    Args:
        detected_drifts: List of detected drift positions
        true_drifts: List of true drift positions
        stream_length: Total stream length
        tolerance: Window matching tolerance
        delay_penalty: Delay penalty coefficient
        
    Returns:
        DriftMetrics with evaluation results
    """
    evaluator = DriftEvaluator(
        tolerance=tolerance,
        delay_penalty=delay_penalty
    )
    
    return evaluator.evaluate(detected_drifts, true_drifts, stream_length)


def compute_roc_like_curve(
    detection_results: List[Dict[str, Any]],
    parameter_name: str = 'threshold'
) -> Dict[str, List[float]]:
    """
    Compute ROC-like curve data for drift detection.
    
    Args:
        detection_results: List of detection results with different parameter values
        parameter_name: Name of the parameter being varied
        
    Returns:
        Dictionary with parameter values, TPR, FPR data
    """
    tprs = []
    fprs = []
    parameters = []
    
    for result in detection_results:
        if 'metrics' in result and parameter_name in result:
            metrics = result['metrics']
            parameters.append(result[parameter_name])
            
            # True Positive Rate (Recall)
            tprs.append(metrics.recall)
            
            # False Positive Rate
            fprs.append(metrics.false_positive_rate)
            
    return {
        'parameters': parameters,
        'tpr': tprs,
        'fpr': fprs
    }