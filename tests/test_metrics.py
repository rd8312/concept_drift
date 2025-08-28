"""
Tests for metrics module - window matching and evaluation metrics.
"""

import pytest
import numpy as np
from src.metrics import (
    WindowMatcher, 
    DriftEvaluator, 
    DriftMetrics,
    DelayAnalyzer,
    CompositeObjective,
    evaluate_detector_performance,
    compute_roc_like_curve
)


class TestWindowMatcher:
    """Test cases for WindowMatcher class."""
    
    def test_perfect_match(self):
        """Test perfect matching of drift points."""
        matcher = WindowMatcher(tolerance=50)
        detected = [100, 200, 300]
        true = [100, 200, 300]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 3
        assert fp == 0
        assert fn == 0
        assert delays == [0, 0, 0]
    
    def test_no_detections(self):
        """Test case with no detections."""
        matcher = WindowMatcher(tolerance=50)
        detected = []
        true = [100, 200, 300]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 0
        assert fp == 0
        assert fn == 3
        assert delays == []
    
    def test_false_positives(self):
        """Test false positive detections."""
        matcher = WindowMatcher(tolerance=50)
        detected = [50, 150, 250, 350, 450]  # Some don't match true drifts
        true = [100, 300]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 2  # 150 matches 100, 350 matches 300
        assert fp == 3  # 50, 250, 450 are false positives
        assert fn == 0  # All true drifts detected
    
    def test_delayed_detection(self):
        """Test detection with delays."""
        matcher = WindowMatcher(tolerance=50)
        detected = [120, 230, 280]  # Delayed detections
        true = [100, 200, 300]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 3
        assert fp == 0
        assert fn == 0
        assert delays == [20, 30, 0]  # Only positive delays counted
    
    def test_tolerance_window(self):
        """Test tolerance window matching."""
        matcher = WindowMatcher(tolerance=25)
        detected = [120, 180]  # One within tolerance, one outside
        true = [100, 200]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 1  # Only 120 is within tolerance of 100
        assert fp == 1  # 180 is outside tolerance of 200
        assert fn == 1  # 200 is unmatched
    
    def test_multiple_detections_single_drift(self):
        """Test multiple detections near single drift point."""
        matcher = WindowMatcher(tolerance=50)
        detected = [90, 110, 120]  # Multiple detections near 100
        true = [100]
        
        tp, fp, fn, delays = matcher.match_drifts(detected, true)
        
        assert tp == 1  # Only one can match
        assert fp == 2  # Others are false positives
        assert fn == 0
        assert len(delays) == 1


class TestDriftEvaluator:
    """Test cases for DriftEvaluator class."""
    
    def test_basic_evaluation(self):
        """Test basic drift evaluation."""
        evaluator = DriftEvaluator(tolerance=50, delay_penalty=0.001)
        detected = [120, 220, 320]
        true = [100, 200, 300]
        
        metrics = evaluator.evaluate(detected, true, stream_length=500)
        
        assert metrics.tp == 3
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.mean_delay > 0
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        evaluator = DriftEvaluator(tolerance=50, delay_penalty=0.002)
        detected = [150]  # High delay
        true = [100]
        
        metrics = evaluator.evaluate(detected, true, stream_length=1000)
        
        # Composite score should be F1 - penalty * normalized_delay
        expected_delay_penalty = 0.002 * (50 / 1000.0)
        expected_composite = metrics.f1_score - expected_delay_penalty
        
        assert abs(metrics.composite_score - expected_composite) < 1e-6
    
    def test_false_positive_rate_calculation(self):
        """Test false positive rate calculation with stream length."""
        evaluator = DriftEvaluator()
        detected = [50, 100, 150, 250]  # 2 FP, 2 TP
        true = [100, 250]
        stream_length = 1000
        
        metrics = evaluator.evaluate(detected, true, stream_length)
        
        # FP rate = FP / (stream_length - true_drifts)
        expected_fp_rate = 2 / (1000 - 2)
        
        assert abs(metrics.false_positive_rate - expected_fp_rate) < 1e-6
    
    def test_empty_results(self):
        """Test evaluation with empty inputs."""
        evaluator = DriftEvaluator()
        
        # No detections, no true drifts
        metrics = evaluator.evaluate([], [], 1000)
        assert metrics.tp == 0
        assert metrics.fp == 0  
        assert metrics.fn == 0
        assert metrics.f1_score == 0.0
        
        # No detections, some true drifts
        metrics = evaluator.evaluate([], [100, 200], 1000)
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 2
        assert metrics.recall == 0.0
        
        # Some detections, no true drifts
        metrics = evaluator.evaluate([100, 200], [], 1000)
        assert metrics.tp == 0
        assert metrics.fp == 2
        assert metrics.fn == 0
        assert metrics.precision == 0.0


class TestDelayAnalyzer:
    """Test cases for DelayAnalyzer class."""
    
    def test_delay_statistics(self):
        """Test delay statistics computation."""
        delays = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        stats = DelayAnalyzer.compute_delay_statistics(delays)
        
        assert stats['mean_delay'] == 30.0
        assert stats['median_delay'] == 30.0
        assert stats['std_delay'] == pytest.approx(15.811, rel=1e-3)
        assert stats['min_delay'] == 10.0
        assert stats['max_delay'] == 50.0
        assert stats['q25_delay'] == 20.0
        assert stats['q75_delay'] == 40.0
    
    def test_empty_delays(self):
        """Test delay statistics with empty input."""
        stats = DelayAnalyzer.compute_delay_statistics([])
        
        for key in stats:
            assert stats[key] == 0.0
    
    def test_delay_distribution_analysis(self):
        """Test delay distribution analysis."""
        delays = [1, 2, 3, 4, 5] * 10  # Create distribution
        
        distribution = DelayAnalyzer.delay_distribution_analysis(delays, bins=5)
        
        assert len(distribution['histogram']) == 5
        assert len(distribution['bin_edges']) == 6  # n+1 edges for n bins
        assert distribution['total_delays'] == 50


class TestCompositeObjective:
    """Test cases for CompositeObjective class."""
    
    def test_composite_score_computation(self):
        """Test composite objective score computation."""
        objective = CompositeObjective(
            f1_weight=0.6,
            fp_weight=0.2,
            delay_weight=0.2,
            delay_penalty=0.002
        )
        
        # Create mock metrics
        metrics = DriftMetrics(
            f1_score=0.8,
            false_positive_rate=0.1,
            mean_delay=100.0
        )
        
        score = objective.compute_score(metrics)
        
        # Manual calculation
        normalized_weights_sum = 0.6 + 0.2 + 0.2
        f1_component = (0.6 / normalized_weights_sum) * 0.8
        fp_penalty = (0.2 / normalized_weights_sum) * 0.1
        delay_penalty = (0.2 / normalized_weights_sum) * 0.002 * (100.0 / 1000.0)
        
        expected_score = f1_component - fp_penalty - delay_penalty
        
        assert abs(score - expected_score) < 1e-6
    
    def test_batch_scores(self):
        """Test batch score computation."""
        objective = CompositeObjective()
        
        metrics_list = [
            DriftMetrics(f1_score=0.8, false_positive_rate=0.1, mean_delay=50.0),
            DriftMetrics(f1_score=0.7, false_positive_rate=0.2, mean_delay=100.0)
        ]
        
        scores = objective.compute_batch_scores(metrics_list)
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestEvaluateDetectorPerformance:
    """Test cases for the convenience evaluation function."""
    
    def test_convenience_function(self):
        """Test the convenience evaluation function."""
        detected = [105, 205]
        true = [100, 200]
        
        metrics = evaluate_detector_performance(
            detected, true,
            stream_length=1000,
            tolerance=10,
            delay_penalty=0.001
        )
        
        assert metrics.tp == 2
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.f1_score == 1.0
        assert metrics.mean_delay == 5.0


class TestComputeROCLikeCurve:
    """Test cases for ROC-like curve computation."""
    
    def test_roc_curve_computation(self):
        """Test ROC-like curve data computation."""
        # Create mock detection results
        detection_results = [
            {
                'threshold': 0.1,
                'metrics': DriftMetrics(recall=0.9, false_positive_rate=0.3)
            },
            {
                'threshold': 0.5,
                'metrics': DriftMetrics(recall=0.7, false_positive_rate=0.1)
            },
            {
                'threshold': 0.9,
                'metrics': DriftMetrics(recall=0.5, false_positive_rate=0.05)
            }
        ]
        
        curve_data = compute_roc_like_curve(detection_results, 'threshold')
        
        assert curve_data['parameters'] == [0.1, 0.5, 0.9]
        assert curve_data['tpr'] == [0.9, 0.7, 0.5]
        assert curve_data['fpr'] == [0.3, 0.1, 0.05]
    
    def test_empty_results(self):
        """Test ROC curve with empty results."""
        curve_data = compute_roc_like_curve([], 'threshold')
        
        assert curve_data['parameters'] == []
        assert curve_data['tpr'] == []
        assert curve_data['fpr'] == []


# Integration tests
class TestMetricsIntegration:
    """Integration tests for metrics components."""
    
    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        # Simulate drift detection scenario
        detected_drifts = [95, 105, 195, 210, 295, 305, 400]  # Some accurate, some delayed, some FP
        true_drifts = [100, 200, 300]
        stream_length = 1000
        
        # Use evaluator
        evaluator = DriftEvaluator(tolerance=20, delay_penalty=0.001)
        metrics = evaluator.evaluate(detected_drifts, true_drifts, stream_length)
        
        # Verify reasonable results
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1  
        assert 0 <= metrics.f1_score <= 1
        assert metrics.tp + metrics.fn == len(true_drifts)
        assert metrics.tp + metrics.fp == len([d for d in detected_drifts if any(abs(d - t) <= 20 for t in true_drifts)]) + sum(1 for d in detected_drifts if not any(abs(d - t) <= 20 for t in true_drifts))
    
    def test_multiple_runs_aggregation(self):
        """Test aggregation across multiple evaluation runs."""
        evaluator = DriftEvaluator()
        
        # Create multiple mock results
        results = []
        for i in range(5):
            metrics = DriftMetrics(
                f1_score=0.7 + i * 0.05,
                precision=0.6 + i * 0.05,
                recall=0.8 + i * 0.02,
                mean_delay=50 + i * 10,
                false_positive_rate=0.1 + i * 0.02
            )
            results.append({'metrics': metrics})
        
        aggregated = evaluator.evaluate_multiple_runs(results)
        
        # Check aggregated statistics exist
        assert 'f1_scores_mean' in aggregated
        assert 'f1_scores_std' in aggregated
        assert 'precisions_mean' in aggregated
        assert 'recalls_median' in aggregated
        
        # Verify reasonable values
        assert 0.7 <= aggregated['f1_scores_mean'] <= 0.9
        assert aggregated['f1_scores_std'] >= 0