"""
Drift Detector Tuner - Hyperparameter optimization for concept drift detectors.

This package provides a comprehensive framework for evaluating and optimizing
drift detection algorithms using River's streaming ML library.
"""

__version__ = "0.1.0"
__author__ = "Drift Detector Tuner Team"

from .detectors import (
    create_detector,
    get_all_detector_names,
    get_detector_parameter_ranges,
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    PageHinkleyDetector,
    KSWINDetector
)

from .datasets import create_dataset
from .metrics import evaluate_detector_performance, DriftMetrics
from .search import HybridSearch
from .evaluate import EvaluationFramework, run_comprehensive_evaluation
from .presets import generate_detector_templates

__all__ = [
    'create_detector',
    'get_all_detector_names', 
    'get_detector_parameter_ranges',
    'ADWINDetector',
    'DDMDetector',
    'EDDMDetector',
    'PageHinkleyDetector',
    'KSWINDetector',
    'create_dataset',
    'evaluate_detector_performance',
    'DriftMetrics',
    'HybridSearch',
    'EvaluationFramework',
    'run_comprehensive_evaluation',
    'generate_detector_templates'
]