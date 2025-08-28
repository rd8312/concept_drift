"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.detectors import create_detector
from src.datasets import create_dataset
from src.metrics import DriftEvaluator


@pytest.fixture
def sample_drift_data():
    """Generate sample drift detection data for testing."""
    np.random.seed(42)
    
    # Generate data with known drift points
    data = []
    drift_positions = [100, 200, 300]
    
    for i in range(400):
        if i < 100:
            # Concept 1
            value = np.random.normal(0.3, 0.1)
        elif i < 200:
            # Concept 2  
            value = np.random.normal(0.7, 0.1)
        elif i < 300:
            # Concept 3
            value = np.random.normal(0.5, 0.15)
        else:
            # Concept 4
            value = np.random.normal(0.9, 0.1)
            
        is_drift = i in drift_positions
        label = 1 if value > 0.5 else 0
        
        data.append((value, label, is_drift))
    
    return data, drift_positions


@pytest.fixture
def sample_detectors():
    """Create sample detectors for testing."""
    return {
        'adwin': create_detector('adwin', delta=0.002),
        'ddm': create_detector('ddm', min_num_instances=30),
        'eddm': create_detector('eddm', min_num_instances=30),
        'pagehinkley': create_detector('pagehinkley', threshold=50),
        'kswin': create_detector('kswin', alpha=0.005, window_size=100)
    }


@pytest.fixture
def drift_evaluator():
    """Create a drift evaluator for testing."""
    return DriftEvaluator(tolerance=25, delay_penalty=0.001)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sea_dataset_small():
    """Generate small SEA dataset for testing."""
    config = {
        'drift_positions': [50, 100],
        'n_samples': 150,
        'noise_level': 0.02,
        'seed': 42
    }
    return list(create_dataset('sea', config))


@pytest.fixture
def detection_results_sample():
    """Sample detection results for testing evaluation functions."""
    return [
        {
            'detector': 'adwin',
            'dataset': 'sea',
            'parameters': {'delta': 0.002},
            'detected_drifts': [55, 105],
            'true_drifts': [50, 100],
            'metrics': {
                'tp': 2,
                'fp': 0,
                'fn': 0,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'mean_delay': 5.0,
                'false_positive_rate': 0.01,
                'composite_score': 0.99
            }
        },
        {
            'detector': 'ddm',
            'dataset': 'sea', 
            'parameters': {'warning_level': 2.0},
            'detected_drifts': [48, 102, 130],
            'true_drifts': [50, 100],
            'metrics': {
                'tp': 2,
                'fp': 1,
                'fn': 0,
                'precision': 0.67,
                'recall': 1.0,
                'f1_score': 0.8,
                'mean_delay': 0.0,
                'false_positive_rate': 0.007,
                'composite_score': 0.79
            }
        }
    ]


@pytest.fixture(scope="session")
def optimization_results_mock():
    """Mock optimization results for testing template generation."""
    from src.search import SearchResult
    from src.metrics import DriftMetrics
    
    # Create mock search results for ADWIN
    adwin_results = []
    
    # High sensitivity configuration
    metrics1 = DriftMetrics(
        tp=9, fp=3, fn=1,
        precision=0.75, recall=0.9, f1_score=0.82,
        mean_delay=30.0, false_positive_rate=0.15,
        composite_score=0.81
    )
    result1 = SearchResult(
        parameters={'delta': 0.0001, 'clock': 32},
        score=0.81,
        metrics=metrics1,
        detector_type='adwin'
    )
    
    # Balanced configuration
    metrics2 = DriftMetrics(
        tp=8, fp=1, fn=2,
        precision=0.89, recall=0.8, f1_score=0.84,
        mean_delay=45.0, false_positive_rate=0.05,
        composite_score=0.83
    )
    result2 = SearchResult(
        parameters={'delta': 0.002, 'clock': 64},
        score=0.83,
        metrics=metrics2,
        detector_type='adwin'
    )
    
    # High stability configuration
    metrics3 = DriftMetrics(
        tp=7, fp=0, fn=3,
        precision=1.0, recall=0.7, f1_score=0.82,
        mean_delay=65.0, false_positive_rate=0.02,
        composite_score=0.81
    )
    result3 = SearchResult(
        parameters={'delta': 0.01, 'clock': 128},
        score=0.81,
        metrics=metrics3,
        detector_type='adwin'
    )
    
    adwin_results = [result1, result2, result3]
    
    return {'adwin': adwin_results}


# Helper functions for tests
def assert_drift_metrics_valid(metrics):
    """Assert that DriftMetrics object has valid values."""
    assert 0 <= metrics.precision <= 1.0
    assert 0 <= metrics.recall <= 1.0
    assert 0 <= metrics.f1_score <= 1.0
    assert metrics.mean_delay >= 0
    assert metrics.false_positive_rate >= 0
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0


def generate_synthetic_drift_stream(n_samples=1000, drift_positions=None, seed=42):
    """Generate synthetic data stream with concept drift."""
    np.random.seed(seed)
    
    if drift_positions is None:
        drift_positions = [n_samples // 3, 2 * n_samples // 3]
    
    data = []
    current_concept = 0
    
    for i in range(n_samples):
        # Check for drift
        is_drift = i in drift_positions
        if is_drift:
            current_concept += 1
        
        # Generate data based on current concept
        if current_concept == 0:
            value = np.random.normal(0.3, 0.1)
        elif current_concept == 1:
            value = np.random.normal(0.7, 0.1)
        else:
            value = np.random.normal(0.5, 0.2)
        
        data.append((value, is_drift))
    
    return data, drift_positions


# Pytest markers for test organization
pytest_markers = [
    "unit: mark test as a unit test",
    "integration: mark test as an integration test", 
    "slow: mark test as slow running",
    "requires_data: mark test as requiring external data"
]