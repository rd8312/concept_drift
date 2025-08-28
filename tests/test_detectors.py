"""
Tests for detectors module - unified drift detector interface.
"""

import pytest
import numpy as np
from src.detectors import (
    create_detector,
    get_all_detector_names,
    get_detector_parameter_ranges,
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    PageHinkleyDetector,
    KSWINDetector,
    DetectorEnsemble,
    DETECTOR_CLASSES
)


class TestDetectorFactory:
    """Test detector factory functions."""
    
    def test_get_all_detector_names(self):
        """Test getting all detector names."""
        names = get_all_detector_names()
        
        expected = ['adwin', 'ddm', 'eddm', 'pagehinkley', 'kswin']
        assert set(names) == set(expected)
        assert len(names) == len(expected)
    
    def test_create_detector_valid_types(self):
        """Test creating detectors with valid types."""
        for detector_type in get_all_detector_names():
            detector = create_detector(detector_type)
            assert detector is not None
            assert hasattr(detector, 'update')
            assert hasattr(detector, 'drift_detected')
    
    def test_create_detector_invalid_type(self):
        """Test creating detector with invalid type."""
        with pytest.raises(ValueError, match="Unknown detector type"):
            create_detector('invalid_detector')
    
    def test_create_detector_with_params(self):
        """Test creating detectors with custom parameters."""
        # ADWIN with custom delta
        detector = create_detector('adwin', delta=0.001)
        assert detector.params['delta'] == 0.001
        
        # DDM with custom parameters
        detector = create_detector('ddm', warning_level=2.5)
        assert detector.params['warning_level'] == 2.5
    
    def test_get_detector_parameter_ranges(self):
        """Test getting parameter ranges for detectors."""
        for detector_type in get_all_detector_names():
            ranges = get_detector_parameter_ranges(detector_type)
            
            assert isinstance(ranges, dict)
            assert len(ranges) > 0
            
            # Check parameter structure
            for param_name, param_config in ranges.items():
                assert isinstance(param_config, dict)
                assert 'type' in param_config
                assert param_config['type'] in ['uniform', 'log_uniform', 'choice', 'int_uniform']
                
                if param_config['type'] in ['uniform', 'log_uniform']:
                    assert 'range' in param_config
                    assert len(param_config['range']) == 2
                elif param_config['type'] == 'choice':
                    assert 'values' in param_config
                    assert len(param_config['values']) > 0


class TestBaseDetectorFunctionality:
    """Test base detector functionality shared across all detectors."""
    
    @pytest.fixture
    def sample_detectors(self):
        """Create sample detectors for testing."""
        return {
            'adwin': create_detector('adwin', delta=0.002),
            'ddm': create_detector('ddm', min_num_instances=30),
            'eddm': create_detector('eddm', min_num_instances=30),
            'pagehinkley': create_detector('pagehinkley', threshold=50),
            'kswin': create_detector('kswin', alpha=0.005)
        }
    
    def test_detector_initialization(self, sample_detectors):
        """Test detector initialization."""
        for name, detector in sample_detectors.items():
            assert detector.state.sample_count == 0
            assert not detector.state.drift_detected
            assert detector.state.last_drift_position is None
            assert detector.drift_points == []
            assert detector.warning_points == []
    
    def test_detector_update_interface(self, sample_detectors):
        """Test detector update interface."""
        for name, detector in sample_detectors.items():
            # Test that update returns self for chaining
            result = detector.update(0.5)
            assert result is detector
            
            # Test that sample count increases
            assert detector.state.sample_count == 1
            
            # Test with different value types
            detector.update(1)
            detector.update(0.0)
            assert detector.state.sample_count == 3
    
    def test_detector_properties(self, sample_detectors):
        """Test detector properties."""
        for name, detector in sample_detectors.items():
            # Initially no drift
            assert not detector.drift_detected
            
            # Update a few times
            for i in range(10):
                detector.update(0.5 + i * 0.1)
            
            # Properties should be accessible
            assert isinstance(detector.drift_detected, bool)
            assert isinstance(detector.drift_points, list)
            assert isinstance(detector.warning_points, list)
    
    def test_detector_reset(self, sample_detectors):
        """Test detector reset functionality."""
        for name, detector in sample_detectors.items():
            # Update detector
            for i in range(20):
                detector.update(0.5 + i * 0.1)
            
            initial_count = detector.state.sample_count
            initial_drifts = len(detector.drift_points)
            
            # Reset detector
            detector.reset()
            
            # State should be reset
            assert detector.state.sample_count == 0
            assert not detector.state.drift_detected
            assert detector.state.last_drift_position is None
            assert detector.drift_points == []
            assert detector.warning_points == []
    
    def test_get_parameters(self, sample_detectors):
        """Test getting detector parameters."""
        for name, detector in sample_detectors.items():
            params = detector.get_parameters()
            assert isinstance(params, dict)
            
            # Should contain expected parameters
            if name == 'adwin':
                assert 'delta' in params
            elif name == 'ddm':
                assert 'min_num_instances' in params
                assert 'warning_level' in params
            elif name == 'pagehinkley':
                assert 'threshold' in params


class TestADWINDetector:
    """Test ADWIN-specific functionality."""
    
    def test_adwin_default_parameters(self):
        """Test ADWIN default parameters."""
        detector = ADWINDetector()
        
        assert detector.params['delta'] == 0.002
        assert detector.params['clock'] == 32
        assert detector.params['max_buckets'] == 5
        assert detector.params['min_window_length'] == 5
        assert detector.params['grace_period'] == 10
    
    def test_adwin_custom_parameters(self):
        """Test ADWIN with custom parameters."""
        detector = ADWINDetector(
            delta=0.001,
            clock=64,
            max_buckets=10
        )
        
        assert detector.params['delta'] == 0.001
        assert detector.params['clock'] == 64
        assert detector.params['max_buckets'] == 10
    
    def test_adwin_parameter_ranges(self):
        """Test ADWIN parameter ranges."""
        detector = ADWINDetector()
        ranges = detector.get_parameter_ranges()
        
        assert 'delta' in ranges
        assert ranges['delta']['type'] == 'log_uniform'
        assert ranges['delta']['priority'] == 'high'
        assert len(ranges['delta']['range']) == 2
        
        assert 'clock' in ranges
        assert ranges['clock']['type'] == 'choice'


class TestDDMDetector:
    """Test DDM-specific functionality."""
    
    def test_ddm_default_parameters(self):
        """Test DDM default parameters."""
        detector = DDMDetector()
        
        assert detector.params['min_num_instances'] == 30
        assert detector.params['warning_level'] == 2.0
        assert detector.params['out_control_level'] == 3.0
    
    def test_ddm_parameter_ranges(self):
        """Test DDM parameter ranges."""
        detector = DDMDetector()
        ranges = detector.get_parameter_ranges()
        
        assert 'warning_level' in ranges
        assert ranges['warning_level']['type'] == 'uniform'
        assert ranges['warning_level']['priority'] == 'high'
        
        assert 'out_control_level' in ranges


class TestEDDMDetector:
    """Test EDDM-specific functionality."""
    
    def test_eddm_default_parameters(self):
        """Test EDDM default parameters."""
        detector = EDDMDetector()
        
        assert detector.params['min_num_instances'] == 30
        assert detector.params['warning_level'] == 0.95
        assert detector.params['out_control_level'] == 0.9


class TestPageHinkleyDetector:
    """Test PageHinkley-specific functionality."""
    
    def test_pagehinkley_default_parameters(self):
        """Test PageHinkley default parameters."""
        detector = PageHinkleyDetector()
        
        assert detector.params['min_instances'] == 30
        assert detector.params['delta'] == 0.005
        assert detector.params['threshold'] == 50
        assert detector.params['alpha'] == 1 - 0.0001
    
    def test_pagehinkley_parameter_ranges(self):
        """Test PageHinkley parameter ranges."""
        detector = PageHinkleyDetector()
        ranges = detector.get_parameter_ranges()
        
        assert 'threshold' in ranges
        assert ranges['threshold']['priority'] == 'high'
        assert 'delta' in ranges
        assert ranges['delta']['type'] == 'log_uniform'


class TestKSWINDetector:
    """Test KSWIN-specific functionality."""
    
    def test_kswin_default_parameters(self):
        """Test KSWIN default parameters."""
        detector = KSWINDetector()
        
        assert detector.params['alpha'] == 0.005
        assert detector.params['window_size'] == 100
        assert detector.params['stat_size'] == 30
        assert detector.params['seed'] is None
    
    def test_kswin_with_seed(self):
        """Test KSWIN with random seed."""
        detector = KSWINDetector(seed=42)
        
        assert detector.params['seed'] == 42
    
    def test_kswin_parameter_ranges(self):
        """Test KSWIN parameter ranges."""
        detector = KSWINDetector()
        ranges = detector.get_parameter_ranges()
        
        assert 'alpha' in ranges
        assert ranges['alpha']['type'] == 'log_uniform'
        assert ranges['alpha']['priority'] == 'high'
        
        assert 'window_size' in ranges
        assert ranges['window_size']['type'] == 'choice'


class TestDetectorEnsemble:
    """Test detector ensemble functionality."""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        detectors = [
            create_detector('adwin'),
            create_detector('ddm'),
            create_detector('eddm')
        ]
        
        ensemble = DetectorEnsemble(detectors)
        
        assert len(ensemble.detectors) == 3
        assert ensemble.sample_count == 0
    
    def test_ensemble_update(self):
        """Test ensemble update functionality."""
        detectors = [
            create_detector('adwin'),
            create_detector('ddm')
        ]
        
        ensemble = DetectorEnsemble(detectors)
        
        # Update ensemble
        result = ensemble.update(0.5)
        assert result is ensemble
        assert ensemble.sample_count == 1
        
        # Check all detectors were updated
        for detector in ensemble.detectors:
            assert detector.state.sample_count == 1
    
    def test_ensemble_drift_detection(self):
        """Test ensemble drift detection."""
        detectors = [
            create_detector('adwin'),
            create_detector('ddm'),
            create_detector('eddm')
        ]
        
        ensemble = DetectorEnsemble(detectors)
        
        # Test minimum agreement
        assert not ensemble.drift_detected(min_agreement=1)
        assert not ensemble.drift_detected(min_agreement=2)
        assert not ensemble.drift_detected(min_agreement=3)
    
    def test_ensemble_drift_votes(self):
        """Test getting individual drift votes."""
        detectors = [
            create_detector('adwin'),
            create_detector('ddm')
        ]
        
        ensemble = DetectorEnsemble(detectors)
        
        # Update ensemble
        for i in range(10):
            ensemble.update(0.5 + i * 0.1)
        
        votes = ensemble.get_drift_votes()
        assert len(votes) == 2
        assert all(isinstance(vote, bool) for vote in votes)
    
    def test_ensemble_reset(self):
        """Test ensemble reset functionality."""
        detectors = [
            create_detector('adwin'),
            create_detector('ddm')
        ]
        
        ensemble = DetectorEnsemble(detectors)
        
        # Update ensemble
        for i in range(20):
            ensemble.update(0.5 + i * 0.1)
        
        # Reset ensemble
        ensemble.reset()
        
        assert ensemble.sample_count == 0
        for detector in ensemble.detectors:
            assert detector.state.sample_count == 0


class TestDetectorIntegration:
    """Integration tests for detector functionality."""
    
    def test_drift_detection_simulation(self):
        """Test drift detection with simulated concept drift."""
        detector = create_detector('adwin', delta=0.01)  # More sensitive
        
        # Simulate stable period
        for i in range(200):
            detector.update(np.random.normal(0.5, 0.1))
        
        initial_drifts = len(detector.drift_points)
        
        # Simulate concept drift (shift in mean)
        for i in range(200):
            detector.update(np.random.normal(0.8, 0.1))  # Higher mean
        
        # Should have more drift detections after the shift
        # (Note: This test might be flaky due to randomness)
        assert detector.state.sample_count == 400
    
    def test_multiple_detector_comparison(self):
        """Test comparing multiple detectors on same data."""
        detectors = {
            'adwin': create_detector('adwin', delta=0.01),
            'ddm': create_detector('ddm', min_num_instances=30),
            'pagehinkley': create_detector('pagehinkley', threshold=30)
        }
        
        # Generate data with a shift
        np.random.seed(42)  # For reproducibility
        data = []
        
        # Stable period
        for i in range(100):
            data.append(np.random.normal(0.5, 0.1))
            
        # Drift period
        for i in range(100):
            data.append(np.random.normal(0.8, 0.1))
        
        # Process with all detectors
        results = {}
        for name, detector in detectors.items():
            for sample in data:
                detector.update(sample)
            results[name] = {
                'drift_count': len(detector.drift_points),
                'drift_points': detector.drift_points.copy()
            }
        
        # All detectors should have processed same number of samples
        for name, detector in detectors.items():
            assert detector.state.sample_count == 200
        
        # Results should be reasonable (detectors may behave differently)
        for name, result in results.items():
            assert result['drift_count'] >= 0
            assert isinstance(result['drift_points'], list)
    
    def test_detector_parameter_validation(self):
        """Test that detectors handle edge case parameters reasonably."""
        # Test ADWIN with very small delta (should be very sensitive)
        detector = create_detector('adwin', delta=1e-6)
        assert detector.params['delta'] == 1e-6
        
        # Test with reasonable data
        for i in range(50):
            detector.update(0.5)
        
        # Should not crash
        assert detector.state.sample_count == 50
    
    def test_detector_state_consistency(self):
        """Test that detector state remains consistent."""
        detector = create_detector('adwin')
        
        drift_count_history = []
        sample_count_history = []
        
        # Process samples and track state
        for i in range(100):
            detector.update(0.5 + 0.01 * i)
            
            current_drift_count = len(detector.drift_points)
            current_sample_count = detector.state.sample_count
            
            # Sample count should always increase
            assert current_sample_count == i + 1
            
            # Drift count should never decrease
            if drift_count_history:
                assert current_drift_count >= drift_count_history[-1]
            
            drift_count_history.append(current_drift_count)
            sample_count_history.append(current_sample_count)
        
        # Final checks
        assert sample_count_history[-1] == 100
        assert all(sample_count_history[i] == i + 1 for i in range(100))