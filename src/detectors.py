"""
Unified drift detector interface wrapping River's drift detection algorithms.
Provides consistent .update(x) interface for ADWIN, DDM, EDDM, PageHinkley, and KSWIN.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np

from river import drift


@dataclass
class DetectorState:
    """State information for a drift detector."""
    drift_detected: bool = False
    warning_detected: bool = False
    sample_count: int = 0
    last_drift_position: Optional[int] = None
    

class BaseDriftDetector(ABC):
    """
    Abstract base class for drift detectors with unified interface.
    """
    
    def __init__(self, **kwargs):
        """Initialize detector with parameters."""
        self.detector = self._create_detector(**kwargs)
        self.state = DetectorState()
        self._drift_points = []
        self._warning_points = []
        
    @abstractmethod
    def _create_detector(self, **kwargs):
        """Create the underlying River detector."""
        pass
    
    @abstractmethod 
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get valid parameter ranges for hyperparameter search."""
        pass
        
    def update(self, x: Union[float, int, Dict[str, float]]) -> 'BaseDriftDetector':
        """
        Update detector with new sample.
        
        Args:
            x: New sample value (float, int, or dict of features)
            
        Returns:
            Self for method chaining
        """
        # Handle different input formats
        if isinstance(x, dict):
            # For feature dictionaries, extract a single value for drift detection
            # Use the first feature value or compute a summary statistic
            if x:
                # Use the first feature value as the primary signal
                feature_values = list(x.values())
                if feature_values:
                    # Use mean of all features as a single drift signal
                    x_value = np.mean(feature_values)
                else:
                    x_value = 0.0
            else:
                x_value = 0.0
        else:
            x_value = x
            
        self.detector.update(x_value)
        self.state.sample_count += 1
        
        # Update state
        self.state.drift_detected = self.detector.drift_detected
        self.state.warning_detected = hasattr(self.detector, 'warning_detected') and self.detector.warning_detected
        
        # Record drift points
        if self.state.drift_detected:
            self.state.last_drift_position = self.state.sample_count
            self._drift_points.append(self.state.sample_count)
            
        # Record warning points
        if self.state.warning_detected:
            self._warning_points.append(self.state.sample_count)
            
        return self
    
    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected."""
        return self.state.drift_detected
        
    @property
    def warning_detected(self) -> bool:
        """Check if warning was detected."""
        return self.state.warning_detected
        
    @property
    def drift_points(self) -> List[int]:
        """Get list of all drift points."""
        return self._drift_points.copy()
        
    @property
    def warning_points(self) -> List[int]:
        """Get list of all warning points."""
        return self._warning_points.copy()
        
    def reset(self):
        """Reset detector state."""
        self.detector = self._create_detector(**self.get_parameters())
        self.state = DetectorState()
        self._drift_points = []
        self._warning_points = []
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get current detector parameters."""
        if hasattr(self.detector, '__dict__'):
            # Filter out internal attributes
            params = {}
            for key, value in self.detector.__dict__.items():
                if not key.startswith('_') and key not in ['drift_detected', 'warning_detected']:
                    params[key] = value
            return params
        return {}


class ADWINDetector(BaseDriftDetector):
    """
    ADWIN (Adaptive Windowing) drift detector wrapper.
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        **kwargs
    ):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (smaller = more sensitive)
            clock: Clock period for bucket updates
            max_buckets: Maximum number of buckets
            min_window_length: Minimum window length
            grace_period: Grace period before detection can start
        """
        self.params = {
            'delta': delta,
            'clock': clock,
            'max_buckets': max_buckets,
            'min_window_length': min_window_length,
            'grace_period': grace_period
        }
        super().__init__(**self.params)
        
    def _create_detector(self, **kwargs):
        """Create ADWIN detector instance."""
        return drift.ADWIN(**kwargs)
        
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get ADWIN parameter search ranges."""
        return {
            'delta': {
                'type': 'log_uniform',
                'range': [1e-5, 5e-2],
                'priority': 'high'
            },
            'clock': {
                'type': 'choice',
                'values': [16, 32, 64, 128],
                'priority': 'medium'
            },
            'max_buckets': {
                'type': 'choice',
                'values': [3, 5, 7, 10],
                'priority': 'low'
            },
            'min_window_length': {
                'type': 'choice',
                'values': [5, 10, 20, 30],
                'priority': 'low'
            }
        }


class DDMDetector(BaseDriftDetector):
    """
    DDM (Drift Detection Method) detector wrapper.
    """
    
    def __init__(
        self,
        warm_start: int = 30,
        warning_threshold: float = 2.0,
        drift_threshold: float = 3.0,
        **kwargs
    ):
        """
        Initialize DDM detector.
        
        Args:
            warm_start: The minimum required number of analyzed samples so change can be detected. Warm start parameter for the drift detector.
            warning_threshold: Threshold to decide if the detector is in a warning zone. The default value gives 95% of confidence level to the warning assessment.
            drift_threshold: Threshold to decide if a drift was detected. The default value gives a 99% of confidence level to the drift assessment.
        """
        self.params = {
            'warm_start': warm_start,
            'warning_threshold': warning_threshold,
            'drift_threshold': drift_threshold
        }
        super().__init__(**self.params)
        
    def _create_detector(self, **kwargs):
        """Create DDM detector instance.""" 
        return drift.binary.DDM(**kwargs)
        
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get DDM parameter search ranges."""
        return {
            'warm_start': {
                'type': 'choice',
                'values': [20, 30, 40, 50, 60],
                'priority': 'medium'
            },
            'warning_threshold': {
                'type': 'uniform',
                'range': [1.5, 2.5],
                'priority': 'high'
            },
            'drift_threshold': {
                'type': 'uniform',
                'range': [2.5, 3.5],
                'priority': 'high'
            }
        }


class EDDMDetector(BaseDriftDetector):
    """
    EDDM (Early Drift Detection Method) detector wrapper.
    """
    
    def __init__(
        self,
        warm_start: int = 30,
        alpha: float = 0.95,
        beta: float = 0.9,
        **kwargs
    ):
        """
        Initialize EDDM detector.
        
        Args:
            warm_start: The minimum required number of monitored errors/failures so change can be detected. Warm start parameter for the drift detector.
            alpha: Threshold for triggering a warning. Must be between 0 and 1. The smaller the value, the more conservative the detector becomes.
            beta: Threshold for triggering a drift. Must be between 0 and 1. The smaller the value, the more conservative the detector becomes.
        """
        self.params = {
            'warm_start': warm_start,
            'alpha': alpha,
            'beta': beta
        }
        super().__init__(**self.params)
        
    def _create_detector(self, **kwargs):
        """Create EDDM detector instance."""
        return drift.binary.EDDM(**kwargs)
        
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get EDDM parameter search ranges."""
        return {
            'warm_start': {
                'type': 'choice',
                'values': [20, 30, 40, 50, 60],
                'priority': 'medium'  
            },
            'alpha': {
                'type': 'uniform',
                'range': [0.9, 0.99],
                'priority': 'high'
            },
            'beta': {
                'type': 'uniform',
                'range': [0.85, 0.95],
                'priority': 'high'
            }
        }


class PageHinkleyDetector(BaseDriftDetector):
    """
    Page-Hinkley drift detector wrapper.
    """
    
    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50,
        alpha: float = 1 - 0.0001,
        **kwargs
    ):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            min_instances: Minimum instances before detection starts
            delta: Detection margin (sensitivity parameter)
            threshold: Threshold value (λ parameter)
            alpha: Forgetting factor
        """
        self.params = {
            'min_instances': min_instances,
            'delta': delta,
            'threshold': threshold,
            'alpha': alpha
        }
        super().__init__(**self.params)
        
    def _create_detector(self, **kwargs):
        """Create Page-Hinkley detector instance."""
        return drift.PageHinkley(**kwargs)
        
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get Page-Hinkley parameter search ranges."""
        return {
            'threshold': {
                'type': 'uniform',
                'range': [10, 200],
                'priority': 'high'
            },
            'delta': {
                'type': 'log_uniform',
                'range': [1e-4, 1e-2],
                'priority': 'high'
            },
            'alpha': {
                'type': 'uniform',
                'range': [0.9, 0.9999],
                'priority': 'medium'
            },
            'min_instances': {
                'type': 'choice',
                'values': [20, 30, 40, 50],
                'priority': 'low'
            }
        }


class KSWINDetector(BaseDriftDetector):
    """
    KSWIN (Kolmogorov-Smirnov Windowing) detector wrapper.
    """
    
    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = None,
        **kwargs
    ):
        """
        Initialize KSWIN detector.
        
        Args:
            alpha: Significance level for KS test
            window_size: Size of reference window
            stat_size: Size of test window
            seed: Random seed for reproducibility
        """
        self.params = {
            'alpha': alpha,
            'window_size': window_size,
            'stat_size': stat_size,
            'seed': seed
        }
        super().__init__(**self.params)
        
    def _create_detector(self, **kwargs):
        """Create KSWIN detector instance."""
        return drift.KSWIN(**kwargs)
        
    def get_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get KSWIN parameter search ranges."""
        return {
            'alpha': {
                'type': 'log_uniform',
                'range': [0.001, 0.2],
                'priority': 'high'
            },
            'window_size': {
                'type': 'choice',
                'values': [50, 100, 150, 200, 250, 300],
                'priority': 'high'
            },
            'stat_size': {
                'type': 'choice',
                'values': [20, 30, 50, 80, 100, 120],
                'priority': 'medium'
            }
        }


# Factory dictionary for detector creation
DETECTOR_CLASSES = {
    'adwin': ADWINDetector,
    'ddm': DDMDetector,
    'eddm': EDDMDetector,
    'pagehinkley': PageHinkleyDetector,
    'kswin': KSWINDetector
}


def create_detector(detector_type: str, **kwargs) -> BaseDriftDetector:
    """
    Factory function to create drift detectors.
    
    Args:
        detector_type: Type of detector ('adwin', 'ddm', 'eddm', 'pagehinkley', 'kswin')
        **kwargs: Detector-specific parameters
        
    Returns:
        Initialized drift detector instance
        
    Raises:
        ValueError: If detector_type is not supported
    """
    detector_type = detector_type.lower()
    
    if detector_type not in DETECTOR_CLASSES:
        available = ', '.join(DETECTOR_CLASSES.keys())
        raise ValueError(f"Unknown detector type '{detector_type}'. Available: {available}")
        
    return DETECTOR_CLASSES[detector_type](**kwargs)


def get_all_detector_names() -> List[str]:
    """Get list of all available detector names."""
    return list(DETECTOR_CLASSES.keys())


def get_detector_parameter_ranges(detector_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get parameter ranges for a specific detector type.
    
    Args:
        detector_type: Type of detector
        
    Returns:
        Dictionary with parameter ranges
    """
    if detector_type.lower() not in DETECTOR_CLASSES:
        raise ValueError(f"Unknown detector type: {detector_type}")
        
    detector_class = DETECTOR_CLASSES[detector_type.lower()]
    # Create temporary instance to get parameter ranges
    temp_detector = detector_class()
    return temp_detector.get_parameter_ranges()


class DetectorEnsemble:
    """
    Ensemble of drift detectors for robust detection.
    """
    
    def __init__(self, detectors: List[BaseDriftDetector]):
        """
        Initialize detector ensemble.
        
        Args:
            detectors: List of initialized drift detectors
        """
        self.detectors = detectors
        self.sample_count = 0
        
    def update(self, x: Union[float, int]) -> 'DetectorEnsemble':
        """Update all detectors with new sample."""
        self.sample_count += 1
        for detector in self.detectors:
            detector.update(x)
        return self
        
    def drift_detected(self, min_agreement: int = 1) -> bool:
        """
        Check if drift is detected by ensemble.
        
        Args:
            min_agreement: Minimum number of detectors that must agree
            
        Returns:
            True if at least min_agreement detectors detected drift
        """
        agreements = sum(1 for detector in self.detectors if detector.drift_detected)
        return agreements >= min_agreement
        
    def get_drift_votes(self) -> List[bool]:
        """Get drift votes from all detectors."""
        return [detector.drift_detected for detector in self.detectors]
        
    def get_ensemble_drift_points(self, min_agreement: int = 1) -> List[int]:
        """
        Get drift points where at least min_agreement detectors agreed.
        
        Args:
            min_agreement: Minimum agreement threshold
            
        Returns:
            List of drift positions with sufficient agreement
        """
        all_drift_points = {}
        
        # Collect all drift points
        for detector in self.detectors:
            for point in detector.drift_points:
                if point not in all_drift_points:
                    all_drift_points[point] = 0
                all_drift_points[point] += 1
                
        # Filter by agreement threshold
        ensemble_drifts = [
            point for point, votes in all_drift_points.items()
            if votes >= min_agreement
        ]
        
        return sorted(ensemble_drifts)
        
    def reset(self):
        """Reset all detectors in ensemble."""
        for detector in self.detectors:
            detector.reset()
        self.sample_count = 0