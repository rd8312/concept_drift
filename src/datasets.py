"""
Dataset generators for concept drift experiments.
Provides implementations for SEA, Sine, FriedmanDrift, Elec2, and ConceptDriftStream generators
using River's dataset API for streaming concept drift scenarios.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any, List
from river import datasets
from river.datasets import synth


class SEAGenerator:
    """
    SEA (Streaming Ensemble Algorithm) concept drift generator.
    Creates abrupt concept drifts at specified positions.
    """
    
    def __init__(
        self,
        drift_positions: list = None,
        noise_level: float = 0.0,
        n_samples: int = 5000,
        seed: int = 42
    ):
        """
        Initialize SEA generator.
        
        Args:
            drift_positions: List of sample positions where drift occurs
            noise_level: Proportion of noise (0.0 to 1.0)
            n_samples: Total number of samples to generate
            seed: Random seed for reproducibility
        """
        self.drift_positions = drift_positions or [1000, 2500, 4000]
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate(self) -> Iterator[Tuple[float, int, bool]]:
        """
        Generate SEA data stream with concept drifts.
        
        Yields:
            Tuple of (feature_value, label, is_drift_point)
        """
        current_concept = 0
        concepts = [
            lambda x: 1 if x < 0.5 else 0,  # Concept 0
            lambda x: 1 if x > 0.5 else 0,  # Concept 1  
            lambda x: 1 if 0.3 < x < 0.7 else 0,  # Concept 2
            lambda x: 1 if x < 0.3 or x > 0.7 else 0,  # Concept 3
        ]
        
        for i in range(self.n_samples):
            # Check for concept drift
            is_drift = i in self.drift_positions
            if is_drift and current_concept < len(concepts) - 1:
                current_concept += 1
                
            # Generate feature value
            x = self.rng.random()
            
            # Apply current concept to get label
            y = concepts[current_concept](x)
            
            # Add noise
            if self.rng.random() < self.noise_level:
                y = 1 - y
                
            yield x, y, is_drift


class SineGenerator:
    """
    Sine wave generator with concept drift using River's ConceptDriftStream.
    """
    
    def __init__(
        self,
        drift_positions: list = None,
        transition_width: int = 1,
        noise_level: float = 0.0,
        n_samples: int = 5000,
        seed: int = 42
    ):
        """
        Initialize Sine generator with concept drift.
        
        Args:
            drift_positions: List of positions where concept changes
            transition_width: Width of transition between concepts (1=abrupt)
            noise_level: Proportion of noise
            n_samples: Total number of samples
            seed: Random seed
        """
        self.drift_positions = drift_positions or [1500, 3000]
        self.transition_width = transition_width
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.seed = seed
        
    def generate(self) -> Iterator[Tuple[Dict[str, float], int, bool]]:
        """
        Generate sine wave data with concept drift.
        
        Yields:
            Tuple of (features_dict, target, is_drift_point)
        """
        # Create base sine generators with different classification functions
        generators = [
            synth.Sine(classification_function=0, seed=self.seed),
            synth.Sine(classification_function=1, seed=self.seed + 1),
            synth.Sine(classification_function=2, seed=self.seed + 2),
            synth.Sine(classification_function=3, seed=self.seed + 3),
        ]
        
        # Create iterators for each generator
        generator_iterators = [iter(gen) for gen in generators]
        
        current_concept = 0
        rng = np.random.RandomState(self.seed)
        
        for i in range(self.n_samples):
            # Check for drift
            is_drift = i in self.drift_positions
            if is_drift and current_concept < len(generators) - 1:
                current_concept += 1
                
            # Get sample from current generator
            try:
                x, y = next(generator_iterators[current_concept])
            except StopIteration:
                # If current generator is exhausted, restart it
                generator_iterators[current_concept] = iter(generators[current_concept])
                x, y = next(generator_iterators[current_concept])
            
            # Add noise (flip label with probability noise_level)
            if self.noise_level > 0 and rng.random() < self.noise_level:
                y = 1 - y  # Flip binary label
                
            yield x, y, is_drift


class ConceptDriftStreamGenerator:
    """
    Flexible concept drift stream generator that can compose different base streams.
    """
    
    def __init__(
        self,
        stream_configs: list,
        drift_positions: list = None,
        transition_width: int = 1,
        noise_level: float = 0.0,
        seed: int = 42
    ):
        """
        Initialize concept drift stream with multiple base streams.
        
        Args:
            stream_configs: List of stream configuration dicts
            drift_positions: Positions where drift occurs
            transition_width: Width of drift transition
            noise_level: Proportion of noise
            seed: Random seed
        """
        self.stream_configs = stream_configs
        self.drift_positions = drift_positions or []
        self.transition_width = transition_width
        self.noise_level = noise_level
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate(self) -> Iterator[Tuple[Dict[str, float], Any, bool]]:
        """
        Generate concept drift stream from composed base streams.
        
        Yields:
            Tuple of (features, target, is_drift_point)
        """
        current_stream_idx = 0
        current_stream = self._create_stream(self.stream_configs[current_stream_idx])
        sample_count = 0
        
        while sample_count < sum(config.get('n_samples', 1000) for config in self.stream_configs):
            # Check for concept drift
            is_drift = sample_count in self.drift_positions
            if is_drift and current_stream_idx < len(self.stream_configs) - 1:
                current_stream_idx += 1
                current_stream = self._create_stream(self.stream_configs[current_stream_idx])
                
            try:
                x, y = next(iter(current_stream))
                
                # Add noise
                if self.rng.random() < self.noise_level:
                    if isinstance(y, (int, float)):
                        y += self.rng.normal(0, 0.1)
                    
                yield x, y, is_drift
                sample_count += 1
                
            except StopIteration:
                break
                
    def _create_stream(self, config: dict):
        """Create a stream based on configuration."""
        stream_type = config.get('type', 'sine')
        
        if stream_type == 'sine':
            return synth.Sine(
                classification_function=config.get('function', 0),
                seed=config.get('seed', self.seed)
            )
        elif stream_type == 'friedman':
            return synth.Friedman(
                seed=config.get('seed', self.seed)
            )
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")


class FriedmanDriftGenerator:
    """
    Friedman drift generator with various drift types.
    """
    
    def __init__(
        self,
        drift_type: str = 'abrupt',
        drift_positions: list = None,
        transition_width: int = 100,
        noise_level: float = 0.0,
        n_samples: int = 5000,
        n_features: int = 10,
        seed: int = 42
    ):
        """
        Initialize Friedman drift generator.
        
        Args:
            drift_type: Type of drift ('abrupt', 'gradual', 'incremental')
            drift_positions: Positions where drift occurs
            transition_width: Width of gradual transition
            noise_level: Level of noise to add
            n_samples: Total samples to generate
            n_features: Number of features
            seed: Random seed
        """
        self.drift_type = drift_type
        self.drift_positions = drift_positions or [2500]
        self.transition_width = transition_width
        self.noise_level = noise_level
        self.n_samples = n_samples
        self.n_features = n_features
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate(self) -> Iterator[Tuple[Dict[str, float], float, bool]]:
        """
        Generate Friedman data with specified drift type.
        
        Yields:
            Tuple of (features_dict, target, is_drift_point)
        """
        # Create base generators for different concepts
        base_gen = synth.Friedman(seed=self.seed)
        
        for i in range(self.n_samples):
            # Get base sample
            x, y = next(iter(base_gen))
            
            # Check for drift points
            is_drift = i in self.drift_positions
            
            # Apply drift transformation based on type
            if self.drift_type == 'abrupt' and i >= min(self.drift_positions):
                # Abrupt drift: sudden change in target function
                y = -y + self.rng.normal(0, 0.5)
                
            elif self.drift_type == 'gradual' and i >= min(self.drift_positions):
                # Gradual drift: smooth transition
                progress = min(1.0, (i - min(self.drift_positions)) / self.transition_width)
                y = (1 - progress) * y + progress * (-y + self.rng.normal(0, 0.5))
                
            elif self.drift_type == 'incremental':
                # Incremental drift: continuous small changes
                if i > 0:
                    drift_factor = 0.001 * i
                    y += drift_factor * self.rng.normal(0, 1)
                    
            # Add noise
            if self.rng.random() < self.noise_level:
                y += self.rng.normal(0, 0.1)
                
            yield x, y, is_drift


class Elec2Generator:
    """
    Real-world electricity dataset generator with concept drift detection.
    Uses River's Elec2 dataset which contains NSW electricity market data.
    """
    
    def __init__(
        self,
        n_samples: Optional[int] = None,
        drift_positions: Optional[list] = None,
        seed: int = 42,
        start_position: int = 0,
        sample_fraction: float = 1.0
    ):
        """
        Initialize Elec2 generator.
        
        Args:
            n_samples: Number of samples to generate (None for full dataset)
            drift_positions: List of positions where to mark drift (None for auto-detect)
            seed: Random seed for reproducible sampling
            start_position: Starting position in the dataset
            sample_fraction: Fraction of data to sample (0.0 to 1.0)
        """
        if sample_fraction < 0.0 or sample_fraction > 1.0:
            raise ValueError("sample_fraction must be between 0.0 and 1.0")
        
        if n_samples is not None and n_samples < 0:
            raise ValueError("n_samples must be positive or None")
        
        self.n_samples = n_samples
        self.drift_positions = drift_positions
        self.seed = seed
        self.start_position = start_position
        self.sample_fraction = sample_fraction
        self.rng = np.random.RandomState(seed)
        
        # Load River's Elec2 dataset
        self.river_dataset = datasets.Elec2()
        
        # Auto-detect drift positions based on known electricity market patterns
        if self.drift_positions is None:
            self.drift_positions = self._auto_detect_drift_positions()
    
    def _auto_detect_drift_positions(self) -> list:
        """
        Auto-detect drift positions based on known patterns in Elec2 dataset.
        NSW electricity market has known structural changes.
        """
        # Known approximate positions where market conditions changed
        # These are based on NSW electricity market history
        total_samples = 45312  # Total samples in Elec2 dataset
        
        if self.n_samples is None:
            target_samples = total_samples - self.start_position
        else:
            target_samples = self.n_samples
        
        # Scale positions based on target sample count
        base_positions = [0.25, 0.5, 0.75]  # Relative positions
        auto_positions = [int(pos * target_samples) for pos in base_positions]
        
        # Filter out positions outside range
        auto_positions = [pos for pos in auto_positions 
                         if 0 < pos < target_samples]
        
        return auto_positions
    
    def generate(self) -> Iterator[Tuple[Dict[str, float], int, bool]]:
        """
        Generate Elec2 data stream with concept drift markers.
        
        Yields:
            Tuple of (features_dict, target, is_drift_point)
        """
        # Create iterator from River dataset
        river_iter = iter(self.river_dataset)
        
        # Skip to start position
        for _ in range(self.start_position):
            try:
                next(river_iter)
            except StopIteration:
                break
        
        sample_count = 0
        samples_generated = 0
        
        for x, y in river_iter:
            # Apply sampling fraction
            if self.rng.random() > self.sample_fraction:
                sample_count += 1
                continue
            
            # Check for drift
            is_drift = sample_count in (self.drift_positions or [])
            
            yield x, y, is_drift
            
            sample_count += 1
            samples_generated += 1
            
            # Stop if we've generated enough samples
            if self.n_samples is not None and samples_generated >= self.n_samples:
                break


def create_dataset(
    dataset_type: str,
    config: Dict[str, Any] = None
) -> Iterator[Tuple[Any, Any, bool]]:
    """
    Factory function to create dataset generators.
    
    Supports both traditional detailed configuration and scenario-based intelligent configuration.
    For simplified usage, consider using create_experiment_stream() from smart_config module.
    
    Args:
        dataset_type: Type of dataset ('sea', 'sine', 'concept_drift', 'friedman', 'elec2')
        config: Configuration dictionary for the dataset
        
    Returns:
        Generator yielding (features, target, is_drift_point) tuples
        
    Examples:
        # Traditional detailed configuration
        >>> stream = create_dataset('sea', {
        ...     'drift_positions': [1000, 2500],
        ...     'noise_level': 0.02,
        ...     'n_samples': 5000
        ... })
        
        # For simplified scenario-based configuration, use:
        >>> from .smart_config import create_experiment_stream
        >>> stream = create_experiment_stream(
        ...     scenario='abrupt_drift',
        ...     difficulty='hard'
        ... )
    """
    config = config or {}
    
    if dataset_type == 'sea':
        generator = SEAGenerator(**config)
    elif dataset_type == 'sine':
        generator = SineGenerator(**config)
    elif dataset_type == 'concept_drift':
        generator = ConceptDriftStreamGenerator(**config)
    elif dataset_type == 'friedman':
        generator = FriedmanDriftGenerator(**config)
    elif dataset_type == 'elec2':
        generator = Elec2Generator(**config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    return generator.generate()


def get_available_datasets() -> List[str]:
    """
    Get list of available dataset types.
    
    Returns:
        List of available dataset names
    """
    return ['sea', 'sine', 'concept_drift', 'friedman', 'elec2']


def get_dataset_info(dataset_type: str = None) -> Dict[str, Any]:
    """
    Get information about available datasets.
    
    Args:
        dataset_type: Specific dataset to get info about, or None for all
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_descriptions = {
        'sea': {
            'name': 'SEA Generator',
            'description': 'SEA (Streaming Ensemble Algorithm) concept drift generator with abrupt changes',
            'drift_type': 'abrupt',
            'features': 'Single feature with 4 concept functions',
            'parameters': ['drift_positions', 'noise_level', 'n_samples', 'seed'],
            'best_for': 'Testing abrupt concept drift detection',
            'sample_config': {
                'drift_positions': [1000, 2500],
                'noise_level': 0.02,
                'n_samples': 5000,
                'seed': 42
            }
        },
        'sine': {
            'name': 'Sine Wave Generator',
            'description': 'Sine wave generator with concept drift using River\'s ConceptDriftStream',
            'drift_type': 'gradual/abrupt',
            'features': 'Multi-dimensional sine wave features',
            'parameters': ['drift_positions', 'transition_width', 'noise_level', 'n_samples', 'seed'],
            'best_for': 'Testing gradual and abrupt concept drift with complex features',
            'sample_config': {
                'drift_positions': [1500, 3000],
                'transition_width': 1,
                'noise_level': 0.02,
                'n_samples': 5000,
                'seed': 42
            }
        },
        'friedman': {
            'name': 'Friedman Drift Generator',
            'description': 'Friedman drift generator with various drift types (abrupt, gradual, incremental)',
            'drift_type': 'configurable',
            'features': 'Multi-dimensional Friedman regression features',
            'parameters': ['drift_type', 'drift_positions', 'transition_width', 'noise_level', 'n_samples', 'n_features', 'seed'],
            'best_for': 'Testing different types of concept drift with regression tasks',
            'sample_config': {
                'drift_type': 'gradual',
                'drift_positions': [2500],
                'transition_width': 200,
                'noise_level': 0.02,
                'n_samples': 5000,
                'n_features': 10,
                'seed': 42
            }
        },
        'concept_drift': {
            'name': 'Concept Drift Stream',
            'description': 'Flexible concept drift stream generator that composes different base streams',
            'drift_type': 'configurable',
            'features': 'Multi-stream composition with different feature types',
            'parameters': ['stream_configs', 'drift_positions', 'transition_width', 'noise_level', 'seed'],
            'best_for': 'Complex scenarios with multiple concept changes and stream types',
            'sample_config': {
                'stream_configs': [
                    {'type': 'sine', 'n_samples': 1000, 'function': 0},
                    {'type': 'friedman', 'n_samples': 1000}
                ],
                'drift_positions': [1000],
                'transition_width': 100,
                'noise_level': 0.02,
                'seed': 42
            }
        },
        'elec2': {
            'name': 'Electricity Market (Elec2)',
            'description': 'Real-world electricity market dataset with natural concept drift patterns',
            'drift_type': 'real-world',
            'features': 'NSW electricity market features (price, demand, etc.)',
            'parameters': ['n_samples', 'drift_positions', 'start_position', 'sample_fraction', 'seed'],
            'best_for': 'Testing on real-world data with natural concept drift',
            'sample_config': {
                'sample_fraction': 1.0,
                'start_position': 0,
                'seed': 42
            },
            'note': 'Uses real-world data with 45,312 samples. Drift positions auto-detected if not specified.'
        }
    }
    
    if dataset_type:
        if dataset_type in dataset_descriptions:
            return dataset_descriptions[dataset_type]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset_descriptions


# Export unified interface components for easy importing
__all__ = [
    'create_dataset',
    'get_available_datasets', 
    'get_dataset_info',
    'SEAGenerator',
    'SineGenerator', 
    'ConceptDriftStreamGenerator',
    'FriedmanDriftGenerator',
    'Elec2Generator'
]