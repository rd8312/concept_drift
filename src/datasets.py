"""
Dataset generators for concept drift experiments.
Provides implementations for SEA, Sine, ConceptDriftStream, and FriedmanDrift generators
using River's dataset API for streaming concept drift scenarios.
"""

import numpy as np
from typing import Iterator, Tuple, Optional, Dict, Any
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
        
    def generate(self) -> Iterator[Tuple[Dict[str, float], float, bool]]:
        """
        Generate sine wave data with concept drift.
        
        Yields:
            Tuple of (features_dict, target, is_drift_point)
        """
        # Create base sine generators with different frequencies
        generators = [
            synth.Sine(classification_function=0, seed=self.seed),
            synth.Sine(classification_function=1, seed=self.seed + 1),
            synth.Sine(classification_function=2, seed=self.seed + 2),
        ]
        
        current_concept = 0
        sample_count = 0
        
        for i in range(self.n_samples):
            # Check for drift
            is_drift = i in self.drift_positions
            if is_drift and current_concept < len(generators) - 1:
                current_concept += 1
                
            # Get sample from current generator
            x, y = next(iter(generators[current_concept]))
            
            # Add noise
            if self.noise_level > 0 and np.random.random() < self.noise_level:
                y = np.random.randint(0, 4)  # Random class for noise
                
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


def create_dataset(
    dataset_type: str,
    config: Dict[str, Any] = None
) -> Iterator[Tuple[Any, Any, bool]]:
    """
    Factory function to create dataset generators.
    
    Args:
        dataset_type: Type of dataset ('sea', 'sine', 'concept_drift', 'friedman')
        config: Configuration dictionary for the dataset
        
    Returns:
        Generator yielding (features, target, is_drift_point) tuples
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
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
        
    return generator.generate()