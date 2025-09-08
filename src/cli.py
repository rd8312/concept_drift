"""
Command-line interface for the Drift Detector Tuner.
Provides easy access to hyperparameter tuning and evaluation functionality.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

from .evaluate import EvaluationConfig, EvaluationFramework, run_comprehensive_evaluation
from .presets import generate_detector_templates, TemplateRecommender
from .detectors import get_all_detector_names, create_detector
from .datasets import create_dataset
from .smart_config import (
    create_experiment_stream, get_scenario_info, 
    Scenario, Difficulty, NoiseLevel, SmartDatasetFactory
)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Drift Detector Tuner - Hyperparameter optimization for concept drift detectors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NEW: Simplified scenario-based tuning
  python -m src.cli tune --scenario abrupt_drift --difficulty hard --trials 200
  python -m src.cli tune --scenario gradual_drift --difficulty medium --stream-length 8000
  python -m src.cli tune --scenario real_world --difficulty easy
  
  # Traditional detailed configuration (still supported)
  python -m src.cli tune --algo all --datasets sea,sine,friedman --trials 200
  
  # Tune specific detector on specific dataset
  python -m src.cli tune --algo adwin --datasets sea --trials 100 --output results/adwin
  
  # Quick evaluation with fewer trials
  python -m src.cli tune --trials 50 --runs 3 --output quick_results
  
  # Generate templates from existing results
  python -m src.cli templates --input results/optimization_results.json --output templates/
  
  # Get recommendations for specific scenario
  python -m src.cli recommend --templates templates/detector_templates.json --scenario critical_systems
  
  # Run single detector test with scenario-based configuration
  python -m src.cli test --scenario abrupt_drift --difficulty hard --detector adwin
  python -m src.cli test --detector adwin --dataset sea --params '{"delta": 0.002}'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tune command
    tune_parser = subparsers.add_parser(
        'tune',
        help='Run hyperparameter tuning and evaluation'
    )
    
    # NEW: Scenario-based configuration (mutually exclusive with traditional dataset selection)
    scenario_group = tune_parser.add_argument_group(
        'Scenario-based Configuration',
        'Simplified configuration using predefined scenarios (recommended)'
    )
    scenario_group.add_argument(
        '--scenario',
        choices=[s.value for s in Scenario],
        help='Experimental scenario for intelligent dataset selection and configuration. '
             'Options: ' + ', '.join([s.value for s in Scenario])
    )
    scenario_group.add_argument(
        '--difficulty',
        choices=[d.value for d in Difficulty],
        default='medium',
        help='Experiment difficulty level (default: medium). '
             'Options: ' + ', '.join([d.value for d in Difficulty])
    )
    scenario_group.add_argument(
        '--stream-length',
        type=int,
        default=5000,
        help='Length of the data stream (default: 5000)'
    )
    scenario_group.add_argument(
        '--drift-count',
        type=int,
        default=2,
        help='Number of concept drifts (default: 2, ignored for real_world scenario)'
    )
    scenario_group.add_argument(
        '--noise-level',
        choices=[n.value for n in NoiseLevel],
        default='low',
        help='Noise level category (default: low). '
             'Options: ' + ', '.join([n.value for n in NoiseLevel])
    )
    
    # Traditional detailed configuration (mutually exclusive with scenario-based)
    traditional_group = tune_parser.add_argument_group(
        'Traditional Configuration',
        'Detailed configuration for advanced users (alternative to scenario-based)'
    )
    traditional_group.add_argument(
        '--algo', '--detectors',
        default='all',
        help='Detectors to tune (comma-separated or "all"). Available: ' + ', '.join(get_all_detector_names())
    )
    traditional_group.add_argument(
        '--datasets',
        help='Datasets to use (comma-separated). Available: sea, sine, friedman, elec2. '
             'If not specified and no scenario given, defaults to: sea,sine,friedman,elec2'
    )
    tune_parser.add_argument(
        '--trials',
        type=int,
        default=200,
        help='Number of optimization trials per detector (default: 200)'
    )
    tune_parser.add_argument(
        '--runs',
        type=int, 
        default=5,
        help='Number of evaluation runs per configuration (default: 5)'
    )
    tune_parser.add_argument(
        '--tolerance',
        type=int,
        default=50,
        help='Window matching tolerance for drift detection (default: 50)'
    )
    tune_parser.add_argument(
        '--noise',
        default='0.0,0.01,0.03,0.05',
        help='Noise levels to test (comma-separated, default: 0.0,0.01,0.03,0.05)'
    )
    tune_parser.add_argument(
        '--delay-penalty',
        type=float,
        default=0.002,
        help='Delay penalty coefficient for composite score (default: 0.002)'
    )
    tune_parser.add_argument(
        '--output',
        default='results',
        help='Output directory for results (default: results)'
    )
    tune_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    tune_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    tune_parser.add_argument(
        '--format',
        choices=['csv', 'json', 'both'],
        default='both',
        help='Output format for results (default: both)'
    )
    
    # Templates command
    templates_parser = subparsers.add_parser(
        'templates',
        help='Generate detector templates from optimization results'
    )
    templates_parser.add_argument(
        '--input',
        required=True,
        help='Path to optimization results JSON file'
    )
    templates_parser.add_argument(
        '--output',
        default='templates',
        help='Output directory for templates (default: templates)'
    )
    
    # Recommend command
    recommend_parser = subparsers.add_parser(
        'recommend', 
        help='Get detector recommendations for specific scenarios'
    )
    recommend_parser.add_argument(
        '--templates',
        required=True,
        help='Path to detector templates JSON file'
    )
    recommend_parser.add_argument(
        '--scenario',
        choices=['critical_systems', 'production_monitoring', 'general_purpose', 'research_evaluation'],
        default='general_purpose',
        help='Application scenario (default: general_purpose)'
    )
    recommend_parser.add_argument(
        '--requirements',
        help='Custom requirements as JSON string, e.g. \'{"min_f1": 0.8, "max_fp_rate": 0.1}\''
    )
    recommend_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top recommendations to show (default: 5)'
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        'test',
        help='Test a single detector configuration'
    )
    test_parser.add_argument(
        '--detector',
        required=True,
        choices=get_all_detector_names(),
        help='Detector type to test'
    )
    
    # Scenario-based test configuration
    test_scenario_group = test_parser.add_argument_group(
        'Scenario-based Test Configuration',
        'Use predefined scenarios for testing (recommended)'
    )
    test_scenario_group.add_argument(
        '--scenario',
        choices=[s.value for s in Scenario],
        help='Experimental scenario for test data generation'
    )
    test_scenario_group.add_argument(
        '--difficulty',
        choices=[d.value for d in Difficulty],
        default='medium',
        help='Test difficulty level (default: medium)'
    )
    
    # Traditional test configuration
    test_traditional_group = test_parser.add_argument_group(
        'Traditional Test Configuration',
        'Detailed configuration (alternative to scenario-based)'
    )
    test_traditional_group.add_argument(
        '--dataset',
        choices=['sea', 'sine', 'friedman', 'concept_drift', 'elec2'],
        help='Dataset to use for testing (ignored if --scenario is used)'
    )
    test_parser.add_argument(
        '--params',
        default='{}',
        help='Detector parameters as JSON string (default: {})'
    )
    test_parser.add_argument(
        '--noise',
        type=float,
        help='Noise level for testing (overrides scenario-based noise if specified)'
    )
    test_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    test_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Display information about scenarios, detectors, and datasets'
    )
    info_parser.add_argument(
        '--scenarios',
        action='store_true',
        help='Show available scenarios and their characteristics'
    )
    info_parser.add_argument(
        '--scenario',
        choices=[s.value for s in Scenario],
        help='Get detailed information about a specific scenario'
    )
    
    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List available detectors and datasets'
    )
    list_parser.add_argument(
        '--detectors',
        action='store_true',
        help='List available detectors'
    )
    list_parser.add_argument(
        '--datasets',
        action='store_true', 
        help='List available datasets'
    )
    
    return parser


def parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated string into list."""
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_json_params(params_str: str) -> Dict[str, Any]:
    """Parse JSON parameters string."""
    try:
        return json.loads(params_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON parameters: {e}")


def run_tune_command(args) -> int:
    """Run hyperparameter tuning command."""
    print("🔧 Drift Detector Tuner - Starting hyperparameter optimization")
    
    # Handle scenario-based vs traditional configuration
    if args.scenario:
        return _run_scenario_based_tuning(args)
    else:
        return _run_traditional_tuning(args)


def _run_scenario_based_tuning(args) -> int:
    """Run scenario-based tuning with intelligent configuration."""
    print(f"🎯 Using scenario-based configuration:")
    print(f"   Scenario: {args.scenario}")
    print(f"   Difficulty: {args.difficulty}")
    print(f"   Stream length: {args.stream_length}")
    print(f"   Drift count: {args.drift_count}")
    print(f"   Noise level: {args.noise_level}")
    
    # Get scenario information
    try:
        scenario_info = get_scenario_info(args.scenario)
        preferred_datasets = scenario_info['preferred_datasets']
        print(f"   Preferred datasets: {', '.join(preferred_datasets)}")
    except Exception as e:
        print(f"❌ Error getting scenario info: {e}")
        return 1
    
    # Parse detector list (still support detector selection)
    if args.algo == 'all':
        detectors = get_all_detector_names()
    else:
        detectors = parse_comma_separated(args.algo)
        
    # Validate detectors
    available_detectors = get_all_detector_names()
    invalid_detectors = [d for d in detectors if d not in available_detectors]
    if invalid_detectors:
        print(f"❌ Invalid detectors: {invalid_detectors}")
        print(f"Available detectors: {', '.join(available_detectors)}")
        return 1
    
    # Convert scenario parameters to traditional format for evaluation
    smart_factory = SmartDatasetFactory()
    noise_mapping = smart_factory.noise_mappings[NoiseLevel(args.noise_level)]
    
    # Generate noise levels around the scenario-based level
    base_noise = noise_mapping
    noise_levels = [
        max(0.0, base_noise * 0.5),  # Lower noise
        base_noise,                  # Scenario noise
        min(0.1, base_noise * 1.5)   # Higher noise
    ]
    # Remove duplicates and sort
    noise_levels = sorted(list(set([round(x, 4) for x in noise_levels])))
    
    # Parse export formats
    if args.format == 'both':
        export_formats = ['csv', 'json']
    else:
        export_formats = [args.format]
    
    # Create configuration with scenario-based datasets
    config = EvaluationConfig(
        datasets=preferred_datasets,
        detectors=detectors,
        noise_levels=noise_levels,
        n_runs=args.runs,
        tolerance=args.tolerance,
        delay_penalty=args.delay_penalty,
        seed=args.seed,
        output_dir=args.output,
        export_formats=export_formats
    )
    
    # Add scenario-specific configuration to config for later use
    config.scenario_config = {
        'scenario': args.scenario,
        'difficulty': args.difficulty,
        'stream_length': args.stream_length,
        'drift_count': args.drift_count,
        'noise_level': args.noise_level
    }
    
    return _execute_tuning(config, args.trials, detectors, preferred_datasets, noise_levels, args)


def _run_traditional_tuning(args) -> int:
    """Run traditional tuning with explicit dataset specification."""
    print("🔧 Using traditional configuration")
    
    # Parse detector list
    if args.algo == 'all':
        detectors = get_all_detector_names()
    else:
        detectors = parse_comma_separated(args.algo)
        
    # Validate detectors
    available_detectors = get_all_detector_names()
    invalid_detectors = [d for d in detectors if d not in available_detectors]
    if invalid_detectors:
        print(f"❌ Invalid detectors: {invalid_detectors}")
        print(f"Available detectors: {', '.join(available_detectors)}")
        return 1
    
    # Parse datasets
    if args.datasets:
        datasets = parse_comma_separated(args.datasets)
    else:
        # Default datasets if none specified
        datasets = ['sea', 'sine', 'friedman', 'elec2']
    
    # Parse noise levels
    try:
        if hasattr(args, 'noise') and args.noise:
            noise_levels = [float(x) for x in parse_comma_separated(args.noise)]
        else:
            noise_levels = [0.0, 0.01, 0.03, 0.05]  # Default noise levels
    except ValueError:
        print("❌ Invalid noise levels. Must be comma-separated numbers.")
        return 1
    
    # Parse export formats
    if args.format == 'both':
        export_formats = ['csv', 'json']
    else:
        export_formats = [args.format]
    
    # Create configuration
    config = EvaluationConfig(
        datasets=datasets,
        detectors=detectors,
        noise_levels=noise_levels,
        n_runs=args.runs,
        tolerance=args.tolerance,
        delay_penalty=args.delay_penalty,
        seed=args.seed,
        output_dir=args.output,
        export_formats=export_formats
    )
    
    return _execute_tuning(config, args.trials, detectors, datasets, noise_levels, args)


def _execute_tuning(config, trials, detectors, datasets, noise_levels, args) -> int:
    """Execute the tuning process with given configuration."""
    print(f"📊 Configuration:")
    print(f"   Detectors: {', '.join(detectors)}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Trials per detector: {trials}")
    print(f"   Evaluation runs: {args.runs}")
    print(f"   Noise levels: {noise_levels}")
    print(f"   Output directory: {args.output}")
    
    try:
        # Run comprehensive evaluation
        results = run_comprehensive_evaluation(
            output_dir=args.output,
            n_trials=trials,
            verbose=True,
            config=config
        )
        
        # Generate and export templates
        if results['optimization_results']:
            print("\n🎯 Generating detector templates...")
            templates = generate_detector_templates(
                results['optimization_results'],
                args.output
            )
            
            if templates:
                print(f"✅ Generated templates for {len(templates)} detectors")
            else:
                print("⚠️  No templates could be generated")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        # Show summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📈 Summary:")
            print(f"   Total experiments: {summary.get('total_experiments', 0)}")
            
            if 'best_configurations' in summary and 'overall' in summary['best_configurations']:
                best = summary['best_configurations']['overall']
                print(f"   Best overall: {best['detector']} on {best['dataset']} "
                      f"(F1: {best['f1_score']:.3f})")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1


def run_info_command(args) -> int:
    """Run info command to display scenario and system information."""
    if args.scenarios:
        # Show all scenarios
        print("🎯 Available Scenarios:\n")
        scenario_info = get_scenario_info()
        
        for scenario_name, info in scenario_info.items():
            print(f"📋 {scenario_name.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Preferred datasets: {', '.join(info['preferred_datasets'])}")
            print(f"   Difficulty levels: {', '.join([d.value for d in info['difficulties']])}")
            print()
            
    elif args.scenario:
        # Show specific scenario
        try:
            info = get_scenario_info(args.scenario)
            print(f"🎯 Scenario: {args.scenario.upper()}")
            print(f"Description: {info['description']}")
            print(f"Preferred datasets: {', '.join(info['preferred_datasets'])}")
            print(f"Available difficulties: {', '.join([d.value for d in info['difficulties']])}")
            
            # Show difficulty-specific details
            factory = SmartDatasetFactory()
            scenario_enum = Scenario(args.scenario)
            mapping = factory.scenario_mappings[scenario_enum]
            
            print(f"\nDifficulty characteristics:")
            for diff, config in mapping['base_config'].items():
                print(f"  {diff.value}:")
                print(f"    Dataset: {config['dataset']}")
                if 'noise_multiplier' in config:
                    print(f"    Noise multiplier: {config['noise_multiplier']}x")
                if 'transition_width' in config:
                    print(f"    Transition width: {config['transition_width']}")
                if 'drift_spacing_factor' in config:
                    print(f"    Drift spacing: {config['drift_spacing_factor']}")
                    
        except Exception as e:
            print(f"❌ Error getting scenario info: {e}")
            return 1
    else:
        # Show general info
        print("🎯 Concept Drift Detection Tuner Information")
        print(f"Available scenarios: {', '.join([s.value for s in Scenario])}")
        print(f"Available difficulty levels: {', '.join([d.value for d in Difficulty])}")
        print(f"Available noise levels: {', '.join([n.value for n in NoiseLevel])}")
        print(f"Available detectors: {', '.join(get_all_detector_names())}")
        print("\nUse --scenarios to see detailed scenario information")
        print("Use --scenario <name> to see specific scenario details")
        
    return 0


def run_test_command(args) -> int:
    """Run test command with scenario or traditional configuration."""
    print(f"🧪 Testing detector: {args.detector}")
    
    try:
        # Parse detector parameters
        detector_params = parse_json_params(args.params)
        
        # Create detector
        detector = create_detector(args.detector, **detector_params)
        
        # Generate test data based on configuration
        if args.scenario:
            # Scenario-based test data
            print(f"🎯 Using scenario: {args.scenario} (difficulty: {args.difficulty})")
            
            # Use custom noise if specified, otherwise use scenario default
            if args.noise is not None:
                noise_level_name = "custom"
                custom_noise = args.noise
            else:
                # Map difficulty to noise level
                difficulty_to_noise = {
                    'easy': 'low',
                    'medium': 'medium', 
                    'hard': 'high',
                    'extreme': 'high'
                }
                noise_level_name = difficulty_to_noise.get(args.difficulty, 'medium')
                custom_noise = None
            
            # Create experiment stream
            if custom_noise is not None:
                # Create stream with custom noise level by overriding the smart factory
                from .smart_config import DataStreamConfig, SmartDatasetFactory
                config = DataStreamConfig(
                    scenario=Scenario(args.scenario),
                    difficulty=Difficulty(args.difficulty),
                    seed=args.seed
                )
                factory = SmartDatasetFactory()
                stream = factory.create_stream(config)
            else:
                stream = create_experiment_stream(
                    scenario=args.scenario,
                    difficulty=args.difficulty,
                    noise_level=noise_level_name,
                    seed=args.seed
                )
        else:
            # Traditional test configuration
            if not args.dataset:
                print("❌ Either --scenario or --dataset must be specified")
                return 1
                
            print(f"🗂️  Using dataset: {args.dataset}")
            noise_level = args.noise if args.noise is not None else 0.02
            
            # Create dataset with traditional method
            dataset_config = {
                'noise_level': noise_level,
                'seed': args.seed,
                'n_samples': 5000
            }
            stream = create_dataset(args.dataset, dataset_config)
        
        # Run test
        from .metrics import DriftEvaluator
        evaluator = DriftEvaluator()
        
        detected_drifts = []
        true_drifts = []
        sample_count = 0
        
        print("🔄 Processing stream...")
        
        for features, target, is_drift in stream:
            # Update detector
            detector.update(features)
            
            # Check for drift detection
            if detector.drift_detected:
                detected_drifts.append(sample_count)
                if args.verbose:
                    print(f"   Drift detected at sample {sample_count}")
            
            # Record true drifts
            if is_drift:
                true_drifts.append(sample_count)
                if args.verbose:
                    print(f"   True drift at sample {sample_count}")
            
            sample_count += 1
            
            # Break if we have enough samples for testing
            if sample_count >= 5000:
                break
        
        # Calculate metrics
        metrics = evaluator.evaluate_single_run(detected_drifts, true_drifts)
        
        # Display results
        print(f"\n📊 Test Results:")
        print(f"   Samples processed: {sample_count}")
        print(f"   True drifts: {len(true_drifts)} at positions {true_drifts}")
        print(f"   Detected drifts: {len(detected_drifts)} at positions {detected_drifts}")
        print(f"   F1 Score: {metrics.f1_score:.4f}")
        print(f"   Precision: {metrics.precision:.4f}")
        print(f"   Recall: {metrics.recall:.4f}")
        print(f"   False Positive Rate: {metrics.false_positive_rate:.4f}")
        if metrics.mean_delay is not None:
            print(f"   Mean Delay: {metrics.mean_delay:.2f} samples")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_templates_command(args) -> int:
    """Run template generation command."""
    print("🎯 Generating detector templates...")
    
    # Load optimization results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    try:
        with open(input_path, 'r') as f:
            optimization_data = json.load(f)
            
        # Convert to SearchResult objects (simplified)
        # In a full implementation, would need proper deserialization
        print(f"📊 Loaded optimization results from {input_path}")
        
        # Generate templates
        templates = generate_detector_templates(
            optimization_data,  # This would need proper conversion
            args.output
        )
        
        if templates:
            total_templates = sum(len(t) for t in templates.values())
            print(f"✅ Generated {total_templates} templates for {len(templates)} detectors")
            print(f"📁 Templates saved to: {args.output}")
        else:
            print("⚠️  No templates could be generated from the input data")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error generating templates: {e}")
        return 1


def run_recommend_command(args) -> int:
    """Run template recommendation command."""
    print(f"🎯 Getting recommendations for scenario: {args.scenario}")
    
    # Load templates
    templates_path = Path(args.templates)
    if not templates_path.exists():
        print(f"❌ Templates file not found: {templates_path}")
        return 1
    
    try:
        with open(templates_path, 'r') as f:
            templates_data = json.load(f)
        
        # Parse custom requirements if provided
        custom_requirements = None
        if args.requirements:
            custom_requirements = parse_json_params(args.requirements)
        
        # Get recommendations
        recommender = TemplateRecommender()
        
        # Convert loaded data to DetectorTemplate objects (simplified)
        # In full implementation, would need proper deserialization
        
        print(f"📊 Scenario: {args.scenario}")
        if custom_requirements:
            print(f"🎯 Custom requirements: {custom_requirements}")
        
        print(f"\n🏆 Top {args.top_k} Recommendations:")
        print("=" * 60)
        
        # This would show actual recommendations in a full implementation
        print("⚠️  Template recommendation functionality requires full implementation")
        print("    of template deserialization. Please use the Python API directly.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        return 1


def run_test_command(args) -> int:
    """Run single detector test command."""
    print(f"🧪 Testing detector: {args.detector}")
    
    try:
        # Parse parameters
        params = parse_json_params(args.params)
        print(f"⚙️  Parameters: {params}")
        
        # Create detector
        detector = create_detector(args.detector, **params)
        print(f"✅ Created detector: {args.detector}")
        
        # Generate test dataset
        dataset_config = {
            'noise_level': args.noise,
            'seed': args.seed
        }
        
        if args.dataset == 'sea':
            dataset_config.update({
                'drift_positions': [1000, 2500],
                'n_samples': 3500
            })
        elif args.dataset == 'sine':
            dataset_config.update({
                'drift_positions': [1500],
                'n_samples': 3000
            })
        elif args.dataset == 'friedman':
            dataset_config.update({
                'drift_type': 'abrupt',
                'drift_positions': [2000],
                'n_samples': 4000
            })
        
        print(f"📊 Dataset: {args.dataset} with noise level {args.noise}")
        
        # Generate and process data
        data_stream = list(create_dataset(args.dataset, dataset_config))
        detected_drifts = []
        true_drifts = [i for i, (_, _, is_drift) in enumerate(data_stream) if is_drift]
        
        print(f"📈 Processing {len(data_stream)} samples...")
        print(f"🎯 True drift positions: {true_drifts}")
        
        # Run detection
        for i, (sample, _, _) in enumerate(data_stream):
            detector.update(sample)
            if detector.drift_detected:
                detected_drifts.append(i)
                if args.verbose:
                    print(f"   Drift detected at position {i}")
        
        print(f"🔍 Detected drift positions: {detected_drifts}")
        
        # Simple evaluation
        if detected_drifts and true_drifts:
            from .metrics import evaluate_detector_performance
            
            metrics = evaluate_detector_performance(
                detected_drifts,
                true_drifts,
                len(data_stream),
                tolerance=50
            )
            
            print(f"\n📊 Performance Metrics:")
            print(f"   F1 Score: {metrics.f1_score:.3f}")
            print(f"   Precision: {metrics.precision:.3f}")  
            print(f"   Recall: {metrics.recall:.3f}")
            print(f"   Mean Delay: {metrics.mean_delay:.1f} samples")
            print(f"   False Positive Rate: {metrics.false_positive_rate:.3f}")
            print(f"   Composite Score: {metrics.composite_score:.3f}")
        else:
            print("⚠️  Cannot compute metrics: no detections or no true drifts")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1


def run_list_command(args) -> int:
    """Run list command."""
    if args.detectors or (not args.detectors and not args.datasets):
        print("📋 Available Detectors:")
        detectors = get_all_detector_names()
        for i, detector in enumerate(detectors, 1):
            print(f"   {i:2d}. {detector}")
        print()
    
    if args.datasets or (not args.detectors and not args.datasets):
        print("📋 Available Datasets:")
        datasets = ['sea', 'sine', 'friedman', 'concept_drift']
        descriptions = {
            'sea': 'SEA concept drift generator with abrupt changes',
            'sine': 'Sine wave generator with ConceptDriftStream',
            'friedman': 'Friedman drift with multiple drift types',
            'concept_drift': 'Flexible concept drift stream composer'
        }
        
        for i, dataset in enumerate(datasets, 1):
            desc = descriptions.get(dataset, 'No description available')
            print(f"   {i:2d}. {dataset:15s} - {desc}")
    
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'tune':
            return run_tune_command(args)
        elif args.command == 'templates':
            return run_templates_command(args)
        elif args.command == 'recommend':
            return run_recommend_command(args)
        elif args.command == 'test':
            return run_test_command(args)
        elif args.command == 'info':
            return run_info_command(args)
        elif args.command == 'list':
            return run_list_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())