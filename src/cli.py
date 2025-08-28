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


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Drift Detector Tuner - Hyperparameter optimization for concept drift detectors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive evaluation with all detectors and datasets
  python -m src.cli tune --algo all --datasets sea,sine,friedman --trials 200
  
  # Tune specific detector on specific dataset
  python -m src.cli tune --algo adwin --datasets sea --trials 100 --output results/adwin
  
  # Quick evaluation with fewer trials
  python -m src.cli tune --trials 50 --runs 3 --output quick_results
  
  # Generate templates from existing results
  python -m src.cli templates --input results/optimization_results.json --output templates/
  
  # Get recommendations for specific scenario
  python -m src.cli recommend --templates templates/detector_templates.json --scenario critical_systems
  
  # Run single detector test
  python -m src.cli test --detector adwin --dataset sea --params '{"delta": 0.002}'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tune command
    tune_parser = subparsers.add_parser(
        'tune',
        help='Run hyperparameter tuning and evaluation'
    )
    tune_parser.add_argument(
        '--algo', '--detectors',
        default='all',
        help='Detectors to tune (comma-separated or "all"). Available: ' + ', '.join(get_all_detector_names())
    )
    tune_parser.add_argument(
        '--datasets',
        default='sea,sine,friedman',
        help='Datasets to use (comma-separated). Available: sea, sine, friedman, concept_drift'
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
    test_parser.add_argument(
        '--dataset',
        default='sea',
        choices=['sea', 'sine', 'friedman', 'concept_drift'],
        help='Dataset to use for testing (default: sea)'
    )
    test_parser.add_argument(
        '--params',
        default='{}',
        help='Detector parameters as JSON string (default: {})'
    )
    test_parser.add_argument(
        '--noise',
        type=float,
        default=0.02,
        help='Noise level for testing (default: 0.02)'
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
    datasets = parse_comma_separated(args.datasets)
    
    # Parse noise levels
    try:
        noise_levels = [float(x) for x in parse_comma_separated(args.noise)]
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
    
    print(f"📊 Configuration:")
    print(f"   Detectors: {', '.join(detectors)}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Trials per detector: {args.trials}")
    print(f"   Evaluation runs: {args.runs}")
    print(f"   Noise levels: {noise_levels}")
    print(f"   Output directory: {args.output}")
    
    try:
        # Run comprehensive evaluation
        results = run_comprehensive_evaluation(
            output_dir=args.output,
            n_trials=args.trials,
            verbose=True
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