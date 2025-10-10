#!/usr/bin/env python3
"""
Unified CLI for nugget evaluation

This script provides a single entry point for running nugget matching evaluations
with configurable parameters and support for both single and multi-run modes.

Usage:
    nugget-eval --config config/thinking_mode.yaml
    nugget-eval --config config/multi_run.yaml --num-runs 20
    nugget-eval --generate-config config/my_config.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

from .config import Config, generate_config_template
from .evaluator import NuggetEvaluator
from .exceptions import ConfigurationError, ModelAPIError


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Nugget Matching Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with configuration file
  nugget-eval --config config/thinking_mode.yaml
  
  # Run multi-run evaluation
  nugget-eval --config config/multi_run.yaml --num-runs 20
  
  # Override specific parameters
  nugget-eval --config config/base.yaml \\
    --input-path /path/to/input.jsonl \\
    --enable-thinking \\
    --batch-size 5
  
  # Generate configuration template
  nugget-eval --generate-config config/my_config.yaml --template-type base
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    # Configuration generation
    parser.add_argument(
        "--generate-config",
        type=str,
        help="Generate a configuration template and save to specified path"
    )
    
    parser.add_argument(
        "--template-type",
        choices=["base", "thinking", "multi_run"],
        default="base",
        help="Type of configuration template to generate"
    )
    
    # Data paths
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to input JSONL file (overrides config)"
    )
    
    parser.add_argument(
        "--gold-path",
        type=str,
        help="Path to gold JSONL file (overrides config)"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name/path (overrides config)"
    )
    
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode (overrides config)"
    )
    
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking mode (overrides config)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Model temperature (overrides config)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens for model response (overrides config)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Nugget batch size (overrides config)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers (overrides config)"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Number of evaluation runs for statistical analysis (overrides config)"
    )
    
    # Logging
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save prediction results to files"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output except errors"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def setup_logging(level: str, quiet: bool = False, verbose: bool = False):
    """Setup basic logging"""
    if quiet:
        level = "ERROR"
    elif verbose:
        level = "DEBUG"
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_and_override_config(args) -> Config:
    """Load configuration and apply command line overrides"""
    if not args.config:
        print("Error: Configuration file is required. Use --config or --generate-config")
        sys.exit(1)
    
    try:
        config = Config.from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.input_path:
        config.data.input_path = args.input_path
    
    if args.gold_path:
        config.data.gold_path = args.gold_path
    
    if args.model_name:
        config.model.name = args.model_name
    
    if args.enable_thinking:
        config.model.enable_thinking = True
    
    if args.disable_thinking:
        config.model.enable_thinking = False
    
    if args.temperature is not None:
        config.model.temperature = args.temperature
    
    if args.max_tokens:
        config.model.max_tokens = args.max_tokens
    
    if args.batch_size:
        config.evaluation.batch_size = args.batch_size
    
    if args.num_workers:
        config.evaluation.num_workers = args.num_workers
    
    if args.num_runs:
        config.evaluation.num_runs = args.num_runs
    
    if args.save_predictions:
        config.logging.save_predictions = True
    
    config.logging.log_level = args.log_level
    
    return config


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup basic logging
    setup_logging(args.log_level, args.quiet, args.verbose)
    
    # Handle configuration generation
    if args.generate_config:
        try:
            generate_config_template(args.generate_config, args.template_type)
            print(f"Configuration template generated: {args.generate_config}")
            print(f"Template type: {args.template_type}")
            print("\nNext steps:")
            print(f"1. Edit the configuration file: {args.generate_config}")
            print("2. Update the paths and model settings")
            print(f"3. Run evaluation: nugget-eval --config {args.generate_config}")
            return
        except Exception as e:
            print(f"Error generating configuration: {e}")
            sys.exit(1)
    
    # Load and validate configuration
    config = load_and_override_config(args)
    
    try:
        config.validate()
    except ConfigurationError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Print configuration summary
    if not args.quiet:
        print(f"\n=== Evaluation Configuration ===")
        print(f"Model: {config.model.name}")
        print(f"Thinking mode: {config.model.enable_thinking}")
        print(f"Input data: {config.data.input_path}")
        print(f"Gold data: {config.data.gold_path}")
        print(f"Batch size: {config.evaluation.batch_size}")
        print(f"Num workers: {config.evaluation.num_workers}")
        print(f"Num runs: {config.evaluation.num_runs}")
        print(f"Save predictions: {config.logging.save_predictions}")
        print("=" * 35)
    
    # Run evaluation
    try:
        evaluator = NuggetEvaluator(config)
        results = evaluator.run_evaluation()
        
        if not args.quiet:
            print(f"\n=== Evaluation Complete ===")
            print(f"Evaluation type: {results['type']}")
            if results['type'] == 'single_run':
                print(f"Truncation rate: {results['truncation_stats']['truncation_rate']:.2f}%")
            else:
                print(f"Number of runs: {results['num_runs']}")
            print("Detailed results are shown above.")
            
            if config.logging.save_predictions:
                print(f"Results saved to: {config.logging.eval_log_dir}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()