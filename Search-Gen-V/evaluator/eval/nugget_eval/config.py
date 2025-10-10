"""Configuration management for nugget evaluation"""

import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import requests

from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    name: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024
    enable_thinking: bool = True
    
    # Prompt type configuration
    prompt_type: str = "legacy"  # legacy|no_reasoning|short_cot|long_cot
    
    # Output format configuration (works with all prompt types)
    format_type: str = "adaptive"  # adaptive|json|markdown|csv|yaml|xml|tsv|numbered|comma_separated|pipe_separated
    
    # Error handling strategy
    error_handling: str = "sequential"  # sequential|strict
    partial_recovery: bool = True  # Whether to recover partial successful predictions


@dataclass
class DataConfig:
    input_path: str = ""
    gold_path: str = ""


@dataclass
class EvaluationConfig:
    batch_size: int = 10
    num_workers: int = 8
    num_runs: int = 1


@dataclass
class MetricsConfig:
    avg_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    pass_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    maj_k: List[int] = field(default_factory=lambda: [3, 5, 7])


@dataclass
class LoggingConfig:
    eval_log_dir: str = "logs/eval"
    llm_log_dir: str = "logs/llm"
    save_predictions: bool = True
    log_level: str = "INFO"
    
    # New detailed reporting configuration
    report_detail_level: str = "full"  # full|standard|minimal
    show_format_analysis: bool = True
    show_error_breakdown: bool = True
    show_batch_recovery_stats: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'data' in data:
            for key, value in data['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'evaluation' in data:
            for key, value in data['evaluation'].items():
                if hasattr(config.evaluation, key):
                    setattr(config.evaluation, key, value)
        
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                if hasattr(config.metrics, key):
                    setattr(config.metrics, key, value)
        
        if 'logging' in data:
            for key, value in data['logging'].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'base_url': self.model.base_url,
                'api_key': self.model.api_key,
                'name': self.model.name,
                'temperature': self.model.temperature,
                'top_p': self.model.top_p,
                'max_tokens': self.model.max_tokens,
                'enable_thinking': self.model.enable_thinking,
                'format_type': self.model.format_type,
                'error_handling': self.model.error_handling,
                'partial_recovery': self.model.partial_recovery
            },
            'data': {
                'input_path': self.data.input_path,
                'gold_path': self.data.gold_path
            },
            'evaluation': {
                'batch_size': self.evaluation.batch_size,
                'num_workers': self.evaluation.num_workers,
                'num_runs': self.evaluation.num_runs
            },
            'metrics': {
                'avg_k': self.metrics.avg_k,
                'pass_k': self.metrics.pass_k,
                'maj_k': self.metrics.maj_k
            },
            'logging': {
                'eval_log_dir': self.logging.eval_log_dir,
                'llm_log_dir': self.logging.llm_log_dir,
                'save_predictions': self.logging.save_predictions,
                'log_level': self.logging.log_level,
                'report_detail_level': self.logging.report_detail_level,
                'show_format_analysis': self.logging.show_format_analysis,
                'show_error_breakdown': self.logging.show_error_breakdown,
                'show_batch_recovery_stats': self.logging.show_batch_recovery_stats
            }
        }
    
    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self, strict: bool = True):
        """
        Validate configuration values
        
        Args:
            strict: If True, perform network connectivity checks and strict validation
        """
        errors = []
        warnings = []
        
        # Model configuration validation
        if not self.model.name:
            errors.append("Model name cannot be empty")
        
        # URL validation
        if not self.model.base_url:
            errors.append("Model base_url cannot be empty")
        else:
            try:
                parsed_url = urlparse(self.model.base_url)
                if not parsed_url.scheme:
                    errors.append(f"Invalid base_url format (missing scheme): {self.model.base_url}")
                elif parsed_url.scheme not in ['http', 'https']:
                    errors.append(f"Unsupported URL scheme: {parsed_url.scheme}")
                elif not parsed_url.netloc:
                    errors.append(f"Invalid base_url format (missing host): {self.model.base_url}")
            except Exception as e:
                errors.append(f"Failed to parse base_url: {e}")
        
        # Model parameter validation
        if not (0.0 <= self.model.temperature <= 2.0):
            errors.append(f"Temperature must be between 0.0 and 2.0, got {self.model.temperature}")
        
        if not (0.0 <= self.model.top_p <= 1.0):
            errors.append(f"top_p must be between 0.0 and 1.0, got {self.model.top_p}")
        
        if self.model.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {self.model.max_tokens}")
        elif self.model.max_tokens > 4096:
            warnings.append(f"max_tokens is very high ({self.model.max_tokens}), may cause high latency")
        
        # New format configuration validation
        valid_formats = {
            "adaptive", "json", "markdown", "csv", "yaml", "xml", "tsv", 
            "numbered", "comma_separated", "pipe_separated", "python_list"
        }
        if self.model.format_type not in valid_formats:
            errors.append(f"Invalid format_type '{self.model.format_type}', must be one of {valid_formats}")
        
        
        # Validate error handling strategy
        valid_error_handling = {"sequential", "strict"}
        if self.model.error_handling not in valid_error_handling:
            errors.append(f"Invalid error_handling '{self.model.error_handling}', must be one of {valid_error_handling}")
        
        # Validate report detail level
        valid_detail_levels = {"full", "standard", "minimal"}
        if self.logging.report_detail_level not in valid_detail_levels:
            errors.append(f"Invalid report_detail_level '{self.logging.report_detail_level}', must be one of {valid_detail_levels}")
        
        # Data path validation
        if not self.data.input_path:
            errors.append("Input data path cannot be empty")
        elif not Path(self.data.input_path).exists():
            errors.append(f"Input data file not found: {self.data.input_path}")
        elif not Path(self.data.input_path).is_file():
            errors.append(f"Input path is not a file: {self.data.input_path}")
        elif not self.data.input_path.endswith(('.jsonl', '.json')):
            warnings.append(f"Input file should be JSONL format: {self.data.input_path}")
        
        if not self.data.gold_path:
            errors.append("Gold data path cannot be empty")
        elif not Path(self.data.gold_path).exists():
            errors.append(f"Gold data file not found: {self.data.gold_path}")
        elif not Path(self.data.gold_path).is_file():
            errors.append(f"Gold path is not a file: {self.data.gold_path}")
        elif not self.data.gold_path.endswith(('.jsonl', '.json')):
            warnings.append(f"Gold file should be JSONL format: {self.data.gold_path}")
        
        # Evaluation parameters validation
        if self.evaluation.batch_size <= 0:
            errors.append("Batch size must be positive")
        elif self.evaluation.batch_size > 50:
            warnings.append(f"Batch size is very high ({self.evaluation.batch_size}), may cause API rate limiting")
        
        if self.evaluation.num_workers <= 0:
            errors.append("Number of workers must be positive")
        elif self.evaluation.num_workers > 32:
            warnings.append(f"Number of workers is very high ({self.evaluation.num_workers}), may overwhelm the API")
        
        if self.evaluation.num_runs <= 0:
            errors.append("Number of runs must be positive")
        elif self.evaluation.num_runs > 1 and self.model.enable_thinking:
            warnings.append("Multi-run evaluation with thinking mode may be very slow")
        
        # Metrics validation
        if any(k <= 0 for k in self.metrics.avg_k):
            errors.append("All avg_k values must be positive")
        
        if any(k <= 0 for k in self.metrics.pass_k):
            errors.append("All pass_k values must be positive")
        
        if any(k <= 0 for k in self.metrics.maj_k):
            errors.append("All maj_k values must be positive")
        
        # Check k values don't exceed num_runs for multi-run scenarios
        if self.evaluation.num_runs > 1:
            max_avg_k = max(self.metrics.avg_k) if self.metrics.avg_k else 0
            max_pass_k = max(self.metrics.pass_k) if self.metrics.pass_k else 0
            max_maj_k = max(self.metrics.maj_k) if self.metrics.maj_k else 0
            
            if max_avg_k > self.evaluation.num_runs:
                errors.append(f"max(avg_k) = {max_avg_k} cannot exceed num_runs = {self.evaluation.num_runs}")
            
            if max_pass_k > self.evaluation.num_runs:
                errors.append(f"max(pass_k) = {max_pass_k} cannot exceed num_runs = {self.evaluation.num_runs}")
            
            if max_maj_k > self.evaluation.num_runs:
                errors.append(f"max(maj_k) = {max_maj_k} cannot exceed num_runs = {self.evaluation.num_runs}")
            
            # Warn about even numbers in maj_k (potential ties)
            even_maj_k = [k for k in self.metrics.maj_k if k % 2 == 0]
            if even_maj_k:
                warnings.append(f"maj_k contains even numbers {even_maj_k} which may result in ties")
        
        # Log directory validation
        try:
            Path(self.logging.eval_log_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create eval log directory: {e}")
        
        try:
            Path(self.logging.llm_log_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create LLM log directory: {e}")
        
        # Network connectivity check (if strict mode)
        if strict and self.model.base_url:
            try:
                # Try a simple HEAD request with timeout
                response = requests.head(self.model.base_url, timeout=5)
                if response.status_code >= 500:
                    warnings.append(f"API server may be down (status {response.status_code})")
            except requests.exceptions.ConnectionError:
                warnings.append(f"Cannot connect to API server: {self.model.base_url}")
            except requests.exceptions.Timeout:
                warnings.append(f"API server connection timeout: {self.model.base_url}")
            except Exception as e:
                warnings.append(f"API connectivity check failed: {e}")
        
        # Print warnings if any
        if warnings:
            print("⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"    - {warning}")
        
        # Raise errors if any
        if errors:
            error_msg = "❌ Configuration validation failed:\n" + "\n".join(f"    - {error}" for error in errors)
            raise ConfigurationError(error_msg)
    
    def override_from_args(self, args):
        """Override configuration with command line arguments"""
        if hasattr(args, 'input_path') and args.input_path:
            self.data.input_path = args.input_path
        
        if hasattr(args, 'gold_path') and args.gold_path:
            self.data.gold_path = args.gold_path
        
        if hasattr(args, 'model_name') and args.model_name:
            self.model.name = args.model_name
        
        if hasattr(args, 'batch_size') and args.batch_size:
            self.evaluation.batch_size = args.batch_size
        
        if hasattr(args, 'num_workers') and args.num_workers:
            self.evaluation.num_workers = args.num_workers
        
        if hasattr(args, 'num_runs') and args.num_runs:
            self.evaluation.num_runs = args.num_runs
        
        if hasattr(args, 'enable_thinking') and args.enable_thinking is not None:
            self.model.enable_thinking = args.enable_thinking
        
        if hasattr(args, 'temperature') and args.temperature is not None:
            self.model.temperature = args.temperature
        
        if hasattr(args, 'max_tokens') and args.max_tokens:
            self.model.max_tokens = args.max_tokens
        
        # New configuration overrides
        if hasattr(args, 'format_type') and args.format_type:
            self.model.format_type = args.format_type
        
        if hasattr(args, 'error_handling') and args.error_handling:
            self.model.error_handling = args.error_handling
        
        if hasattr(args, 'partial_recovery') and args.partial_recovery is not None:
            self.model.partial_recovery = args.partial_recovery
        
        if hasattr(args, 'report_detail_level') and args.report_detail_level:
            self.logging.report_detail_level = args.report_detail_level


def generate_config_template(output_path: str, config_type: str = "base"):
    """Generate a configuration template file"""
    config = Config()
    
    if config_type == "thinking":
        config.model.enable_thinking = True
        config.model.max_tokens = 1024
        config.evaluation.num_runs = 1
    elif config_type == "multi_run":
        config.model.enable_thinking = False
        config.model.max_tokens = 64
        config.evaluation.num_runs = 16
    
    # Set example paths
    config.data.input_path = "/path/to/your/input.jsonl"
    config.data.gold_path = "/path/to/your/gold.jsonl"
    config.model.name = "/path/to/your/model"
    
    config.save_yaml(output_path)
    print(f"Configuration template saved to: {output_path}")