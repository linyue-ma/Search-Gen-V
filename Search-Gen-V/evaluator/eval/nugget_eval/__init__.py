"""Nugget evaluation framework with enhanced multi-format support"""

from .config import Config, ModelConfig, DataConfig, EvaluationConfig, MetricsConfig, LoggingConfig
from .evaluator import NuggetEvaluator
from .metrics import MetricsCalculator
from .common import (
    ModelClient, 
    parse_labels_multi_format,
    build_multi_format_prompt,
    process_single_item,
    aggregate_assignment,
    create_gold_data_index,
    load_jsonl_data,
    save_jsonl_data
)
from .exceptions import (
    NuggetEvalError,
    ModelAPIError, 
    NetworkTimeoutError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    DataLoadError,
    ConfigurationError
)

__version__ = "2.0.0"
__all__ = [
    # Configuration
    "Config", "ModelConfig", "DataConfig", "EvaluationConfig", "MetricsConfig", "LoggingConfig",
    
    # Core classes
    "NuggetEvaluator", "MetricsCalculator", "ModelClient",
    
    # Multi-format parsing functions
    "parse_labels_multi_format", "build_multi_format_prompt",
    
    # Utility functions
    "process_single_item", "aggregate_assignment", "create_gold_data_index",
    "load_jsonl_data", "save_jsonl_data",
    
    # Exception classes
    "NuggetEvalError", "ModelAPIError", "NetworkTimeoutError",
    "AuthenticationError", "RateLimitError", "ModelNotFoundError",
    "DataLoadError", "ConfigurationError"
]
