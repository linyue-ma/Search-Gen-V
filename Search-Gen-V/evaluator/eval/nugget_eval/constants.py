"""Configuration constants for nugget evaluation framework"""

from typing import Dict, Set, List


# Run ID and Logging Constants
class RunIdConfig:
    """Constants for run ID generation and handling"""
    UUID_LENGTH = 8  # Length of UUID suffix for run_id
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Format for timestamp prefix
    SEPARATOR = "_"  # Separator between timestamp and UUID
    
    # Validation patterns
    MIN_RUN_ID_LENGTH = 17  # Minimum expected length for run_id
    MAX_RUN_ID_LENGTH = 30  # Maximum expected length for run_id


# Log Management Constants
class LogConfig:
    """Constants for log file management and merging"""
    MERGE_BATCH_SIZE = 1000  # Number of entries to process in batch during merge
    LOG_FILE_EXTENSION = ".jsonl"  # Standard log file extension
    MERGED_LOG_SUFFIX = "merged"  # Suffix for merged log files
    
    # Log level mappings
    LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    DEFAULT_LOG_LEVEL = "INFO"
    
    # File naming patterns
    LLM_LOG_PATTERN = "*llm_calls.jsonl"
    EVAL_LOG_PATTERN = "*evaluation.log"
    META_FILE_NAME = "run_meta.json"


# API and Model Constants
class ModelConfig:
    """Constants for model configuration and API handling"""
    # Default API parameters
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.95
    DEFAULT_MAX_TOKENS = 1024
    
    # Rate limiting and retry
    DEFAULT_RETRY_ATTEMPTS = 3
    BACKOFF_BASE_DELAY = 1.0  # seconds
    MAX_BACKOFF_DELAY = 60.0  # seconds
    
    # Response validation
    MIN_CONTENT_LENGTH = 1
    MAX_RESPONSE_SIZE = 16384  # Maximum expected response size in chars


# Batch Processing Constants  
class BatchConfig:
    """Constants for batch processing and parallel execution"""
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 50
    DEFAULT_BATCH_SIZE = 10
    
    # Worker configuration
    MIN_WORKERS = 1
    MAX_WORKERS = 32
    DEFAULT_WORKERS = 8
    
    # Resource management
    WORKER_TIMEOUT = 300  # seconds
    CLEANUP_TIMEOUT = 10  # seconds for resource cleanup


# Validation and Error Handling Constants
class ValidationConfig:
    """Constants for validation and error handling"""
    # Supported formats and modes
    VALID_PROMPT_TYPES = {
        "legacy", "no_reasoning", "short_cot", "long_cot", 
        "optimized_prompt", "template_based"
    }
    
    VALID_FORMAT_TYPES = {
        "adaptive", "json", "markdown", "csv", "yaml", "xml", 
        "tsv", "numbered", "comma_separated", "pipe_separated", 
        "python_list"
    }
    
    VALID_ERROR_HANDLING = {"sequential", "strict"}
    VALID_REPORT_LEVELS = {"full", "standard", "minimal"}
    
    # Label validation
    VALID_LABELS = {"support", "partial_support", "not_support", "error"}
    ORDERED_LABELS = ["support", "partial_support", "not_support"]
    
    # Error categories
    ERROR_CATEGORIES = {
        "api_error", "parsing_error", "timeout_error", 
        "validation_error", "unexpected_error"
    }


# Metrics and Analysis Constants
class MetricsConfig:
    """Constants for metrics calculation and analysis"""
    # Default k values for different metrics
    DEFAULT_AVG_K = [1, 5, 10]
    DEFAULT_PASS_K = [1, 5, 10] 
    DEFAULT_MAJ_K = [3, 5, 7]
    
    # Statistical thresholds
    MIN_SAMPLE_SIZE = 10  # Minimum samples for meaningful statistics
    CONFIDENCE_LEVEL = 0.95  # For confidence intervals
    
    # Performance benchmarks
    FAST_RESPONSE_THRESHOLD = 1.0  # seconds
    SLOW_RESPONSE_THRESHOLD = 10.0  # seconds
    HIGH_TOKEN_USAGE_THRESHOLD = 2000  # tokens per request


# File Path and Directory Constants
class PathConfig:
    """Constants for file paths and directory structure"""
    DEFAULT_EVAL_LOG_DIR = "logs/eval"
    DEFAULT_LLM_LOG_DIR = "logs/llm"
    
    # Subdirectory patterns
    WORKER_SUBDIR_PREFIX = "worker_"
    RUN_SUBDIR_PATTERN = "{run_id}"
    
    # Required file extensions
    SUPPORTED_DATA_EXTENSIONS = {".jsonl", ".json"}
    SUPPORTED_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json"}


# Case Study and Analysis Constants
class AnalysisConfig:
    """Constants for case study and analysis functionality"""
    # Report generation
    REPORT_FORMATS = {"html", "markdown", "json", "csv"}
    DEFAULT_REPORT_FORMAT = "markdown"
    
    # Correlation analysis
    CORRELATION_TIME_WINDOW = 300  # seconds - window for correlating logs
    MIN_CORRELATION_CONFIDENCE = 0.8  # Minimum confidence for log correlation
    
    # Performance analysis thresholds
    ERROR_RATE_THRESHOLD = 0.05  # 5% error rate threshold
    LATENCY_PERCENTILES = [50, 90, 95, 99]  # Percentiles to calculate
    
    # Case study filtering
    MAX_CASES_PER_CATEGORY = 100  # Maximum cases to include per error category
    MIN_CASE_FREQUENCY = 3  # Minimum occurrences to consider pattern


# Network and Connectivity Constants
class NetworkConfig:
    """Constants for network operations and API connectivity"""
    CONNECTION_TIMEOUT = 10.0  # seconds
    READ_TIMEOUT = 30.0  # seconds  
    
    # Health check parameters
    HEALTH_CHECK_INTERVAL = 60  # seconds
    MAX_HEALTH_CHECK_FAILURES = 3
    
    # Supported URL schemes
    SUPPORTED_SCHEMES = {"http", "https"}
    DEFAULT_PORT_HTTP = 80
    DEFAULT_PORT_HTTPS = 443


# Export commonly used constants
__all__ = [
    "RunIdConfig",
    "LogConfig", 
    "ModelConfig",
    "BatchConfig",
    "ValidationConfig",
    "MetricsConfig",
    "PathConfig",
    "AnalysisConfig",
    "NetworkConfig"
]