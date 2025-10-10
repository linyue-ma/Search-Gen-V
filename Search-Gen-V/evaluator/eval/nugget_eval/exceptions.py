"""Custom exceptions for nugget evaluation framework"""


class NuggetEvalError(Exception):
    """Base exception for nugget evaluation framework"""
    pass


class ConfigurationError(NuggetEvalError):
    """Configuration validation or loading error"""
    pass


class ModelAPIError(NuggetEvalError):
    """Model API call error"""
    
    def __init__(self, message: str, is_temporary: bool = False, retry_after: int = None):
        super().__init__(message)
        self.is_temporary = is_temporary
        self.retry_after = retry_after


class DataLoadError(NuggetEvalError):
    """Data loading or parsing error"""
    pass


class EvaluationError(NuggetEvalError):
    """General evaluation process error"""
    pass


class NetworkTimeoutError(ModelAPIError):
    """Network timeout during API call"""
    
    def __init__(self, message: str, retry_after: int = 5):
        super().__init__(message, is_temporary=True, retry_after=retry_after)


class AuthenticationError(ModelAPIError):
    """API authentication error"""
    
    def __init__(self, message: str):
        super().__init__(message, is_temporary=False)


class RateLimitError(ModelAPIError):
    """API rate limit exceeded"""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, is_temporary=True, retry_after=retry_after)


class ModelNotFoundError(ModelAPIError):
    """Requested model not found"""
    
    def __init__(self, message: str):
        super().__init__(message, is_temporary=False)