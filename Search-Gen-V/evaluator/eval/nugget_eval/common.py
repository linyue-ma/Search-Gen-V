"""Common functions extracted from the original evaluation scripts"""

import json
import re
import logging
import time
import yaml
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from math import ceil
from typing import List, Dict, Any, Tuple, Optional, cast
# NOTE: Import OpenAI lazily inside methods to avoid linter errors when package is unavailable at analysis time
from collections import defaultdict

from .exceptions import (
    ModelAPIError, NetworkTimeoutError, AuthenticationError, 
    RateLimitError, ModelNotFoundError, DataLoadError
)
from .prompts import build_template_prompt, prompt_template_manager


# Priority mapping for nugget match labels
MATCH_PRIORITY = {
    "not_support": 0,
    "partial_support": 1,
    "support": 2,
    "error": -1
}

PRIORITY_TO_LABEL = {v: k for k, v in MATCH_PRIORITY.items()}
LABELS_ORDERED = ['support', 'partial_support', 'not_support']


def normalize_label(label: str) -> str:
    
    return label.strip().lower()


def get_valid_labels_mapping():
    
    mapping = {
        
        "support": "support",
        "partial_support": "partial_support",
        "not_support": "not_support",
        
        "supported": "support",
        "supports": "support",
        "yes": "support",
        "true": "support",
        
        "partial": "partial_support",
        "partly_support": "partial_support",
        "partial_supported": "partial_support",
        "partially_support": "partial_support",
        "partially_supported": "partial_support",
        "maybe": "partial_support",
        
        "not_supported": "not_support",
        "no_support": "not_support",
        "unsupported": "not_support",
        "no": "not_support",
        "false": "not_support",
        "none": "not_support",
    }
    return mapping


class ModelClient:
    """OpenAI-compatible API client for model inference with resource management"""
    
    def __init__(self, base_url: str, api_key: str, model_name: str, llm_log_dir: str = "logs/llm", run_id: Optional[str] = None):
        # Lazy import to avoid static linter errors if dependency is not installed
        try:
            from openai import OpenAI as _OpenAI  # type: ignore
        except Exception as e:
            raise ImportError("openai package is required for ModelClient") from e
        self.client = _OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.llm_log_dir = Path(llm_log_dir)
        self.llm_log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        
        # Create LLM call log file
        from .constants import RunIdConfig
        timestamp = datetime.now().strftime(RunIdConfig.TIMESTAMP_FORMAT)
        name_part = f"{timestamp}_{self.run_id}_" if self.run_id else f"{timestamp}_"
        self.llm_log_file = self.llm_log_dir / f"{name_part}llm_calls.jsonl"
        # Ensure the log file exists early to avoid empty worker directories during runtime
        try:
            self.llm_log_file.touch(exist_ok=True)
        except Exception as e:
            # Non-fatal: logging directory exists but file creation failed
            self.logger.warning(f"Failed to create llm_calls log file early: {e}")
        
        # Resource management
        self._closed = False
        self._log_buffer = []
        self._buffer_size = 50  # Flush every 50 entries
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup"""
        self.close()
    
    def close(self):
        """Clean up resources"""
        if self._closed:
            return
            
        try:
            # Flush any remaining logs
            self._flush_log_buffer()
            self.logger.info(f"ModelClient closed, logs saved to {self.llm_log_file}")
        except Exception as e:
            self.logger.error(f"Error during ModelClient cleanup: {e}")
        finally:
            self._closed = True

    def flush_logs(self):
        """Public method to flush buffered LLM logs to disk immediately.
        Safe to call frequently; no-op if buffer is empty.
        """
        try:
            self._flush_log_buffer()
        except Exception as e:
            self.logger.error(f"Failed to flush LLM logs: {e}")
    
    def predict_batch(self, query: str, block_text: str, nuggets_text_list: List[str], 
                     temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 1024,
                     enable_thinking: bool = True, prompt_type: str = "legacy",
                     format_type: str = "adaptive", context: Optional[Dict[str, Any]] = None,
                     nuggets_with_assignment: Optional[List[Dict[str, Any]]] = None,
                     case_sensitive: bool = False) -> Tuple[List[Optional[str]], Dict[str, Any]]:
        """
        Predict nugget matching labels for a batch of nuggets with multi-format support
        
        Args:
            query: The search query
            block_text: The text passage
            nuggets_text_list: List of nugget texts to judge
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            enable_thinking: Whether to enable thinking mode
            prompt_type: Prompt type (legacy|no_reasoning|short_cot|long_cot|optimized)
            format_type: Output format type (json|csv|markdown|yaml|xml|tsv|numbered|adaptive)
            context: Additional context for logging
            nuggets_with_assignment: List of nugget dicts with 'text' and 'assignment' keys for true labels
            case_sensitive: Whether to use case sensitive label matching (default: False)
        
        Returns:
            Tuple of (predictions, batch_stats)
            batch_stats contains detailed information about parsing results
        """
        # Choose prompt building method based on prompt_type
        if prompt_type == "legacy":
            messages = build_multi_format_prompt(query, block_text, nuggets_text_list, format_type)
        else:
            # Use template-based prompts for no_reasoning, short_cot, long_cot
            messages = build_template_prompt(prompt_type, query, block_text, nuggets_text_list, format_type)
        
        # Extract true labels if nuggets_with_assignment is provided
        true_labels = None
        if nuggets_with_assignment:
            try:
                true_labels = [nug.get("assignment") for nug in nuggets_with_assignment[:len(nuggets_text_list)]]
            except (AttributeError, KeyError, TypeError):
                self.logger.warning("Failed to extract true labels from nuggets_with_assignment")
                true_labels = None
        
        # Log the API call
        call_timestamp = datetime.now().isoformat()
        call_log = {
            "timestamp": call_timestamp,
            "run_id": self.run_id,
            "context": dict(context) if context is not None else {},
            "model": self.model_name,
            "messages": messages,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "format_type": format_type,
                "case_sensitive": case_sensitive
            },
            "nuggets_count": len(nuggets_text_list)
        }
        # Enrich context with query and block_text for downstream traceability
        try:
            ctx = call_log.get("context", {}) or {}
            ctx.update({"query": query, "block_text": block_text})
            call_log["context"] = ctx
        except Exception:
            # Non-fatal: keep existing context if enrichment fails
            pass
        
        try:
            response = self._make_api_call(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking
            )
            
            # Safely extract content and finish_reason
            finish_reason = None
            content = ""
            labels = None
            raw_choice = None
            
            if response is not None and getattr(response, 'choices', None):
                try:
                    raw_choice = response.choices[0] if response.choices else None
                except (IndexError, AttributeError, TypeError) as e:
                    self.logger.warning(f"Failed to extract choice from response: {e}")
                    raw_choice = None
            
            if raw_choice is not None and getattr(raw_choice, 'message', None) is not None:
                raw_content = raw_choice.message.content
                reasoning_content = getattr(raw_choice.message, 'reasoning_content', None)
                if enable_thinking and reasoning_content is not None:
                    content = str(reasoning_content).strip()
                    
                    reasoning_match = re.search(r'</reasoning>\s*(.+?)$', reasoning_content, re.DOTALL)
                    if reasoning_match:
                        labels_text = reasoning_match.group(1).strip()
                        labels = safe_parse_labels(labels_text, case_sensitive)
                    else:
                        labels = safe_parse_labels(content, case_sensitive)
                        
                elif raw_content is not None:
                    content = str(raw_content).strip()
                    labels = safe_parse_labels(content, case_sensitive)
                else:
                    self.logger.warning(f"API returned None content and reasoning_content, {response}")
                    content = ""
                    labels = None
            
            try:
                finish_reason = getattr(raw_choice, 'finish_reason', None)
            except (AttributeError, TypeError) as e:
                self.logger.debug(f"Failed to extract finish_reason: {e}")
                finish_reason = None
            
            # Add response to log
            usage = getattr(response, 'usage', None) if response is not None else None
            usage_dict = None
            if usage is not None:
                try:
                    # Convert CompletionUsage object to dictionary
                    if hasattr(usage, 'model_dump'):
                        # Pydantic v2 style
                        usage_dict = usage.model_dump()
                    elif hasattr(usage, 'dict'):
                        # Pydantic v1 style
                        usage_dict = usage.dict()
                    else:
                        # Fallback: manually extract common attributes
                        usage_dict = {
                            "completion_tokens": getattr(usage, 'completion_tokens', None),
                            "prompt_tokens": getattr(usage, 'prompt_tokens', None),
                            "total_tokens": getattr(usage, 'total_tokens', None)
                        }
                except Exception:
                    # If conversion fails, use None
                    usage_dict = None
            
            # For template-based prompts, first strip thinking/reasoning blocks
            if prompt_type in ["short_cot", "long_cot", "optimized","optimizednum","optimizedhot"]:
                # Strip thinking and reasoning blocks before parsing
                clean_content = strip_thinking_reasoning_blocks(content)
            else:
                # Legacy mode uses content as-is
                clean_content = content
            
            # Use multi-format parsing for all modes
            parsed_labels, parsed_count, format_used, error_type = parse_labels_multi_format(
                clean_content, len(nuggets_text_list), case_sensitive=case_sensitive
            )
            
            # Log the API call for debugging
            self.logger.debug(f"Model response: {content}")
            self.logger.debug(f"Parsed labels: {parsed_labels}")
            self.logger.debug(f"Format used: {format_used}, Error type: {error_type}")
            
            # Sequential judgment logic - key improvement
            expected_count = len(nuggets_text_list)
            
            labels_list = parsed_labels or []
            if parsed_count == 0:
                # Complete parsing failure
                result_labels = ["error"] * expected_count
                batch_status = "complete_failure"
            elif parsed_count < expected_count:
                # Partial success - preserve parsed results, mark rest as error
                result_labels = ["error"] * (expected_count - parsed_count)+labels_list[:parsed_count] 
                batch_status = "partial_success"
            else:
                # Complete success (may have extra labels, truncate to expected count)
                result_labels = labels_list[-expected_count:]
                batch_status = "success"
            
            # For non-thinking mode, convert errors to None
            if not enable_thinking:
                result_labels = [None if label == "error" else label for label in result_labels]
            
            # Prepare detailed batch statistics
            batch_stats = {
                "batch_status": batch_status,
                "parsed_count": parsed_count,
                "expected_count": expected_count,
                "format_used": format_used,
                "error_type": error_type,
                "finish_reason": finish_reason,
                "recovery_rate": parsed_count / expected_count if expected_count > 0 else 0,
                "prompt_type": prompt_type,
                "case_sensitive": case_sensitive,
                "run_id": self.run_id,
                "qid": (context or {}).get("qid"),
                "batch_index": (context or {}).get("batch_index"),
            }
            
            # Update call log with final results including true labels
            call_log.update({
                "response": {
                    "content": content,
                    "parsed_labels": parsed_labels,
                    "result_labels": result_labels,
                    "true_labels": true_labels,
                    "usage": usage_dict,
                    "batch_stats": batch_stats
                },
                "success": True,
                "error": None
            })
            
            # Save to LLM log file
            self._save_llm_log(call_log)
            
            # Ensure return type matches List[Optional[str]] for type checkers
            result_labels_typed = cast(List[Optional[str]], result_labels)
            return result_labels_typed, batch_stats
                
        except ModelAPIError as e:
            # Log the specific API error
            call_log.update({
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "is_temporary": getattr(e, 'is_temporary', False),
                    "retry_after": getattr(e, 'retry_after', None)
                },
                "response": None
            })
            self._save_llm_log(call_log)
            
            self.logger.error(f"API call failed: {e}")
            
            # For temporary errors, we might want to retry in the future
            if getattr(e, 'is_temporary', False):
                self.logger.info(f"Temporary error detected, may retry after {getattr(e, 'retry_after', 0)}s")
            
            # Return consistent error response with batch stats
            err_labels_api: List[Optional[str]] = []
            if enable_thinking:
                err_labels_api = ["error" for _ in range(len(nuggets_text_list))]
            else:
                err_labels_api = [None for _ in range(len(nuggets_text_list))]
            error_stats = {
                "batch_status": "api_error",
                "parsed_count": 0,
                "expected_count": len(nuggets_text_list),
                "format_used": "none",
                "error_type": "api_error",
                "finish_reason": None,
                "recovery_rate": 0,
                "case_sensitive": case_sensitive,
                "api_error": type(e).__name__
            }
            
            return err_labels_api, error_stats
        
        except Exception as e:
            # Log unexpected errors
            call_log.update({
                "success": False,
                "error": {
                    "type": "UnexpectedError",
                    "message": str(e),
                    "is_temporary": False
                },
                "response": None
            })
            self._save_llm_log(call_log)
            
            self.logger.error(f"Unexpected error in API call: {e}")
            
            # Return consistent error response with batch stats
            err_labels_unexpected: List[Optional[str]] = []
            if enable_thinking:
                err_labels_unexpected = ["error" for _ in range(len(nuggets_text_list))]
            else:
                err_labels_unexpected = [None for _ in range(len(nuggets_text_list))]
            error_stats = {
                "batch_status": "unexpected_error",
                "parsed_count": 0,
                "expected_count": len(nuggets_text_list),
                "format_used": "none",
                "error_type": "unexpected_error",
                "finish_reason": None,
                "recovery_rate": 0,
                "case_sensitive": case_sensitive,
                "exception": str(e)
            }
            
            return err_labels_unexpected, error_stats
    
    def _make_api_call(self, messages: List[Dict], temperature: float, 
                      top_p: float, max_tokens: int, enable_thinking: bool):
        """Make API call with proper error handling and retries"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body={
                        "top_k": -1, 
                        "separate_reasoning": enable_thinking,
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Classify the error
                if "timeout" in error_str or "connection" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"Network timeout (attempt {attempt + 1}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise NetworkTimeoutError(f"Network timeout after {max_retries} attempts: {e}")
                
                elif "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        # Extract retry-after if possible, otherwise use exponential backoff
                        delay = 60 * (2 ** attempt)  # Start with 60s for rate limits
                        self.logger.warning(f"Rate limit hit (attempt {attempt + 1}), retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError(f"Rate limit exceeded after {max_retries} attempts: {e}")
                
                elif "unauthorized" in error_str or "authentication" in error_str:
                    raise AuthenticationError(f"Authentication failed: {e}")
                
                elif "model not found" in error_str or "model does not exist" in error_str:
                    raise ModelNotFoundError(f"Model not found: {self.model_name}")
                
                else:
                    # For other errors, retry once then fail
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"API error (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise ModelAPIError(f"API call failed after {max_retries} attempts: {e}")
    
    def _save_llm_log(self, call_log: Dict[str, Any]):
        """Save LLM call log to buffer and flush when needed"""
        if self._closed:
            return
            
        try:
            # Add to buffer
            self._log_buffer.append(call_log)
            
            # Flush if buffer is full
            if len(self._log_buffer) >= self._buffer_size:
                self._flush_log_buffer()
                
        except Exception as e:
            self.logger.error(f"Failed to buffer LLM log: {e}")
    
    def _flush_log_buffer(self):
        """Flush the log buffer to file"""
        if not self._log_buffer:
            return
            
        try:
            with open(self.llm_log_file, 'a', encoding='utf-8') as f:
                for log_entry in self._log_buffer:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            self._log_buffer.clear()
        except Exception as e:
            self.logger.error(f"Failed to flush LLM log buffer: {e}")


def build_multi_format_prompt(query: str, block: str, nuggets: List[str], format_type: str = "markdown") -> List[Dict[str, str]]:
    """Build prompt messages for nugget matching with different output formats
    
    Args:
        query: The search query
        block: The text passage
        nuggets: List of nugget texts to judge
        format_type: Output format type (json|csv|markdown|yaml|xml|tsv|numbered|adaptive)
    
    Returns:
        List of message dicts for the API call
    """
    system_msg = (
        "You are NuggetMatchJudge, an intelligent assistant.\n"
        "Your task is to read a query, a passage, and nuggets, "
        "and then decide whether the passage supports the nugget in the context of the query.\n"
        "You need to label the nugget as one of the following: support, partial_support, or not_support.\n"
    )
    
    # Format-specific instructions
    format_instructions = {
        "json": (
            'Respond with a JSON array containing exactly one label for each nugget:\n'
            '["support", "partial_support", "not_support"]\n'
            'Example: ["support", "not_support", "partial_support"]'
        ),
        "csv": (
            'Respond with comma-separated values, one label for each nugget:\n'
            'support,partial_support,not_support\n'
            'Example: support,not_support,partial_support'
        ),
        "python_list": (
            'Respond with a Python list containing exactly one label for each nugget:\n'
            "['support', 'partial_support', 'not_support']\n"
            "Example: ['support', 'not_support', 'partial_support']"
        ),
        "yaml": (
            'Respond with a YAML list, one label for each nugget:\n'
            '- support\n'
            '- partial_support\n'
            '- not_support\n'
            'Example:\n- support\n- not_support\n- partial_support'
        ),
        "markdown": (
            'Respond with a Markdown unordered list, one label for each nugget:\n'
            '* support\n'
            '* partial_support\n'
            '* not_support\n'
            'Example:\n* support\n* not_support\n* partial_support'
        ),
        "xml": (
            'Respond with XML format, one label for each nugget:\n'
            '<labels>\n'
            '  <label>support</label>\n'
            '  <label>partial_support</label>\n'
            '  <label>not_support</label>\n'
            '</labels>\n'
            'Example:\n<labels>\n  <label>support</label>\n  <label>not_support</label>\n</labels>'
        ),
        "tsv": (
            'Respond with tab-separated values, one label for each nugget:\n'
            'support\tpartial_support\tnot_support\n'
            'Example: support\tnot_support\tpartial_support'
        ),
        "numbered": (
            'Respond with a numbered list, one label for each nugget:\n'
            '1. support\n'
            '2. partial_support\n'
            '3. not_support\n'
            'Example:\n1. support\n2. not_support\n3. partial_support'
        ),
        "comma_separated": (
            'Respond with comma-separated values with spaces, one label for each nugget:\n'
            'support, partial_support, not_support\n'
            'Example: support, not_support, partial_support'
        ),
        "pipe_separated": (
            'Respond with pipe-separated values, one label for each nugget:\n'
            'support|partial_support|not_support\n'
            'Example: support|not_support|partial_support'
        ),
        "adaptive": (
            'Respond using ANY of these formats (choose the one that works best for you):\n'
            '1. JSON array: ["support", "partial_support", "not_support"]\n'
            '2. Markdown list: * support\n* partial_support\n* not_support\n'
            '3. Comma-separated: support, partial_support, not_support\n'
            '4. Numbered list: 1. support\n2. partial_support\n3. not_support\n'
            '\nProvide exactly one label for each nugget in the same order.'
        )
    }
    
    # Get format instruction or default to adaptive
    format_instruction = format_instructions.get(format_type, format_instructions["adaptive"])
    
    nugget_section = "\n".join([f"- nugget: \"{nug}\"\n" for nug in nuggets])
    user_msg = (
        f"{format_instruction}\n\n"
        "The list should be in the same order as the input nuggets. "
        "Make sure to provide exactly one label for each nugget.\n\n" 
        f"Query: {query}\n"
        f"Passage: {block}\n"
        f"Nuggets:\n{nugget_section}\n\n"  
    )
    
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_multi_nugget_prompt(query: str, block: str, nuggets: List[str]) -> List[Dict[str, str]]:
    """Legacy function maintained for backward compatibility"""
    return build_multi_format_prompt(query, block, nuggets, "markdown")


# Multi-format parsing functions with case sensitivity support
def parse_json_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse JSON array format: ["support", "partial_support", "not_support"]"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Try to find JSON array pattern
    json_pattern = r'\[([^\]]+)\]'
    match = re.search(json_pattern, text)
    if not match:
        return None
    
    try:
        # Extract and parse the JSON array
        json_str = '[' + match.group(1) + ']'
        labels_raw = json.loads(json_str)
        
        # Filter and normalize valid labels
        labels = []
        for label in labels_raw:
            if isinstance(label, str):
                cleaned_label = label.strip('"\' ')
                
                if case_sensitive:
                    if cleaned_label in valid_labels_set:
                        labels.append(cleaned_label)
                else:
                    normalized = normalize_label(cleaned_label)
                    if normalized in valid_labels_mapping:
                        labels.append(valid_labels_mapping[normalized])
        
        return labels if labels else None
    except (json.JSONDecodeError, TypeError):
        return None


def parse_csv_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse comma-separated format (CSV): support,partial_support,not_support or support, partial_support, not_support"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Look for comma-separated values - ensure we have at least one comma
    if ',' not in text:
        return None
    
    if case_sensitive:
        csv_pattern = r'(?:support|partial_support|not_support)(?:\s*,\s*(?:support|partial_support|not_support))+'
        match = re.search(csv_pattern, text)
    else:
        # Build case insensitive pattern from all valid mappings
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        csv_pattern = rf'(?:{label_patterns})(?:\s*,\s*(?:{label_patterns}))+'
        match = re.search(csv_pattern, text, re.IGNORECASE)
    
    if not match:
        return None
    
    labels = []
    for label in match.group().split(','):
        cleaned_label = label.strip()
        
        if case_sensitive:
            if cleaned_label in valid_labels_set:
                labels.append(cleaned_label)
        else:
            normalized = normalize_label(cleaned_label)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_python_list_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse Python list format: ['support', 'partial_support', 'not_support']"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Look for Python list pattern with single or double quotes
    py_pattern = r"\[([^\]]+)\]"
    match = re.search(py_pattern, text)
    if not match:
        return None
    
    try:
        # Try to evaluate as Python literal
        import ast
        list_str = '[' + match.group(1) + ']'
        labels_raw = ast.literal_eval(list_str)
        
        if not isinstance(labels_raw, list):
            return None
        
        labels = []
        for label in labels_raw:
            cleaned_label = str(label).strip()
            
            if case_sensitive:
                if cleaned_label in valid_labels_set:
                    labels.append(cleaned_label)
            else:
                normalized = normalize_label(cleaned_label)
                if normalized in valid_labels_mapping:
                    labels.append(valid_labels_mapping[normalized])
        
        return labels if labels else None
    except (ValueError, SyntaxError, TypeError):
        return None


def parse_yaml_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse YAML list format: - support\n- partial_support\n- not_support (strict)"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # YAML should only use '-' and not have markdown indicators like '*' or '+'
    if '- ' not in text or '*' in text or '+' in text:
        return None
    
    if case_sensitive:
        yaml_pattern = r'(?:^|\n)\s*-\s+(support|partial_support|not_support)'
        matches = re.findall(yaml_pattern, text, re.MULTILINE)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        yaml_pattern = rf'(?:^|\n)\s*-\s+({label_patterns})'
        matches = re.findall(yaml_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    if not matches:
        return None
    
    labels = []
    for match in matches:
        cleaned_match = match.strip()
        
        if case_sensitive:
            if cleaned_match in valid_labels_set:
                labels.append(cleaned_match)
        else:
            normalized = normalize_label(cleaned_match)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_markdown_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse markdown list format: * support or + support (but not - support which is YAML)"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Markdown should prefer * and + over - (leave - for YAML)
    if not any(indicator in text for indicator in ['* ', '+ ']):
        return None
    
    if case_sensitive:
        md_pattern = r'(?:^|\n)\s*[*+]\s*(support|partial_support|not_support)'
        matches = re.findall(md_pattern, text, re.MULTILINE)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        md_pattern = rf'(?:^|\n)\s*[*+]\s*({label_patterns})'
        matches = re.findall(md_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    if not matches:
        return None
    
    labels = []
    for match in matches:
        cleaned_match = match.strip()
        
        if case_sensitive:
            if cleaned_match in valid_labels_set:
                labels.append(cleaned_match)
        else:
            normalized = normalize_label(cleaned_match)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_xml_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse XML format: <labels><label>support</label><label>partial_support</label></labels>"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    if case_sensitive:
        xml_pattern = r'<label>(support|partial_support|not_support)</label>'
        matches = re.findall(xml_pattern, text)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        xml_pattern = f'<label>({label_patterns})</label>'
        matches = re.findall(xml_pattern, text, re.IGNORECASE)
    
    if not matches:
        return None
    
    labels = []
    for match in matches:
        cleaned_match = match.strip()
        
        if case_sensitive:
            if cleaned_match in valid_labels_set:
                labels.append(cleaned_match)
        else:
            normalized = normalize_label(cleaned_match)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_tsv_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse TSV format: support\tpartial_support\tnot_support"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Look for tab-separated values - ensure we have at least one tab
    if '\t' not in text:
        return None
    
    if case_sensitive:
        tsv_pattern = r'(?:support|partial_support|not_support)(?:\t(?:support|partial_support|not_support))+'
        match = re.search(tsv_pattern, text)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        tsv_pattern = f'(?:{label_patterns})(?:\t(?:{label_patterns}))+'
        match = re.search(tsv_pattern, text, re.IGNORECASE)
    
    if not match:
        return None
    
    labels = []
    for label in match.group().split('\t'):
        cleaned_label = label.strip()
        
        if case_sensitive:
            if cleaned_label in valid_labels_set:
                labels.append(cleaned_label)
        else:
            normalized = normalize_label(cleaned_label)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_numbered_list_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse numbered list format: 1. support\n2. partial_support\n3. not_support"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    if case_sensitive:
        numbered_pattern = r'(?:^|\n)\s*\d+\.\s*(support|partial_support|not_support)'
        matches = re.findall(numbered_pattern, text, re.MULTILINE)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        numbered_pattern = rf'(?:^|\n)\s*\d+\.\s*({label_patterns})'
        matches = re.findall(numbered_pattern, text, re.MULTILINE | re.IGNORECASE)
    
    if not matches:
        return None
    
    labels = []
    for match in matches:
        cleaned_match = match.strip()
        
        if case_sensitive:
            if cleaned_match in valid_labels_set:
                labels.append(cleaned_match)
        else:
            normalized = normalize_label(cleaned_match)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def parse_pipe_separated_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Parse pipe-separated format: support|partial_support|not_support"""
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Look for pipe-separated values - ensure we have at least one pipe
    if '|' not in text:
        return None
    
    if case_sensitive:
        pipe_pattern = r'(?:support|partial_support|not_support)(?:\|(?:support|partial_support|not_support))+'
        match = re.search(pipe_pattern, text)
    else:
        label_patterns = '|'.join(re.escape(label) for label in valid_labels_mapping.keys())
        pipe_pattern = rf'(?:{label_patterns})(?:\|(?:{label_patterns}))+'
        match = re.search(pipe_pattern, text, re.IGNORECASE)
    
    if not match:
        return None
    
    labels = []
    for label in match.group().split('|'):
        cleaned_label = label.strip()
        
        if case_sensitive:
            if cleaned_label in valid_labels_set:
                labels.append(cleaned_label)
        else:
            normalized = normalize_label(cleaned_label)
            if normalized in valid_labels_mapping:
                labels.append(valid_labels_mapping[normalized])
    
    return labels if labels else None


def strip_thinking_reasoning_blocks(text: str) -> str:
    """
    Remove <think> and <reasoning> blocks from text
    
    Args:
        text: Raw model response text
        
    Returns:
        Text with thinking and reasoning blocks removed
    """
    # Remove <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove <reasoning>...</reasoning> blocks  
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    
    return text.strip()


def parse_labels_multi_format(text: str, expected_count: int, case_sensitive: bool = False) -> Tuple[Optional[List[str]], int, str, Optional[str]]:
    """Parse labels using multiple format parsers with fixed priority order and case sensitivity
    
    Args:
        text: The text to parse
        expected_count: Expected number of labels
        case_sensitive: Whether to use case sensitive matching (default: False)
    
    Returns:
        Tuple of (labels, parsed_count, format_used, error_type)
        error_type: None | "truncation" | "format_error" | "count_mismatch"
    """
    if not text or not text.strip():
        return None, 0, "empty", "format_error"
    
    # Fixed parser priority order - most specific first
    parsers = [
        ("json", lambda t: parse_json_labels(t, case_sensitive)),
        ("python_list", lambda t: parse_python_list_labels(t, case_sensitive)),
        ("xml", lambda t: parse_xml_labels(t, case_sensitive)),
        ("numbered_list", lambda t: parse_numbered_list_labels(t, case_sensitive)),
        ("markdown", lambda t: parse_markdown_labels(t, case_sensitive)),
        ("yaml", lambda t: parse_yaml_labels(t, case_sensitive)),
        ("pipe_separated", lambda t: parse_pipe_separated_labels(t, case_sensitive)),
        ("tsv", lambda t: parse_tsv_labels(t, case_sensitive)),
        ("csv", lambda t: parse_csv_labels(t, case_sensitive)),
    ]
    
    # Try each parser
    for format_name, parser_func in parsers:
        try:
            labels = parser_func(text)
            if labels and isinstance(labels, list):
                parsed_count = len(labels)
                
                # Determine error type if any
                error_type = None
                if parsed_count < expected_count:
                    error_type = "truncation"
                elif parsed_count > expected_count:
                    error_type = "count_mismatch"
                
                return labels, parsed_count, format_name, error_type
        except Exception as e:
            # Log parsing error but continue to next parser
            logging.debug(f"Parser {format_name} failed: {e}")
            continue
    
    # If all parsers failed, try fallback extraction
    fallback_labels = fallback_keyword_extraction(text, case_sensitive)
    if fallback_labels:
        parsed_count = len(fallback_labels)
        error_type = "truncation" if parsed_count < expected_count else None
        return fallback_labels, parsed_count, "fallback", error_type
    
    return None, 0, "failed", "format_error"


def fallback_keyword_extraction(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Fallback: extract valid labels anywhere in text with case sensitivity support"""
    if not text or not text.strip():
        return None
    
    valid_labels_mapping = get_valid_labels_mapping()
    valid_labels_set = {"support", "partial_support", "not_support"}
    
    # Find all valid labels in order of appearance
    found_labels = []
    for line in text.splitlines():
        for word in line.split():
            # Clean word (remove punctuation)
            clean_word = re.sub(r'[^a-zA-Z_]', '', word)
            
            if case_sensitive:
                if clean_word in valid_labels_set:
                    found_labels.append(clean_word)
            else:
                normalized = normalize_label(clean_word)
                if normalized in valid_labels_mapping:
                    found_labels.append(valid_labels_mapping[normalized])
    
    return found_labels if found_labels else None


def safe_parse_labels(text: str, case_sensitive: bool = False) -> Optional[List[str]]:
    """Legacy function maintained for backward compatibility with case sensitivity support"""
    labels, _, _, _ = parse_labels_multi_format(text, 999, case_sensitive=case_sensitive)  # Use large expected count
    return labels


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl_data(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_single_item(client: ModelClient, item: Dict[str, Any], batch_size: int,
                       model_config: Dict[str, Any], run_context_base: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """
    Process a single evaluation item (query + block + nuggets) with enhanced error tracking
    
    Returns:
        Tuple of (results, num_batches, batch_stats_list)
    """
    qid = item["qid"]
    query = item["query"]
    block_text = item["block"][0]
    nuggets_list = item["block_nuggets_assignment"]
    
    all_preds = []
    all_batch_stats = []
    num_batches = ceil(len(nuggets_list) / batch_size)
    
    for i in range(num_batches):
        batch = nuggets_list[i * batch_size: (i + 1) * batch_size]
        batch_texts = [n["text"] for n in batch]
        
        # Build trace context for this batch
        context = dict(run_context_base or {})
        context.update({"qid": qid, "batch_index": i, "nuggets_in_batch": len(batch_texts)})
        
        pred_labels, batch_stats = client.predict_batch(
            query, block_text, batch_texts,
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.95),
            max_tokens=model_config.get("max_tokens", 1024),
            enable_thinking=model_config.get("enable_thinking", False),
            prompt_type=model_config.get("prompt_type", "legacy"),
            format_type=model_config.get("format_type", "adaptive"),
            context=context
        )
        
        all_preds.extend(pred_labels)
        all_batch_stats.append(batch_stats)
    
    # Create results
    results = []
    for nug, pred in zip(nuggets_list, all_preds):
        results.append({
            "qid": qid,
            "query": query,
            "block_text": block_text,
            "nugget_text": nug["text"],
            "block_pred": pred,
            "block_true": nug["assignment"]
        })
    
    return results, num_batches, all_batch_stats

def aggregate_assignment(predicted_results: List[Dict[str, Any]], 
                           gold_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Aggregate block-level predictions to overall match predictions
    
    Returns:
        Tuple of (overall_y_true, overall_y_pred)
    """
    overall_y_true, overall_y_pred = [], []
    
    # Group predictions by qid
    predicted_data_by_qid = defaultdict(list)
    for res in predicted_results:
        predicted_data_by_qid[res["qid"]].append(res)
    
    for qid, pred_entries in predicted_data_by_qid.items():
        if qid not in gold_data:
            continue
        
        # Group predictions by nugget text and find max priority
        nugget_preds_for_qid = defaultdict(list)
        for entry in pred_entries:
            if entry["block_pred"] is not None:
                nugget_preds_for_qid[entry["nugget_text"]].append(entry["block_pred"])
        
        predicted_assignment = {}
        for text, preds in nugget_preds_for_qid.items():
            if preds:
                max_score = max(MATCH_PRIORITY.get(p, -1) for p in preds)
                predicted_assignment[text] = PRIORITY_TO_LABEL.get(max_score, "not_support")
        
        # Get gold standard overall match
        gold_item = gold_data[qid]
        gold_assignment_dict = {
            nug["text"]: nug["assignment"]
            for nug in gold_item.get("global_nuggets_assignment", [])
        }
        
        # Collect predictions for evaluation
        for text, true_label in gold_assignment_dict.items():
            pred_label = predicted_assignment.get(text)
            if pred_label is not None:
                overall_y_true.append(true_label)
                overall_y_pred.append(pred_label)
    
    return overall_y_true, overall_y_pred


def create_gold_data_index(gold_data_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create an index of gold data by qid"""
    return {item["qid"]: item for item in gold_data_list}