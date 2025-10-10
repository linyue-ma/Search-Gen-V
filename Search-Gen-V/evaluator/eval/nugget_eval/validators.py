"""Validation utilities for nugget evaluation framework"""

import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime

from .constants import RunIdConfig, ValidationConfig, LogConfig


class RunIdValidator:
    """Validator for run_id format and consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Build regex pattern for run_id validation
        # Format: YYYYMMDD_HHMMSS_<8-char-hex>
        timestamp_pattern = r"\d{8}_\d{6}"
        uuid_pattern = r"[a-f0-9]{" + str(RunIdConfig.UUID_LENGTH) + "}"
        self.run_id_pattern = re.compile(f"^{timestamp_pattern}_{uuid_pattern}$")
    
    def validate_run_id_format(self, run_id: str) -> bool:
        """
        Validate run_id format
        
        Args:
            run_id: The run_id to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        if not run_id or not isinstance(run_id, str):
            self.logger.warning(f"Invalid run_id type: {type(run_id)}")
            return False
        
        if not (RunIdConfig.MIN_RUN_ID_LENGTH <= len(run_id) <= RunIdConfig.MAX_RUN_ID_LENGTH):
            self.logger.warning(f"run_id length {len(run_id)} outside expected range")
            return False
        
        if not self.run_id_pattern.match(run_id):
            self.logger.warning(f"run_id '{run_id}' does not match expected pattern")
            return False
        
        return True
    
    def extract_timestamp_from_run_id(self, run_id: str) -> Optional[datetime]:
        """
        Extract timestamp from run_id
        
        Args:
            run_id: The run_id to parse
            
        Returns:
            datetime object if successful, None otherwise
        """
        if not self.validate_run_id_format(run_id):
            return None
        
        try:
            # Extract timestamp part (first 15 characters: YYYYMMDD_HHMMSS)
            timestamp_str = run_id[:15]
            return datetime.strptime(timestamp_str, RunIdConfig.TIMESTAMP_FORMAT)
        except ValueError as e:
            self.logger.warning(f"Failed to parse timestamp from run_id '{run_id}': {e}")
            return None
    
    def validate_run_id_consistency(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Check consistency across multiple run_ids
        
        Args:
            run_ids: List of run_ids to check
            
        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            "total_count": len(run_ids),
            "valid_count": 0,
            "invalid_count": 0,
            "invalid_run_ids": [],
            "duplicate_count": 0,
            "duplicates": [],
            "time_range": None,
            "warnings": []
        }
        
        if not run_ids:
            results["warnings"].append("No run_ids provided for validation")
            return results
        
        # Check format and collect valid timestamps
        valid_timestamps = []
        seen_run_ids = set()
        
        for run_id in run_ids:
            # Check for duplicates
            if run_id in seen_run_ids:
                results["duplicate_count"] += 1
                results["duplicates"].append(run_id)
                continue
            seen_run_ids.add(run_id)
            
            # Validate format
            if self.validate_run_id_format(run_id):
                results["valid_count"] += 1
                timestamp = self.extract_timestamp_from_run_id(run_id)
                if timestamp:
                    valid_timestamps.append(timestamp)
            else:
                results["invalid_count"] += 1
                results["invalid_run_ids"].append(run_id)
        
        # Calculate time range if we have valid timestamps
        if valid_timestamps:
            min_time = min(valid_timestamps)
            max_time = max(valid_timestamps)
            results["time_range"] = {
                "earliest": min_time.isoformat(),
                "latest": max_time.isoformat(),
                "duration_seconds": (max_time - min_time).total_seconds()
            }
        
        # Add warnings for concerning patterns
        if results["duplicate_count"] > 0:
            results["warnings"].append(f"Found {results['duplicate_count']} duplicate run_ids")
        
        if results["invalid_count"] > results["valid_count"] * 0.1:  # More than 10% invalid
            results["warnings"].append(f"High invalid rate: {results['invalid_count']}/{results['total_count']}")
        
        return results


class LogCorrelationValidator:
    """Validator for log correlation and consistency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_log_correlation(self, eval_log_dir: Path, llm_log_dir: Path, 
                                 run_id: str) -> Dict[str, Any]:
        """
        Validate that eval and llm logs are properly correlated for a run_id
        
        Args:
            eval_log_dir: Directory containing evaluation logs
            llm_log_dir: Directory containing LLM logs
            run_id: The run_id to validate
            
        Returns:
            Dictionary with correlation validation results
        """
        results = {
            "run_id": run_id,
            "eval_files": [],
            "llm_files": [],
            "correlation_valid": False,
            "issues": [],
            "statistics": {}
        }
        
        # Find files for this run_id
        eval_run_dir = eval_log_dir / run_id
        llm_run_dir = llm_log_dir / run_id
        
        # Check if run directories exist
        if not eval_run_dir.exists():
            results["issues"].append(f"Evaluation log directory not found: {eval_run_dir}")
        else:
            results["eval_files"] = [f.name for f in eval_run_dir.glob("*")]
        
        if not llm_run_dir.exists():
            results["issues"].append(f"LLM log directory not found: {llm_run_dir}")
        else:
            results["llm_files"] = [f.name for f in llm_run_dir.glob("*")]
        
        # Validate run metadata consistency
        meta_file = eval_run_dir / LogConfig.META_FILE_NAME
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                stored_run_id = meta_data.get("run_id")
                if stored_run_id != run_id:
                    results["issues"].append(
                        f"run_id mismatch: directory={run_id}, metadata={stored_run_id}"
                    )
            except Exception as e:
                results["issues"].append(f"Failed to read metadata file: {e}")
        else:
            results["issues"].append("Run metadata file not found")
        
        # Analyze LLM log entries for run_id consistency
        llm_stats = self._analyze_llm_log_consistency(llm_run_dir, run_id)
        results["statistics"]["llm_log"] = llm_stats
        
        # Determine overall correlation validity
        results["correlation_valid"] = (
            len(results["issues"]) == 0 and 
            len(results["eval_files"]) > 0 and 
            len(results["llm_files"]) > 0
        )
        
        return results
    
    def _analyze_llm_log_consistency(self, llm_log_dir: Path, expected_run_id: str) -> Dict[str, Any]:
        """Analyze LLM log files for run_id consistency"""
        stats = {
            "total_entries": 0,
            "entries_with_run_id": 0,
            "matching_run_id": 0,
            "missing_run_id": 0,
            "mismatched_run_id": 0,
            "unique_run_ids": set(),
            "files_analyzed": 0
        }
        
        # Find all JSONL files
        jsonl_files = list(llm_log_dir.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            stats["files_analyzed"] += 1
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            entry = json.loads(line)
                            stats["total_entries"] += 1
                            
                            entry_run_id = entry.get("run_id")
                            if entry_run_id:
                                stats["entries_with_run_id"] += 1
                                stats["unique_run_ids"].add(entry_run_id)
                                
                                if entry_run_id == expected_run_id:
                                    stats["matching_run_id"] += 1
                                else:
                                    stats["mismatched_run_id"] += 1
                            else:
                                stats["missing_run_id"] += 1
                        
                        except json.JSONDecodeError:
                            # Skip malformed entries
                            continue
                            
            except Exception as e:
                self.logger.warning(f"Failed to analyze log file {jsonl_file}: {e}")
        
        # Convert set to list for JSON serialization
        stats["unique_run_ids"] = list(stats["unique_run_ids"])
        
        return stats
    
    def validate_batch_logs_correlation(self, log_dirs: List[Tuple[Path, Path]], 
                                        run_ids: List[str]) -> Dict[str, Any]:
        """
        Validate log correlation for multiple runs
        
        Args:
            log_dirs: List of (eval_log_dir, llm_log_dir) pairs
            run_ids: List of run_ids to validate
            
        Returns:
            Dictionary with batch validation results
        """
        batch_results = {
            "total_runs": len(run_ids),
            "valid_correlations": 0,
            "invalid_correlations": 0,
            "run_results": {},
            "common_issues": [],
            "summary": {}
        }
        
        issue_counts = {}
        
        for run_id in run_ids:
            run_valid = True
            run_issues = []
            
            for eval_log_dir, llm_log_dir in log_dirs:
                result = self.validate_log_correlation(eval_log_dir, llm_log_dir, run_id)
                
                if not result["correlation_valid"]:
                    run_valid = False
                
                run_issues.extend(result["issues"])
                
                # Count issue types
                for issue in result["issues"]:
                    issue_type = issue.split(":")[0] if ":" in issue else issue
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            batch_results["run_results"][run_id] = {
                "valid": run_valid,
                "issues": run_issues
            }
            
            if run_valid:
                batch_results["valid_correlations"] += 1
            else:
                batch_results["invalid_correlations"] += 1
        
        # Identify common issues (affecting >10% of runs)
        threshold = max(1, len(run_ids) * 0.1)
        batch_results["common_issues"] = [
            f"{issue_type}: {count} runs affected"
            for issue_type, count in issue_counts.items() 
            if count >= threshold
        ]
        
        # Generate summary
        success_rate = batch_results["valid_correlations"] / batch_results["total_runs"]
        batch_results["summary"] = {
            "success_rate": success_rate,
            "recommendation": "OK" if success_rate > 0.9 else "NEEDS_ATTENTION"
        }
        
        return batch_results


class IntegratedValidator:
    """Combined validator for comprehensive validation"""
    
    def __init__(self):
        self.run_id_validator = RunIdValidator()
        self.log_validator = LogCorrelationValidator()
        self.logger = logging.getLogger(__name__)
    
    def comprehensive_validation(self, eval_log_dir: Path, llm_log_dir: Path, 
                                 run_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of run system
        
        Args:
            eval_log_dir: Directory containing evaluation logs
            llm_log_dir: Directory containing LLM logs  
            run_ids: Optional list of specific run_ids to validate
            
        Returns:
            Comprehensive validation report
        """
        if run_ids is None:
            # Auto-discover run_ids from directory structure
            run_ids = []
            for subdir in eval_log_dir.iterdir():
                if subdir.is_dir():
                    run_ids.append(subdir.name)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "eval_log_dir": str(eval_log_dir),
            "llm_log_dir": str(llm_log_dir),
            "run_id_validation": None,
            "log_correlation_validation": None,
            "overall_status": "UNKNOWN",
            "recommendations": []
        }
        
        try:
            # Validate run_id formats
            run_id_results = self.run_id_validator.validate_run_id_consistency(run_ids)
            report["run_id_validation"] = run_id_results
            
            # Validate log correlations
            correlation_results = self.log_validator.validate_batch_logs_correlation(
                [(eval_log_dir, llm_log_dir)], run_ids
            )
            report["log_correlation_validation"] = correlation_results
            
            # Determine overall status
            run_id_ok = run_id_results["invalid_count"] == 0
            correlation_ok = correlation_results["summary"]["recommendation"] == "OK"
            
            if run_id_ok and correlation_ok:
                report["overall_status"] = "HEALTHY"
            elif run_id_ok or correlation_ok:
                report["overall_status"] = "PARTIAL"
                report["recommendations"].append("Some validation checks failed - review detailed results")
            else:
                report["overall_status"] = "UNHEALTHY"
                report["recommendations"].append("Multiple validation failures detected - system needs attention")
            
            # Add specific recommendations
            if run_id_results["invalid_count"] > 0:
                report["recommendations"].append(
                    f"Fix {run_id_results['invalid_count']} invalid run_ids"
                )
            
            if correlation_results["invalid_correlations"] > 0:
                report["recommendations"].append(
                    f"Investigate {correlation_results['invalid_correlations']} correlation issues"
                )
            
        except Exception as e:
            report["overall_status"] = "ERROR"
            report["error"] = str(e)
            self.logger.error(f"Validation failed with error: {e}")
        
        return report


# Convenience functions for easy validation
def validate_run_id(run_id: str) -> bool:
    """Quick run_id format validation"""
    validator = RunIdValidator()
    return validator.validate_run_id_format(run_id)


def validate_system_health(eval_log_dir: str, llm_log_dir: str) -> Dict[str, Any]:
    """Quick system health check"""
    validator = IntegratedValidator()
    return validator.comprehensive_validation(Path(eval_log_dir), Path(llm_log_dir))


# Export main classes and functions
__all__ = [
    "RunIdValidator",
    "LogCorrelationValidator", 
    "IntegratedValidator",
    "validate_run_id",
    "validate_system_health"
]