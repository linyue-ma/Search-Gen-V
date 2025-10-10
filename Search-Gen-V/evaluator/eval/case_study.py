#!/usr/bin/env python3
"""
Case Study Analysis Tool for Nugget Evaluation Framework

This script provides comprehensive analysis of evaluation runs by correlating 
eval_log and llm_log files through run_id, enabling deep insights into model 
performance, error patterns, and system behavior.

Usage:
    python case_study.py --run-id <run_id> --log-dir <path>
    python case_study.py --analyze-all --log-dir <path>
    python case_study.py --compare-runs <run_id1> <run_id2> --log-dir <path>
"""

import argparse
import json
import logging
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nugget_eval.constants import (
    RunIdConfig, LogConfig, ValidationConfig, 
    AnalysisConfig, MetricsConfig
)
from nugget_eval.validators import (
    RunIdValidator, LogCorrelationValidator, 
    validate_system_health
)


class LogAnalyzer:
    """Core log analysis engine for correlating eval and LLM logs"""
    
    def __init__(self, log_base_dir: Path):
        self.log_base_dir = Path(log_base_dir)
        self.eval_log_dir = self.log_base_dir / "eval" 
        self.llm_log_dir = self.log_base_dir / "llm"
        self.logger = logging.getLogger(__name__)
        
        # Validate log directories exist
        if not self.eval_log_dir.exists():
            raise FileNotFoundError(f"Evaluation log directory not found: {self.eval_log_dir}")
        if not self.llm_log_dir.exists():
            raise FileNotFoundError(f"LLM log directory not found: {self.llm_log_dir}")
    
    def discover_run_ids(self) -> List[str]:
        """Discover all available run_ids from log directories"""
        run_ids = set()
        
        # Check eval log directory
        for subdir in self.eval_log_dir.iterdir():
            if subdir.is_dir():
                run_ids.add(subdir.name)
        
        # Check llm log directory 
        for subdir in self.llm_log_dir.iterdir():
            if subdir.is_dir():
                run_ids.add(subdir.name)
        
        # Validate discovered run_ids
        validator = RunIdValidator()
        valid_run_ids = [
            run_id for run_id in run_ids 
            if validator.validate_run_id_format(run_id)
        ]
        
        if len(valid_run_ids) != len(run_ids):
            self.logger.warning(f"Found {len(run_ids) - len(valid_run_ids)} invalid run_ids")
        
        return sorted(valid_run_ids)
    
    def load_run_data(self, run_id: str) -> Dict[str, Any]:
        """
        Load all data for a specific run_id
        
        Args:
            run_id: The run identifier to load
            
        Returns:
            Dictionary containing all run data including logs and metadata
        """
        run_data = {
            "run_id": run_id,
            "metadata": {},
            "eval_logs": [],
            "llm_logs": [],
            "predictions": [],
            "statistics": {},
            "load_timestamp": datetime.now().isoformat()
        }
        
        eval_run_dir = self.eval_log_dir / run_id
        llm_run_dir = self.llm_log_dir / run_id
        
        # Load metadata
        meta_file = eval_run_dir / "run_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    run_data["metadata"] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {run_id}: {e}")
        
        # Load evaluation logs (text logs)
        for log_file in eval_run_dir.glob("*.log"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    run_data["eval_logs"].append({
                        "file": log_file.name,
                        "content": f.read(),
                        "size": log_file.stat().st_size
                    })
            except Exception as e:
                self.logger.warning(f"Failed to load eval log {log_file}: {e}")
        
        # Load prediction files
        for pred_file in eval_run_dir.glob("*predictions.jsonl"):
            try:
                predictions = []
                with open(pred_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            predictions.append(json.loads(line))
                run_data["predictions"].append({
                    "file": pred_file.name,
                    "predictions": predictions,
                    "count": len(predictions)
                })
            except Exception as e:
                self.logger.warning(f"Failed to load predictions {pred_file}: {e}")
        
        # Load LLM logs (JSONL)
        for llm_file in llm_run_dir.glob("*.jsonl"):
            try:
                llm_entries = []
                with open(llm_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                llm_entries.append(entry)
                            except json.JSONDecodeError:
                                continue
                run_data["llm_logs"].append({
                    "file": llm_file.name,
                    "entries": llm_entries,
                    "count": len(llm_entries)
                })
            except Exception as e:
                self.logger.warning(f"Failed to load LLM log {llm_file}: {e}")
        
        # Calculate basic statistics
        run_data["statistics"] = self._calculate_run_statistics(run_data)
        
        return run_data
    
    def _calculate_run_statistics(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic statistics for a run"""
        stats = {
            "total_llm_calls": 0,
            "total_predictions": 0,
            "api_success_rate": 0.0,
            "average_latency": 0.0,
            "total_tokens": {"input": 0, "output": 0, "total": 0},
            "error_rate": 0.0,
            "batch_statistics": {},
            "time_range": {}
        }
        
        all_llm_entries = []
        for log_group in run_data["llm_logs"]:
            all_llm_entries.extend(log_group["entries"])
        
        if not all_llm_entries:
            return stats
        
        stats["total_llm_calls"] = len(all_llm_entries)
        
        # Analyze API calls
        successful_calls = 0
        total_latency = []
        total_input_tokens = 0
        total_output_tokens = 0
        error_count = 0
        timestamps = []
        
        for entry in all_llm_entries:
            # Success rate
            if entry.get("success", False):
                successful_calls += 1
            else:
                error_count += 1
            
            # Latency (if available)
            if "latency" in entry:
                total_latency.append(entry["latency"])
            
            # Token usage
            if "response" in entry and "usage" in entry["response"]:
                usage = entry["response"]["usage"]
                if isinstance(usage, dict):
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
            
            # Timestamps
            if "timestamp" in entry:
                timestamps.append(entry["timestamp"])
        
        # Calculate derived statistics
        if stats["total_llm_calls"] > 0:
            stats["api_success_rate"] = successful_calls / stats["total_llm_calls"]
            stats["error_rate"] = error_count / stats["total_llm_calls"]
        
        if total_latency:
            stats["average_latency"] = statistics.mean(total_latency)
            stats["latency_percentiles"] = {
                "p50": statistics.median(total_latency),
                "p90": self._percentile(total_latency, 90),
                "p95": self._percentile(total_latency, 95),
                "p99": self._percentile(total_latency, 99)
            }
        
        stats["total_tokens"] = {
            "input": total_input_tokens,
            "output": total_output_tokens, 
            "total": total_input_tokens + total_output_tokens
        }
        
        # Time range
        if timestamps:
            try:
                parsed_times = [
                    datetime.fromisoformat(ts.replace("Z", "+00:00")) 
                    for ts in timestamps
                ]
                stats["time_range"] = {
                    "start": min(parsed_times).isoformat(),
                    "end": max(parsed_times).isoformat(),
                    "duration_seconds": (max(parsed_times) - min(parsed_times)).total_seconds()
                }
            except Exception:
                pass
        
        # Count predictions
        for pred_group in run_data["predictions"]:
            stats["total_predictions"] += pred_group["count"]
        
        return stats
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def correlate_logs(self, run_id: str) -> Dict[str, Any]:
        """
        Correlate eval and LLM logs for detailed analysis
        
        Args:
            run_id: The run identifier to analyze
            
        Returns:
            Dictionary with detailed correlation analysis
        """
        correlation_data = {
            "run_id": run_id,
            "correlation_success": False,
            "qid_correlations": {},
            "batch_analysis": {},
            "error_analysis": {},
            "performance_analysis": {},
            "recommendations": []
        }
        
        run_data = self.load_run_data(run_id)
        
        if not run_data["llm_logs"]:
            correlation_data["recommendations"].append("No LLM logs found for correlation")
            return correlation_data
        
        # Build QID-based correlation map
        qid_map = defaultdict(list)
        all_llm_entries = []
        
        for log_group in run_data["llm_logs"]:
            for entry in log_group["entries"]:
                all_llm_entries.append(entry)
                context = entry.get("context", {})
                qid = context.get("qid")
                if qid:
                    qid_map[qid].append(entry)
        
        correlation_data["qid_correlations"] = {
            "total_qids": len(qid_map),
            "qid_details": {}
        }
        
        # Analyze each QID
        for qid, entries in qid_map.items():
            qid_analysis = {
                "total_calls": len(entries),
                "batches": defaultdict(list),
                "success_rate": 0.0,
                "errors": [],
                "performance": {}
            }
            
            # Group by batch_index
            for entry in entries:
                batch_idx = entry.get("context", {}).get("batch_index", 0)
                qid_analysis["batches"][batch_idx].append(entry)
            
            # Calculate success rate
            successful = sum(1 for e in entries if e.get("success", False))
            qid_analysis["success_rate"] = successful / len(entries)
            
            # Collect errors
            for entry in entries:
                if not entry.get("success", False):
                    error_info = {
                        "error": entry.get("error"),
                        "batch_index": entry.get("context", {}).get("batch_index"),
                        "timestamp": entry.get("timestamp")
                    }
                    qid_analysis["errors"].append(error_info)
            
            correlation_data["qid_correlations"]["qid_details"][qid] = qid_analysis
        
        # Overall error analysis
        error_types = Counter()
        error_patterns = defaultdict(int)
        
        for entry in all_llm_entries:
            if not entry.get("success", False):
                error_type = entry.get("error", {}).get("type", "unknown")
                error_types[error_type] += 1
                
                # Look for patterns
                context = entry.get("context", {})
                batch_idx = context.get("batch_index", 0)
                if batch_idx is not None:
                    error_patterns[f"batch_{batch_idx}"] += 1
        
        correlation_data["error_analysis"] = {
            "error_types": dict(error_types),
            "error_patterns": dict(error_patterns),
            "total_errors": sum(error_types.values())
        }
        
        # Performance analysis
        latencies = []
        tokens_per_call = []
        
        for entry in all_llm_entries:
            if entry.get("success", False):
                if "latency" in entry:
                    latencies.append(entry["latency"])
                
                usage = entry.get("response", {}).get("usage")
                if isinstance(usage, dict):
                    total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                    if total_tokens > 0:
                        tokens_per_call.append(total_tokens)
        
        if latencies:
            correlation_data["performance_analysis"]["latency"] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        
        if tokens_per_call:
            correlation_data["performance_analysis"]["tokens"] = {
                "mean_per_call": statistics.mean(tokens_per_call),
                "median_per_call": statistics.median(tokens_per_call),
                "total_tokens": sum(tokens_per_call)
            }
        
        correlation_data["correlation_success"] = True
        return correlation_data


class CaseStudyRunner:
    """Main case study analysis runner"""
    
    def __init__(self, log_base_dir: Path, output_dir: Optional[Path] = None):
        self.log_analyzer = LogAnalyzer(log_base_dir)
        self.output_dir = output_dir or Path("case_study_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def analyze_specific_run(self, run_id: str, detailed: bool = True) -> Dict[str, Any]:
        """
        Analyze a specific run in detail
        
        Args:
            run_id: The run to analyze
            detailed: Whether to include detailed correlation analysis
            
        Returns:
            Complete analysis report for the run
        """
        analysis = {
            "analysis_type": "single_run",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "basic_stats": {},
            "correlation_analysis": {},
            "findings": [],
            "recommendations": []
        }
        
        try:
            # Load basic run data
            run_data = self.log_analyzer.load_run_data(run_id)
            analysis["basic_stats"] = run_data["statistics"]
            
            if detailed:
                # Perform detailed correlation analysis
                correlation = self.log_analyzer.correlate_logs(run_id)
                analysis["correlation_analysis"] = correlation
                
                # Generate findings and recommendations
                findings, recommendations = self._generate_insights(run_data, correlation)
                analysis["findings"] = findings
                analysis["recommendations"] = recommendations
            
        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(f"Failed to analyze run {run_id}: {e}")
        
        return analysis
    
    def analyze_all_runs(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze all available runs for patterns and trends
        
        Args:
            limit: Optional limit on number of runs to analyze
            
        Returns:
            Comprehensive analysis across all runs
        """
        run_ids = self.log_analyzer.discover_run_ids()
        if limit:
            run_ids = run_ids[:limit]
        
        analysis = {
            "analysis_type": "multi_run",
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(run_ids),
            "run_summaries": {},
            "aggregate_statistics": {},
            "trends": {},
            "common_issues": [],
            "system_health": {}
        }
        
        # Analyze each run
        all_stats = []
        all_errors = []
        
        for run_id in run_ids:
            try:
                run_analysis = self.analyze_specific_run(run_id, detailed=False)
                analysis["run_summaries"][run_id] = {
                    "basic_stats": run_analysis.get("basic_stats", {}),
                    "analysis_success": "error" not in run_analysis
                }
                
                if "basic_stats" in run_analysis:
                    all_stats.append(run_analysis["basic_stats"])
                
                # Collect errors for pattern analysis
                if "correlation_analysis" in run_analysis:
                    error_analysis = run_analysis["correlation_analysis"].get("error_analysis", {})
                    all_errors.extend(error_analysis.get("error_types", {}).keys())
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze run {run_id}: {e}")
                analysis["run_summaries"][run_id] = {"error": str(e)}
        
        # Calculate aggregate statistics
        if all_stats:
            analysis["aggregate_statistics"] = self._calculate_aggregate_stats(all_stats)
        
        # Identify trends and patterns
        analysis["trends"] = self._identify_trends(all_stats)
        
        # Common issues analysis
        error_counter = Counter(all_errors)
        analysis["common_issues"] = [
            {"error_type": error_type, "frequency": count}
            for error_type, count in error_counter.most_common(10)
        ]
        
        # System health check
        try:
            health_report = validate_system_health(
                str(self.log_analyzer.eval_log_dir),
                str(self.log_analyzer.llm_log_dir)
            )
            analysis["system_health"] = health_report
        except Exception as e:
            analysis["system_health"] = {"error": str(e)}
        
        return analysis
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple runs for differences and patterns
        
        Args:
            run_ids: List of run_ids to compare
            
        Returns:
            Comparative analysis report
        """
        comparison = {
            "analysis_type": "comparison",
            "timestamp": datetime.now().isoformat(),
            "run_ids": run_ids,
            "individual_analyses": {},
            "comparative_metrics": {},
            "differences": {},
            "recommendations": []
        }
        
        # Analyze each run individually
        for run_id in run_ids:
            try:
                comparison["individual_analyses"][run_id] = self.analyze_specific_run(run_id, detailed=True)
            except Exception as e:
                comparison["individual_analyses"][run_id] = {"error": str(e)}
        
        # Compare metrics across runs
        successful_analyses = {
            run_id: analysis for run_id, analysis in comparison["individual_analyses"].items()
            if "error" not in analysis
        }
        
        if len(successful_analyses) >= 2:
            comparison["comparative_metrics"] = self._compare_metrics(successful_analyses)
            comparison["differences"] = self._identify_differences(successful_analyses)
            comparison["recommendations"] = self._generate_comparison_recommendations(
                successful_analyses, comparison["differences"]
            )
        else:
            comparison["error"] = "Insufficient successful analyses for comparison"
        
        return comparison
    
    def _calculate_aggregate_stats(self, all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics across runs"""
        if not all_stats:
            return {}
        
        # Extract numeric metrics
        success_rates = [s.get("api_success_rate", 0) for s in all_stats]
        error_rates = [s.get("error_rate", 0) for s in all_stats]
        total_calls = [s.get("total_llm_calls", 0) for s in all_stats]
        latencies = [s.get("average_latency", 0) for s in all_stats if s.get("average_latency", 0) > 0]
        
        aggregate = {
            "runs_analyzed": len(all_stats),
            "success_rate": {
                "mean": statistics.mean(success_rates) if success_rates else 0,
                "min": min(success_rates) if success_rates else 0,
                "max": max(success_rates) if success_rates else 0
            },
            "error_rate": {
                "mean": statistics.mean(error_rates) if error_rates else 0,
                "min": min(error_rates) if error_rates else 0,
                "max": max(error_rates) if error_rates else 0
            },
            "total_api_calls": sum(total_calls),
            "average_calls_per_run": statistics.mean(total_calls) if total_calls else 0
        }
        
        if latencies:
            aggregate["latency"] = {
                "mean": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies)
            }
        
        return aggregate
    
    def _identify_trends(self, all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends across runs"""
        trends = {
            "performance_trend": "stable",
            "error_trend": "stable", 
            "volume_trend": "stable",
            "insights": []
        }
        
        if len(all_stats) < 2:
            return trends
        
        # Analyze success rate trend
        success_rates = [s.get("api_success_rate", 0) for s in all_stats]
        if success_rates:
            if success_rates[-1] > success_rates[0] + 0.05:
                trends["performance_trend"] = "improving"
                trends["insights"].append("API success rate is improving over time")
            elif success_rates[-1] < success_rates[0] - 0.05:
                trends["performance_trend"] = "declining"
                trends["insights"].append("API success rate is declining over time")
        
        # Analyze error rate trend  
        error_rates = [s.get("error_rate", 0) for s in all_stats]
        if error_rates:
            if error_rates[-1] < error_rates[0] - 0.05:
                trends["error_trend"] = "improving"
                trends["insights"].append("Error rate is decreasing over time")
            elif error_rates[-1] > error_rates[0] + 0.05:
                trends["error_trend"] = "declining"
                trends["insights"].append("Error rate is increasing over time")
        
        return trends
    
    def _generate_insights(self, run_data: Dict[str, Any], 
                           correlation: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate insights and recommendations for a run"""
        findings = []
        recommendations = []
        
        stats = run_data.get("statistics", {})
        
        # Performance findings
        if stats.get("api_success_rate", 0) < 0.95:
            findings.append(f"API success rate is {stats['api_success_rate']:.2%}, below recommended 95%")
            recommendations.append("Investigate API failures and implement retry logic")
        
        if stats.get("error_rate", 0) > 0.05:
            findings.append(f"Error rate is {stats['error_rate']:.2%}, above acceptable 5% threshold")
            recommendations.append("Analyze error patterns and improve error handling")
        
        # Token usage findings
        total_tokens = stats.get("total_tokens", {}).get("total", 0)
        if total_tokens > MetricsConfig.HIGH_TOKEN_USAGE_THRESHOLD:
            findings.append(f"High token usage detected: {total_tokens:,} tokens")
            recommendations.append("Consider optimizing prompts to reduce token consumption")
        
        # Latency findings
        avg_latency = stats.get("average_latency", 0)
        if avg_latency > MetricsConfig.SLOW_RESPONSE_THRESHOLD:
            findings.append(f"High average latency: {avg_latency:.2f}s")
            recommendations.append("Investigate API response times and consider optimization")
        
        # Correlation-specific findings
        if correlation.get("correlation_success"):
            error_analysis = correlation.get("error_analysis", {})
            total_errors = error_analysis.get("total_errors", 0)
            
            if total_errors > 0:
                findings.append(f"Found {total_errors} errors with detailed context")
                error_types = error_analysis.get("error_types", {})
                most_common_error = max(error_types.items(), key=lambda x: x[1]) if error_types else None
                if most_common_error:
                    findings.append(f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)")
                    recommendations.append(f"Focus on resolving {most_common_error[0]} errors first")
        
        return findings, recommendations
    
    def _compare_metrics(self, analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare metrics across multiple analyses"""
        comparison_metrics = {}
        
        # Extract key metrics from each analysis
        for run_id, analysis in analyses.items():
            stats = analysis.get("basic_stats", {})
            comparison_metrics[run_id] = {
                "success_rate": stats.get("api_success_rate", 0),
                "error_rate": stats.get("error_rate", 0),
                "total_calls": stats.get("total_llm_calls", 0),
                "total_tokens": stats.get("total_tokens", {}).get("total", 0),
                "avg_latency": stats.get("average_latency", 0)
            }
        
        return comparison_metrics
    
    def _identify_differences(self, analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify key differences between analyses"""
        differences = {
            "performance_differences": {},
            "configuration_differences": {},
            "significant_differences": []
        }
        
        if len(analyses) < 2:
            return differences
        
        run_ids = list(analyses.keys())
        metrics = self._compare_metrics(analyses)
        
        # Compare success rates
        success_rates = [metrics[run_id]["success_rate"] for run_id in run_ids]
        if max(success_rates) - min(success_rates) > 0.1:  # 10% difference
            differences["significant_differences"].append({
                "type": "success_rate_variance",
                "description": f"Success rate varies from {min(success_rates):.2%} to {max(success_rates):.2%}",
                "runs_affected": {
                    "best": run_ids[success_rates.index(max(success_rates))],
                    "worst": run_ids[success_rates.index(min(success_rates))]
                }
            })
        
        # Compare token usage
        token_counts = [metrics[run_id]["total_tokens"] for run_id in run_ids]
        if max(token_counts) > min(token_counts) * 1.5:  # 50% difference
            differences["significant_differences"].append({
                "type": "token_usage_variance",
                "description": f"Token usage varies from {min(token_counts):,} to {max(token_counts):,}",
                "runs_affected": {
                    "highest": run_ids[token_counts.index(max(token_counts))],
                    "lowest": run_ids[token_counts.index(min(token_counts))]
                }
            })
        
        return differences
    
    def _generate_comparison_recommendations(self, analyses: Dict[str, Dict[str, Any]], 
                                             differences: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison analysis"""
        recommendations = []
        
        significant_diffs = differences.get("significant_differences", [])
        
        for diff in significant_diffs:
            if diff["type"] == "success_rate_variance":
                best_run = diff["runs_affected"]["best"]
                worst_run = diff["runs_affected"]["worst"]
                recommendations.append(
                    f"Investigate why run {best_run} has higher success rate than {worst_run}. "
                    f"Consider adopting successful configuration."
                )
            
            elif diff["type"] == "token_usage_variance":
                lowest_run = diff["runs_affected"]["lowest"]
                highest_run = diff["runs_affected"]["highest"]
                recommendations.append(
                    f"Run {lowest_run} used significantly fewer tokens than {highest_run}. "
                    f"Consider optimizing prompts based on efficient run."
                )
        
        if not recommendations:
            recommendations.append("Runs show similar performance characteristics - system appears stable")
        
        return recommendations
    
    def generate_report(self, analysis: Dict[str, Any], 
                        format_type: str = "markdown") -> str:
        """
        Generate formatted report from analysis
        
        Args:
            analysis: Analysis results to format
            format_type: Output format (markdown, html, json)
            
        Returns:
            Formatted report string
        """
        if format_type == "json":
            return json.dumps(analysis, indent=2, default=str)
        
        elif format_type == "markdown":
            return self._generate_markdown_report(analysis)
        
        elif format_type == "html":
            return self._generate_html_report(analysis)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown report"""
        lines = []
        
        # Header
        analysis_type = analysis.get("analysis_type", "unknown")
        timestamp = analysis.get("timestamp", "")
        
        lines.extend([
            f"# Case Study Analysis Report",
            f"",
            f"**Analysis Type:** {analysis_type.replace('_', ' ').title()}",
            f"**Generated:** {timestamp}",
            f""
        ])
        
        if analysis_type == "single_run":
            return self._generate_single_run_markdown(analysis, lines)
        elif analysis_type == "multi_run":
            return self._generate_multi_run_markdown(analysis, lines)
        elif analysis_type == "comparison":
            return self._generate_comparison_markdown(analysis, lines)
        
        return "\n".join(lines)
    
    def _generate_single_run_markdown(self, analysis: Dict[str, Any], lines: List[str]) -> str:
        """Generate markdown for single run analysis"""
        run_id = analysis.get("run_id", "")
        basic_stats = analysis.get("basic_stats", {})
        correlation = analysis.get("correlation_analysis", {})
        findings = analysis.get("findings", [])
        recommendations = analysis.get("recommendations", [])
        
        lines.extend([
            f"**Run ID:** `{run_id}`",
            f"",
            f"## Summary Statistics",
            f"",
            f"- **Total API Calls:** {basic_stats.get('total_llm_calls', 0):,}",
            f"- **Success Rate:** {basic_stats.get('api_success_rate', 0):.2%}",
            f"- **Error Rate:** {basic_stats.get('error_rate', 0):.2%}",
            f"- **Total Tokens:** {basic_stats.get('total_tokens', {}).get('total', 0):,}",
            f"- **Average Latency:** {basic_stats.get('average_latency', 0):.2f}s",
            f""
        ])
        
        if correlation.get("correlation_success"):
            qid_corr = correlation.get("qid_correlations", {})
            lines.extend([
                f"## Correlation Analysis",
                f"",
                f"- **Total Questions (QIDs):** {qid_corr.get('total_qids', 0)}",
                f"- **Correlation Success:** ✅ Yes",
                f""
            ])
            
            error_analysis = correlation.get("error_analysis", {})
            if error_analysis.get("total_errors", 0) > 0:
                lines.extend([
                    f"### Error Analysis",
                    f"",
                    f"**Total Errors:** {error_analysis['total_errors']}",
                    f""
                ])
                
                error_types = error_analysis.get("error_types", {})
                for error_type, count in error_types.items():
                    lines.append(f"- **{error_type}:** {count} occurrences")
                lines.append("")
        
        if findings:
            lines.extend([
                f"## Key Findings",
                f""
            ])
            for finding in findings:
                lines.append(f"- {finding}")
            lines.append("")
        
        if recommendations:
            lines.extend([
                f"## Recommendations",
                f""
            ])
            for rec in recommendations:
                lines.append(f"1. {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_multi_run_markdown(self, analysis: Dict[str, Any], lines: List[str]) -> str:
        """Generate markdown for multi-run analysis"""
        total_runs = analysis.get("total_runs", 0)
        aggregate_stats = analysis.get("aggregate_statistics", {})
        trends = analysis.get("trends", {})
        common_issues = analysis.get("common_issues", [])
        system_health = analysis.get("system_health", {})
        
        lines.extend([
            f"**Total Runs Analyzed:** {total_runs}",
            f"",
            f"## Aggregate Statistics",
            f""
        ])
        
        if aggregate_stats:
            lines.extend([
                f"- **Total API Calls:** {aggregate_stats.get('total_api_calls', 0):,}",
                f"- **Average Success Rate:** {aggregate_stats.get('success_rate', {}).get('mean', 0):.2%}",
                f"- **Average Error Rate:** {aggregate_stats.get('error_rate', {}).get('mean', 0):.2%}",
                f"- **Average Calls per Run:** {aggregate_stats.get('average_calls_per_run', 0):.0f}",
                f""
            ])
        
        if trends.get("insights"):
            lines.extend([
                f"## Trends Analysis",
                f""
            ])
            for insight in trends["insights"]:
                lines.append(f"- {insight}")
            lines.append("")
        
        if common_issues:
            lines.extend([
                f"## Common Issues",
                f""
            ])
            for issue in common_issues:
                lines.append(f"- **{issue['error_type']}:** {issue['frequency']} occurrences")
            lines.append("")
        
        health_status = system_health.get("overall_status", "UNKNOWN")
        lines.extend([
            f"## System Health",
            f"",
            f"**Overall Status:** {health_status}",
            f""
        ])
        
        health_recommendations = system_health.get("recommendations", [])
        if health_recommendations:
            lines.append("**Health Recommendations:**")
            for rec in health_recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_comparison_markdown(self, analysis: Dict[str, Any], lines: List[str]) -> str:
        """Generate markdown for comparison analysis"""
        run_ids = analysis.get("run_ids", [])
        comparative_metrics = analysis.get("comparative_metrics", {})
        differences = analysis.get("differences", {})
        recommendations = analysis.get("recommendations", [])
        
        lines.extend([
            f"**Runs Compared:** {', '.join(f'`{rid}`' for rid in run_ids)}",
            f"",
            f"## Comparative Metrics",
            f""
        ])
        
        if comparative_metrics:
            # Create comparison table
            lines.append("| Run ID | Success Rate | Error Rate | Total Calls | Total Tokens | Avg Latency |")
            lines.append("|--------|-------------|------------|-------------|--------------|-------------|")
            
            for run_id, metrics in comparative_metrics.items():
                lines.append(
                    f"| `{run_id}` | "
                    f"{metrics.get('success_rate', 0):.2%} | "
                    f"{metrics.get('error_rate', 0):.2%} | "
                    f"{metrics.get('total_calls', 0):,} | "
                    f"{metrics.get('total_tokens', 0):,} | "
                    f"{metrics.get('avg_latency', 0):.2f}s |"
                )
            lines.append("")
        
        significant_diffs = differences.get("significant_differences", [])
        if significant_diffs:
            lines.extend([
                f"## Significant Differences",
                f""
            ])
            for diff in significant_diffs:
                lines.extend([
                    f"### {diff['type'].replace('_', ' ').title()}",
                    f"",
                    f"{diff['description']}",
                    f""
                ])
                
                if "runs_affected" in diff:
                    affected = diff["runs_affected"]
                    for role, run_id in affected.items():
                        lines.append(f"- **{role.title()} performing:** `{run_id}`")
                    lines.append("")
        
        if recommendations:
            lines.extend([
                f"## Recommendations",
                f""
            ])
            for rec in recommendations:
                lines.append(f"1. {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report"""
        # Convert markdown to HTML (simplified version)
        markdown_content = self._generate_markdown_report(analysis)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Case Study Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div id="content">
{markdown_content.replace('`', '<code>').replace('</code><code>', '').replace('**', '<strong>').replace('</strong><strong>', '').replace('*', '<em>').replace('</em><em>', '')}
    </div>
</body>
</html>
"""
        return html_template
    
    def save_report(self, analysis: Dict[str, Any], 
                    filename: Optional[str] = None, 
                    format_type: str = "markdown") -> Path:
        """
        Save analysis report to file
        
        Args:
            analysis: Analysis results to save
            filename: Optional custom filename
            format_type: Output format
            
        Returns:
            Path to saved report file
        """
        if not filename:
            analysis_type = analysis.get("analysis_type", "analysis")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if analysis_type == "single_run":
                run_id = analysis.get("run_id", "unknown")
                filename = f"case_study_{run_id}_{timestamp}"
            elif analysis_type == "comparison":
                run_ids = analysis.get("run_ids", [])
                run_suffix = "_".join(run_ids[:2])  # First 2 run IDs
                if len(run_ids) > 2:
                    run_suffix += f"_and_{len(run_ids)-2}_more"
                filename = f"comparison_{run_suffix}_{timestamp}"
            else:
                filename = f"multi_run_analysis_{timestamp}"
        
        # Add extension based on format
        extensions = {"markdown": ".md", "html": ".html", "json": ".json"}
        extension = extensions.get(format_type, ".txt")
        if not filename.endswith(extension):
            filename += extension
        
        # Generate report content
        report_content = self.generate_report(analysis, format_type)
        
        # Save to file
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return output_path


def main():
    """Main CLI interface for case study analysis"""
    parser = argparse.ArgumentParser(
        description="Case Study Analysis Tool for Nugget Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze specific run
    python case_study.py --run-id 20240101_120000_abc12345 --log-dir logs/

    # Analyze all runs
    python case_study.py --analyze-all --log-dir logs/ --limit 10

    # Compare runs
    python case_study.py --compare-runs 20240101_120000_abc12345 20240101_130000_def67890 --log-dir logs/

    # Generate HTML report
    python case_study.py --run-id 20240101_120000_abc12345 --log-dir logs/ --format html

    # Save report to custom location
    python case_study.py --analyze-all --log-dir logs/ --output-dir ./reports/
        """
    )
    
    # Main action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--run-id", 
        type=str,
        help="Analyze specific run by run_id"
    )
    action_group.add_argument(
        "--analyze-all",
        action="store_true", 
        help="Analyze all available runs"
    )
    action_group.add_argument(
        "--compare-runs",
        nargs="+",
        metavar="RUN_ID",
        help="Compare multiple runs (specify 2 or more run_ids)"
    )
    action_group.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available run_ids and exit"
    )
    action_group.add_argument(
        "--health-check",
        action="store_true",
        help="Perform system health check and exit"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Base directory containing eval/ and llm/ log directories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save reports (default: ./case_study_reports/)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format for reports (default: markdown)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of runs to analyze (for --analyze-all)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed correlation analysis"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save analysis report to file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize case study runner
        log_base_dir = Path(args.log_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        runner = CaseStudyRunner(log_base_dir, output_dir)
        
        # Handle different actions
        if args.list_runs:
            run_ids = runner.log_analyzer.discover_run_ids()
            print(f"Found {len(run_ids)} valid run_ids:")
            for run_id in run_ids:
                print(f"  - {run_id}")
            return
        
        if args.health_check:
            print("Performing system health check...")
            health_report = validate_system_health(
                str(runner.log_analyzer.eval_log_dir),
                str(runner.log_analyzer.llm_log_dir)
            )
            print(f"System Status: {health_report.get('overall_status', 'UNKNOWN')}")
            
            recommendations = health_report.get("recommendations", [])
            if recommendations:
                print("\nRecommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
            
            if args.format == "json":
                print("\nDetailed Report:")
                print(json.dumps(health_report, indent=2, default=str))
            return
        
        # Perform main analysis
        analysis_result = None
        
        if args.run_id:
            logger.info(f"Analyzing run: {args.run_id}")
            analysis_result = runner.analyze_specific_run(args.run_id, detailed=args.detailed)
            
        elif args.analyze_all:
            logger.info(f"Analyzing all runs (limit: {args.limit or 'none'})")
            analysis_result = runner.analyze_all_runs(limit=args.limit)
            
        elif args.compare_runs:
            if len(args.compare_runs) < 2:
                parser.error("--compare-runs requires at least 2 run_ids")
            logger.info(f"Comparing runs: {', '.join(args.compare_runs)}")
            analysis_result = runner.compare_runs(args.compare_runs)
        
        if not analysis_result:
            logger.error("No analysis result generated")
            return 1
        
        # Handle output
        if args.save_report:
            report_path = runner.save_report(analysis_result, format_type=args.format)
            print(f"Report saved to: {report_path}")
        else:
            # Print to stdout
            report_content = runner.generate_report(analysis_result, format_type=args.format)
            print(report_content)
        
        # Check for errors in analysis
        if "error" in analysis_result:
            logger.error(f"Analysis completed with errors: {analysis_result['error']}")
            return 1
        
        logger.info("Analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())