#!/usr/bin/env python3
"""
Interactive Analysis Tool for Nugget Evaluation Framework

This tool provides an interactive command-line interface for exploring 
evaluation runs, querying specific cases, and performing ad-hoc analysis.

Usage:
    python interactive_analysis.py --log-dir <path>
    python interactive_analysis.py --log-dir <path> --web-interface
"""

import argparse
import cmd
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import readline  # Enable command history and editing

from case_study import LogAnalyzer, CaseStudyRunner
from nugget_eval.validators import validate_run_id, validate_system_health
from nugget_eval.constants import RunIdConfig, ValidationConfig


class InteractiveAnalyzer(cmd.Cmd):
    """Interactive command-line interface for analysis"""
    
    intro = """
🔍 Interactive Nugget Evaluation Analysis Tool
==============================================

Type 'help' or '?' to list available commands.
Type 'help <command>' for detailed help on a specific command.
Type 'quit' or 'exit' to leave.

"""
    
    prompt = "analysis> "
    
    def __init__(self, log_base_dir: Path):
        super().__init__()
        self.log_analyzer = LogAnalyzer(log_base_dir)
        self.case_runner = CaseStudyRunner(log_base_dir)
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._run_ids_cache = None
        self._current_run_data = None
        self._current_run_id = None
        
        # Initialize
        self._refresh_run_ids()
        print(f"📊 Connected to log directory: {log_base_dir}")
        print(f"✅ Found {len(self.run_ids)} valid runs")
    
    def _refresh_run_ids(self):
        """Refresh the cache of available run_ids"""
        self._run_ids_cache = self.log_analyzer.discover_run_ids()
    
    @property
    def run_ids(self) -> List[str]:
        """Get cached run_ids"""
        if self._run_ids_cache is None:
            self._refresh_run_ids()
        return self._run_ids_cache
    
    def do_list_runs(self, args: str):
        """
        List available run_ids with optional filtering
        
        Usage: list_runs [pattern]
        
        Examples:
            list_runs                    # List all runs
            list_runs 20240101           # List runs from specific date
            list_runs 2024*_1200*        # List runs with pattern matching
        """
        pattern = args.strip() if args.strip() else "*"
        
        if pattern == "*":
            matching_runs = self.run_ids
        else:
            import fnmatch
            matching_runs = [
                run_id for run_id in self.run_ids 
                if fnmatch.fnmatch(run_id, pattern)
            ]
        
        if not matching_runs:
            print(f"❌ No runs found matching pattern: {pattern}")
            return
        
        print(f"📋 Found {len(matching_runs)} runs matching '{pattern}':")
        print()
        
        # Group by date for better readability
        grouped_runs = {}
        for run_id in matching_runs:
            date_part = run_id[:8]  # YYYYMMDD
            if date_part not in grouped_runs:
                grouped_runs[date_part] = []
            grouped_runs[date_part].append(run_id)
        
        for date, runs in sorted(grouped_runs.items()):
            try:
                formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
                print(f"📅 {formatted_date}:")
            except:
                print(f"📅 {date}:")
            
            for run_id in sorted(runs):
                print(f"   {run_id}")
            print()
    
    def do_load_run(self, args: str):
        """
        Load a specific run for detailed analysis
        
        Usage: load_run <run_id>
        
        Example:
            load_run 20240101_120000_abc12345
        """
        run_id = args.strip()
        
        if not run_id:
            print("❌ Please provide a run_id")
            print("Usage: load_run <run_id>")
            return
        
        if not validate_run_id(run_id):
            print(f"❌ Invalid run_id format: {run_id}")
            return
        
        if run_id not in self.run_ids:
            print(f"❌ Run not found: {run_id}")
            print("💡 Use 'list_runs' to see available runs")
            return
        
        try:
            print(f"⏳ Loading run: {run_id}")
            run_data = self.log_analyzer.load_run_data(run_id)
            self._current_run_data = run_data
            self._current_run_id = run_id
            
            stats = run_data.get("statistics", {})
            print(f"✅ Loaded run: {run_id}")
            print(f"   📊 API Calls: {stats.get('total_llm_calls', 0):,}")
            print(f"   📈 Success Rate: {stats.get('api_success_rate', 0):.2%}")
            print(f"   🔍 Predictions: {stats.get('total_predictions', 0):,}")
            print()
            print("💡 Use 'show_summary', 'show_errors', or 'query_qid' to analyze further")
            
        except Exception as e:
            print(f"❌ Failed to load run: {e}")
    
    def do_show_summary(self, args: str):
        """
        Show summary of currently loaded run
        
        Usage: show_summary [detailed]
        
        Examples:
            show_summary            # Basic summary
            show_summary detailed   # Detailed summary with performance metrics
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        detailed = args.strip().lower() == "detailed"
        run_id = self._current_run_id
        stats = self._current_run_data.get("statistics", {})
        
        print(f"📊 Summary for run: {run_id}")
        print("=" * 50)
        
        # Basic statistics
        print(f"🔢 Total API Calls: {stats.get('total_llm_calls', 0):,}")
        print(f"📈 Success Rate: {stats.get('api_success_rate', 0):.2%}")
        print(f"❌ Error Rate: {stats.get('error_rate', 0):.2%}")
        print(f"🔍 Total Predictions: {stats.get('total_predictions', 0):,}")
        
        # Token usage
        tokens = stats.get("total_tokens", {})
        print(f"🪙 Token Usage:")
        print(f"   📥 Input: {tokens.get('input', 0):,}")
        print(f"   📤 Output: {tokens.get('output', 0):,}")
        print(f"   📊 Total: {tokens.get('total', 0):,}")
        
        # Performance metrics (if detailed)
        if detailed:
            avg_latency = stats.get("average_latency", 0)
            if avg_latency > 0:
                print(f"⏱️  Average Latency: {avg_latency:.2f}s")
                
                latency_percentiles = stats.get("latency_percentiles", {})
                if latency_percentiles:
                    print(f"📊 Latency Percentiles:")
                    for p, value in latency_percentiles.items():
                        print(f"   {p}: {value:.2f}s")
            
            time_range = stats.get("time_range", {})
            if time_range:
                duration = time_range.get("duration_seconds", 0)
                print(f"⏰ Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        
        # File information
        eval_logs = self._current_run_data.get("eval_logs", [])
        llm_logs = self._current_run_data.get("llm_logs", [])
        predictions = self._current_run_data.get("predictions", [])
        
        print(f"📁 Files:")
        print(f"   📜 Eval logs: {len(eval_logs)}")
        print(f"   🤖 LLM logs: {len(llm_logs)}")
        print(f"   📋 Prediction files: {len(predictions)}")
        
        print()
    
    def do_show_errors(self, args: str):
        """
        Show error analysis for currently loaded run
        
        Usage: show_errors [limit]
        
        Examples:
            show_errors        # Show all errors
            show_errors 10     # Show first 10 errors
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        limit = None
        if args.strip():
            try:
                limit = int(args.strip())
            except ValueError:
                print("❌ Invalid limit. Please provide a number")
                return
        
        # Collect all errors from LLM logs
        all_errors = []
        llm_logs = self._current_run_data.get("llm_logs", [])
        
        for log_group in llm_logs:
            for entry in log_group["entries"]:
                if not entry.get("success", False):
                    error_info = {
                        "timestamp": entry.get("timestamp"),
                        "error": entry.get("error", {}),
                        "context": entry.get("context", {}),
                        "qid": entry.get("context", {}).get("qid"),
                        "batch_index": entry.get("context", {}).get("batch_index")
                    }
                    all_errors.append(error_info)
        
        if not all_errors:
            print("✅ No errors found in this run!")
            return
        
        # Apply limit if specified
        if limit and limit < len(all_errors):
            displayed_errors = all_errors[:limit]
            print(f"❌ Showing first {limit} of {len(all_errors)} errors:")
        else:
            displayed_errors = all_errors
            print(f"❌ Found {len(all_errors)} errors:")
        
        print()
        
        # Group errors by type for summary
        error_types = {}
        for error in all_errors:
            error_type = error["error"].get("type", "unknown")
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        print("📊 Error Type Summary:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count} occurrences")
        print()
        
        # Show detailed errors
        print("🔍 Detailed Error List:")
        for i, error in enumerate(displayed_errors, 1):
            print(f"{i:3d}. [{error.get('timestamp', 'N/A')}]")
            print(f"     QID: {error.get('qid', 'N/A')}")
            print(f"     Batch: {error.get('batch_index', 'N/A')}")
            print(f"     Type: {error['error'].get('type', 'unknown')}")
            print(f"     Message: {error['error'].get('message', 'No message')}")
            print()
    
    def do_query_qid(self, args: str):
        """
        Query specific QID for detailed analysis
        
        Usage: query_qid <qid>
        
        Example:
            query_qid Q12345
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        qid = args.strip()
        if not qid:
            print("❌ Please provide a QID")
            print("Usage: query_qid <qid>")
            return
        
        # Find all entries for this QID
        qid_entries = []
        llm_logs = self._current_run_data.get("llm_logs", [])
        
        for log_group in llm_logs:
            for entry in log_group["entries"]:
                entry_qid = entry.get("context", {}).get("qid")
                if entry_qid == qid:
                    qid_entries.append(entry)
        
        if not qid_entries:
            print(f"❌ No entries found for QID: {qid}")
            
            # Suggest similar QIDs
            all_qids = set()
            for log_group in llm_logs:
                for entry in log_group["entries"]:
                    entry_qid = entry.get("context", {}).get("qid")
                    if entry_qid:
                        all_qids.add(entry_qid)
            
            if all_qids:
                similar_qids = [q for q in all_qids if qid.lower() in q.lower()][:5]
                if similar_qids:
                    print(f"💡 Similar QIDs found: {', '.join(similar_qids)}")
            return
        
        print(f"🔍 Analysis for QID: {qid}")
        print("=" * 40)
        print(f"📊 Total API calls: {len(qid_entries)}")
        
        # Analyze by batch
        batch_groups = {}
        for entry in qid_entries:
            batch_idx = entry.get("context", {}).get("batch_index", 0)
            if batch_idx not in batch_groups:
                batch_groups[batch_idx] = []
            batch_groups[batch_idx].append(entry)
        
        print(f"🔢 Number of batches: {len(batch_groups)}")
        print()
        
        # Show batch details
        for batch_idx in sorted(batch_groups.keys()):
            entries = batch_groups[batch_idx]
            successful = sum(1 for e in entries if e.get("success", False))
            
            print(f"📦 Batch {batch_idx}:")
            print(f"   🔢 Calls: {len(entries)}")
            print(f"   ✅ Success: {successful}/{len(entries)} ({successful/len(entries):.1%})")
            
            # Show errors if any
            errors = [e for e in entries if not e.get("success", False)]
            if errors:
                print(f"   ❌ Errors: {len(errors)}")
                for error in errors:
                    error_type = error.get("error", {}).get("type", "unknown")
                    print(f"      - {error_type}")
            
            # Show response details for successful calls
            successful_entries = [e for e in entries if e.get("success", False)]
            if successful_entries:
                # Show token usage
                total_tokens = 0
                for entry in successful_entries:
                    usage = entry.get("response", {}).get("usage")
                    if isinstance(usage, dict):
                        total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                
                if total_tokens > 0:
                    print(f"   🪙 Tokens: {total_tokens:,}")
            
            print()
    
    def do_search(self, args: str):
        """
        Search across all loaded run data
        
        Usage: search <pattern> [in_field]
        
        Examples:
            search "timeout"                    # Search in all fields
            search "temperature" parameters     # Search in specific field
            search "error" content              # Search in content fields
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        parts = args.strip().split()
        if not parts:
            print("❌ Please provide a search pattern")
            print("Usage: search <pattern> [in_field]")
            return
        
        pattern = parts[0].strip('"\'')  # Remove quotes
        field_filter = parts[1] if len(parts) > 1 else None
        
        print(f"🔍 Searching for '{pattern}' in run: {self._current_run_id}")
        if field_filter:
            print(f"   📋 Field filter: {field_filter}")
        print()
        
        # Search in LLM logs
        matches = []
        llm_logs = self._current_run_data.get("llm_logs", [])
        
        for log_group in llm_logs:
            for entry in log_group["entries"]:
                match_info = self._search_in_entry(entry, pattern, field_filter)
                if match_info:
                    matches.extend(match_info)
        
        if not matches:
            print(f"❌ No matches found for '{pattern}'")
            return
        
        print(f"✅ Found {len(matches)} matches:")
        print()
        
        # Show matches
        for i, match in enumerate(matches[:20], 1):  # Limit to first 20
            print(f"{i:2d}. {match['location']}")
            print(f"    QID: {match.get('qid', 'N/A')}")
            print(f"    Match: {match['match_text'][:100]}...")
            print()
        
        if len(matches) > 20:
            print(f"... and {len(matches) - 20} more matches")
    
    def _search_in_entry(self, entry: Dict[str, Any], pattern: str, 
                         field_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Search for pattern within a log entry"""
        matches = []
        pattern_lower = pattern.lower()
        
        # Helper function to search in a value
        def search_value(value, location):
            if isinstance(value, str) and pattern_lower in value.lower():
                matches.append({
                    "location": location,
                    "match_text": value,
                    "qid": entry.get("context", {}).get("qid"),
                    "timestamp": entry.get("timestamp")
                })
            elif isinstance(value, dict):
                for k, v in value.items():
                    search_value(v, f"{location}.{k}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    search_value(item, f"{location}[{i}]")
        
        # Search in different parts of the entry
        if not field_filter or field_filter == "error":
            if "error" in entry:
                search_value(entry["error"], "error")
        
        if not field_filter or field_filter == "response":
            if "response" in entry:
                search_value(entry["response"], "response")
        
        if not field_filter or field_filter == "parameters":
            if "parameters" in entry:
                search_value(entry["parameters"], "parameters")
        
        if not field_filter or field_filter == "messages":
            if "messages" in entry:
                search_value(entry["messages"], "messages")
        
        if not field_filter or field_filter == "context":
            if "context" in entry:
                search_value(entry["context"], "context")
        
        return matches
    
    def do_compare(self, args: str):
        """
        Compare current run with another run
        
        Usage: compare <other_run_id>
        
        Example:
            compare 20240101_130000_def67890
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        other_run_id = args.strip()
        if not other_run_id:
            print("❌ Please provide another run_id to compare with")
            return
        
        if other_run_id not in self.run_ids:
            print(f"❌ Run not found: {other_run_id}")
            return
        
        try:
            print(f"⏳ Loading run for comparison: {other_run_id}")
            comparison_result = self.case_runner.compare_runs([self._current_run_id, other_run_id])
            
            print("📊 Comparison Results:")
            print("=" * 40)
            
            # Show comparative metrics
            metrics = comparison_result.get("comparative_metrics", {})
            if metrics:
                print("📈 Performance Comparison:")
                print(f"{'Metric':<20} {'Current':<15} {'Other':<15} {'Difference':<15}")
                print("-" * 65)
                
                current_metrics = metrics.get(self._current_run_id, {})
                other_metrics = metrics.get(other_run_id, {})
                
                metric_names = {
                    "success_rate": "Success Rate",
                    "error_rate": "Error Rate", 
                    "total_calls": "Total Calls",
                    "total_tokens": "Total Tokens",
                    "avg_latency": "Avg Latency"
                }
                
                for metric_key, metric_name in metric_names.items():
                    current_val = current_metrics.get(metric_key, 0)
                    other_val = other_metrics.get(metric_key, 0)
                    
                    if metric_key in ["success_rate", "error_rate"]:
                        current_str = f"{current_val:.2%}"
                        other_str = f"{other_val:.2%}"
                        diff = current_val - other_val
                        diff_str = f"{diff:+.2%}"
                    elif metric_key == "avg_latency":
                        current_str = f"{current_val:.2f}s"
                        other_str = f"{other_val:.2f}s"
                        diff = current_val - other_val
                        diff_str = f"{diff:+.2f}s"
                    else:
                        current_str = f"{current_val:,}"
                        other_str = f"{other_val:,}"
                        diff = current_val - other_val
                        diff_str = f"{diff:+,}"
                    
                    print(f"{metric_name:<20} {current_str:<15} {other_str:<15} {diff_str:<15}")
            
            # Show significant differences
            differences = comparison_result.get("differences", {})
            significant_diffs = differences.get("significant_differences", [])
            
            if significant_diffs:
                print("\n⚠️  Significant Differences:")
                for diff in significant_diffs:
                    print(f"   • {diff['description']}")
            else:
                print("\n✅ No significant differences found")
            
            # Show recommendations
            recommendations = comparison_result.get("recommendations", [])
            if recommendations:
                print("\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"   • {rec}")
            
            print()
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    
    def do_health_check(self, args: str):
        """
        Perform system health check
        
        Usage: health_check
        """
        try:
            print("⏳ Performing system health check...")
            health_report = validate_system_health(
                str(self.log_analyzer.eval_log_dir),
                str(self.log_analyzer.llm_log_dir)
            )
            
            status = health_report.get("overall_status", "UNKNOWN")
            status_icons = {
                "HEALTHY": "✅",
                "PARTIAL": "⚠️",
                "UNHEALTHY": "❌",
                "ERROR": "💥"
            }
            
            print(f"{status_icons.get(status, '❓')} System Status: {status}")
            print()
            
            # Show validation results
            run_id_validation = health_report.get("run_id_validation", {})
            if run_id_validation:
                print(f"🆔 Run ID Validation:")
                print(f"   Valid: {run_id_validation.get('valid_count', 0)}")
                print(f"   Invalid: {run_id_validation.get('invalid_count', 0)}")
                print()
            
            correlation_validation = health_report.get("log_correlation_validation", {})
            if correlation_validation:
                print(f"🔗 Log Correlation:")
                print(f"   Valid: {correlation_validation.get('valid_correlations', 0)}")
                print(f"   Invalid: {correlation_validation.get('invalid_correlations', 0)}")
                print()
            
            # Show recommendations
            recommendations = health_report.get("recommendations", [])
            if recommendations:
                print("💡 Recommendations:")
                for rec in recommendations:
                    print(f"   • {rec}")
                print()
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
    
    def do_export(self, args: str):
        """
        Export analysis results to file
        
        Usage: export <type> [filename] [format]
        
        Types: summary, errors, comparison, full
        Formats: json, markdown, csv
        
        Examples:
            export summary                          # Export summary as markdown
            export errors error_analysis.json json # Export errors as JSON
            export full complete_analysis.md        # Export full analysis
        """
        if not self._current_run_data:
            print("❌ No run loaded. Use 'load_run <run_id>' first")
            return
        
        parts = args.strip().split()
        if not parts:
            print("❌ Please specify export type")
            print("Usage: export <type> [filename] [format]")
            print("Types: summary, errors, comparison, full")
            return
        
        export_type = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        format_type = parts[2] if len(parts) > 2 else "markdown"
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = {"json": ".json", "markdown": ".md", "csv": ".csv"}.get(format_type, ".txt")
            filename = f"{export_type}_{self._current_run_id}_{timestamp}{extension}"
        
        try:
            if export_type == "summary":
                data = {
                    "run_id": self._current_run_id,
                    "statistics": self._current_run_data.get("statistics", {}),
                    "timestamp": datetime.now().isoformat()
                }
            elif export_type == "full":
                analysis = self.case_runner.analyze_specific_run(self._current_run_id, detailed=True)
                data = analysis
            else:
                print(f"❌ Unsupported export type: {export_type}")
                return
            
            # Save to file
            output_path = Path(filename)
            if format_type == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format_type == "markdown":
                content = self.case_runner.generate_report(data, format_type)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                print(f"❌ Unsupported format: {format_type}")
                return
            
            print(f"✅ Exported {export_type} to: {output_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def do_refresh(self, args: str):
        """
        Refresh run_ids cache and reload current run
        
        Usage: refresh
        """
        print("⏳ Refreshing data...")
        self._refresh_run_ids()
        print(f"✅ Found {len(self.run_ids)} runs")
        
        if self._current_run_id:
            try:
                self._current_run_data = self.log_analyzer.load_run_data(self._current_run_id)
                print(f"✅ Reloaded current run: {self._current_run_id}")
            except Exception as e:
                print(f"⚠️  Failed to reload current run: {e}")
    
    def do_quit(self, args: str):
        """Exit the interactive analyzer"""
        print("👋 Goodbye!")
        return True
    
    def do_exit(self, args: str):
        """Exit the interactive analyzer"""
        return self.do_quit(args)
    
    def do_EOF(self, args: str):
        """Handle Ctrl+D"""
        print()  # New line for cleaner output
        return self.do_quit(args)
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def default(self, line: str):
        """Handle unknown commands"""
        print(f"❌ Unknown command: {line}")
        print("💡 Type 'help' to see available commands")


def main():
    """Main function for interactive analysis tool"""
    parser = argparse.ArgumentParser(
        description="Interactive Analysis Tool for Nugget Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Commands Available:
    list_runs [pattern]         - List available runs
    load_run <run_id>          - Load a specific run for analysis  
    show_summary [detailed]    - Show summary of loaded run
    show_errors [limit]        - Show error analysis
    query_qid <qid>           - Query specific question ID
    search <pattern> [field]   - Search in loaded run data
    compare <other_run_id>     - Compare with another run
    health_check              - Check system health
    export <type> [file]      - Export analysis results
    refresh                   - Refresh data cache
    help [command]            - Show help for commands
    quit/exit                 - Exit the tool

Examples:
    python interactive_analysis.py --log-dir logs/
        """
    )
    
    parser.add_argument(
        "--log-dir", 
        type=str,
        required=True,
        help="Base directory containing eval/ and llm/ log directories"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        log_base_dir = Path(args.log_dir)
        if not log_base_dir.exists():
            print(f"❌ Log directory not found: {log_base_dir}")
            return 1
        
        # Start interactive analyzer
        analyzer = InteractiveAnalyzer(log_base_dir)
        analyzer.cmdloop()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start interactive analyzer: {e}")
        return 1


if __name__ == "__main__":
    exit(main())