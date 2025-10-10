"""Unified evaluator for nugget matching tasks"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import multiprocessing
from collections import defaultdict

from .config import Config
from .common import (
    ModelClient, load_jsonl_data, save_jsonl_data, 
    process_single_item, aggregate_assignment, create_gold_data_index
)
from .metrics import MetricsCalculator


# Global variable for worker processes
_worker_client = None


def _init_worker(base_url: str, api_key: str, model_name: str, llm_log_dir: str, run_id: str):
    """Initialize a worker process with its own ModelClient"""
    global _worker_client
    # Create a unique log directory for this worker process
    worker_log_dir = f"{llm_log_dir}/worker_{os.getpid()}"
    _worker_client = ModelClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        llm_log_dir=worker_log_dir,
        run_id=run_id
    )
    # Ensure logs are flushed when worker exits
    try:
        import atexit
        atexit.register(lambda: _worker_client.close())
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("atexit module not available, manual cleanup required")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to register cleanup handler: {e}")


def _process_item_worker(item_and_config):
    """Process a single item using the worker's ModelClient"""
    global _worker_client
    # Support both legacy 3-tuple and new 4-tuple (with run_context_base)
    try:
        item, batch_size, model_config, run_context_base = item_and_config
    except ValueError:
        item, batch_size, model_config = item_and_config
        run_context_base = None
    results, num_batches, batch_stats = process_single_item(
        _worker_client, item, batch_size, model_config, run_context_base
    )
    # Ensure logs are flushed to disk for this worker to avoid data loss on pool teardown
    try:
        if hasattr(_worker_client, "flush_logs"):
            _worker_client.flush_logs()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Worker {os.getpid()} failed to flush logs: {e}")
    return results, num_batches, batch_stats


class NuggetEvaluator:
    """Unified evaluator supporting both single and multi-run evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        # Create a unique run_id and per-run directories
        from datetime import datetime
        import uuid
        from .constants import RunIdConfig
        from .validators import validate_run_id
        
        self.run_id = datetime.now().strftime(RunIdConfig.TIMESTAMP_FORMAT) + RunIdConfig.SEPARATOR + uuid.uuid4().hex[:RunIdConfig.UUID_LENGTH]
        
        # Validate the generated run_id
        if not validate_run_id(self.run_id):
            raise RuntimeError(f"Generated invalid run_id: {self.run_id}")
        self.eval_run_dir = Path(config.logging.eval_log_dir) / self.run_id
        self.llm_run_dir = Path(config.logging.llm_log_dir) / self.run_id
        self.eval_run_dir.mkdir(parents=True, exist_ok=True)
        self.llm_run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calc = MetricsCalculator()
        self.logger = self._setup_logging()
        
        # Initialize model client
        self.model_client = ModelClient(
            base_url=config.model.base_url,
            api_key=config.model.api_key,
            model_name=config.model.name,
            llm_log_dir=str(self.llm_run_dir),
            run_id=self.run_id
        )
        
        # Create base output directories as well (for safety)
        Path(config.logging.eval_log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging.llm_log_dir).mkdir(parents=True, exist_ok=True)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation based on configuration"""
        self.logger.info(f"Starting evaluation with {self.config.evaluation.num_runs} run(s)")
        
        # Load data
        input_data = load_jsonl_data(self.config.data.input_path)
        gold_data_list = load_jsonl_data(self.config.data.gold_path)
        gold_data_index = create_gold_data_index(gold_data_list)
        
        self.logger.info(f"Loaded {len(input_data)} input items and {len(gold_data_list)} gold items")
        
        if self.config.evaluation.num_runs == 1:
            results = self._run_single_evaluation(input_data, gold_data_index)
        else:
            results = self._run_multi_evaluation(input_data, gold_data_index)
        
        # Finalize and gather logs
        self._finalize_and_gather()
        return results
    
    def _run_single_evaluation(self, input_data: List[Dict], gold_data: Dict) -> Dict[str, Any]:
        """Run single evaluation (like eval_with_thinking.py)"""
        self.logger.info("Running single evaluation...")
        
        # Run predictions
        all_results, total_batches, all_batch_stats = self._run_predictions(input_data)
        
        # Calculate block-level metrics with batch statistics
        nugget_y_true = [res["block_true"] for res in all_results]
        nugget_y_pred = [res["block_pred"] for res in all_results]
        
        nugget_metrics = self.metrics_calc.calculate_single_run_metrics(
            nugget_y_true, nugget_y_pred, "Nugget-level Match", all_batch_stats
        )
        
        # Calculate sample-level metrics (max support aggregation)
        sample_y_true, sample_y_pred = self.metrics_calc.aggregate_sample_level_predictions_by_max_support(all_results)
        
        sample_metrics = self.metrics_calc.calculate_single_run_metrics(
            sample_y_true, sample_y_pred, "Sample-level Match (Max Support)"
        )
        
        # Calculate overall-level metrics
        overall_y_true, overall_y_pred = aggregate_assignment(all_results, gold_data)
        
        overall_metrics = {}
        if overall_y_true:
            overall_metrics = self.metrics_calc.calculate_single_run_metrics(
                overall_y_true, overall_y_pred, "Overall Match"
            )
        
        # Print results
        self.metrics_calc.print_single_run_metrics(nugget_metrics)
        self.metrics_calc.print_single_run_metrics(sample_metrics)
        if overall_metrics:
            self.metrics_calc.print_single_run_metrics(overall_metrics)
        
        # Print detailed error analysis
        if total_batches > 0:
            self._print_error_analysis_report(all_batch_stats)
        
        # Save results if requested (after all reports)
        if self.config.logging.save_predictions:
            filepath = self._save_predictions_silent(all_results, "single_run")
            print(f"\n📁 Predictions saved to: {filepath}")
        
        return {
            "type": "single_run",
            "nugget_metrics": nugget_metrics,
            "sample_metrics": sample_metrics,
            "overall_metrics": overall_metrics,
            "batch_stats": all_batch_stats,
            "total_batches": total_batches,
            "predictions": all_results
        }
    
    def _run_multi_evaluation(self, input_data: List[Dict], gold_data: Dict) -> Dict[str, Any]:
        """Run multiple evaluations for statistical analysis (like eval_without_thinking.py)"""
        self.logger.info(f"Running {self.config.evaluation.num_runs} evaluations...")
        
        nugget_metrics_runs = []
        sample_metrics_runs = []
        overall_metrics_runs = []
        all_predictions_runs = []
        all_batch_stats_runs = []  # Collect batch stats from all runs for analysis
        
        for run_idx in tqdm(range(self.config.evaluation.num_runs), desc="Evaluation runs"):
            self.logger.info(f"Starting run {run_idx + 1}/{self.config.evaluation.num_runs}")
            
            # Run predictions for this iteration
            all_results, total_batches, batch_stats_list = self._run_predictions(input_data, run_idx)
            
            # Calculate block-level metrics with batch statistics
            nugget_y_true = [res["block_true"] for res in all_results]
            nugget_y_pred = [res["block_pred"] for res in all_results]
            
            nugget_metrics = self.metrics_calc.calculate_single_run_metrics(
                nugget_y_true, nugget_y_pred, f"Nugget-level Run {run_idx + 1}", batch_stats_list
            )
            nugget_metrics_runs.append(nugget_metrics)
            
            # Calculate sample-level metrics (max support aggregation)
            sample_y_true, sample_y_pred = self.metrics_calc.aggregate_sample_level_predictions_by_max_support(all_results)
            sample_metrics = self.metrics_calc.calculate_single_run_metrics(
                sample_y_true, sample_y_pred, f"Sample-level Run {run_idx + 1}"
            )
            sample_metrics_runs.append(sample_metrics)
            
            # Calculate overall-level metrics
            overall_y_true, overall_y_pred = aggregate_assignment(all_results, gold_data)
            
            overall_metrics = {}
            if overall_y_true:
                overall_metrics = self.metrics_calc.calculate_single_run_metrics(
                    overall_y_true, overall_y_pred, f"Overall Run {run_idx + 1}"
                )
            overall_metrics_runs.append(overall_metrics)
            
            all_predictions_runs.append(all_results)
            all_batch_stats_runs.extend(batch_stats_list)  # Collect all batch stats across runs
        
        # Calculate statistics across runs
        nugget_statistics = self.metrics_calc.calculate_multi_run_statistics(nugget_metrics_runs)
        sample_statistics = self.metrics_calc.calculate_multi_run_statistics(sample_metrics_runs)
        overall_statistics = self.metrics_calc.calculate_multi_run_statistics(overall_metrics_runs)
        
        # Calculate extended statistics (pass@k, avg@k, maj@k)
        extended_statistics = self.metrics_calc.calculate_multi_run_extended_statistics(
            all_predictions_runs, self.config
        )
        
        # Print results with clear level distinction
        self.metrics_calc.print_multi_run_statistics(nugget_statistics, "Nugget-Level")
        self.metrics_calc.print_multi_run_statistics(sample_statistics, "Sample-Level (Max Support)")
        if overall_statistics.get("valid_metrics_stats") or overall_statistics.get("all_metrics_stats"):  # Print overall match statistics (still nugget-based)
            self.metrics_calc.print_multi_run_statistics(overall_statistics, "Overall Match (Nugget-Level)")
        
        # Print extended statistics (both nugget and sample level)
        self.metrics_calc.print_extended_statistics(extended_statistics, "Nugget-Level")
        self.metrics_calc.print_extended_statistics(extended_statistics, "Sample-Level")
        
        # Print detailed error analysis if enabled and we have batch statistics
        if (getattr(self.config.logging, 'show_error_breakdown', False) or
            getattr(self.config.logging, 'show_format_analysis', False)) and all_batch_stats_runs:
            self._print_error_analysis_report(all_batch_stats_runs)
        
        # Save results if requested (batch save without interrupting reports)
        saved_files = []
        if self.config.logging.save_predictions:
            for run_idx, predictions in enumerate(all_predictions_runs):
                filepath = self._save_predictions_silent(predictions, f"multi_run_{run_idx + 1}")
                saved_files.append(filepath)
        
        # Report saved files after all statistics
        if saved_files:
            print("\n📁 Prediction Files Saved:")
            for filepath in saved_files:
                print(f"  • {filepath}")
        
        return {
            "type": "multi_run",
            "num_runs": self.config.evaluation.num_runs,
            "nugget_statistics": nugget_statistics,
            "sample_statistics": sample_statistics,
            "overall_statistics": overall_statistics,
            "extended_statistics": extended_statistics,
            "individual_metrics": {
                "nugget": nugget_metrics_runs,
                "sample": sample_metrics_runs,
                "overall": overall_metrics_runs
            },
            "predictions": all_predictions_runs,
            "batch_stats": all_batch_stats_runs  # Include all batch statistics for analysis
        }
    
    def _run_predictions(self, input_data: List[Dict], run_idx: int = 0) -> Tuple[List[Dict], int, List[Dict[str, Any]]]:
        """Run predictions on input data with enhanced error tracking"""
        all_results = []
        total_batches = 0
        all_batch_stats = []  # Collect all batch statistics for analysis
        
        # Prepare model configuration
        model_config = {
            "temperature": self.config.model.temperature,
            "top_p": self.config.model.top_p,
            "max_tokens": self.config.model.max_tokens,
            "enable_thinking": self.config.model.enable_thinking,
            "prompt_type": getattr(self.config.model, 'prompt_type', 'legacy'),
            "format_type": getattr(self.config.model, 'format_type', 'adaptive')
        }
        
        if self.config.evaluation.num_workers > 1:
            # Parallel processing with worker-specific clients
            with multiprocessing.Pool(
                processes=self.config.evaluation.num_workers,
                initializer=_init_worker,
                initargs=(
                    self.config.model.base_url,
                    self.config.model.api_key,
                    self.config.model.name,
                    str(self.llm_run_dir),
                    self.run_id
                )
            ) as pool:
                # Prepare tasks without sharing the main client
                run_context_base = {"run_id": self.run_id}
                tasks = [
                    (item, self.config.evaluation.batch_size, model_config, run_context_base)
                    for item in input_data
                ]
                
                for result, num_batches, batch_stats_list in tqdm(
                    pool.map(_process_item_worker, tasks),
                    total=len(input_data),
                    desc=f"Running predictions (Run {run_idx + 1})" if run_idx > 0 else "Running predictions"
                ):
                    all_results.extend(result)
                    total_batches += num_batches
                    all_batch_stats.extend(batch_stats_list)
        else:
            # Sequential processing
            for item in tqdm(
                input_data, 
                desc=f"Running predictions (Run {run_idx + 1})" if run_idx > 0 else "Running predictions"
            ):
                result, num_batches, batch_stats_list = process_single_item(
                    self.model_client, item, self.config.evaluation.batch_size, model_config, {"run_id": self.run_id}
                )
                all_results.extend(result)
                total_batches += num_batches
                all_batch_stats.extend(batch_stats_list)
        
        return all_results, total_batches, all_batch_stats
    
    def _save_predictions(self, predictions: List[Dict], run_name: str):
        """Save predictions to file"""
        from .constants import RunIdConfig
        timestamp = datetime.now().strftime(RunIdConfig.TIMESTAMP_FORMAT)
        filename = f"{timestamp}_{run_name}_predictions.jsonl"
        filepath = self.eval_run_dir / filename
        
        save_jsonl_data(predictions, str(filepath))
        self.logger.info(f"Predictions saved to: {filepath}")
    
    def _save_predictions_silent(self, predictions: List[Dict], run_name: str) -> str:
        """Save predictions to file silently and return filepath"""
        from .constants import RunIdConfig
        timestamp = datetime.now().strftime(RunIdConfig.TIMESTAMP_FORMAT)
        filename = f"{timestamp}_{run_name}_predictions.jsonl"
        filepath = self.eval_run_dir / filename
        
        save_jsonl_data(predictions, str(filepath))
        return str(filepath)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("nugget_evaluator")
        logger.setLevel(getattr(logging, self.config.logging.log_level))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        from .constants import RunIdConfig
        timestamp = datetime.now().strftime(RunIdConfig.TIMESTAMP_FORMAT)
        log_filename = f"{timestamp}_{self.run_id}_evaluation.log"
        log_filepath = self.eval_run_dir / log_filename
        
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, self.config.logging.log_level))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Save run meta for traceability
        try:
            with open(self.eval_run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump({"run_id": self.run_id, "config": self.config.to_dict()}, f, ensure_ascii=False, indent=2)
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Failed to save run metadata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving run metadata: {e}")
        return logger

    def _finalize_and_gather(self):
        """Close main client and gather worker logs into a single merged JSONL file"""
        try:
            if hasattr(self, "model_client") and self.model_client:
                self.model_client.close()
        except Exception as e:
            self.logger.warning(f"Failed to close model client: {e}")
        merged_path = self._gather_llm_logs()
        if merged_path:
            try:
                import shutil
                shutil.copyfile(merged_path, self.eval_run_dir / "llm_calls.merged.jsonl")
            except (IOError, OSError, PermissionError) as e:
                self.logger.warning(f"Failed to copy merged log file: {e}")
            except ImportError:
                self.logger.debug("shutil module not available for log file copying")
            except Exception as e:
                self.logger.error(f"Unexpected error copying merged log file: {e}")

    def _gather_llm_logs(self) -> Optional[str]:
        """Merge all worker and main-process JSONL logs into a single time-ordered file"""
        import json as _json
        import heapq as _heapq
        import glob as _glob
        from datetime import datetime as _dt
        # Find all jsonl files under llm_run_dir (including main and worker subdirs)
        pattern = str(self.llm_run_dir / "**/*llm_calls.jsonl")
        files = _glob.glob(pattern, recursive=True)
        if not files:
            return None
        # Initialize heap with first line of each file
        heap = []
        fps = []
        def parse_ts(ts: str) -> float:
            try:
                return _dt.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError) as e:
                # Log parsing issues for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to parse timestamp '{ts}': {e}")
                return float("inf")
        for idx, path in enumerate(files):
            try:
                fp = open(path, "r", encoding="utf-8")
                line = fp.readline()
                if line:
                    obj = _json.loads(line)
                    ts = obj.get("timestamp") or obj.get("time") or ""
                    _heapq.heappush(heap, (parse_ts(ts), idx, obj, fp))
                    fps.append(fp)
                else:
                    fp.close()
            except (IOError, OSError, PermissionError) as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to read log file {path}: {e}")
                continue
            except (_json.JSONDecodeError, KeyError, ValueError) as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to parse log entry in {path}: {e}")
                try:
                    fp.close()
                except:
                    pass
                continue
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Unexpected error processing log file {path}: {e}")
                try:
                    fp.close()
                except:
                    pass
                continue
        from .constants import RunIdConfig
        out_name = _dt.now().strftime(RunIdConfig.TIMESTAMP_FORMAT) + f"_{self.run_id}_llm_calls.merged.jsonl"
        out_path = str(self.llm_run_dir / out_name)
        with open(out_path, "w", encoding="utf-8") as fout:
            while heap:
                _, idx, obj, fp = _heapq.heappop(heap)
                fout.write(_json.dumps(obj, ensure_ascii=False) + "\n")
                nxt = fp.readline()
                if nxt:
                    try:
                        o2 = _json.loads(nxt)
                        ts2 = o2.get("timestamp") or o2.get("time") or ""
                        _heapq.heappush(heap, (parse_ts(ts2), idx, o2, fp))
                    except (_json.JSONDecodeError, KeyError, ValueError) as e:
                        logger = logging.getLogger(__name__)
                        logger.debug(f"Failed to parse next log entry: {e}")
                        pass
                else:
                    try:
                        fp.close()
                    except Exception:
                        pass
        return out_path
    
    def _print_error_analysis_report(self, batch_stats_collection: List[Dict[str, Any]]):
        """Print comprehensive error analysis report"""
        if not batch_stats_collection:
            return
        
        print("\n--- Detailed Error Analysis Report ---")
        
        total_batches = len(batch_stats_collection)
        
        # 1. Batch Status Summary
        status_counts = defaultdict(int)
        for stats in batch_stats_collection:
            status = stats.get("batch_status", "unknown")
            status_counts[status] += 1
        
        print(f"  Total Batches: {total_batches}")
        print(f"  \n  Batch Status Distribution:")
        for status, count in sorted(status_counts.items()):
            percentage = (count / total_batches) * 100
            print(f"    {status:<20}: {count:4} batches ({percentage:.1f}%)")
        
        # 2. Format Usage Analysis
        format_counts = defaultdict(int)
        for stats in batch_stats_collection:
            format_used = stats.get("format_used", "unknown")
            format_counts[format_used] += 1
        
        print(f"\n  Output Format Usage:")
        for fmt, count in sorted(format_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_batches) * 100
            print(f"    {fmt:<20}: {count:4} batches ({percentage:.1f}%)")
        
        # 3. Error Type Analysis
        error_type_counts = defaultdict(int)
        for stats in batch_stats_collection:
            error_type = stats.get("error_type")
            if error_type:
                error_type_counts[error_type] += 1
        
        if error_type_counts:
            print(f"\n  Error Type Distribution:")
            for error_type, count in sorted(error_type_counts.items()):
                percentage = (count / total_batches) * 100
                print(f"    {error_type:<20}: {count:4} batches ({percentage:.1f}%)")
        
        # 4. Recovery Rate Analysis
        partial_successes = [s for s in batch_stats_collection 
                            if s.get("batch_status") == "partial_success"]
        
        if partial_successes:
            total_expected = sum(s.get("expected_count", 0) for s in partial_successes)
            total_parsed = sum(s.get("parsed_count", 0) for s in partial_successes)
            
            recovery_rates = [s.get("recovery_rate", 0) for s in partial_successes]
            avg_recovery_rate = np.mean(recovery_rates) if recovery_rates else 0.0
            
            print(f"\n  Partial Success Recovery Analysis:")
            print(f"    Partial Success Batches: {len(partial_successes)}")
            print(f"    Total Expected Nuggets: {total_expected}")
            print(f"    Total Recovered Nuggets: {total_parsed}")
            print(f"    Overall Recovery Rate: {(total_parsed / total_expected * 100) if total_expected > 0 else 0:.1f}%")
            print(f"    Average Recovery Rate: {avg_recovery_rate * 100:.1f}%")
        
        # 5. Performance Impact Summary
        success_count = status_counts.get("success", 0)
        partial_count = status_counts.get("partial_success", 0)
        failure_count = total_batches - success_count - partial_count
        
        print(f"\n  Performance Impact Summary:")
        print(f"    Complete Success Rate: {(success_count / total_batches * 100):.1f}%")
        print(f"    Partial Success Rate:  {(partial_count / total_batches * 100):.1f}%")
        print(f"    Failure Rate:          {(failure_count / total_batches * 100):.1f}%")
        
        # Calculate data recovery statistics
        total_expected_all = sum(s.get("expected_count", 0) for s in batch_stats_collection)
        total_parsed_all = sum(s.get("parsed_count", 0) for s in batch_stats_collection)
        
        if total_expected_all > 0:
            overall_data_recovery = (total_parsed_all / total_expected_all) * 100
            print(f"    Overall Data Recovery: {overall_data_recovery:.1f}% ({total_parsed_all}/{total_expected_all} nuggets)")
        
        print()