"""Extended metrics calculation for nugget evaluation"""

import numpy as np
from math import sqrt
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from sklearn.metrics import (
    precision_recall_fscore_support, 
    accuracy_score, 
    classification_report, 
    confusion_matrix
)


LABELS_ORDERED = ['support', 'partial_support', 'not_support']


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics"""
    
    def __init__(self):
        pass
    
    def calculate_single_run_metrics(self, y_true: List[str], y_pred: List[str], 
                                   name: str = "", batch_stats: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Calculate metrics for a single run with dual metric support
        
        Calculates both:
        1. Metrics on valid predictions only (excluding None/"error")
        2. Metrics on all predictions (mapping None/"error" to special categories)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            name: Name for this evaluation
            batch_stats: Optional list of batch statistics for error analysis
        """
        # Count different types of predictions
        total_predictions = len(y_pred)
        valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) 
                       if p is not None and p != "error"]
        valid_predictions = len(valid_pairs)
        error_predictions = sum(1 for p in y_pred if p == "error")
        none_predictions = sum(1 for p in y_pred if p is None)
        
        # Calculate validity rate
        validity_rate = valid_predictions / total_predictions if total_predictions > 0 else 0.0
        
        if total_predictions == 0:
            return {
                "error": "No predictions",
                "total_predictions": 0,
                "valid_predictions": 0,
                "error_predictions": 0,
                "none_predictions": 0,
                "validity_rate": 0.0
            }
        
        # === METRICS ON VALID PREDICTIONS ONLY ===
        valid_metrics = {}
        if valid_pairs:
            y_true_filtered, y_pred_filtered = zip(*valid_pairs)
            
            # Basic metrics on valid predictions
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_filtered, y_pred_filtered, average='micro', zero_division=0
            )
            acc = accuracy_score(y_true_filtered, y_pred_filtered)
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true_filtered, y_pred_filtered, average='macro', zero_division=0
            )
            
            # Per-class metrics on valid predictions
            report = classification_report(
                y_true_filtered, y_pred_filtered, output_dict=True, zero_division=0
            )
            
            # Confusion matrix on valid predictions
            cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=LABELS_ORDERED)
            
            valid_metrics = {
                "micro_accuracy": acc,
                "micro_precision": p,
                "micro_recall": r,
                "micro_f1": f1,
                "macro_precision": p_macro,
                "macro_recall": r_macro,
                "macro_f1": f1_macro,
                "confusion_matrix": cm.tolist(),
                "per_class_metrics": {}
            }
            
            # Add per-class metrics for valid predictions
            for label in LABELS_ORDERED:
                if label in report:
                    valid_metrics["per_class_metrics"][label] = {
                        "precision": report[label]["precision"],
                        "recall": report[label]["recall"],
                        "f1": report[label]["f1-score"],
                        "support": report[label]["support"]
                    }
        
        # === METRICS ON ALL PREDICTIONS ===
        # Create extended label sets including error categories
        extended_labels = LABELS_ORDERED + ["_none", "_error"]
        
        # Map predictions: None -> "_none", "error" -> "_error", others unchanged
        y_pred_extended = []
        for p in y_pred:
            if p is None:
                y_pred_extended.append("_none")
            elif p == "error":
                y_pred_extended.append("_error")
            else:
                y_pred_extended.append(p)
        
        # Ground truth remains the same (no None or "error" in ground truth)
        y_true_extended = y_true[:]
        
        # Basic metrics on all predictions
        p_all, r_all, f1_all, _ = precision_recall_fscore_support(
            y_true_extended, y_pred_extended, average='micro', zero_division=0
        )
        acc_all = accuracy_score(y_true_extended, y_pred_extended)
        p_macro_all, r_macro_all, f1_macro_all, _ = precision_recall_fscore_support(
            y_true_extended, y_pred_extended, average='macro', zero_division=0
        )
        
        # Per-class metrics on all predictions
        report_all = classification_report(
            y_true_extended, y_pred_extended, output_dict=True, zero_division=0,
            labels=extended_labels
        )
        
        # Confusion matrix on all predictions
        cm_all = confusion_matrix(y_true_extended, y_pred_extended, labels=extended_labels)
        
        all_metrics = {
            "micro_accuracy": acc_all,
            "micro_precision": p_all,
            "micro_recall": r_all,
            "micro_f1": f1_all,
            "macro_precision": p_macro_all,
            "macro_recall": r_macro_all,
            "macro_f1": f1_macro_all,
            "confusion_matrix": cm_all.tolist(),
            "per_class_metrics": {}
        }
        
        # Add per-class metrics for all predictions
        for label in extended_labels:
            if label in report_all:
                all_metrics["per_class_metrics"][label] = {
                    "precision": report_all[label]["precision"],
                    "recall": report_all[label]["recall"],
                    "f1": report_all[label]["f1-score"],
                    "support": report_all[label]["support"]
                }
        
        # === COMBINE RESULTS ===
        metrics = {
            "name": name,
            "num_predictions": valid_predictions,
            "num_total": total_predictions,
            "error_predictions": error_predictions,
            "none_predictions": none_predictions,
            "validity_rate": validity_rate,
            
            # Metrics on valid predictions only
            "valid_metrics": valid_metrics,
            
            # Metrics on all predictions
            "all_metrics": all_metrics
        }
        
        # Add batch statistics analysis if available
        if batch_stats:
            batch_analysis = self._analyze_batch_statistics(batch_stats)
            metrics.update(batch_analysis)
        
        return metrics
    
    def _analyze_batch_statistics(self, batch_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch statistics to provide detailed error breakdown"""
        if not batch_stats:
            return {}
        
        total_batches = len(batch_stats)
        
        # Count different batch statuses
        status_counts = defaultdict(int)
        format_usage = defaultdict(int)
        error_type_counts = defaultdict(int)
        
        total_expected = 0
        total_parsed = 0
        recovery_rates = []
        
        for stats in batch_stats:
            # Count batch status
            status = stats.get("batch_status", "unknown")
            status_counts[status] += 1
            
            # Count format usage
            format_used = stats.get("format_used", "unknown")
            format_usage[format_used] += 1
            
            # Count error types
            error_type = stats.get("error_type")
            if error_type:
                error_type_counts[error_type] += 1
            
            # Calculate recovery statistics
            expected = stats.get("expected_count", 0)
            parsed = stats.get("parsed_count", 0)
            total_expected += expected
            total_parsed += parsed
            
            recovery_rate = stats.get("recovery_rate", 0)
            recovery_rates.append(recovery_rate)
        
        # Calculate overall statistics
        overall_recovery_rate = total_parsed / total_expected if total_expected > 0 else 0.0
        avg_recovery_rate = np.mean(recovery_rates) if recovery_rates else 0.0
        
        # Calculate success rates
        success_rate = status_counts.get("success", 0) / total_batches
        partial_success_rate = status_counts.get("partial_success", 0) / total_batches
        failure_rate = (status_counts.get("complete_failure", 0) + 
                       status_counts.get("api_error", 0) + 
                       status_counts.get("unexpected_error", 0)) / total_batches
        
        return {
            "batch_analysis": {
                "total_batches": total_batches,
                "success_rate": success_rate,
                "partial_success_rate": partial_success_rate,
                "failure_rate": failure_rate,
                "overall_recovery_rate": overall_recovery_rate,
                "avg_recovery_rate": avg_recovery_rate,
                "status_distribution": dict(status_counts),
                "format_distribution": dict(format_usage),
                "error_type_distribution": dict(error_type_counts)
            }
        }
    
    def calculate_multi_run_statistics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics across multiple runs with dual metric support"""
        if not metrics_list:
            return {"error": "No metrics to analyze"}
        
        n_runs = len(metrics_list)
        
        # Collect all metric values for both valid and all predictions
        valid_metrics_all = defaultdict(list)
        all_metrics_all = defaultdict(list)
        valid_cms = []
        all_cms = []
        
        for run_metrics in metrics_list:
            if "error" in run_metrics:
                continue
            
            # === COLLECT VALID PREDICTION METRICS ===
            valid_metrics = run_metrics.get('valid_metrics', {})
            if valid_metrics:
                # Collect scalar metrics for valid predictions
                for key in ["micro_accuracy", "micro_precision", "micro_recall", "micro_f1",
                           "macro_precision", "macro_recall", "macro_f1"]:
                    if key in valid_metrics:
                        valid_metrics_all[key].append(valid_metrics[key])
                
                # Collect per-class metrics for valid predictions
                if "per_class_metrics" in valid_metrics:
                    for label, class_metrics in valid_metrics["per_class_metrics"].items():
                        for metric_name, value in class_metrics.items():
                            if metric_name != "support":  # Skip support counts
                                valid_metrics_all[f"{label}_{metric_name}"].append(value)
                
                # Collect confusion matrices for valid predictions
                if "confusion_matrix" in valid_metrics:
                    valid_cms.append(np.array(valid_metrics["confusion_matrix"]))
            
            # === COLLECT ALL PREDICTION METRICS ===
            all_metrics = run_metrics.get('all_metrics', {})
            if all_metrics:
                # Collect scalar metrics for all predictions
                for key in ["micro_accuracy", "micro_precision", "micro_recall", "micro_f1",
                           "macro_precision", "macro_recall", "macro_f1"]:
                    if key in all_metrics:
                        all_metrics_all[key].append(all_metrics[key])
                
                # Collect per-class metrics for all predictions
                if "per_class_metrics" in all_metrics:
                    for label, class_metrics in all_metrics["per_class_metrics"].items():
                        for metric_name, value in class_metrics.items():
                            if metric_name != "support":  # Skip support counts
                                all_metrics_all[f"{label}_{metric_name}"].append(value)
                
                # Collect confusion matrices for all predictions
                if "confusion_matrix" in all_metrics:
                    all_cms.append(np.array(all_metrics["confusion_matrix"]))
        
        # === CALCULATE STATISTICS ===
        statistics = {
            "num_runs": n_runs,
            "valid_metrics_stats": {},
            "all_metrics_stats": {},
            "valid_confusion_matrix_stats": {},
            "all_confusion_matrix_stats": {}
        }
        
        # Calculate statistics for valid prediction metrics
        for metric_name, values in valid_metrics_all.items():
            if values:
                mean, ci_half_width = self._calculate_confidence_interval(values)
                std_dev = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                statistics["valid_metrics_stats"][metric_name] = {
                    "mean": mean,
                    "std": std_dev,
                    "ci_lower": mean - ci_half_width,
                    "ci_upper": mean + ci_half_width,
                    "values": values
                }
        
        # Calculate statistics for all prediction metrics
        for metric_name, values in all_metrics_all.items():
            if values:
                mean, ci_half_width = self._calculate_confidence_interval(values)
                std_dev = np.std(values, ddof=1) if len(values) > 1 else 0.0
                
                statistics["all_metrics_stats"][metric_name] = {
                    "mean": mean,
                    "std": std_dev,
                    "ci_lower": mean - ci_half_width,
                    "ci_upper": mean + ci_half_width,
                    "values": values
                }
        
        # Calculate confusion matrix statistics for valid predictions
        if valid_cms:
            avg_cm = np.mean(valid_cms, axis=0)
            std_cm = np.std(valid_cms, axis=0, ddof=1) if len(valid_cms) > 1 else np.zeros_like(avg_cm)
            
            statistics["valid_confusion_matrix_stats"] = {
                "average": avg_cm.tolist(),
                "std": std_cm.tolist(),
                "labels": LABELS_ORDERED
            }
        
        # Calculate confusion matrix statistics for all predictions
        if all_cms:
            avg_cm_all = np.mean(all_cms, axis=0)
            std_cm_all = np.std(all_cms, axis=0, ddof=1) if len(all_cms) > 1 else np.zeros_like(avg_cm_all)
            extended_labels = LABELS_ORDERED + ["_none", "_error"]
            
            statistics["all_confusion_matrix_stats"] = {
                "average": avg_cm_all.tolist(),
                "std": std_cm_all.tolist(),
                "labels": extended_labels
            }
        
        return statistics
    
    def calculate_multi_run_extended_statistics(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                                              config) -> Dict[str, Any]:
        """
        Calculate extended statistics including pass@k, avg@k, and maj@k for multi-run evaluation
        Support both nugget-level and sample-level calculations
        """
        if len(all_predictions_runs) <= 1:
            return {"error": "Extended statistics require multiple runs"}
        
        extended_stats = {}
        
        # Nugget-level calculations (original behavior)
        if config.metrics.pass_k:
            pass_at_k_results = self.calculate_pass_at_k(all_predictions_runs, config.metrics.pass_k)
            extended_stats.update(pass_at_k_results)
        
        if config.metrics.avg_k:
            avg_at_k_results = self.calculate_avg_at_k(all_predictions_runs, config.metrics.avg_k)
            extended_stats.update(avg_at_k_results)
        
        if config.metrics.maj_k:
            maj_at_k_results = self.calculate_majority_voting_at_k(all_predictions_runs, config.metrics.maj_k)
            extended_stats.update(maj_at_k_results)
        
        # Sample-level calculations (new feature)
        if config.metrics.pass_k:
            sample_pass_at_k = self.calculate_sample_level_pass_at_k(all_predictions_runs, config.metrics.pass_k)
            # Add _sample suffix to distinguish from nugget-level
            for k, v in sample_pass_at_k.items():
                extended_stats[f"{k}_sample"] = v
        
        if config.metrics.avg_k:
            sample_avg_at_k = self.calculate_sample_level_avg_at_k(all_predictions_runs, config.metrics.avg_k)
            for k, v in sample_avg_at_k.items():
                extended_stats[f"{k}_sample"] = v
        
        if config.metrics.maj_k:
            sample_maj_at_k = self.calculate_sample_level_majority_voting_at_k(all_predictions_runs, config.metrics.maj_k)
            for k, v in sample_maj_at_k.items():
                extended_stats[f"{k}_sample"] = v
        
        return extended_stats
    
    def calculate_sample_level_pass_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                                       k_values: List[int]) -> Dict[str, float]:
        """
        Calculate pass@k at sample (query) level: probability that at least one of k runs 
        gets ALL nuggets correct for a query
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by qid (sample level)
        sample_results = defaultdict(list)  # qid -> [run1_all_correct, run2_all_correct, ...]
        
        for run_predictions in all_predictions_runs:
            # Group by qid within this run
            run_by_qid = defaultdict(list)
            for pred in run_predictions:
                run_by_qid[pred['qid']].append(pred)
            
            # For each qid, check if ALL nuggets are correct in this run
            for qid, qid_preds in run_by_qid.items():
                all_correct = all(
                    pred['block_pred'] == pred['block_true'] and pred['block_pred'] is not None 
                    for pred in qid_preds
                )
                sample_results[qid].append(all_correct)
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
                
            pass_at_k_scores = []
            
            # For each sample, calculate pass@k
            for qid, correctness_list in sample_results.items():
                # Take first k runs for this sample
                k_results = correctness_list[:k]
                # pass@k = 1 if at least one run got all nuggets correct, 0 otherwise
                pass_at_k_sample = 1.0 if any(k_results) else 0.0
                pass_at_k_scores.append(pass_at_k_sample)
            
            # Average pass@k across all samples
            results[f'pass@{k}'] = np.mean(pass_at_k_scores) if pass_at_k_scores else 0.0
        
        return results
    
    def calculate_sample_level_avg_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                                      k_values: List[int]) -> Dict[str, float]:
        """
        Calculate avg@k at sample (query) level: average probability that a sample 
        is completely correct (all nuggets correct) over k runs
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by qid (sample level)
        sample_results = defaultdict(list)  # qid -> [run1_all_correct, run2_all_correct, ...]
        
        for run_predictions in all_predictions_runs:
            # Group by qid within this run
            run_by_qid = defaultdict(list)
            for pred in run_predictions:
                run_by_qid[pred['qid']].append(pred)
            
            # For each qid, check if ALL nuggets are correct in this run
            for qid, qid_preds in run_by_qid.items():
                all_correct = all(
                    pred['block_pred'] == pred['block_true'] and pred['block_pred'] is not None 
                    for pred in qid_preds
                )
                # Convert boolean to float (1.0 if all correct, 0.0 otherwise)
                sample_results[qid].append(1.0 if all_correct else 0.0)
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
                
            avg_at_k_scores = []
            
            # For each sample, calculate avg@k
            for qid, correctness_list in sample_results.items():
                # Take first k runs for this sample
                k_results = correctness_list[:k]
                # avg@k = average probability of complete correctness over k runs
                avg_at_k_sample = np.mean(k_results) if k_results else 0.0
                avg_at_k_scores.append(avg_at_k_sample)
            
            # Average avg@k across all samples
            results[f'avg@{k}'] = np.mean(avg_at_k_scores) if avg_at_k_scores else 0.0
        
        return results
    
    def calculate_sample_level_majority_voting_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                                                  k_values: List[int]) -> Dict[str, float]:
        """
        Calculate maj@k at sample (query) level: for each nugget in a query, use majority voting 
        across k runs, then check if ALL nuggets are correct (sample completely correct probability)
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by qid and nugget_text
        sample_nugget_results = defaultdict(lambda: defaultdict(list))  # qid -> nugget_text -> [pred1, pred2, ...]
        sample_ground_truth = defaultdict(dict)  # qid -> nugget_text -> true_label
        
        for run_predictions in all_predictions_runs:
            for pred in run_predictions:
                qid = pred['qid']
                nugget_text = pred['nugget_text']
                sample_nugget_results[qid][nugget_text].append(pred['block_pred'])
                sample_ground_truth[qid][nugget_text] = pred['block_true']
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
            
            sample_correctness = []
            
            for qid, nugget_dict in sample_nugget_results.items():
                all_nuggets_correct = True
                has_valid_nuggets = False
                
                for nugget_text, predictions in nugget_dict.items():
                    # Take first k predictions for this nugget
                    k_predictions = predictions[:k]
                    
                    # Majority voting - exclude None predictions
                    valid_predictions = [p for p in k_predictions if p is not None]
                    
                    if valid_predictions:
                        has_valid_nuggets = True
                        vote_counts = Counter(valid_predictions)
                        majority_pred = vote_counts.most_common(1)[0][0]
                        true_label = sample_ground_truth[qid][nugget_text]
                        
                        if majority_pred != true_label:
                            all_nuggets_correct = False
                            break  # Early exit - sample is not completely correct
                    else:
                        # If no valid predictions for any nugget, sample is not correct
                        all_nuggets_correct = False
                        break
                
                # Only add to results if we have valid nuggets to evaluate
                if has_valid_nuggets:
                    sample_correctness.append(1.0 if all_nuggets_correct else 0.0)
            
            # Calculate probability of complete sample correctness
            if sample_correctness:
                results[f'maj@{k}'] = np.mean(sample_correctness)
            else:
                results[f'maj@{k}'] = 0.0
        
        return results
    
    def calculate_pass_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                           k_values: List[int]) -> Dict[str, float]:
        """
        Calculate pass@k: probability that at least one of k runs is correct for each sample
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by (qid, nugget_text) - sample level
        sample_results = defaultdict(list)
        
        for run_predictions in all_predictions_runs:
            for pred in run_predictions:
                key = (pred['qid'], pred['nugget_text'])
                is_correct = (pred['block_pred'] == pred['block_true'] and pred['block_pred'] is not None)
                sample_results[key].append(is_correct)
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
                
            pass_at_k_scores = []
            
            # For each sample, calculate pass@k
            for sample_key, correctness_list in sample_results.items():
                # Take first k runs for this sample
                k_results = correctness_list[:k]
                # pass@k = 1 if at least one is correct, 0 otherwise
                pass_at_k_sample = 1.0 if any(k_results) else 0.0
                pass_at_k_scores.append(pass_at_k_sample)
            
            # Average pass@k across all samples
            results[f'pass@{k}'] = np.mean(pass_at_k_scores) if pass_at_k_scores else 0.0
        
        return results
    
    def calculate_avg_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                          k_values: List[int]) -> Dict[str, float]:
        """
        Calculate avg@k: average accuracy over k runs for each sample
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by (qid, nugget_text)
        sample_results = defaultdict(list)
        
        for run_predictions in all_predictions_runs:
            for pred in run_predictions:
                key = (pred['qid'], pred['nugget_text'])
                is_correct = (pred['block_pred'] == pred['block_true'] and pred['block_pred'] is not None)
                sample_results[key].append(is_correct)
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
                
            avg_at_k_scores = []
            
            # For each sample, calculate avg@k
            for sample_key, correctness_list in sample_results.items():
                # Take first k runs for this sample  
                k_results = correctness_list[:k]
                # avg@k = average correctness over k runs
                avg_at_k_sample = np.mean(k_results) if k_results else 0.0
                avg_at_k_scores.append(avg_at_k_sample)
            
            # Average avg@k across all samples
            results[f'avg@{k}'] = np.mean(avg_at_k_scores) if avg_at_k_scores else 0.0
        
        return results
    
    def calculate_majority_voting_at_k(self, all_predictions_runs: List[List[Dict[str, Any]]], 
                                      k_values: List[int]) -> Dict[str, float]:
        """
        Calculate maj@k: simple majority voting accuracy over k runs
        """
        total_runs = len(all_predictions_runs)
        
        # Group predictions by (qid, nugget_text)
        sample_results = defaultdict(list)
        sample_ground_truth = {}
        
        for run_predictions in all_predictions_runs:
            for pred in run_predictions:
                key = (pred['qid'], pred['nugget_text'])
                sample_results[key].append(pred['block_pred'])
                sample_ground_truth[key] = pred['block_true']  # Ground truth is same across runs
        
        results = {}
        
        for k in k_values:
            if k > total_runs:
                continue
                
            correct_predictions = 0
            total_predictions = 0
            
            for sample_key, predictions in sample_results.items():
                # Take first k predictions for this sample
                k_predictions = predictions[:k]
                
                # Simple majority voting - exclude None predictions
                valid_predictions = [p for p in k_predictions if p is not None]
                
                if valid_predictions:
                    vote_counts = Counter(valid_predictions)
                    # Get most common prediction (ties resolved by first occurrence)
                    majority_pred = vote_counts.most_common(1)[0][0]
                    true_label = sample_ground_truth[sample_key]
                    
                    if majority_pred == true_label:
                        correct_predictions += 1
                    total_predictions += 1
            
            if total_predictions > 0:
                results[f'maj@{k}'] = correct_predictions / total_predictions
            else:
                results[f'maj@{k}'] = 0.0
        
        return results
    
    def print_single_run_metrics(self, metrics: Dict[str, Any]):
        """Pretty print single run metrics with dual metric support"""
        if "error" in metrics:
            print(f"Error in metrics: {metrics['error']}")
            if "total_predictions" in metrics:
                print(f"  Total Predictions: {metrics['total_predictions']}")
                print(f"  Error Predictions: {metrics.get('error_predictions', 0)}")
                print(f"  None Predictions: {metrics.get('none_predictions', 0)}")
            return
        
        name = metrics.get("name", "")
        print(f"\n--- {name} Evaluation ---")
        
        # Basic prediction statistics
        total_preds = metrics['num_total']
        valid_preds = metrics['num_predictions']
        error_preds = metrics.get('error_predictions', 0)
        none_preds = metrics.get('none_predictions', 0)
        validity_rate = metrics.get('validity_rate', 0)
        
        print(f"  Total Predictions: {total_preds}")
        print(f"  Valid Predictions: {valid_preds} ({validity_rate:.1%})")
        print(f"  Error Predictions: {error_preds}")
        print(f"  None Predictions:  {none_preds}")
        print()
        
        # === PERFORMANCE METRICS ON VALID PREDICTIONS ===
        valid_metrics = metrics.get('valid_metrics', {})
        if valid_metrics and valid_preds > 0:
            print(f"  Performance Metrics (on valid predictions):")
            print(f"    Micro Accuracy:  {valid_metrics['micro_accuracy']:.4f}")
            print(f"    Micro Precision: {valid_metrics['micro_precision']:.4f}")
            print(f"    Micro Recall:    {valid_metrics['micro_recall']:.4f}")
            print(f"    Micro F1 Score:  {valid_metrics['micro_f1']:.4f}")
            print(f"    Macro Precision: {valid_metrics['macro_precision']:.4f}")
            print(f"    Macro Recall:    {valid_metrics['macro_recall']:.4f}")
            print(f"    Macro F1 Score:  {valid_metrics['macro_f1']:.4f}")
        else:
            print(f"  Performance Metrics (on valid predictions): N/A (no valid predictions)")
        
        # === PERFORMANCE METRICS ON ALL PREDICTIONS ===
        all_metrics = metrics.get('all_metrics', {})
        if all_metrics:
            print(f"\n  Performance Metrics (on all predictions):")
            print(f"    Micro Accuracy:  {all_metrics['micro_accuracy']:.4f}")
            print(f"    Micro Precision: {all_metrics['micro_precision']:.4f}")
            print(f"    Micro Recall:    {all_metrics['micro_recall']:.4f}")
            print(f"    Micro F1 Score:  {all_metrics['micro_f1']:.4f}")
            print(f"    Macro Precision: {all_metrics['macro_precision']:.4f}")
            print(f"    Macro Recall:    {all_metrics['macro_recall']:.4f}")
            print(f"    Macro F1 Score:  {all_metrics['macro_f1']:.4f}")
        
        # === PER-CLASS METRICS (on valid predictions) ===
        if valid_metrics.get("per_class_metrics") and valid_preds > 0:
            print("\n  Per-class Metrics (on valid predictions):")
            for label, class_metrics in valid_metrics["per_class_metrics"].items():
                print(f"    {label:<15}: "
                      f"precision={class_metrics['precision']:.4f}, "
                      f"recall={class_metrics['recall']:.4f}, "
                      f"f1-score={class_metrics['f1']:.4f}, "
                      f"support={class_metrics['support']}")
        
        # === PER-CLASS METRICS (on all predictions) ===
        if all_metrics.get("per_class_metrics"):
            print("\n  Per-class Metrics (on all predictions):")
            for label, class_metrics in all_metrics["per_class_metrics"].items():
                if label.startswith("_"):  # Error categories
                    continue  # Skip error categories in per-class display for brevity
                print(f"    {label:<15}: "
                      f"precision={class_metrics['precision']:.4f}, "
                      f"recall={class_metrics['recall']:.4f}, "
                      f"f1-score={class_metrics['f1']:.4f}, "
                      f"support={class_metrics['support']}")
        
        # === CONFUSION MATRIX (on valid predictions) ===
        if valid_metrics.get("confusion_matrix") and valid_preds > 0:
            cm = np.array(valid_metrics["confusion_matrix"])
            print("\n  Confusion Matrix (valid predictions only):")
            header_label = "True\\Pred"
            print(f"    {header_label:<15}", end="")
            for label in LABELS_ORDERED:
                print(f"{label:>15}", end="")
            print()
            
            for i, true_label in enumerate(LABELS_ORDERED):
                print(f"    {true_label:<15}", end="")
                for j in range(len(LABELS_ORDERED)):
                    print(f"{cm[i, j]:>15}", end="")
                print()
        
        # === CONFUSION MATRIX (on all predictions) ===
        if all_metrics.get("confusion_matrix"):
            cm_all = np.array(all_metrics["confusion_matrix"])
            extended_labels = LABELS_ORDERED + ["_none", "_error"]
            print("\n  Confusion Matrix (all predictions):")
            header_label = "True\\Pred"
            print(f"    {header_label:<15}", end="")
            for label in extended_labels:
                print(f"{label:>12}", end="")
            print()
            
            for i, true_label in enumerate(LABELS_ORDERED):  # Only show true labels from original set
                print(f"    {true_label:<15}", end="")
                for j in range(len(extended_labels)):
                    print(f"{cm_all[i, j]:>12}", end="")
                print()
        
        # === BATCH ANALYSIS ===
        if "batch_analysis" in metrics:
            self._print_batch_analysis(metrics["batch_analysis"])
        
        print()
    
    def _print_batch_analysis(self, batch_analysis: Dict[str, Any]):
        """Print detailed batch analysis"""
        print("\n  Batch Analysis:")
        
        ba = batch_analysis
        print(f"    Total Batches: {ba['total_batches']}")
        print(f"    Success Rate: {ba['success_rate']:.1%}")
        print(f"    Partial Success Rate: {ba['partial_success_rate']:.1%}")
        print(f"    Failure Rate: {ba['failure_rate']:.1%}")
        print(f"    Overall Recovery Rate: {ba['overall_recovery_rate']:.1%}")
        
        # Format usage distribution
        if ba.get('format_distribution'):
            print("\n    Format Usage:")
            for fmt, count in sorted(ba['format_distribution'].items()):
                percentage = (count / ba['total_batches']) * 100
                print(f"      {fmt:<15}: {count:3} batches ({percentage:.1f}%)")
        
        # Error type distribution
        if ba.get('error_type_distribution'):
            print("\n    Error Types:")
            for error_type, count in sorted(ba['error_type_distribution'].items()):
                percentage = (count / ba['total_batches']) * 100
                print(f"      {error_type:<15}: {count:3} batches ({percentage:.1f}%)")
    
    def print_multi_run_statistics(self, statistics: Dict[str, Any], level_name: str = ""):
        """Pretty print multi-run statistics with dual metric support"""
        if "error" in statistics:
            print(f"Error in statistics: {statistics['error']}")
            return
        
        n_runs = statistics["num_runs"]
        level_header = f" ({level_name})" if level_name else ""
        print(f"\n--- Multi-Run Statistics ({n_runs} runs){level_header} ---")
        
        # === METRICS ON VALID PREDICTIONS ===
        if "valid_metrics_stats" in statistics and statistics["valid_metrics_stats"]:
            print("\n  Metric Statistics (on valid predictions):")
            for metric_name, stats in statistics["valid_metrics_stats"].items():
                mean = stats["mean"]
                std = stats["std"]
                ci_lower = stats["ci_lower"]
                ci_upper = stats["ci_upper"]
                
                print(f"    {metric_name.replace('_', ' ').title():<28}: "
                      f"Mean = {mean:.4f}, Std = {std:.4f}, "
                      f"95% CI = ({ci_lower:.4f}, {ci_upper:.4f})")
        
        # === METRICS ON ALL PREDICTIONS ===
        if "all_metrics_stats" in statistics and statistics["all_metrics_stats"]:
            print("\n  Metric Statistics (on all predictions):")
            for metric_name, stats in statistics["all_metrics_stats"].items():
                # Skip error categories in main display for brevity
                if "_none_" in metric_name or "_error_" in metric_name:
                    continue
                    
                mean = stats["mean"]
                std = stats["std"]
                ci_lower = stats["ci_lower"]
                ci_upper = stats["ci_upper"]
                
                print(f"    {metric_name.replace('_', ' ').title():<28}: "
                      f"Mean = {mean:.4f}, Std = {std:.4f}, "
                      f"95% CI = ({ci_lower:.4f}, {ci_upper:.4f})")
        
        # === CONFUSION MATRIX FOR VALID PREDICTIONS ===
        if "valid_confusion_matrix_stats" in statistics and statistics["valid_confusion_matrix_stats"]:
            cm_stats = statistics["valid_confusion_matrix_stats"]
            avg_cm = np.array(cm_stats["average"])
            std_cm = np.array(cm_stats["std"])
            labels = cm_stats["labels"]
            
            print("\n  Average Confusion Matrix (valid predictions):")
            print("  Format: mean(std_dev) - averaged across all runs")
            header_label = "True\\Pred"
            print(f"    {header_label:<15}", end="")
            for label in labels:
                print(f"{label:>15}", end="")
            print()
            
            for i, true_label in enumerate(labels):
                print(f"    {true_label:<15}", end="")
                for j, pred_label in enumerate(labels):
                    avg_val = avg_cm[i, j]
                    std_val = std_cm[i, j]
                    print(f"{avg_val:9.1f}({std_val:3.1f})", end="")
                print()
        
        # === CONFUSION MATRIX FOR ALL PREDICTIONS ===
        if "all_confusion_matrix_stats" in statistics and statistics["all_confusion_matrix_stats"]:
            cm_stats_all = statistics["all_confusion_matrix_stats"]
            avg_cm_all = np.array(cm_stats_all["average"])
            std_cm_all = np.array(cm_stats_all["std"])
            extended_labels = cm_stats_all["labels"]
            
            print("\n  Average Confusion Matrix (all predictions):")
            print("  Format: mean(std_dev) - averaged across all runs")
            header_label = "True\\Pred"
            print(f"    {header_label:<15}", end="")
            for label in extended_labels:
                print(f"{label:>12}", end="")
            print()
            
            for i, true_label in enumerate(LABELS_ORDERED):  # Only show true labels from original set
                print(f"    {true_label:<15}", end="")
                for j in range(len(extended_labels)):
                    avg_val = avg_cm_all[i, j]
                    std_val = std_cm_all[i, j]
                    print(f"{avg_val:7.1f}({std_val:3.1f})", end="")
                print()
        
        print()
    
    def print_extended_statistics(self, extended_stats: Dict[str, Any], level_name: str = ""):
        """Pretty print extended statistics (pass@k, avg@k, maj@k)"""
        if "error" in extended_stats:
            print(f"Error in extended statistics: {extended_stats['error']}")
            return
        
        # Separate nugget-level and sample-level metrics
        nugget_metrics = {}
        sample_metrics = {}
        
        for key, value in extended_stats.items():
            if key.endswith('_sample'):
                # Sample-level metric
                clean_key = key.replace('_sample', '')
                sample_metrics[clean_key] = value
            else:
                # Nugget-level metric (default)
                nugget_metrics[key] = value
        
        # Print based on what we have
        if nugget_metrics and level_name == "Nugget-Level":
            self._print_extended_metrics_section(nugget_metrics, "Nugget-Level Extended Statistics")
        elif sample_metrics and level_name == "Sample-Level":
            self._print_extended_metrics_section(sample_metrics, "Sample-Level Extended Statistics")
        elif not level_name:
            # Backward compatibility - print all available metrics
            if nugget_metrics:
                self._print_extended_metrics_section(nugget_metrics, "Extended Multi-Run Statistics")
            if sample_metrics:
                self._print_extended_metrics_section(sample_metrics, "Sample-Level Extended Statistics")
    
    def _print_extended_metrics_section(self, metrics_dict: Dict[str, float], section_title: str):
        """Helper method to print a section of extended metrics"""
        # Group by metric type
        pass_at_k_metrics = {k: v for k, v in metrics_dict.items() if k.startswith('pass@')}
        avg_at_k_metrics = {k: v for k, v in metrics_dict.items() if k.startswith('avg@')}
        maj_at_k_metrics = {k: v for k, v in metrics_dict.items() if k.startswith('maj@')}
        
        print(f"\n--- {section_title} ---")
        
        # Print pass@k metrics
        if pass_at_k_metrics:
            print("  Pass@K (at least one correct):")
            for k, v in sorted(pass_at_k_metrics.items(), key=lambda x: int(x[0].split('@')[1])):
                print(f"    {k}: {v:.4f}")
        
        # Print avg@k metrics  
        if avg_at_k_metrics:
            print("  Avg@K (average accuracy):")
            for k, v in sorted(avg_at_k_metrics.items(), key=lambda x: int(x[0].split('@')[1])):
                print(f"    {k}: {v:.4f}")
        
        # Print maj@k metrics
        if maj_at_k_metrics:
            print("  Maj@K (majority voting):")
            for k, v in sorted(maj_at_k_metrics.items(), key=lambda x: int(x[0].split('@')[1])):
                print(f"    {k}: {v:.4f}")
        
        print()
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values"""
        a = np.array(data)
        n = len(a)
        
        if n < 2:
            return np.mean(a), 0.0
        
        mean = np.mean(a)
        std_err = np.std(a, ddof=1) / sqrt(n)
        
        # Use t-distribution critical value for small samples, normal for large
        if n < 30:
            from scipy import stats
            t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_error = std_err * t_critical
        else:
            # For large samples, use normal approximation
            z_critical = 1.96  # 95% confidence
            margin_error = std_err * z_critical
        
        return mean, margin_error
    
    def aggregate_sample_level_predictions_by_max_support(self, predictions: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Aggregate nugget-level predictions to sample-level using maximum support strategy
        
        For each nugget in each sample (qid):
        - Collect all predictions AND ground truths for that nugget across different blocks
        - Select the prediction with highest support level (support > partial_support > not_support)
        - Select the ground truth with highest support level (support > partial_support > not_support)
        - Return sample-level y_true and y_pred lists for metrics calculation
        
        Args:
            predictions: List of nugget-level prediction dictionaries with keys:
                        'qid', 'nugget_text', 'block_pred', 'block_true'
        
        Returns:
            Tuple of (y_true, y_pred) lists for sample-level metrics
        """
        # Support priority mapping (higher value = higher priority)
        SUPPORT_PRIORITY = {
            "support": 2,
            "partial_support": 1, 
            "not_support": 0,
            None: -1,  # None or error has lowest priority
            "error": -1
        }
        
        # Group predictions and ground truths by qid and nugget_text
        sample_nugget_preds = defaultdict(lambda: defaultdict(list))
        sample_nugget_truths = defaultdict(lambda: defaultdict(list))
        
        for pred in predictions:
            qid = pred['qid']
            nugget_text = pred['nugget_text']
            sample_nugget_preds[qid][nugget_text].append(pred['block_pred'])
            sample_nugget_truths[qid][nugget_text].append(pred['block_true'])
        
        # Aggregate using maximum support strategy for BOTH predictions and ground truth
        sample_y_true = []
        sample_y_pred = []
        
        for qid in sample_nugget_preds:
            for nugget_text in sample_nugget_preds[qid]:
                preds = sample_nugget_preds[qid][nugget_text]
                truths = sample_nugget_truths[qid][nugget_text]
                
                # Select prediction with highest support level
                best_pred = max(preds, key=lambda x: SUPPORT_PRIORITY.get(x, -1))
                
                # Select ground truth with highest support level
                best_true = max(truths, key=lambda x: SUPPORT_PRIORITY.get(x, -1))
                
                sample_y_true.append(best_true)
                sample_y_pred.append(best_pred)
        
        return sample_y_true, sample_y_pred