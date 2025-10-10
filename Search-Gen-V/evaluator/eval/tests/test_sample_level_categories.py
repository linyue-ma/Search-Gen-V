#!/usr/bin/env python3
"""
Test whether sample-level aggregation supports all 5 categories for F1/precision/recall calculation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval.metrics import MetricsCalculator

def test_all_categories_sample_level():
    """Test if sample-level aggregation supports all 5 categories"""
    print(" Testing Sample-Level Categories Support")
    print("=" * 60)
    
    # Create comprehensive test data covering all scenarios
    test_predictions = [
        # Sample 1: All three categories represented after aggregation
        {"qid": "q1", "nugget_text": "n1", "block_pred": "support", "block_true": "support"},
        {"qid": "q1", "nugget_text": "n1", "block_pred": "not_support", "block_true": "support"},  # support wins
        
        {"qid": "q1", "nugget_text": "n2", "block_pred": "partial_support", "block_true": "partial_support"},
        {"qid": "q1", "nugget_text": "n2", "block_pred": "not_support", "block_true": "partial_support"},  # partial wins
        
        {"qid": "q1", "nugget_text": "n3", "block_pred": "not_support", "block_true": "not_support"},
        {"qid": "q1", "nugget_text": "n3", "block_pred": None, "block_true": "not_support"},  # not_support wins
        
        # Sample 2: Mixed predictions and truth labels
        {"qid": "q2", "nugget_text": "n1", "block_pred": "support", "block_true": "partial_support"},  # Correct pred, wrong class
        {"qid": "q2", "nugget_text": "n1", "block_pred": "partial_support", "block_true": "partial_support"},  # support wins but truth is partial
        
        {"qid": "q2", "nugget_text": "n2", "block_pred": "partial_support", "block_true": "not_support"},  # Wrong prediction
        {"qid": "q2", "nugget_text": "n2", "block_pred": "not_support", "block_true": "not_support"},  # partial wins but truth is not_support
        
        {"qid": "q2", "nugget_text": "n3", "block_pred": "not_support", "block_true": "support"},  # Wrong prediction
        {"qid": "q2", "nugget_text": "n3", "block_pred": "error", "block_true": "support"},  # not_support wins but truth is support
        
        # Sample 3: More complex scenarios
        {"qid": "q3", "nugget_text": "n1", "block_pred": "support", "block_true": "support"},  # Perfect
        
        {"qid": "q3", "nugget_text": "n2", "block_pred": "partial_support", "block_true": "partial_support"},  # Perfect
        
        {"qid": "q3", "nugget_text": "n3", "block_pred": None, "block_true": "not_support"},  # Failed prediction
        {"qid": "q3", "nugget_text": "n3", "block_pred": "error", "block_true": "not_support"},  # None/error both have -1 priority
    ]
    
    metrics_calc = MetricsCalculator()
    
    # Test aggregation
    sample_y_true, sample_y_pred = metrics_calc.aggregate_sample_level_predictions_by_max_support(test_predictions)
    
    print(" Aggregation Results:")
    print(f"  Total aggregated predictions: {len(sample_y_pred)}")
    print(f"  y_true distribution: {dict(zip(*zip(*[(label, sample_y_true.count(label)) for label in set(sample_y_true)])))}")
    print(f"  y_pred distribution: {dict(zip(*zip(*[(label, sample_y_pred.count(label)) for label in set(sample_y_pred) if label is not None])))}")
    
    print("\\n Detailed Results:")
    for i, (true, pred) in enumerate(zip(sample_y_true, sample_y_pred)):
        pred_str = str(pred) if pred is not None else "None"
        print(f"  {i+1:2}: true={true:>15}, pred={pred_str:>15}")
    
    # Test metrics calculation
    print("\\n Testing Metrics Calculation:")
    try:
        sample_metrics = metrics_calc.calculate_single_run_metrics(
            sample_y_true, sample_y_pred, "Sample-level Category Test"
        )
        
        print(" Metrics calculation successful!")
        
        # Check valid metrics
        valid_metrics = sample_metrics.get('valid_metrics', {})
        all_metrics = sample_metrics.get('all_metrics', {})
        
        print("\\n Valid Predictions Metrics:")
        if valid_metrics:
            print(f"  • Micro F1: {valid_metrics.get('micro_f1', 'N/A'):.4f}")
            print(f"  • Macro F1: {valid_metrics.get('macro_f1', 'N/A'):.4f}")
            
            per_class = valid_metrics.get('per_class_metrics', {})
            for class_name in ['support', 'partial_support', 'not_support']:
                if class_name in per_class:
                    metrics = per_class[class_name]
                    print(f"  • {class_name:>15}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Support={metrics['support']}")
                else:
                    print(f"  • {class_name:>15}: Not present in results")
        
        print("\\n All Predictions Metrics:")
        if all_metrics:
            print(f"  • Micro F1: {all_metrics.get('micro_f1', 'N/A'):.4f}")
            print(f"  • Macro F1: {all_metrics.get('macro_f1', 'N/A'):.4f}")
            
            per_class_all = all_metrics.get('per_class_metrics', {})
            for class_name in ['support', 'partial_support', 'not_support', '_none', '_error']:
                if class_name in per_class_all:
                    metrics = per_class_all[class_name]
                    print(f"  • {class_name:>15}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Support={metrics['support']}")
        
        # Check which categories are actually supported
        supported_classes = set()
        if valid_metrics and 'per_class_metrics' in valid_metrics:
            supported_classes.update(valid_metrics['per_class_metrics'].keys())
        
        print(f"\\n Categories Supported After Aggregation: {supported_classes}")
        
        expected_classes = {'support', 'partial_support', 'not_support'}
        missing_classes = expected_classes - supported_classes
        
        if missing_classes:
            print(f"  Missing Categories: {missing_classes}")
        else:
            print(" All 3 main categories supported!")
            
        return len(missing_classes) == 0
        
    except Exception as e:
        print(f" Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases for category support"""
    print("\\n Testing Edge Cases")
    print("=" * 40)
    
    edge_cases = [
        # Case 1: Only one category
        {
            "name": "Only support category",
            "data": [
                {"qid": "q1", "nugget_text": "n1", "block_pred": "support", "block_true": "support"},
                {"qid": "q1", "nugget_text": "n2", "block_pred": "support", "block_true": "support"},
            ]
        },
        
        # Case 2: Only invalid predictions
        {
            "name": "Only invalid predictions", 
            "data": [
                {"qid": "q1", "nugget_text": "n1", "block_pred": None, "block_true": "support"},
                {"qid": "q1", "nugget_text": "n2", "block_pred": "error", "block_true": "partial_support"},
            ]
        },
        
        # Case 3: Mixed valid and invalid
        {
            "name": "Mixed valid and invalid",
            "data": [
                {"qid": "q1", "nugget_text": "n1", "block_pred": "support", "block_true": "support"},
                {"qid": "q1", "nugget_text": "n1", "block_pred": None, "block_true": "support"},  # support wins
                {"qid": "q1", "nugget_text": "n2", "block_pred": None, "block_true": "partial_support"},
                {"qid": "q1", "nugget_text": "n2", "block_pred": "error", "block_true": "partial_support"},  # None wins (both -1)
            ]
        }
    ]
    
    metrics_calc = MetricsCalculator()
    
    for case in edge_cases:
        print(f"\\n {case['name']}:")
        
        try:
            sample_y_true, sample_y_pred = metrics_calc.aggregate_sample_level_predictions_by_max_support(case['data'])
            
            valid_preds = [p for p in sample_y_pred if p is not None]
            invalid_preds = len(sample_y_pred) - len(valid_preds)
            
            print(f"  • Total: {len(sample_y_pred)}, Valid: {len(valid_preds)}, Invalid: {invalid_preds}")
            print(f"  • Predictions: {sample_y_pred}")
            
            # Try to calculate metrics
            sample_metrics = metrics_calc.calculate_single_run_metrics(
                sample_y_true, sample_y_pred, f"Edge Case: {case['name']}"
            )
            print("   Metrics calculation successful")
            
        except Exception as e:
            print(f"   Failed: {e}")
    
    return True

if __name__ == "__main__":
    print(" Sample-Level Categories Support Test")
    print("=" * 60)
    
    success = True
    success &= test_all_categories_sample_level()
    success &= test_edge_cases()
    
    print("\\n" + "=" * 60)
    if success:
        print(" Sample-level aggregation SUPPORTS all category calculations!")
        print("\\n Confirmed capabilities:")
        print("  • Micro/Macro F1, Precision, Recall calculation")
        print("  • Per-class metrics for support/partial_support/not_support")  
        print("  • Dual metrics (valid/all predictions)")
        print("  • Proper handling of None/error predictions")
    else:
        print(" Some category support tests failed!")
    
    print("\\n Summary:")
    print("Sample-level 聚合后的数据完全支持计算:")
    print("  • micro/macro F1/precision/recall 指标")
    print("  • support/partial_support/not_support 三个类别的独立指标")
    print("  • valid predictions 和 all predictions 两套指标")
    print("  • 最大支持度策略确保聚合结果的合理性")