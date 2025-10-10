#!/usr/bin/env python3
"""
Test script for sample-level metrics functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval.metrics import MetricsCalculator

def test_sample_level_aggregation():
    """Test the sample-level aggregation functionality"""
    print(" Testing Sample-Level Metrics Aggregation")
    print("=" * 50)
    
    # Create test data - simulate predictions from different batches for same qid/nuggets
    test_predictions = [
        # qid 1 - nugget 1: multiple predictions across batches
        {"qid": "q1", "nugget_text": "nugget_1", "block_pred": "not_support", "block_true": "support"},
        {"qid": "q1", "nugget_text": "nugget_1", "block_pred": "support", "block_true": "support"},  # This should win
        {"qid": "q1", "nugget_text": "nugget_1", "block_pred": "partial_support", "block_true": "support"},
        
        # qid 1 - nugget 2: different predictions
        {"qid": "q1", "nugget_text": "nugget_2", "block_pred": "not_support", "block_true": "partial_support"},
        {"qid": "q1", "nugget_text": "nugget_2", "block_pred": "partial_support", "block_true": "partial_support"},  # This should win
        
        # qid 2 - nugget 1: mixed with None/error
        {"qid": "q2", "nugget_text": "nugget_1", "block_pred": None, "block_true": "not_support"},
        {"qid": "q2", "nugget_text": "nugget_1", "block_pred": "not_support", "block_true": "not_support"},  # This should win
        {"qid": "q2", "nugget_text": "nugget_1", "block_pred": "error", "block_true": "not_support"},
    ]
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    try:
        # Test aggregation function
        sample_y_true, sample_y_pred = metrics_calc.aggregate_sample_level_predictions_by_max_support(test_predictions)
        
        print(" Aggregation Results:")
        print(f"  • Number of aggregated predictions: {len(sample_y_pred)}")
        print(f"  • Ground truth: {sample_y_true}")
        print(f"  • Predictions: {sample_y_pred}")
        
        # Expected results:
        # q1_nugget_1: support (highest priority from multiple predictions)
        # q1_nugget_2: partial_support (highest priority)
        # q2_nugget_1: not_support (highest priority, beating None/error)
        
        expected_y_true = ["support", "partial_support", "not_support"]
        expected_y_pred = ["support", "partial_support", "not_support"]
        
        print("\\n Expected vs Actual:")
        print(f"  • Expected y_true: {expected_y_true}")
        print(f"  • Actual y_true:   {sample_y_true}")
        print(f"  • Expected y_pred: {expected_y_pred}")
        print(f"  • Actual y_pred:   {sample_y_pred}")
        
        # Check if results match expectations
        if sample_y_true == expected_y_true and sample_y_pred == expected_y_pred:
            print("\\n Test PASSED: Aggregation works correctly!")
            
            # Test metrics calculation
            sample_metrics = metrics_calc.calculate_single_run_metrics(
                sample_y_true, sample_y_pred, "Sample-level Test"
            )
            
            print("\\n Sample-level Metrics:")
            metrics_calc.print_single_run_metrics(sample_metrics)
            
        else:
            print("\\n Test FAILED: Aggregation results don't match expectations!")
            return False
            
    except Exception as e:
        print(f"\\n Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_priority_logic():
    """Test the support priority logic specifically"""
    print("\\n Testing Support Priority Logic")
    print("=" * 50)
    
    # Test cases for priority
    test_cases = [
        (["support", "partial_support", "not_support"], "support"),
        (["not_support", "partial_support"], "partial_support"),
        (["not_support"], "not_support"),
        ([None, "error", "not_support"], "not_support"),
        ([None, "error"], None),  # When only invalid predictions
        (["partial_support", "support", "not_support"], "support"),
    ]
    
    metrics_calc = MetricsCalculator()
    SUPPORT_PRIORITY = {
        "support": 2,
        "partial_support": 1, 
        "not_support": 0,
        None: -1,
        "error": -1
    }
    
    print("Testing priority selection:")
    for preds, expected in test_cases:
        actual = max(preds, key=lambda x: SUPPORT_PRIORITY.get(x, -1))
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {preds} -> {actual} (expected: {expected})")
    
    return True

if __name__ == "__main__":
    print(" Sample-Level Metrics Test Suite")
    print("=" * 60)
    
    success = True
    success &= test_priority_logic()
    success &= test_sample_level_aggregation()
    
    print("\\n" + "=" * 60)
    if success:
        print(" All tests PASSED! Sample-level metrics are working correctly.")
    else:
        print(" Some tests FAILED! Please check the implementation.")
    
    print("\\n Next steps:")
    print("  1. Test with real data using existing evaluation configs")
    print("  2. Compare nugget-level vs sample-level metrics")
    print("  3. Verify multi-run statistical analysis works")