#!/usr/bin/env python3
"""
Template-based evaluation test script
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval import Config, NuggetEvaluator

def main():
    print(" TEMPLATE-BASED EVALUATION TEST")
    print("=" * 50)
    
    config = Config()
    
    config.model.prompt_type = "short_cot"  
    config.model.format_type = "json"       
    config.model.error_handling = "sequential"    
    config.model.partial_recovery = True
    
    config.model.base_url = "http://localhost:8000/v1"
    config.model.api_key = "EMPTY"
    config.model.name = "/path/to/your/model"  
    config.model.enable_thinking = True
    config.model.temperature = 0.7
    config.model.max_tokens = 1024
   
    config.data.input_path = "/path/to/your/input.jsonl" 
    config.data.gold_path = "/path/to/your/gold.jsonl"    
    
    config.evaluation.batch_size = 10
    config.evaluation.num_workers = 8
    config.evaluation.num_runs = 1
    
    config.logging.eval_log_dir = "logs/eval_template"
    config.logging.llm_log_dir = "logs/llm_template"
    config.logging.save_predictions = True
    config.logging.log_level = "INFO"
    config.logging.show_format_analysis = True     
    config.logging.show_error_breakdown = True      
    config.logging.show_batch_recovery_stats = True 
    
    print(" Configuration Summary:")
    print(f"  • Prompt Type: {config.model.prompt_type}")
    print(f"  • Format Type: {config.model.format_type}")
    print(f"  • Error Handling: {config.model.error_handling}")
    print(f"  • Partial Recovery: {config.model.partial_recovery}")
    print(f"  • Log Directory: {config.logging.eval_log_dir}")
    print()
   
    try:
        print(" Validating configuration...")
        config.validate(strict=False)
        print(" Configuration validation passed!")
    except Exception as e:
        print(f" Configuration validation failed: {e}")
        print(" Please update the model and data paths in the script")
        return

    print("\\n Starting template-based evaluation...")
    evaluator = NuggetEvaluator(config)
    
    try:
        results = evaluator.run_evaluation()
        
        print("\\n TEMPLATE-SPECIFIC ANALYSIS:")
        if "batch_analysis" in results.get("nugget_metrics", {}):
            batch_stats = results["nugget_metrics"]["batch_analysis"]
            format_dist = batch_stats.get("format_distribution", {})
            
            total_batches = batch_stats.get("total_batches", 1)
            
            prompt_types = {}
            for stat in results.get("batch_stats", []):
                pt = stat.get("prompt_type", "unknown")
                prompt_types[pt] = prompt_types.get(pt, 0) + 1
            
            if prompt_types:
                print(f"  • Prompt Type Distribution:")
                for pt, count in prompt_types.items():
                    print(f"    - {pt}: {count} batches")
            
            print(f"  • Format Distribution:")
            for format_name, count in format_dist.items():
                print(f"    - {format_name}: {count} batches")
            
            print(f"  • Success Rate: {batch_stats.get('success_rate', 0)*100:.1f}%")
            print(f"  • Recovery Rate: {batch_stats.get('overall_recovery_rate', 0)*100:.1f}%")
        
        print("\\n Template-based evaluation completed successfully!")
        
    except Exception as e:
        print(f"\\n Evaluation failed: {e}")
        print(" Check your model endpoint and data paths")

if __name__ == "__main__":
    main()