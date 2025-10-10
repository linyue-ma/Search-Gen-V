#!/usr/bin/env python3
"""
Example usage of optimized prompt type
优化版 prompt type 使用示例
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval.prompts import build_template_prompt
from nugget_eval.common import parse_labels_multi_format, strip_thinking_reasoning_blocks

def demonstrate_optimized_prompt():
    """Demonstrate the optimized prompt functionality"""
    print("OPTIMIZED PROMPT DEMONSTRATION")
    print("=" * 50)
    
    # Example data
    query = "What are the health benefits of green tea?"
    passage = """Green tea is one of the most popular beverages worldwide. It contains powerful antioxidants called catechins, particularly EGCG (epigallocatechin gallate), which may help reduce inflammation in the body. Some studies suggest that regular green tea consumption might support weight management by slightly boosting metabolism. However, the evidence for significant weight loss is limited and results vary among individuals. Green tea also contains L-theanine, an amino acid that may promote relaxation."""
    
    nuggets = [
        "Green tea contains powerful antioxidants",
        "Green tea causes significant weight loss", 
        "Green tea helps with weight management",
        "Green tea contains caffeine only",
        "Green tea contains L-theanine"
    ]
    
    # Generate optimized prompt
    print("\n📝 Generated Optimized Prompt:")
    messages = build_template_prompt("optimized", query, passage, nuggets, "json")
    
    print(f"System Message ({len(messages[0]['content'])} chars):")
    print(messages[0]['content'][:300] + "..." if len(messages[0]['content']) > 300 else messages[0]['content'])
    
    print(f"\nUser Message ({len(messages[1]['content'])} chars):")
    print(messages[1]['content'])
    
    # Simulate a model response in the expected format
    simulated_response = '''<reasoning>
Nugget 1: "Green tea contains powerful antioxidants"
Looking at the passage, I can see it explicitly states "It contains powerful antioxidants called catechins, particularly EGCG". This directly matches what the nugget claims about green tea containing powerful antioxidants. The passage provides specific examples (catechins, EGCG) which strongly supports this statement.
Conclusion: support

Nugget 2: "Green tea causes significant weight loss"  
The passage mentions weight management but specifically says "the evidence for significant weight loss is limited and results vary among individuals". This actually contradicts the nugget's claim about "significant weight loss" by explicitly stating the evidence is limited. This is a clear contradiction.
Conclusion: not_support

Nugget 3: "Green tea helps with weight management"
The passage states "might support weight management by slightly boosting metabolism". This provides some support for the nugget, but uses qualifying language like "might" and "slightly". It's not a strong, unqualified statement of support, but it does indicate some potential benefit for weight management.
Conclusion: partial_support

Nugget 4: "Green tea contains caffeine only"
This is clearly contradicted by the passage, which mentions multiple components: "powerful antioxidants called catechins, particularly EGCG" and "L-theanine, an amino acid". The passage shows green tea contains many substances beyond just caffeine.
Conclusion: not_support

Nugget 5: "Green tea contains L-theanine"
The passage explicitly states "Green tea also contains L-theanine, an amino acid that may promote relaxation." This directly supports the nugget's claim about L-theanine content.
Conclusion: support
</reasoning>

support, not_support, partial_support, not_support, support'''
    
    print("\n Simulated Model Response:")
    print("Reasoning Part (will be stripped during parsing):")
    
    # Show the reasoning part for demonstration
    if "<reasoning>" in simulated_response:
        start = simulated_response.find("<reasoning>") + len("<reasoning>")
        end = simulated_response.find("</reasoning>")
        reasoning_content = simulated_response[start:end].strip()
        
        # Show first few lines of reasoning
        reasoning_lines = reasoning_content.split('\n')[:6]
        for line in reasoning_lines:
            if line.strip():
                print(f"  {line.strip()}")
        print("  ... (more reasoning)")
    
    # Strip reasoning blocks to get clean response
    clean_response = strip_thinking_reasoning_blocks(simulated_response)
    print(f"\nClean Response (after removing <reasoning> blocks):")
    print(f"'{clean_response.strip()}'")
    
    # Parse labels using multi-format parser
    labels, count, format_used, error_type = parse_labels_multi_format(simulated_response, len(nuggets))
    
    print("\n Parsing Results:")
    print(f"  • Format detected: {format_used}")
    print(f"  • Parsed {count} labels (expected {len(nuggets)})")
    print(f"  • Error type: {error_type}")
    print(f"  • Labels: {labels}")
    
    # Show expected vs actual
    expected_labels = ["support", "not_support", "partial_support", "not_support", "support"]
    print(f"\n Expected: {expected_labels}")
    print(f" Actual:   {labels}")
    print(f" Match: {labels == expected_labels}")
    
    print("\n Optimized prompt demonstration completed!")
    print("\nKey advantages of optimized prompts:")
    print("  • Clear decision framework reduces judgment ambiguity")
    print("  • <reasoning> blocks provide natural language explanations") 
    print("  • Better definition of essential facts with concrete examples")
    print("  • Clear guidance on safe vs unsafe inference")
    print("  • Freedom for models to think naturally while maintaining structure")
    print("  • Simple and reliable parsing by stripping reasoning blocks")

if __name__ == "__main__":
    demonstrate_optimized_prompt()