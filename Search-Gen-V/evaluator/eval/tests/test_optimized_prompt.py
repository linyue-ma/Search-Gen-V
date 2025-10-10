#!/usr/bin/env python3
"""
Test script for optimized prompt type

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval.prompts import build_template_prompt, prompt_template_manager
from nugget_eval.common import parse_labels_multi_format, strip_thinking_reasoning_blocks

def test_prompt_types():
    """Test all prompt types are available"""
    print(" Testing available prompt types...")
    
    available_types = prompt_template_manager.get_available_prompt_types()
    expected_types = ["legacy", "no_reasoning", "short_cot", "long_cot", "optimized"]
    
    print(f"Available types: {available_types}")
    assert set(expected_types) == set(available_types), f"Expected {expected_types}, got {available_types}"
    print(" All expected prompt types are available")

def test_optimized_prompt_generation():
    """Test optimized prompt generation"""
    print("\n Testing optimized prompt generation...")
    
    query = "What are the health benefits of green tea?"
    passage = "Green tea contains antioxidants that may help reduce inflammation. Some studies suggest it might help with weight management."
    nuggets = [
        "Green tea contains antioxidants",
        "Green tea helps with weight loss",
        "Green tea is good for heart health"
    ]
    
    # Test optimized prompt generation
    messages = build_template_prompt("optimized", query, passage, nuggets, "json")
    
    assert len(messages) == 2, "Should have system and user message"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    
    system_content = messages[0]["content"]
    assert "decision framework" in system_content.lower(), "Should contain decision framework instructions"
    assert "<reasoning>" in system_content.lower(), "Should mention <reasoning> tags"
    
    user_content = messages[1]["content"]
    assert query in user_content, "Query should be in user message"
    assert passage in user_content, "Passage should be in user message"
    assert "1. Green tea contains antioxidants" in user_content, "Nuggets should be numbered"
    
    print(" Optimized prompt generation works correctly")

def test_reasoning_block_stripping():
    """Test reasoning block stripping functionality"""
    print("\n Testing reasoning block stripping...")
    
    # Test response with reasoning block
    test_response = '''
    <reasoning>
    Nugget 1: "Green tea contains antioxidants"
    The passage explicitly states "contains powerful antioxidants called catechins".
    This directly supports the nugget claim.
    Conclusion: support
    
    Nugget 2: "Green tea helps with weight loss"
    The passage mentions "might support weight management" but uses uncertain language.
    This is only partial support, not full support for weight loss claims.
    Conclusion: partial_support
    </reasoning>
    
    Final answer: support, partial_support
    '''
    
    # Test strip_thinking_reasoning_blocks
    cleaned_text = strip_thinking_reasoning_blocks(test_response)
    
    assert "Final answer: support, partial_support" in cleaned_text, "Should preserve non-reasoning content"
    assert "<reasoning>" not in cleaned_text, "Should remove reasoning tags"
    assert "Nugget 1:" not in cleaned_text, "Should remove reasoning content"
    assert "Conclusion: support" not in cleaned_text, "Should remove reasoning content"
    
    print(" Reasoning block stripping works correctly")

def test_backward_compatibility():
    """Test that existing functionality still works"""
    print("\n Testing backward compatibility...")
    
    # Test legacy prompt types still work
    query = "test query"
    passage = "test passage"
    nuggets = ["test nugget"]
    
    for prompt_type in ["no_reasoning", "short_cot", "long_cot"]:
        messages = build_template_prompt(prompt_type, query, passage, nuggets, "adaptive")
        assert len(messages) == 2, f"Prompt type {prompt_type} should work"
    
    # Test existing parsers still work
    test_responses = [
        '["support", "partial_support", "not_support"]',  # JSON
        'support, partial_support, not_support',          # CSV
        '* support\n* partial_support\n* not_support',    # Markdown
    ]
    
    for response in test_responses:
        labels, count, format_used, error_type = parse_labels_multi_format(response, 3)
        assert labels is not None, f"Should parse {response}"
        assert len(labels) == 3, f"Should parse 3 labels from {response}"
    
    print(" Backward compatibility maintained")

def test_standard_parsing_functionality():
    """Test that standard parsing still works correctly"""
    print("\n Testing standard parsing functionality...")
    
    # Response with multiple format possibilities
    mixed_response = '''
    <reasoning>
    Some reasoning content here that should be stripped away.
    </reasoning>
    
    ["support", "partial_support", "not_support"]
    
    Also in other format: support, partial_support, not_support
    '''
    
    # Should parse the JSON array, not the CSV
    labels, count, format_used, error_type = parse_labels_multi_format(mixed_response, 3)
    
    assert format_used == "json", "JSON should have highest priority among standard formats"
    assert labels == ["support", "partial_support", "not_support"], "Should extract from JSON array"
    assert count == 3, "Should parse all 3 labels"
    
    print(" Standard parsing functionality works correctly")

def main():
    """Run all tests"""
    print(" OPTIMIZED PROMPT COMPATIBILITY TEST")
    print("=" * 50)
    
    try:
        test_prompt_types()
        test_optimized_prompt_generation()
        test_reasoning_block_stripping()
        test_backward_compatibility()
        test_standard_parsing_functionality()
        
        print("\n ALL TESTS PASSED!")
        print(" Optimized prompt functionality is working correctly and maintains backward compatibility")
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()