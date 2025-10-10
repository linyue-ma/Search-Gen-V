#!/usr/bin/env python3
"""
Unified prompt_type system usage example
演示如何使用统一的 prompt_type 系统
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nugget_eval import Config, NuggetEvaluator
from nugget_eval.prompts import build_template_prompt


def demonstrate_template_prompts():
    """演示不同 prompt_type 与 format_type 的组合"""
    print(" PROMPT TYPE + FORMAT TYPE DEMONSTRATION")
    print("=" * 60)
    
    query = "What is the capital of France?"
    passage = "Paris is the capital and largest city of France. It is located in northern France."
    nuggets = ["Paris is the capital of France", "France is in Europe"]
    
    # 演示不同组合
    examples = [
        ("no_reasoning", "json"),
        ("short_cot", "markdown"),
        ("long_cot", "yaml")
    ]
    
    for prompt_type, format_type in examples:
        print(f"\n {prompt_type.upper()} + {format_type.upper()}:")
        print("-" * 40)
        
        messages = build_template_prompt(prompt_type, query, passage, nuggets, format_type)
        
        print("System prompt (first 200 chars):")
        print(f"  {messages[0]['content'][:200]}...")
        print(f"\nUser prompt (first 150 chars):")
        print(f"  {messages[1]['content'][:150]}...")
        print()


def create_example_configs():
    """创建不同 prompt_type 配置示例"""
    print("  CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Legacy", "legacy", "adaptive", 512),
        ("No Reasoning", "no_reasoning", "json", 64),
        ("Short CoT", "short_cot", "markdown", 1024), 
        ("Long CoT", "long_cot", "json", 2048)
    ]
    
    for name, prompt_type, format_type, max_tokens in examples:
        print(f"\n {name} Configuration:")
        print(f"""
config = Config()
config.model.prompt_type = "{prompt_type}"
config.model.format_type = "{format_type}"
config.model.max_tokens = {max_tokens}
config.model.enable_thinking = True

# Set your model details
config.model.base_url = "http://localhost:8000/v1"
config.model.name = "/path/to/your/model"

# Set your data paths
config.data.input_path = "/path/to/input.jsonl"
config.data.gold_path = "/path/to/gold.jsonl"
        """)


def show_migration_guide():
    """显示从旧系统迁移的指南"""
    print(" MIGRATION GUIDE")
    print("=" * 60)
    
    print("""
从旧的多字段系统迁移到统一的 prompt_type 系统:

1  简化配置:
   OLD: config.model.use_template_prompts = True
        config.model.reasoning_mode = "short_cot"
        config.model.format_type = "json"
   NEW: config.model.prompt_type = "short_cot"
        config.model.format_type = "json"

2  使用新的 prompt 构建:
   OLD: if use_template_prompts:
            messages = build_template_prompt(reasoning_mode, ...)
        else:
            messages = build_multi_format_prompt(...)
   NEW: if prompt_type == "legacy":
            messages = build_multi_format_prompt(...)
        else:
            messages = build_template_prompt(prompt_type, ...)

3  简化的解析逻辑:
   • 自动移除 <thinking> 和 <reasoning> 块
   • 复用原有的多格式解析器
   • 保持简单高效

4  完全向后兼容:
   • prompt_type = "legacy" 使用原有系统
   • 所有 prompt_type 都支持所有 format_type
    """)


def main():
    """主函数"""
    print(" UNIFIED PROMPT_TYPE EVALUATION SYSTEM")
    print("=" * 70)
    
    demonstrate_template_prompts()
    create_example_configs() 
    show_migration_guide()
    
    print("\n QUICK START:")
    print("1. 复制 config/template_examples.yaml 中的配置")
    print("2. 修改模型和数据路径")
    print("3. 运行: python test_markdown_only.py")
    print("4. 查看 logs/ 目录中的详细日志")
    
    print("\n KEY FEATURES:")
    print("• 统一的 prompt_type: legacy, no_reasoning, short_cot, long_cot")
    print("• 所有 prompt_type 都支持所有 format_type") 
    print("• 4 种 prompts × 10 种格式 = 40 种组合")
    print("• 简化的解析逻辑，复用原有解析器")


if __name__ == "__main__":
    main()