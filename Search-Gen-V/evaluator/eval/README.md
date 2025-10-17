# Nugget Matching Evaluation Framework (v2.0)

A unified, configurable evaluation framework for nugget matching models in the search-gen-v pipeline. This framework eliminates code duplication, provides flexible configuration management, and supports both single-run and multi-run statistical evaluation with comprehensive error analysis.

## ✨ Key Features

- **🔧 Zero Hardcoding**: All configuration externalized to YAML files
- **🎯 Unified CLI**: Single entry point for all evaluation modes
- **📊 Dual Metrics System**: Performance analysis on both valid and all predictions
- **📈 Statistical Analysis**: Multi-run evaluation with 95% confidence intervals
- **🎨 Multi-Format Support**: 10+ output formats with intelligent parsing
- **🔄 Sequential Recovery**: Preserves partial predictions from truncated responses
- **📝 Complete Logging**: Separate evaluation and LLM call logs with run_id correlation
- **⚡ Parallel Processing**: Configurable worker processes
- **🔌 Extensible Design**: Easy to add new metrics and evaluation modes
- **🔍 Advanced Analytics**: Comprehensive case study and interactive analysis tools
- **🛡️ System Validation**: Built-in health checks and log correlation verification
- **🎛️ Interactive Exploration**: Command-line interface for real-time data analysis

## 🆕 **v2.0 Highlights: Dual Metrics & Enhanced Analysis**

### **Dual Metrics System**
- **Valid Predictions**: Metrics calculated only on successfully parsed predictions
- **All Predictions**: Comprehensive metrics including error/None predictions as special categories
- **Complete Comparison**: See both ideal performance and real-world impact of parsing failures

### **Enhanced Multi-Format Support**
- **10+ Formats**: JSON, CSV, Markdown, YAML, XML, TSV, Numbered lists, Pipe-separated, etc.
- **Sequential Parsing**: Intelligent format detection with fixed priority order
- **Partial Recovery**: Preserve valid predictions from partially failed parsing attempts

### **Advanced Statistical Analysis**
- **Extended Metrics**: pass@k, avg@k, maj@k at both nugget and sample levels
- **Confidence Intervals**: 95% CI for all metrics in multi-run mode
- **Error Breakdown**: Detailed analysis of truncation, format errors, and count mismatches

### **🆕 Run ID System & Log Correlation**
- **Unified Tracking**: Every evaluation run gets a unique run_id for complete traceability
- **Log Correlation**: Automatic correlation between evaluation logs and LLM API call logs
- **System Validation**: Built-in health checks for log integrity and run_id consistency
- **Enhanced Diagnostics**: Improved error handling and resource management

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup the environment
./setup.sh
source .venv/bin/activate
```

### 2. Generate Configuration
```bash
# Generate configuration templates
nugget-eval --generate-config config/my_thinking.yaml --template-type thinking
nugget-eval --generate-config config/my_multi.yaml --template-type multi_run
```

### 3. Configure Your Evaluation
Edit your configuration file:
```yaml
model:
  base_url: "http://localhost:8000/v1"
  name: "/path/to/your/model"
  format_type: "adaptive"        # Controls prompt format
  error_handling: "sequential"   # Preserve partial results
  enable_thinking: true

data:
  input_path: "/path/to/input.jsonl"
  gold_path: "/path/to/gold.jsonl"

evaluation:
  num_runs: 1          # Single run (1) or multi-run (16+)
  batch_size: 10
  num_workers: 8
```

### 4. Run Evaluation
```bash
# Single-run evaluation
nugget-eval --config config/my_thinking.yaml

# Multi-run statistical analysis  
nugget-eval --config config/my_multi.yaml --num-runs 16
```

## 📊 Output Examples

### Single-Run Dual Metrics
```
--- Nugget-level Match Evaluation ---
  Total Predictions: 1250
  Valid Predictions: 1180 (94.4%)
  Error Predictions: 45
  None Predictions: 25

  Performance Metrics (on valid predictions):
    Micro Accuracy:  0.8542
    Micro Precision: 0.8542
    Micro Recall:    0.8542
    Micro F1 Score:  0.8542

  Performance Metrics (on all predictions):
    Micro Accuracy:  0.8063    # Lower due to errors
    Micro Precision: 0.8063
    Micro Recall:    0.8063
    Micro F1 Score:  0.8063

  Batch Analysis:
    Success Rate: 84.0%
    Partial Success Rate: 12.0%
    Overall Recovery Rate: 96.8%
    
    Format Usage:
      json           : 67 batches (53.6%)
      markdown       : 35 batches (28.0%)
      csv            : 15 batches (12.0%)
```

### Multi-Run Statistical Analysis
```
--- Multi-Run Statistics (16 runs) (Nugget-Level) ---

  Metric Statistics (on valid predictions):
    Micro F1: Mean = 0.8542, Std = 0.0124, 95% CI = (0.8418, 0.8666)
    Macro F1: Mean = 0.8523, Std = 0.0156, 95% CI = (0.8367, 0.8679)

  Metric Statistics (on all predictions):
    Micro F1: Mean = 0.7234, Std = 0.0178, 95% CI = (0.7056, 0.7412)
    Macro F1: Mean = 0.7156, Std = 0.0189, 95% CI = (0.6967, 0.7345)

--- Nugget-Level Extended Statistics ---
  Pass@K (at least one correct):
    pass@1: 0.7245    pass@5: 0.8967    pass@10: 0.9234
  
  Avg@K (average accuracy):
    avg@1: 0.7245     avg@5: 0.8123     avg@10: 0.8234
  
  Maj@K (majority voting):
    maj@3: 0.8456     maj@5: 0.8567     maj@7: 0.8634

--- Sample-Level Extended Statistics ---
  Pass@K: pass@1: 0.6234, pass@5: 0.8456, pass@10: 0.9012
  Avg@K:  avg@1: 0.6234,  avg@5: 0.7345,  avg@10: 0.7678
  Maj@K:  maj@3: 0.7234,  maj@5: 0.7456,  maj@7: 0.7634
```

## 🔧 Configuration Reference

### Model Configuration
```yaml
model:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"
  name: "/path/to/model"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 1024
  enable_thinking: true
  
  # Multi-format output configuration
  format_type: "adaptive"          # adaptive|json|markdown|csv|yaml|xml|etc.
  error_handling: "sequential"     # sequential|strict
  partial_recovery: true           # Enable partial prediction recovery
```

### Evaluation Configuration
```yaml
evaluation:
  batch_size: 10          # Nuggets per batch
  num_workers: 8          # Parallel workers
  num_runs: 1            # Single run (1) or multi-run (16+)

metrics:
  avg_k: [1, 5, 10]      # avg@k calculation
  pass_k: [1, 5, 10]     # pass@k calculation
  maj_k: [3, 5, 7]       # maj@k calculation (use odd numbers)
```

### Logging Configuration
```yaml
logging:
  eval_log_dir: "logs/eval"
  llm_log_dir: "logs/llm"
  save_predictions: true
  log_level: "INFO"
  
  # Enhanced reporting options
  report_detail_level: "full"      # full|standard|minimal
  show_format_analysis: true       # Show format usage statistics
  show_error_breakdown: true       # Show detailed error analysis
  show_batch_recovery_stats: true  # Show partial recovery stats
```

## 📁 Project Structure

```
Search-Gen-V/evaluator/
├── eval/                    # Main package
│   ├── __init__.py
│   ├── cli.py               # CLI interface
│   ├── config.py            # Configuration management
│   ├── common.py            # Multi-format parsing & model client
│   ├── evaluator.py         # Unified evaluator with dual metrics
│   ├── metrics.py           # Enhanced metrics calculation
│   ├── exceptions.py        # Custom exceptions
│   ├── constants.py         # 🆕 System constants and configuration
│   └── validators.py        # 🆕 Run ID and log validation utilities
├── config/                   # Configuration templates
│   ├── base.yaml           # Base template
│   ├── thinking_mode.yaml  # Single-run with thinking
│   ├── multi_run.yaml      # Multi-run statistical
│   └── demo_v2.yaml        # v2.0 feature demo
├── logs/                    # Output logs (auto-created)
│   ├── eval/              # Evaluation logs (organized by run_id)
│   │   └── <run_id>/      # 🆕 Per-run log isolation
│   └── llm/               # LLM call logs (organized by run_id)
│       └── <run_id>/      # 🆕 Per-run log isolation
├── tests/                   # 🆕 Test suite
│   ├── __init__.py
│   └── test_run_id_system.py
├── case_study.py            # 🆕 Advanced analysis and reporting tool
├── interactive_analysis.py  # 🆕 Interactive command-line analysis
├── run_tests.py            # 🆕 Test runner
├── predictions_to_gold.py   # Prediction format converter
├── setup.sh                # Environment setup script
├── run_eval.sh            # Convenient evaluation runner
├── pyproject.toml         # Package configuration
├── IMPROVEMENTS_SUMMARY.md  # 🆕 Detailed improvement documentation
└── README.md
```

## 🎯 Usage Examples

### Format-Specific Testing
```bash
# Test markdown-only output
nugget-eval --config config/base.yaml --format-type markdown

# Test JSON output with strict error handling
nugget-eval --config config/base.yaml --format-type json --error-handling strict
```

### Statistical Analysis
```bash
# Quick 5-run analysis
nugget-eval --config config/multi_run.yaml --num-runs 5

# Comprehensive 20-run analysis with extended metrics
nugget-eval --config config/multi_run.yaml --num-runs 20
```

### Parameter Overrides
```bash
nugget-eval --config config/base.yaml \
  --input-path /new/data.jsonl \
  --batch-size 5 \
  --num-runs 10 \
  --format-type adaptive \
  --report-detail-level full
```

### 🆕 Advanced Analysis Tools

#### Case Study Analysis
Perform comprehensive analysis of evaluation runs with automatic log correlation:

```bash
# Analyze a specific run in detail
python case_study.py --run-id 20240101_120000_abc12345 --log-dir logs/ --detailed

# Compare multiple runs
python case_study.py --compare-runs 20240101_120000_abc12345 20240101_130000_def67890 --log-dir logs/

# Analyze all available runs for trends
python case_study.py --analyze-all --log-dir logs/ --limit 10

# Generate HTML reports
python case_study.py --run-id 20240101_120000_abc12345 --log-dir logs/ --format html --save-report

# Perform system health check
python case_study.py --health-check --log-dir logs/

# List all available runs
python case_study.py --list-runs --log-dir logs/
```

#### Interactive Analysis
Launch an interactive command-line interface for real-time exploration:

```bash
# Start interactive analyzer
python interactive_analysis.py --log-dir logs/

# Available interactive commands:
analysis> list_runs                    # List available runs
analysis> load_run 20240101_120000_abc12345  # Load specific run
analysis> show_summary detailed        # Show detailed run summary
analysis> show_errors 10              # Show first 10 errors
analysis> query_qid Q12345            # Analyze specific question
analysis> search "timeout"            # Search across loaded data
analysis> compare 20240101_130000_def67890  # Compare with another run
analysis> export summary my_report.json     # Export analysis
analysis> health_check                 # Check system health
analysis> help                         # Show all commands
```

#### System Validation
Validate run_id system integrity and log correlation:

```bash
# Run comprehensive tests
python run_tests.py -v

# Quick health check
python -c "from nugget_eval.validators import validate_system_health; print(validate_system_health('logs/eval', 'logs/llm'))"

# Validate specific run_id format
python -c "from nugget_eval.validators import validate_run_id; print(validate_run_id('20240101_120000_abc12345'))"
```

### Convert eval logs to gold/input JSONL

当你通过 `nugget-eval` 得到逐 nugget 的评估日志（每行必须含 `qid`,`query`,`block_text`,`nugget_text`,`block_pred`,`block_true`）后，可以：

1) 生成 gold（可选从 input 模板注入 `query` 并按模板顺序输出 nuggets）：

```bash
python predictions_to_gold.py \
  --eval-log logs/eval/20250101_120000_single_run_predictions.jsonl \
  --input-template /wuxi_gpfs/user/malinyue/data/release/eval/rag24.jsonl \
  --output-gold /wuxi_gpfs/user/malinyue/data/release/eval/rag24sample.jsonl \
  --fallback not_support
```

2) 基于现有 input 模板回填/覆盖 assignment 生成新的 input：

```bash
python predictions_to_gold.py \
  --eval-log logs/eval/20250101_120000_single_run_predictions.jsonl \
  --input-template /wuxi_gpfs/user/malinyue/data/release/eval/rag24.jsonl \
  --output-input /wuxi_gpfs/user/malinyue/data/release/eval/rag24.updated.jsonl \
  --fallback not_support
```

如果希望对于 eval 缺失的 nuggets 保留模板中的原 assignment 而不是写入 fallback：

```bash
python predictions_to_gold.py \
  --eval-log logs/eval/20250101_120000_single_run_predictions.jsonl \
  --input-template /wuxi_gpfs/user/malinyue/data/release/eval/rag24.jsonl \
  --output-input /wuxi_gpfs/user/malinyue/data/release/eval/rag24.updated.jsonl \
  --keep-existing-when-missing
```

#### 标准格式与严格匹配说明（重要）

- 输入模板中，`block` 字段为：
  - `List[Any]`，其中 `block[0]` 是 passage 文本，`block[1]` 是 `List[{"title": str, "url": str}]`（参考列表）。
  - 或兼容 `str`（仅 passage 文本）。
- 评估日志（predictions JSONL）中若存在 `block_text`，脚本会对同一 `qid` 下执行 per-block 聚合，并在更新 input 时“严格使用”模板的 `block[0]` 与 `block_text` 做精确匹配：
  - 若该 `qid` 存在 per-block 聚合但无法匹配模板的 `block[0]`，脚本将直接报错（不回退）。
  - 若该 `qid` 不存在 per-block 聚合，则回退为 qid 级别的全局聚合。
- gold 输出为 per-qid 的 `global_nuggets_assignment`（可含 `query`），不包含 per-block `blocks` 字段。
 - 评估日志每条记录必须包含 `query` 与 `block_text`；缺失将报错。每个 `qid` 的 `query` 必须能从模板或评估日志补齐，否则报错。

示例命令（严格匹配与 gold 输出对齐标准）：

```bash
python predictions_to_gold.py \
  --eval-log logs/eval/<run_id>/single_run_predictions.jsonl \
  --input-template /path/to/rag24.jsonl \
  --output-gold /path/to/rag24_gold.jsonl \
  --output-input /path/to/rag24_input.updated.jsonl \
  --fallback not_support
```

说明：
- `assignment` 聚合规则为最大支持度优先：support > partial_support > not_support。
- 聚合会忽略 `None` 与 `error`；若全部无效，则使用 `--fallback`（默认 `not_support`）。
- gold 输出按 `qid` 分组；gold 行包含 `query` 字段（若模板缺失则从评估日志获取；无法获取将报错），同时 nuggets 顺序将按模板的 `block_nuggets_assignment.text` 排序；否则按字典序。
- input 更新仅覆盖 `block_nuggets_assignment[].assignment`，保留 `docids`、`importance`、`block`、`query` 等其余字段与结构。

## 📈 Key Metrics Explained

### **Dual Metrics System**
- **Valid Predictions**: Calculated only on successfully parsed outputs
- **All Predictions**: Include None/error as special failure categories
- **Gap Analysis**: Difference shows impact of parsing failures

### **Extended Statistics** (Multi-run only)
- **pass@k**: Probability that ≥1 of k runs succeeds
- **avg@k**: Average accuracy over k runs  
- **maj@k**: Majority voting accuracy over k runs
- **Nugget-Level**: Traditional per-nugget analysis
- **Sample-Level**: Per-query complete correctness analysis

### **Error Analysis**
- **Batch Status**: success/partial_success/complete_failure rates
- **Format Usage**: Which formats models actually use
- **Recovery Rate**: Data saved from partial parsing failures
- **Error Types**: Truncation, format errors, count mismatches

## 🚨 Migration Notes

### From v1.0 Legacy Scripts
```bash
# Old way
python eval_with_thinking.py
python eval_without_thinking.py

# New way (fully compatible)
nugget-eval --config config/thinking_mode.yaml
nugget-eval --config config/multi_run.yaml
```

### Key Changes in v2.0
1. **Dual metrics**: Both valid and all predictions analyzed
2. **Simplified parsing**: Removed fallback_formats configuration
3. **Enhanced reporting**: More detailed error analysis
4. **Extended statistics**: Sample-level pass@k/avg@k/maj@k added

## 💡 Best Practices

### Evaluation Best Practices
1. **Use multi-run for production**: `num_runs >= 16` for reliable statistics
2. **Monitor format compliance**: Check format usage in batch analysis  
3. **Analyze error patterns**: Use detailed error breakdown for model improvement
4. **Compare dual metrics**: Gap indicates parsing robustness issues
5. **Use odd numbers for maj@k**: Avoid ties in majority voting

### 🆕 Analysis and Debugging Best Practices
6. **Start with health checks**: Always run `python case_study.py --health-check` before analysis
7. **Use interactive mode for exploration**: `interactive_analysis.py` for real-time investigation
8. **Correlate logs systematically**: Each run_id automatically correlates eval and LLM logs
9. **Track performance trends**: Use `--analyze-all` to identify performance degradation
10. **Export findings for sharing**: Save analysis reports for team collaboration

### Workflow Recommendations
```bash
# Recommended analysis workflow
1. nugget-eval --config your_config.yaml           # Run evaluation
2. python case_study.py --health-check --log-dir logs/  # Verify system health
3. python interactive_analysis.py --log-dir logs/       # Explore results
4. python case_study.py --analyze-all --log-dir logs/   # Generate trends report
```

### Troubleshooting Guide
- **API errors**: Use `query_qid <qid>` to trace specific failures
- **Performance issues**: Compare runs with `case_study.py --compare-runs`
- **System problems**: Check `validate_system_health()` output
- **Log correlation issues**: Verify run_id consistency with health checks

## 🎯 适用场景与选择指南

### Case Study Analysis 适用于：
- **性能调优**: 深入分析特定run的性能瓶颈和优化机会
- **错误诊断**: 系统化分析API错误、解析失败等问题
- **trend分析**: 跟踪多个run之间的性能变化趋势
- **报告生成**: 为团队或客户生成专业的分析报告
- **系统健康监控**: 定期检查日志系统的完整性和一致性

### Interactive Analysis 适用于：
- **实时调试**: 在开发过程中快速定位和分析问题
- **数据探索**: 交互式查询特定QID或错误模式
- **临时分析**: 无需编写脚本即可进行复杂的数据查询
- **学习系统**: 熟悉数据结构和评估结果的分布
- **快速验证**: 验证修复效果或配置变更的影响

### 工具选择建议：
```
定期监控 → case_study.py --health-check
深度分析 → case_study.py --analyze-all  
问题调试 → interactive_analysis.py
报告生成 → case_study.py --save-report
系统验证 → run_tests.py
```

---

## 🚀 新功能总结

**v2.1 新增功能（基于run_id系统）:**
- 🔗 **完整的日志关联系统**: 通过run_id实现eval_log和llm_log的自动关联
- 📊 **高级分析工具**: case_study.py提供专业级的性能和错误分析
- 🎛️ **交互式探索界面**: interactive_analysis.py支持实时数据查询和分析
- 🛡️ **系统验证框架**: 内置健康检查和完整的测试套件
- 📈 **趋势分析能力**: 跨多个run的性能趋势和模式识别
- 🔍 **精确问题定位**: QID级别的追踪和错误上下文分析

这些工具将evaluation framework从基础的指标计算升级为完整的模型性能分析和调优平台。

---

**v2.1 delivers comprehensive evaluation with dual metrics analysis and advanced analytics tools, making it easy to understand both model capabilities and real-world deployment impact with complete traceability.**