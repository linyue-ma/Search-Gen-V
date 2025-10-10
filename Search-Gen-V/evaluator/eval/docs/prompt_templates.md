# Prompt Templates Documentation

## Overview

This document describes the various prompt templates used in the nugget evaluation framework. The templates are designed to evaluate how well a passage supports specific information nuggets in relation to a search query.

## Template Types

### 1. No Reasoning (`no_reasoning`)
**Purpose**: Quick evaluation without explicit reasoning steps
**Best for**: High-performance models, batch processing, simple nuggets

**Template Structure**:
```text
System: You are NuggetMatchJudge.

Task: Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Evidence rules (use the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

Decision policy:
- Base the decision on verifiable snippets from the passage (identify them internally; do NOT output them).
- If information is missing or contradicted, choose "not_support".
- If some relevant evidence exists but is incomplete, choose "partial_support".
- Paraphrases and synonymous formulations count as support if the meaning clearly aligns.

Output constraints:
- Output only a JSON array (no extra text), length == {num_nuggets}, order matches the input nuggets.
- Elements must be one of: "support", "partial_support", "not_support".
```

### 2. Short Chain-of-Thought (`short_cot`)
**Purpose**: Structured reasoning for instruct models
**Best for**: General-purpose models, moderate complexity nuggets

**Template Structure**:
```text
System: You are NuggetMatchJudge.

Task: Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Definitions (evidence from the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

Rules:
- Use only the passage; ignore external knowledge.
- Base judgments on verifiable snippets from the passage; quote the minimal necessary phrase(s).
- If information is missing or contradicted → not_support.
- If some relevant evidence exists but is incomplete → partial_support.
- Paraphrases count as support if the meaning clearly aligns.

Output format (two-part):
1) Reasoning: Wrap ALL nugget-level reasoning in a single block delimited by <reasoning> and </reasoning>. Inside the block, for each nugget k = 1..{num_nuggets}, output a standardized multi-line item with the four fields in order: nugget, snippet, rationale, decision. Use the following format for every k:
   <reasoning>
   k.
     nugget="<nugget_k>"
     snippet="<verbatim phrase(s) from passage>"
     rationale="<free-form reasoning; one or more sentences/clauses>"
     decision="<support|partial_support|not_support>"
   ...
   </reasoning>
   - The rationale may be any length/format (short clause, sentence(s), or bullet-like clauses separated by ';' or new lines). Do NOT include square brackets [] anywhere in reasoning. Do NOT output any other tags outside <reasoning> ... </reasoning>.
2) Final answer: The LAST line must be ONLY a JSON array of labels in input order, e.g. ["support","not_support","partial_support"].
   - No extra text before/after the array.
   - Array length must equal {num_nuggets}.
```

### 3. Long Chain-of-Thought (`long_cot`)
**Purpose**: Detailed reasoning for reasoning models (e.g., DeepSeek-R1, Qwen3)
**Best for**: Complex reasoning models, difficult nuggets, research scenarios

**Template Structure**:
```text
System: You are NuggetMatchJudge.

Task: Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Definitions (evidence from the passage only):
- support: The passage explicitly and sufficiently affirms the nugget's key facts relevant to the query; no contradiction.
- partial_support: The passage contains directly relevant but incomplete/hedged/ambiguous information; minor inference is needed; at least one essential part is present.
- not_support: The nugget is absent, contradicted, or requires external knowledge beyond the passage.

Rules:
- Use only the passage; ignore external knowledge.
- Base judgments on verifiable snippets from the passage; quote the minimal necessary phrase(s).
- If information is missing or contradicted → not_support.
- If some relevant evidence exists but is incomplete → partial_support.
- Paraphrases count as support if the meaning clearly aligns.

Output format (three-part):
1) Optional long-form chain-of-thought: If you are a long-CoT model (e.g., DeepSeek-R1, Qwen3), place any free-form internal reasoning ONLY inside a single <think> ... </think> block. This block may be arbitrarily long. Do NOT include square brackets [] or any JSON-like structures inside <think>. If you do not need it, omit the block entirely.
2) Structured reasoning: Wrap ALL nugget-level structured reasoning in a single block delimited by <reasoning> and </reasoning>. Inside the block, for each nugget k = 1..{num_nuggets}, output a standardized multi-line item with the four fields in order: nugget, snippet, rationale, decision. Use the following format for every k:
   <reasoning>
   k.
     nugget="<nugget_k>"
     snippet="<verbatim phrase(s) from passage>"
     rationale="<free-form reasoning; one or more sentences/clauses>"
     decision="<support|partial_support|not_support>"
   ...
   </reasoning>
   - The rationale may be any length/format (short clause, sentence(s), or bullet-like clauses separated by ';' or new lines). Do NOT include square brackets [] inside <reasoning>. Do NOT output any other tags outside <think> ... </think> and <reasoning> ... </reasoning>.
3) Final answer: The LAST line must be ONLY a JSON array of labels in input order, e.g. ["support","not_support","partial_support"].
   - No extra text before/after the array.
   - Array length must equal {num_nuggets}.
```

### 4. Legacy (`legacy`)
**Purpose**: Backward compatibility with existing evaluations
**Best for**: Existing datasets, comparison studies

**Template Structure**: Uses the original template format for consistency with previous evaluations.

### 5. Optimized (`optimized`)
**Purpose**: Enhanced template with improved decision framework
**Best for**: Production use, high-accuracy requirements

**Key Features**:
- Improved decision framework with clearer boundary definitions
- Enhanced reasoning block structure for better parseability
- Optimized instruction flow for reduced model confusion
- Built-in error prevention mechanisms

## Label Definitions

### Support Categories
1. **support**: The passage explicitly and sufficiently affirms the nugget's key facts
   - Contains clear, unambiguous evidence
   - No contradictory information
   - Direct or clearly implied information
   
2. **partial_support**: The passage contains relevant but incomplete information
   - Some key elements present but not comprehensive
   - Ambiguous or hedged language
   - Requires minor inference to connect to nugget
   
3. **not_support**: The nugget is not supported by the passage
   - Information is absent
   - Information is contradicted
   - Requires external knowledge beyond the passage

## Format Types Integration

The templates work with various output format types:

### JSON Format
```json
["support", "partial_support", "not_support"]
```

### Markdown Format
```markdown
* support
* partial_support  
* not_support
```

### CSV Format
```text
support, partial_support, not_support
```

### YAML Format
```yaml
- support
- partial_support
- not_support
```

### XML Format
```xml
<labels>
  <label>support</label>
  <label>partial_support</label>
  <label>not_support</label>
</labels>
```

### Adaptive Format
Automatically detects and parses multiple format types from the model response.

## Best Practices

### Template Selection Guidelines

1. **Use `no_reasoning`** when:
   - Processing large batches
   - Model has proven accuracy without reasoning
   - Speed is prioritized over interpretability

2. **Use `short_cot`** when:
   - Need balance between speed and accuracy
   - Working with general-purpose instruct models
   - Want some interpretability of decisions

3. **Use `long_cot`** when:
   - Working with specialized reasoning models
   - Need detailed reasoning traces
   - Handling complex or ambiguous nuggets

4. **Use `optimized`** when:
   - Need highest accuracy
   - Processing critical evaluations
   - Want enhanced error prevention

### Configuration Examples

```python
# High-performance batch processing
config.model.prompt_type = "no_reasoning"
config.model.format_type = "json"

# Balanced accuracy and speed
config.model.prompt_type = "short_cot"
config.model.format_type = "adaptive"

# Maximum accuracy with reasoning
config.model.prompt_type = "optimized"
config.model.format_type = "json"
```

## Parsing and Validation

### Reasoning Block Stripping
All templates with reasoning blocks support automatic reasoning removal:
```python
from nugget_eval.common import strip_thinking_reasoning_blocks

# Automatically removes <think>, <reasoning> blocks
cleaned_response = strip_thinking_reasoning_blocks(model_response)
```

### Multi-format Parsing
```python
from nugget_eval.common import parse_labels_multi_format

labels, count, format_used, error_type = parse_labels_multi_format(response, expected_count)
```

## Template Evolution

### Template Versioning
- **v1.0**: Original templates with basic reasoning
- **v1.1**: Added short/long CoT distinction
- **v1.2**: Introduced optimized template
- **v2.0**: Unified format system with adaptive parsing

### Migration Guidelines
When updating templates:
1. Test with representative samples
2. Compare accuracy metrics with previous versions
3. Validate parsing robustness
4. Document any behavior changes

## Troubleshooting

### Common Issues

1. **Parsing Failures**:
   - Check format_type configuration
   - Verify model follows output constraints
   - Use adaptive format for robust parsing

2. **Inconsistent Reasoning**:
   - Switch to more structured template (short_cot → optimized)
   - Add few-shot examples in user prompts
   - Adjust temperature/sampling parameters

3. **Performance Issues**:
   - Use no_reasoning for speed
   - Optimize batch sizes
   - Consider model-specific templates

### Debug Commands
```bash
# Test template generation
python examples/template_usage.py

# Test optimized template specifically  
python examples/optimized_prompt.py

# Analyze parsing performance
python scripts/interactive_analysis.py
```

## Future Enhancements

### Planned Features
1. **Domain-specific templates**: Specialized templates for different domains
2. **Few-shot integration**: Built-in few-shot example management
3. **Dynamic templates**: Context-aware template selection
4. **Multi-language support**: Templates for non-English evaluations

### Extension Points
- **Custom templates**: Plugin system for domain-specific templates
- **Template optimization**: A/B testing framework for template effectiveness
- **Adaptive prompting**: Dynamic template selection based on nugget complexity