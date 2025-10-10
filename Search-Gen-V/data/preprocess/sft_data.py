import json
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT_INSTRUCTIONS = {
    "json": (
        'Respond with a JSON array containing exactly one label for each nugget:\n'
        '["support", "partial_support", "not_support"]\n'
        'Example: ["support", "not_support", "partial_support"]'
    ),
    "csv": (
        'Respond with comma-separated values, one label for each nugget:\n'
        'support,partial_support,not_support\n'
        'Example: support,not_support,partial_support'
    ),
    "python_list": (
        'Respond with a Python list containing exactly one label for each nugget:\n'
        "['support', 'partial_support', 'not_support']\n"
        "Example: ['support', 'not_support', 'partial_support']"
    ),
    "yaml": (
        'Respond with a YAML list, one label for each nugget:\n'
        '- support\n'
        '- partial_support\n'
        '- not_support\n'
        'Example:\n- support\n- not_support\n- partial_support'
    ),
    "markdown": (
        'Respond with a Markdown unordered list, one label for each nugget:\n'
        '* support\n'
        '* partial_support\n'
        '* not_support\n'
        'Example:\n* support\n* not_support\n* partial_support'
    ),
    "xml": (
        'Respond with XML format, one label for each nugget:\n'
        '<labels>\n'
        '  <label>support</label>\n'
        '  <label>partial_support</label>\n'
        '  <label>not_support</label>\n'
        '</labels>\n'
        'Example:\n<labels>\n  <label>support</label>\n  <label>not_support</label>\n</labels>'
    ),
    "tsv": (
        'Respond with tab-separated values, one label for each nugget:\n'
        'support\tpartial_support\tnot_support\n'
        'Example: support\tnot_support\tpartial_support'
    ),
    "numbered": (
        'Respond with a numbered list, one label for each nugget:\n'
        '1. support\n'
        '2. partial_support\n'
        '3. not_support\n'
        'Example:\n1. support\n2. not_support\n3. partial_support'
    ),
    "comma_separated": (
        'Respond with comma-separated values with spaces, one label for each nugget:\n'
        'support, partial_support, not_support\n'
        'Example: support, not_support, partial_support'
    ),
    "pipe_separated": (
        'Respond with pipe-separated values, one label for each nugget:\n'
        'support|partial_support|not_support\n'
        'Example: support|not_support|partial_support'
    )
}

def format_labels(labels, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(labels, ensure_ascii=False)
    elif fmt == "csv":
        return ",".join(labels)
    elif fmt == "python_list":
        return str(labels)  
    elif fmt == "yaml":
        return "\n".join([f"- {label}" for label in labels])
    elif fmt == "markdown":
        return "\n".join([f"* {label}" for label in labels])
    elif fmt == "xml":
        return "<labels>\n" + "\n".join([f"  <label>{label}</label>" for label in labels]) + "\n</labels>"
    elif fmt == "tsv":
        return "\t".join(labels)
    elif fmt == "numbered":
        return "\n".join([f"{i+1}. {label}" for i, label in enumerate(labels)])
    elif fmt == "comma_separated":
        return ", ".join(labels)
    elif fmt == "pipe_separated":
        return "|".join(labels)
    else:
        raise ValueError(f"Unknown format: {fmt}")

def build_multi_nugget_prompt(query: str, block: str, nuggets: list[str],fmt: str) -> list[dict]:
    num_nuggets = len(nuggets)

    format_instruction = FORMAT_INSTRUCTIONS[fmt]
    system_msg = f"""You are NuggetMatchJudge.

Task:
Given a search query, a passage, and {num_nuggets} nuggets, assign one label to each nugget: "support", "partial_support", or "not_support".

Core Principle:
Your judgment must be based EXCLUSIVELY on the provided passage. Do not use any external knowledge.

Label Definitions & Decision Process:
Please follow this decision framework for each nugget:

1. Check for Contradiction → "not_support"
   - Does the passage explicitly state the opposite of the nugget?
   - If yes, label "not_support".

2. Check for Full Support → "support"  
   - Are ALL essential facts of the nugget explicitly and unambiguously stated in the passage?
   - Essential facts include: subjects, actions, key quantities, dates, conditions, and qualifiers
   - Do all qualifiers (e.g., "always", "some", "may") match perfectly?
   - If yes, label "support".

3. Check for Partial Support → "partial_support"
   - Does the passage support at least one essential fact, but another essential fact is missing, hedged (e.g., "may", "suggests"), or stated ambiguously?
   - Does verifying the nugget require only a minor, safe inference (e.g., treating clear paraphrases like "reached the summit" as equivalent to "climbed the mountain")?
   - If yes, label "partial_support".
   - Safe inference example: Passage says "turnover of $10 million", nugget says "revenue of $10 million"
   - Unsafe inference example: Passage says "CEO bought a new car", nugget says "company is doing well financially"

4. Default → "not_support"
   - If none of the above conditions are met (information entirely absent or only topically related), label "not_support".

Output Format (two parts):
1) Reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. For each nugget, freely express your thought process, including:
   - Restate the nugget to ensure understanding
   - Quote or paraphrase relevant parts from the passage
   - Analyze the relationship and support level
   - Reach a conclusion (support/partial_support/not_support)
   Use any format that helps you think clearly - paragraphs, bullet points, or numbered lists.

2) Final Answer: After the </reasoning> tag, provide the final labels in the requested format.
   - {format_instruction}
   - No extra text after the labels.
   - Count must equal {num_nuggets}.
"""

    nugget_section = "\n".join([f"- nugget: \"{nug}\"" for nug in nuggets])
    user_msg = f"""Search Query: {query}
Passage: {block}
Nuggets ({num_nuggets}): 
{nugget_section}

Please provide your detailed reasoning in <reasoning> tags, then the final labels:"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def process_data_and_save_to_parquet(input_jsonl_path: str, output_parquet_path: str, max_nuggets: int = 10):
    data = []

    with open(input_jsonl_path, encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            query = entry["query"]
            block_text = entry["block_text"]
            nuggets_list = entry["nuggets_list"]
            for i in range(0, len(nuggets_list), max_nuggets):
                batch_nuggets = nuggets_list[i:i+max_nuggets]
                nugget_texts = [nug["text"] for nug in batch_nuggets]
                labels = [nug["match"] for nug in batch_nuggets]

                fmt = random.choice(list(FORMAT_INSTRUCTIONS.keys()))
                conversation = build_multi_nugget_prompt(query, block_text, nugget_texts, fmt)

                labels_formatted = format_labels(labels, fmt)
                conversation.append({"role": "assistant", "content": labels_formatted})
                data.append({
                    "messages": conversation,
                    "tools": [],
                    "enable_thinking": False,
                })

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_parquet_path)
    print(f"成功将数据从 {input_jsonl_path} 转换为 {output_parquet_path}")
    print(f"转换后的数据共 {len(df)} 条，每条最多 {max_nuggets} 个 nuggets。")


if __name__ == "__main__":
    input_file = '/path/to/your/input.jsonl'
    output_file = '/path/to/your/output.parquet'
    process_data_and_save_to_parquet(input_file, output_file, max_nuggets=10)

