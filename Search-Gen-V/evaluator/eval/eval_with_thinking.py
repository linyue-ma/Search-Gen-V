import os
import numpy as np
import json
from math import ceil
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from openai import OpenAI
from collections import defaultdict
import multiprocessing
import re
import time
import sys

def print_metrics(y_true, y_pred, name=""):
    print(f"\n--- {name} Evaluation ---")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"  Micro Accuracy:  {acc:.4f}")
    print(f"  Micro Precision: {p:.4f}")
    print(f"  Micro Recall:    {r:.4f}")
    print(f"  Micro F1 Score:  {f1:.4f}")

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print(f"  Macro Precision: {p_macro:.4f}")
    print(f"  Macro Recall:    {r_macro:.4f}")
    print(f"  Macro F1 Score:  {f1_macro:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("\n  Confusion Matrix:")
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = [""] + [f"Pred {label}" for label in labels]
    print(f"  {' | '.join(header)}")
    print("  " + "-" * (len(" | ".join(header)) + 4))
    for i, label in enumerate(labels):
        row_str = [f"True {label}"] + [str(val) for val in cm[i]]
        print(f"  {' | '.join(row_str)}")
        
    print("\n")


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MATCH_PRIORITY = {
    "not_support": 0,
    "partial_support": 1,
    "support": 2,
    "error": -1
}
PRIORITY_TO_LABEL = {v: k for k, v in MATCH_PRIORITY.items()}

def build_multi_nugget_prompt(query: str, block: str, nuggets: list[str]) -> list[dict]:
    system_msg = (
        "You are NuggetMatchJudge, an intelligent assistant.\n"
        "Your task is to read a query, a passage, and nuggets, "
        "and then decide whether the passage supports the nugget in the context of the query.\n"
        "You need to label the nugget as one of the following: support, partial_support, or not_support.\n"
    )
    nugget_section = "\n".join([f"- nugget: \"{nug}\"\n" for nug in nuggets])
    user_msg = (
        "Please respond **only** using a Markdown unordered list like this:\n"
        "* support\n"
        "* partial_support\n"
        "* not_support\n"
        "The list should be in the same order as the input nuggets. "
        "Make sure to provide a label for each nugget.\n\n" 
        f"Query: {query}\n"
        f"Passage: {block}\n"
        f"Nuggets:\n{nugget_section}\n\n"  
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

def safe_parse_labels(text):
    valid_labels = {"support", "partial_support", "not_support"}
    text = text.strip()
    if not text:
        return None
    labels = []
    for line in text.splitlines():
        match = re.match(r"^\*?\s*(\S+)", line)
        if match:
            label = match.group(1).strip()
            if label in valid_labels:
                labels.append(label)

    return labels if labels else None

def get_prediction(query, block_text, nuggets_text_list):
    messages = build_multi_nugget_prompt(query, block_text, nuggets_text_list)
    try:
        response = client.chat.completions.create(
            model="/wuxi_gpfs/user/malinyue/checkpoints/nugget-matching/nugget-matching-qwen3-1.7b-dapo-aug-sft/merge/global_step_450",
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024, 
            extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": True}}
        )
        content = response.choices[0].message.content.strip()
        labels = safe_parse_labels(content)
        # print("----------------content---------------")
        # print(content)
        # print("-----------------label----------------")
        # print(labels)
        is_truncated = False
        if labels is None or not isinstance(labels, list) or len(labels) != len(nuggets_text_list):
            is_truncated = True
            return ["error"] * len(nuggets_text_list), is_truncated

        return labels, is_truncated
    except Exception as e:
        return ["error"] * len(nuggets_text_list), True

def process_item_task(item):
    qid = item["qid"]
    query = item["query"]
    block_text = item["block"]
    nuggets_list = item["block_nuggets_assignment"]

    all_preds = []
    truncation_count_local = 0
    BATCH_SIZE = 10
    num_batches = ceil(len(nuggets_list) / BATCH_SIZE)

    for i in range(num_batches):
        batch = nuggets_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_texts = [n["text"] for n in batch]
        
        pred_labels, is_truncated = get_prediction(query, block_text, batch_texts)
        if is_truncated:
            truncation_count_local += 1
            
        all_preds.extend(pred_labels)

    results = []
    for nug, pred in zip(nuggets_list, all_preds):
        results.append({
            "qid": qid,
            "nugget_text": nug["text"],
            "block_pred": pred,
            "block_true": nug["assignment"]
        })
    return results, num_batches, truncation_count_local

def run_end_to_end_evaluation(input_path, gold_path, num_workers=8):
    with open(input_path, "r", encoding="utf-8") as fin:
        input_data = [json.loads(line) for line in fin]
    
    all_predicted_results = []
    total_batches = 0
    total_truncations = 0

    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result, num_batches, truncation_count in tqdm(
                pool.imap_unordered(process_item_task, input_data),
                total=len(input_data),
                desc="Step 1: Running predictions in parallel"
            ):
                all_predicted_results.extend(result)
                total_batches += num_batches
                total_truncations += truncation_count
    else: 
        for item in tqdm(input_data, desc="Step 1: Running predictions in sequential mode"):
            result, num_batches, truncation_count = process_item_task(item)
            all_predicted_results.extend(result)
            total_batches += num_batches
            total_truncations += truncation_count

    if total_batches > 0:
        truncation_rate = (total_truncations / total_batches) * 100
        print(f"\n--- 截断分析 ---")
        print(f"  总预测批次数量: {total_batches}")
        print(f"  截断批次数量: {total_truncations}")
        print(f"  截断率: {truncation_rate:.2f}%")

    nugget_y_true = [res["block_true"] for res in all_predicted_results]
    nugget_y_pred = [res["block_pred"] for res in all_predicted_results]
    print_metrics(nugget_y_true, nugget_y_pred, name="Nugget-level Match")
    overall_y_true, overall_y_pred = [], []
    gold_data_by_qid = {}
    with open(gold_path, "r", encoding="utf-8") as fgold:
        for line in fgold:
            obj = json.loads(line)
            gold_data_by_qid[obj["qid"]] = obj
    predicted_data_by_qid = defaultdict(list)
    for res in all_predicted_results:
        predicted_data_by_qid[res["qid"]].append(res)
        
    for qid, pred_entries in tqdm(predicted_data_by_qid.items(), desc="Step 2: Aggregating overall match"):
        if qid not in gold_data_by_qid:
            continue

        nugget_preds_for_qid = defaultdict(list)
        for entry in pred_entries:
            nugget_preds_for_qid[entry["nugget_text"]].append(entry["block_pred"])

        predicted_assignment = {}
        for text, preds in nugget_preds_for_qid.items():
            max_score = max(MATCH_PRIORITY.get(p, -1) for p in preds)
            predicted_assignment[text] = PRIORITY_TO_LABEL.get(max_score, "not_support")

        gold_item = gold_data_by_qid[qid]
        gold_assignment_dict = {
            nug["text"]: nug["assignment"]
            for nug in gold_item.get("global_nuggets_assignment", [])
        }

        for text, true_label in gold_assignment_dict.items():
            pred_label = predicted_assignment.get(text)
            if pred_label is not None:
                overall_y_true.append(true_label)
                overall_y_pred.append(pred_label)
    
    if overall_y_true:
        print_metrics(overall_y_true, overall_y_pred, name="Overall Match")

if __name__ == "__main__":
    input_path = "/path/to/your/input.jsonl"
    gold_path = "/path/to/your/gold.json"
    num_gpus = 8
    run_end_to_end_evaluation(input_path, gold_path, num_workers=num_gpus)