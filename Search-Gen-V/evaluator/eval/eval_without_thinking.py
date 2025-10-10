import json
from math import ceil, sqrt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from openai import OpenAI
from collections import defaultdict
import multiprocessing
import re
import numpy as np
import ast
import yaml
import xml.etree.ElementTree as ET

N_RUNS = 16

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

MATCH_PRIORITY = {
    "not_support": 0,
    "partial_support": 1,
    "support": 2,
}
PRIORITY_TO_LABEL = {v: k for k, v in MATCH_PRIORITY.items()}
LABELS_ORDERED = ['support', 'partial_support', 'not_support']

def build_multi_nugget_prompt(query: str, block: str, nuggets: list[str]) -> list[dict]:
    system_msg = (
        "You are NuggetMatchJudge, an intelligent assistant.\n"
        "Your task is to read a query, a passage, and nuggets, "
        "and then decide whether the passage supports the nugget in the context of the query.\n"
        "You need to label the nugget as one of the following: support, partial_support, or not_support.\n\n"
    )
    nugget_section = "\n".join([f"- nugget: \"{nug}\"\n" for nug in nuggets])
    user_msg = (
        "Please respond **only** using a Markdown unordered list like this:\n"
        "* support\n"
        "* partial_support\n"
        "* not_support\n"
        "The list should be in the same order as the input nuggets. "
        "Make sure to provide a label for each nugget. "
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
            max_tokens=64,
            extra_body={"top_k": -1, "chat_template_kwargs": {"enable_thinking": False}}
        )
        content = response.choices[0].message.content.strip()
        labels = safe_parse_labels(content)
        if labels is None or not isinstance(labels, list) or len(labels) != len(nuggets_text_list):
            return [None] * len(nuggets_text_list)
        valid_labels = ["support", "partial_support", "not_support"]
        cleaned_labels = [label if label in valid_labels else None for label in labels]
        return cleaned_labels

    except Exception as e:
        return [None] * len(nuggets_text_list)

def process_item_task(item):
    qid = item["qid"]
    query = item["query"]
    block_text = item["block"]
    nuggets_list = item["block_nuggets_assignment"]

    all_preds = []
    BATCH_SIZE = 10
    num_batches = ceil(len(nuggets_list) / BATCH_SIZE)

    for i in range(num_batches):
        batch = nuggets_list[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        batch_texts = [n["text"] for n in batch]
        pred_labels = get_prediction(query, block_text, batch_texts)
        all_preds.extend(pred_labels)

    results = []
    for nug, pred in zip(nuggets_list, all_preds):
        results.append({
            "qid": qid,
            "nugget_text": nug["text"],
            "block_pred": pred,
            "block_true": nug["assignment"]
        })
    return results

def calculate_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return np.nan, np.nan
    m, se = np.mean(a), np.std(a, ddof=1) / sqrt(n)
    h = se * 1.96
    return m, h

def print_summary_metrics(metrics_list, cm_list, name=""):
    print(f"\n--- {name} Summary Metrics over {N_RUNS} Runs ---")

    # Identify all unique metric names, including per-class metrics
    all_metric_names = set()
    for run_metrics in metrics_list:
        all_metric_names.update(run_metrics.keys())

    sorted_metric_names = sorted(list(all_metric_names))

    all_metrics = defaultdict(list)
    for run_metrics in metrics_list:
        for metric_name in sorted_metric_names:
            all_metrics[metric_name].append(run_metrics.get(metric_name, 0.0))

    for metric_name in sorted_metric_names:
        data = all_metrics[metric_name]
        mean, ci_half_width = calculate_confidence_interval(data)
        std_dev = np.std(data, ddof=1) if len(data) > 1 else np.nan

        print(f"  {metric_name.replace('_', ' ').title():<28}: "
              f"Mean = {mean:.4f}, Std Dev = {std_dev:.4f}, "
              f"95% CI = ({mean - ci_half_width:.4f}, {mean + ci_half_width:.4f})")

    if cm_list and len(cm_list[0]) > 0: # Check if cm_list is not empty and has a valid shape
        avg_cm = np.mean(cm_list, axis=0)
        std_cm = np.std(cm_list, axis=0, ddof=1) if len(cm_list) > 1 else np.full_like(avg_cm, np.nan)

        print("\n  --- Average Confusion Matrix ---")
        print(f"  (Row = True Label, Column = Predicted Label)")

        # Corrected line
        header = "True\\Pred".ljust(15) + "".join(f"{label:>15}" for label in LABELS_ORDERED)
        print(f"  {header}")

        # Rows
        for i, true_label in enumerate(LABELS_ORDERED):
            row_str = f"  {true_label:<15}"
            for j, pred_label in enumerate(LABELS_ORDERED):
                val_avg = avg_cm[i, j]
                val_std = std_cm[i, j]
                row_str += f"{val_avg:9.1f} ({val_std:3.1f})"
            print(f"  {row_str}")

def compute_and_add_cm(y_true, y_pred):
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid_pairs:
        return np.zeros((len(LABELS_ORDERED), len(LABELS_ORDERED)))

    y_true_filtered, y_pred_filtered = zip(*valid_pairs)
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=LABELS_ORDERED)
    return cm

def get_metrics(y_true, y_pred):
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not valid_pairs:
        return {}
    
    y_true_filtered, y_pred_filtered = zip(*valid_pairs)
    
    p, r, f1, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    
    metrics = {
        "micro_acc": acc,
        "micro_p": p,
        "micro_r": r,
        "micro_f1": f1,
        "macro_p": p_macro,
        "macro_r": r_macro,
        "macro_f1": f1_macro
    }
    
    report = classification_report(y_true_filtered, y_pred_filtered, output_dict=True, zero_division=0)
    for label in ["support", "partial_support", "not_support"]:
        if label in report:
            metrics[f"{label}_precision"] = report[label]["precision"]
            metrics[f"{label}_recall"] = report[label]["recall"]
            metrics[f"{label}_f1"] = report[label]["f1-score"]

    return metrics

def run_and_collect_metrics(input_data, gold_data_by_qid, num_workers):
    all_predicted_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(process_item_task, input_data):
            all_predicted_results.extend(result)
            
    nugget_y_true = [res["block_true"] for res in all_predicted_results]
    nugget_y_pred = [res["block_pred"] for res in all_predicted_results]
    nugget_metrics = get_metrics(nugget_y_true, nugget_y_pred)
    nugget_cm = compute_and_add_cm(nugget_y_true, nugget_y_pred)
    
    overall_y_true, overall_y_pred = [], []
    predicted_data_by_qid = defaultdict(list)
    for res in all_predicted_results:
        predicted_data_by_qid[res["qid"]].append(res)
        
    for qid, pred_entries in predicted_data_by_qid.items():
        if qid not in gold_data_by_qid:
            continue

        nugget_preds_for_qid = defaultdict(list)
        for entry in pred_entries:
            if entry["block_pred"] is not None:
                nugget_preds_for_qid[entry["nugget_text"]].append(entry["block_pred"])

        predicted_assignment = {}
        for text, preds in nugget_preds_for_qid.items():
            if preds:
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
    
    overall_metrics = {}
    overall_cm = np.zeros((len(LABELS_ORDERED), len(LABELS_ORDERED)))
    if overall_y_true:
        overall_metrics = get_metrics(overall_y_true, overall_y_pred)
        overall_cm = compute_and_add_cm(overall_y_true, overall_y_pred)
    
    return nugget_metrics, overall_metrics, nugget_cm, overall_cm

if __name__ == "__main__":
    input_path = "/path/to/your/input.jsonl"
    gold_path = "/path/to/your/gold.json"
    num_gpus = 8
    
    with open(input_path, "r", encoding="utf-8") as fin:
        input_data = [json.loads(line) for line in fin]
    
    gold_data_by_qid = {}
    with open(gold_path, "r", encoding="utf-8") as fgold:
        for line in fgold:
            obj = json.loads(line)
            gold_data_by_qid[obj["qid"]] = obj
    
    nugget_metrics_runs = []
    overall_metrics_runs = []
    nugget_cm_runs = []
    overall_cm_runs = []

    print(f"Starting {N_RUNS} runs of end-to-end evaluation...")
    for i in tqdm(range(N_RUNS), desc="Total Evaluation Progress"):
        nugget_metrics, overall_metrics, nugget_cm, overall_cm = run_and_collect_metrics(input_data, gold_data_by_qid, num_workers=num_gpus)
        nugget_metrics_runs.append(nugget_metrics)
        overall_metrics_runs.append(overall_metrics)
        nugget_cm_runs.append(nugget_cm)
        overall_cm_runs.append(overall_cm)
        
    print_summary_metrics(nugget_metrics_runs, nugget_cm_runs, name="Nugget-level Match")
    print_summary_metrics(overall_metrics_runs, overall_cm_runs, name="Overall Match")


    