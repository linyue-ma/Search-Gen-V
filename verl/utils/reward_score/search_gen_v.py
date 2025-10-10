import json
import re

VALID_LABELS = {"support", "partial_support", "not_support"}


def extract_reasoning_and_labels(solution_str: str) -> tuple[str, str]:
    if not solution_str:
        return "", ""

    solution_str = solution_str.strip()
    reasoning_matches = re.findall(r"<reasoning>", solution_str)
    if len(reasoning_matches) > 1:
        return "", ""
    
    if "<reasoning>" in solution_str and "</reasoning>" in solution_str:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", solution_str, re.DOTALL)
        if not reasoning_match:
            return "", ""
        
        reasoning_part = reasoning_match.group(1).strip()
        after_reasoning = solution_str.split("</reasoning>", 1)
        if len(after_reasoning) > 1:
            remaining_text = after_reasoning[1]
            if "<reasoning>" in remaining_text:
                return "", ""  
        
        labels_part = solution_str.split("</reasoning>")[-1].strip()
        return reasoning_part, labels_part
    else:
        return "", solution_str


def check_reasoning_format(solution_str: str) -> float:
    if not solution_str:
        return 0.0

    solution_str = solution_str.strip()
    reasoning_open_count = len(re.findall(r"<reasoning>", solution_str))
    reasoning_close_count = len(re.findall(r"</reasoning>", solution_str))

    if reasoning_open_count != 1 or reasoning_close_count != 1:
        return 0.0
    reasoning_pattern = r"<reasoning>.*?</reasoning>"
    reasoning_match = re.search(reasoning_pattern, solution_str, re.DOTALL)

    if not reasoning_match:
        return 0.0

    reasoning_content = reasoning_match.group(0)
    inner_content = re.search(r"<reasoning>(.*?)</reasoning>", reasoning_content, re.DOTALL)
    if not inner_content or not inner_content.group(1).strip():
        return 0.5  

    labels_part = solution_str.split("</reasoning>")[-1].strip()
    if not labels_part:
        return 0.5  

    return 1.0


def parse_labels(predict_str: str) -> list[str]:
    if not predict_str:
        return []

    predict_str = predict_str.strip()
    try:
        obj = json.loads(predict_str)
        if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
            return [x.strip() for x in obj["labels"] if x.strip() in VALID_LABELS]
        elif isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip() in VALID_LABELS]
    except Exception:
        pass
    
    if predict_str.startswith("[") and predict_str.endswith("]"):
        try:
            parsed = eval(predict_str, {}, {})
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip() in VALID_LABELS]
        except Exception:
            pass
    
    lines = [line.strip("*- \n").strip() for line in predict_str.splitlines() if line.strip()]
    if all(x in VALID_LABELS for x in lines) and lines:
        return lines
    elif any(x in VALID_LABELS for x in lines):
        return [x for x in lines if x in VALID_LABELS]

    lines = [line.strip("- \n").strip() for line in predict_str.splitlines() if line.strip()]
    if all(x in VALID_LABELS for x in lines) and lines:
        return lines
    elif any(x in VALID_LABELS for x in lines):
        return [x for x in lines if x in VALID_LABELS]

    xml_matches = re.findall(r"<label>(.*?)</label>", predict_str)
    if xml_matches:
        return [x.strip() for x in xml_matches if x.strip() in VALID_LABELS]

    if "," in predict_str:
        parts = [x.strip() for x in predict_str.split(",")]
        return [x for x in parts if x in VALID_LABELS]

    if "\t" in predict_str:
        parts = [x.strip() for x in predict_str.split("\t")]
        return [x for x in parts if x in VALID_LABELS]

    numbered_matches = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in predict_str.splitlines() if line.strip()]
    if all(x in VALID_LABELS for x in numbered_matches) and numbered_matches:
        return numbered_matches
    elif any(x in VALID_LABELS for x in numbered_matches):
        return [x for x in numbered_matches if x in VALID_LABELS]

    if "|" in predict_str:
        parts = [x.strip() for x in predict_str.split("|")]
        return [x for x in parts if x in VALID_LABELS]

    return []


def check_format_reward(solution_str: str, prompt_format_hint: str) -> float:
    try:
        _, labels_part = extract_reasoning_and_labels(solution_str)
        labels_part = labels_part.strip()

        if prompt_format_hint == "json":
            obj = json.loads(labels_part)
            if isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
                return 1.0
            elif isinstance(obj, list):
                return 1.0

        elif prompt_format_hint == "csv":
            if "," in labels_part and not labels_part.strip().startswith("["):
                return 1.0

        elif prompt_format_hint in ("python_list"):
            if labels_part.startswith("[") and labels_part.endswith("]"):
                return 1.0

        elif prompt_format_hint == "yaml":
            lines = labels_part.splitlines()
            if all(line.strip().startswith("- ") for line in lines if line.strip()):
                return 1.0

        elif prompt_format_hint == "markdown":
            lines = labels_part.splitlines()
            if all(line.strip().startswith("* ") for line in lines if line.strip()):
                return 1.0

        elif prompt_format_hint == "xml":
            if labels_part.startswith("<labels>") and labels_part.endswith("</labels>"):
                if re.search(r"<label>.*?</label>", labels_part):
                    return 1.0

        elif prompt_format_hint == "tsv":
            if "\t" in labels_part:
                return 1.0

        elif prompt_format_hint == "numbered":
            lines = labels_part.splitlines()
            if all(re.match(r"^\d+\.\s+\w+", line.strip()) for line in lines if line.strip()):
                return 1.0

        elif prompt_format_hint == "comma_separated":
            if ", " in labels_part and not labels_part.strip().startswith("["):
                return 1.0

        elif prompt_format_hint == "pipe_separated":
            if "|" in labels_part:
                return 1.0

    except Exception:
        pass

    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    **kwargs
) -> float:
    F1_WEIGHT = 0.35
    PERFECT_MATCH_WEIGHT = 0.35
    REASONING_WEIGHT = 0.2
    FORMAT_WEIGHT = 0.1
    
    nuggets_count = extra_info.get("nuggets_count") if extra_info else None
    prompt_format_hint = extra_info.get("prompt_format_hint") if extra_info else None
    
    reasoning_part, labels_part = extract_reasoning_and_labels(solution_str)

    if not reasoning_part and not labels_part and "<reasoning>" in solution_str:
        return 0.0
    
    pred_labels = parse_labels(labels_part)
    true_labels = parse_labels(ground_truth)
    
    if nuggets_count is not None and len(pred_labels) != nuggets_count:
        return 0.0

    if len(pred_labels) != len(true_labels):
        return 0.0
    
    if not pred_labels:
        return 0.0
    
    f1_score = 0.0
    if pred_labels and true_labels:
        n = min(len(pred_labels), len(true_labels))
        pred_labels_for_f1 = pred_labels[:n]
        true_labels_for_f1 = true_labels[:n]

        scores = []
        for label in VALID_LABELS:
            tp = sum((p == label and t == label) for p, t in zip(pred_labels_for_f1, true_labels_for_f1, strict=False))
            fp = sum((p == label and t != label) for p, t in zip(pred_labels_for_f1, true_labels_for_f1, strict=False))
            fn = sum((p != label and t == label) for p, t in zip(pred_labels_for_f1, true_labels_for_f1, strict=False))

            if tp + fp == 0 and tp + fn == 0:
                f1 = 1.0
            else:
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            scores.append(f1)
        f1_score = sum(scores) / len(scores)

    perfect_match_reward = 1.0 if pred_labels == true_labels else 0.0
    reasoning_reward = check_reasoning_format(solution_str)
    format_reward = check_format_reward(solution_str, prompt_format_hint) if prompt_format_hint else 0.0
    
    final_reward = (
        F1_WEIGHT * f1_score +
        PERFECT_MATCH_WEIGHT * perfect_match_reward +
        REASONING_WEIGHT * reasoning_reward +
        FORMAT_WEIGHT * format_reward
    )
    
    if f1_score == 0:
        return 0.0
    
    reasoning_open_count = len(re.findall(r"<reasoning>", solution_str))
    if reasoning_open_count > 1:
        final_reward *= 0.1  
    
    return final_reward


