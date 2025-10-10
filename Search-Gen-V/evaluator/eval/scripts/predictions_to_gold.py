#!/usr/bin/env python3
"""
Utilities to convert evaluator logs into gold/input JSONL formats.

Supported sources:
  - Evaluator predictions JSONL (one record per nugget):
      {"qid": str, "nugget_text": str, "block_pred": str|None, "block_true": str,
       "query": str, "block_text": str}

Outputs:
  - Gold JSONL grouped per qid (optionally with query if provided via template or eval log):
      {"qid": str, "query": str?,
       "global_nuggets_assignment": [{"text": str, "assignment": "support|partial_support|not_support"}, ...]}

  - Input JSONL (updated from a provided template), preserving docids/importance and
    overwriting/setting assignment from aggregated evaluator labels. If per-block aggregation
    is available for the qid, STRICTLY match the template's block[0] (text) to eval's block_text.
    If a per-block aggregation exists for the qid but the block_text cannot be matched, raise an error.
    If no per-block aggregation exists for the qid, fall back to global qid-level aggregation:
      {"qid": str, "query": str, "block": Any, "block_nuggets_assignment": [{"text": str, "docids": List[str], "importance": str, "assignment": "support|partial_support|not_support"}, ...]}

Aggregation policy:
  - support > partial_support > not_support
  - Ignore labels that are None or "error" when aggregating
  - If all labels are invalid/ignored, use a configurable fallback (default: not_support)

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LABEL_PRIORITY: Dict[str, int] = {
    "not_support": 0,
    "partial_support": 1,
    "support": 2,
}


def derive_overall_label(labels: List[Optional[str]], fallback: str) -> str:
    """Derive overall label by max support priority.

    Args:
        labels: List of predicted labels for the same (qid, nugget_text). May include None or "error".
        fallback: Label to use if no valid labels are present.

    Returns:
        The aggregated label among {support, partial_support, not_support}.
    """
    valid_labels = [lb for lb in labels if lb in LABEL_PRIORITY]
    if not valid_labels:
        return fallback
    return max(valid_labels, key=lambda x: LABEL_PRIORITY[x])


def _load_eval_predictions(eval_path: Path) -> Dict[str, Dict[str, List[Optional[str]]]]:
    """Load evaluator predictions JSONL into nested dict structure.

    Returns a mapping: qid -> nugget_text -> List[labels]
    """
    by_qid_by_nugget: Dict[str, Dict[str, List[Optional[str]]]] = defaultdict(lambda: defaultdict(list))
    with eval_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            nugget_text = obj.get("nugget_text")
            block_pred = obj.get("block_pred")
            if qid is None or nugget_text is None:
                continue
            if block_pred is not None:
                block_pred = str(block_pred)
            by_qid_by_nugget[qid][nugget_text].append(block_pred)
    return by_qid_by_nugget


def _load_eval_predictions_rich(
    eval_path: Path,
) -> Tuple[
    Dict[str, Dict[str, List[Optional[str]]]],
    Dict[str, Dict[str, Dict[str, List[Optional[str]]]]],
    Dict[str, str],
]:
    """Load evaluator predictions with optional query and block_text.

    Returns a tuple of:
      - by_qid_by_nugget: qid -> nugget_text -> List[labels]
      - by_qid_by_block_by_nugget: qid -> block_text -> nugget_text -> List[labels]
      - qid_to_query_from_eval: qid -> query (when available)
    """
    by_qid_by_nugget: Dict[str, Dict[str, List[Optional[str]]]] = defaultdict(lambda: defaultdict(list))
    by_qid_by_block_by_nugget: Dict[str, Dict[str, Dict[str, List[Optional[str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    qid_to_query: Dict[str, str] = {}

    with eval_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            nugget_text = obj.get("nugget_text")
            block_pred = obj.get("block_pred")
            query = obj.get("query")
            block_text = obj.get("block_text")
            if qid is None or nugget_text is None:
                continue
            # Strict requirement: query and block_text must be present
            if not isinstance(query, str) or not query:
                raise ValueError(f"Missing 'query' in eval log for qid={qid} nugget='{nugget_text}'")
            if not isinstance(block_text, str) or not block_text:
                raise ValueError(f"Missing 'block_text' in eval log for qid={qid} nugget='{nugget_text}'")

            if qid not in qid_to_query:
                qid_to_query[qid] = query
            if block_pred is not None:
                block_pred = str(block_pred)
            by_qid_by_nugget[qid][nugget_text].append(block_pred)
            by_qid_by_block_by_nugget[qid][block_text][nugget_text].append(block_pred)

    return by_qid_by_nugget, by_qid_by_block_by_nugget, qid_to_query


def _aggregate_labels(
    by_qid_by_nugget: Dict[str, Dict[str, List[Optional[str]]]],
    fallback_on_missing: str,
) -> Dict[str, Dict[str, str]]:
    """Aggregate lists of labels into single assignment per nugget.

    Returns a mapping: qid -> nugget_text -> assignment
    """
    agg: Dict[str, Dict[str, str]] = {}
    for qid, nugget_map in by_qid_by_nugget.items():
        out: Dict[str, str] = {}
        for nugget_text, labels in nugget_map.items():
            out[nugget_text] = derive_overall_label(labels, fallback_on_missing)
        agg[qid] = out
    return agg


def _aggregate_labels_per_block(
    by_qid_by_block_by_nugget: Dict[str, Dict[str, Dict[str, List[Optional[str]]]]],
    fallback_on_missing: str,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Aggregate labels per block into single assignment per nugget.

    Returns: qid -> block_text -> nugget_text -> assignment
    """
    agg: Dict[str, Dict[str, Dict[str, str]]] = {}
    for qid, block_map in by_qid_by_block_by_nugget.items():
        out_blocks: Dict[str, Dict[str, str]] = {}
        for block_text, nugget_map in block_map.items():
            out_nuggets: Dict[str, str] = {}
            for nugget_text, labels in nugget_map.items():
                out_nuggets[nugget_text] = derive_overall_label(labels, fallback_on_missing)
            out_blocks[block_text] = out_nuggets
        agg[qid] = out_blocks
    return agg


def _read_template_for_query_and_nugget_order(
    template_path: Path,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Read input template JSONL to build maps for qid->query and qid->nugget_texts order.

    Returns (qid_to_query, qid_to_nugget_order)
    """
    qid_to_query: Dict[str, str] = {}
    qid_to_order: Dict[str, List[str]] = {}
    with template_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if not qid:
                continue
            query = obj.get("query")
            if isinstance(query, str):
                qid_to_query[qid] = query
            bna = obj.get("block_nuggets_assignment") or []
            order = []
            for item in bna:
                text = (item or {}).get("text")
                if isinstance(text, str):
                    order.append(text)
            if order:
                qid_to_order[qid] = order
    return qid_to_query, qid_to_order


def write_gold_from_eval(
    eval_path: Path,
    output_path: Path,
    fallback_on_missing: str = "not_support",
    template_path: Optional[Path] = None,
) -> None:
    """Convert eval predictions JSONL into gold JSONL grouped by qid.

    If template_path is provided, include "query" and preserve nugget order from template when possible.
    """
    if fallback_on_missing not in LABEL_PRIORITY:
        raise ValueError(
            f"fallback_on_missing must be one of {list(LABEL_PRIORITY.keys())}, got: {fallback_on_missing}"
        )

    by_qid_by_nugget, by_qid_by_block_by_nugget, qid_to_query_from_eval = _load_eval_predictions_rich(eval_path)
    agg = _aggregate_labels(by_qid_by_nugget, fallback_on_missing)
    agg_per_block = _aggregate_labels_per_block(by_qid_by_block_by_nugget, fallback_on_missing)

    qid_to_query: Dict[str, str] = {}
    qid_to_order: Dict[str, List[str]] = {}
    if template_path is not None:
        qid_to_query, qid_to_order = _read_template_for_query_and_nugget_order(template_path)
    # Fill missing queries from eval logs
    for qid, q in qid_to_query_from_eval.items():
        if qid not in qid_to_query:
            qid_to_query[qid] = q
    # Enforce each qid has query present
    for qid in agg.keys():
        if qid not in qid_to_query or not isinstance(qid_to_query[qid], str) or not qid_to_query[qid]:
            raise ValueError(f"Missing 'query' for qid={qid}. Provide via input template or eval log.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for qid, nugget_assignments in agg.items():
            # Determine order: prefer template order, else alphabetical
            order = qid_to_order.get(qid)
            if order:
                texts = [t for t in order if t in nugget_assignments]
                # append any extra texts not present in template
                extra = [t for t in nugget_assignments.keys() if t not in texts]
                texts.extend(sorted(extra))
            else:
                texts = sorted(nugget_assignments.keys())

            results = [{"text": t, "assignment": nugget_assignments[t]} for t in texts]

            line_obj = {"qid": qid, "global_nuggets_assignment": results}
            if qid in qid_to_query:
                line_obj["query"] = qid_to_query[qid]
            fout.write(json.dumps(line_obj, ensure_ascii=False) + "\n")


def write_input_from_template_and_eval(
    template_path: Path,
    output_path: Path,
    agg_labels: Dict[str, Dict[str, str]],
    fallback_on_missing: str = "not_support",
    keep_existing_when_missing: bool = False,
    agg_labels_per_block: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
) -> None:
    """Update an input template JSONL with assignments from aggregated eval labels.

    - Preserves original fields (query, block, docids, importance)
    - Sets/overwrites assignment per nugget using agg_labels
    - When a nugget is missing in agg_labels, uses fallback or keeps existing based on flag
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid")
            if not qid:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            per_qid_global = agg_labels.get(qid, {})
            # Determine block_text from template for per-block override if available (block can be List[str, List[refs]] or str)
            block_field = obj.get("block")
            block_text_from_template: Optional[str] = None
            if isinstance(block_field, list) and block_field:
                if isinstance(block_field[0], str):
                    block_text_from_template = block_field[0]
            elif isinstance(block_field, str):
                block_text_from_template = block_field

            per_qid_block_all: Dict[str, Dict[str, str]] = agg_labels_per_block.get(qid, {}) if agg_labels_per_block else {}
            # If we have any per-block aggregation for this qid, enforce strict matching
            if per_qid_block_all:
                if not isinstance(block_text_from_template, str) or not block_text_from_template:
                    raise ValueError(
                        f"Per-block aggregation exists for qid={qid} but template block_text is missing or invalid."
                    )
                if block_text_from_template not in per_qid_block_all:
                    raise ValueError(
                        f"No per-block aggregation found for qid={qid} and provided block_text. "
                        f"Template block_text startswith='{(block_text_from_template or '')[:80]}...' (len={len(block_text_from_template) if block_text_from_template else 0})."
                    )
                per_qid = per_qid_block_all[block_text_from_template]
            else:
                # Fall back to global qid-level aggregation
                per_qid = per_qid_global
            bna = obj.get("block_nuggets_assignment")
            if isinstance(bna, list):
                new_bna = []
                for item in bna:
                    if not isinstance(item, dict):
                        new_bna.append(item)
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text in per_qid and per_qid[text] in LABEL_PRIORITY:
                        item["assignment"] = per_qid[text]
                    else:
                        if keep_existing_when_missing:
                            pass
                        else:
                            item["assignment"] = fallback_on_missing
                    new_bna.append(item)
                obj["block_nuggets_assignment"] = new_bna
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert eval logs to gold and/or update input JSONL")

    # Primary inputs
    parser.add_argument(
        "--eval-log",
        "-e",
        type=Path,
        required=True,
        help="Path to evaluator predictions JSONL (qid, nugget_text, block_pred, block_true)",
    )

    # Optional template to enrich outputs (query inclusion, nugget order, docids/importance)
    parser.add_argument(
        "--input-template",
        "-t",
        type=Path,
        required=False,
        help="Path to existing input JSONL used for evaluation; will be used to include query in gold and to update input assignments",
    )

    # Outputs
    parser.add_argument(
        "--output-gold",
        "-g",
        type=Path,
        required=False,
        help="Path to output gold JSONL (per qid aggregated assignments)",
    )
    parser.add_argument(
        "--output-input",
        "-i",
        type=Path,
        required=False,
        help="Path to output updated input JSONL (template with assignments overwritten by eval)",
    )

    # Behaviors
    parser.add_argument(
        "--fallback",
        choices=list(LABEL_PRIORITY.keys()),
        default="not_support",
        help="Fallback label when predictions are all invalid for a nugget or when missing",
    )
    parser.add_argument(
        "--keep-existing-when-missing",
        action="store_true",
        help="When updating input, keep existing assignment if eval label is missing for nugget",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # --eval-log is required via argparse

    by_qid_by_nugget, by_qid_by_block_by_nugget, _ = _load_eval_predictions_rich(args.eval_log)
    agg = _aggregate_labels(by_qid_by_nugget, args.fallback)
    agg_per_block = _aggregate_labels_per_block(by_qid_by_block_by_nugget, args.fallback)

    # Generate gold if requested
    if args.output_gold is not None:
        write_gold_from_eval(
            eval_path=args.eval_log,
            output_path=args.output_gold,
            fallback_on_missing=args.fallback,
            template_path=args.input_template,
        )

    # Update input if requested
    if args.output_input is not None:
        if args.input_template is None:
            raise SystemExit("--input-template is required when specifying --output-input")
        write_input_from_template_and_eval(
            template_path=args.input_template,
            output_path=args.output_input,
            agg_labels=agg,
            fallback_on_missing=args.fallback,
            keep_existing_when_missing=args.keep_existing_when_missing,
            agg_labels_per_block=agg_per_block,
        )


if __name__ == "__main__":
    main()


