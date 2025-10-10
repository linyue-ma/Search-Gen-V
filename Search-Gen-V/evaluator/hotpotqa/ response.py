try:
    import openai  # type: ignore
except Exception:
    openai = None  # fallback if package is unavailable
import argparse
import importlib
import json
import logging
import os
import re
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

try:
    _tenacity = importlib.import_module("tenacity")
    retry = _tenacity.retry
    stop_after_attempt = _tenacity.stop_after_attempt
    wait_random_exponential = _tenacity.wait_random_exponential
    retry_if_exception_type = _tenacity.retry_if_exception_type
    before_sleep_log = _tenacity.before_sleep_log
except Exception:
    # Fallback no-op shims to keep runtime/lints happy if tenacity is missing
    def retry(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None

    def before_sleep_log(*args, **kwargs):
        def _noop(*a, **k):
            return None
        return _noop

def _get_openai_client():
    if openai is None:
        raise RuntimeError("openai package is not available")
    return openai.OpenAI(api_key="test", base_url="http://localhost:30000")

DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)
search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "item": {
                        "type": "string"
                    },
                    "description": "A list of fully-formed semantic queries. The tool will return search results for each query."
                }
            },
            "required": ["query_list"]
        }
    }
}
tools = [search_tool_schema]




# ===== Search tool implementation (single-turn) =====
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1

# Default retrieval service configuration
RETRIEVAL_SERVICE_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_TOPK = 3

logger = logging.getLogger(__name__)


def _passages2string(retrieval_result: list[dict[str, Any]]) -> str:
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        try:
            content = doc_item["document"]["contents"]
        except Exception:
            content = str(doc_item)
        parts = (content or "").split("\n")
        title = parts[0] if parts else ""
        text = "\n".join(parts[1:]) if len(parts) > 1 else ""
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


class RetryableError(Exception):
    """Marker exception for retryable failures."""


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_random_exponential(multiplier=INITIAL_RETRY_DELAY, max=30),
    retry=retry_if_exception_type(RetryableError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = DEFAULT_TOPK,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call remote retrieval API with retry via tenacity.

    Retries on network errors and 5xx server errors with exponential backoff and jitter.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    import ssl
    from urllib import error as urlerr
    from urllib import request as urlreq

    ssl_ctx = ssl.create_default_context()

    logger.info(f"{log_prefix}POST {retrieval_service_url}")

    data = json.dumps(payload).encode("utf-8")
    req = urlreq.Request(retrieval_service_url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        try:
            with urlreq.urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
                status_code = getattr(resp, "status", 200)
                body = resp.read().decode("utf-8")
        except urlerr.HTTPError as e:
            status_code = e.code
            body = e.read().decode("utf-8", errors="ignore") if e.fp else ""
            if status_code in (500, 502, 503, 504):
                # Retry on transient server errors
                raise RetryableError(f"{log_prefix}Server error {status_code}")
            # Non-retryable HTTP errors
            raise RuntimeError(f"{log_prefix}HTTP {status_code}: {body[:200]}")
        except urlerr.URLError as e:
            # Network issues / DNS / connection refused
            raise RetryableError(f"{log_prefix}URLError: {e}")
        except TimeoutError as e:
            # Explicit socket timeout
            raise RetryableError(f"{log_prefix}Timeout: {e}")

        if status_code >= 400:
            if status_code in (500, 502, 503, 504):
                raise RetryableError(f"{log_prefix}HTTP {status_code}")
            raise RuntimeError(f"{log_prefix}HTTP {status_code}: {body[:200]}")

        logger.info(f"{log_prefix}Success")
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            # Treat parse errors as non-retryable; upstream should fix
            raise RuntimeError(f"{log_prefix}JSON decode error: {e}")
    except RetryableError:
        # Let tenacity handle retries
        raise
    except Exception:
        # Non-retryable or final error
        raise


def _perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = DEFAULT_TOPK,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Perform a single batch search for multiple queries and return a formatted string.
    The returned string will be wrapped by <tool_response> ... </tool_response> by the caller.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    try:
        api_response = _call_search_api(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            topk=topk,
            return_scores=True,
            timeout=timeout,
        )
    except Exception as e:
        traceback.print_exc()
        return f"Search error: {e}"

    try:
        raw_results = api_response.get("result", [])
        if not raw_results:
            return "No search results found."

        pretty_results: list[str] = []
        for retrieval in raw_results:
            if isinstance(retrieval, list):
                pretty_results.append(_passages2string(retrieval))
            else:
                pretty_results.append(str(retrieval))

        final_result = "\n---\n".join(pretty_results)
        return final_result
    except Exception as e:
        logger.error(f"Error processing search results: {e}")
        return f"Error processing search results: {e}"


def search(query_list: list[str]) -> str:
    """
    Executable search tool to be called on assistant tool_call.

    Returns a string that will be used directly as the tool turn content.
    """
    if isinstance(query_list, str):
        query_list = [query_list]
    if not isinstance(query_list, list) or not query_list:
        return "<tool_response>\nSearch error: invalid or empty query_list.\n</tool_response>"

    result_str = _perform_single_search_batch(
        retrieval_service_url=RETRIEVAL_SERVICE_URL,
        query_list=query_list,
        topk=DEFAULT_TOPK,
        timeout=DEFAULT_TIMEOUT,
    )
    return result_str


# ===== Rollout and parsing utilities =====
DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_CALL_ROUNDS_DEFAULT = 4
MAX_RESPONSE_TOKENS_DEFAULT = 512


def _parse_tag_blocks(text: str, tag: str) -> list[str]:
    """
    Extracts all blocks between <tag> and </tag> (case-insensitive), trimming whitespace.
    """
    if not text:
        return []
    pattern = re.compile(rf"<\s*{tag}\s*>[\s\S]*?<\s*/\s*{tag}\s*>", re.IGNORECASE)
    matches = pattern.findall(text)
    results: list[str] = []
    for block in matches:
        inner = re.sub(rf"^\s*<\s*{tag}\s*>\s*|\s*<\s*/\s*{tag}\s*>\s*$", "", block, flags=re.IGNORECASE)
        results.append(inner.strip())
    return results


def _extract_answer(text: str) -> str | None:
    blocks = _parse_tag_blocks(text, "answer")
    return blocks[0].strip() if blocks else None


def run_query(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tool_rounds: int = MAX_TOOL_CALL_ROUNDS_DEFAULT,
    max_response_tokens: int = MAX_RESPONSE_TOKENS_DEFAULT,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """
    Execute a rollout with optional tool use capped by max_tool_rounds.

    Returns a dict with keys: "messages", "query", "answer".
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": f"{DEFAULT_USER_CONTENT_PREFIX}{query}"},
    ]

    tool_rounds_used = 0
    consecutive_no_tool_turns = 0
    parsed_answer: str | None = None

    # Generate until we have an <answer> or tool cap and a couple of no-tool turns
    while True:
        completion = _get_openai_client().chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_response_tokens,
        )

        choice = completion.choices[0]
        assistant_message = choice.message
        assistant_content = assistant_message.content or ""

        # Append assistant message as-is
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        # Check for final answer in content
        parsed_answer = _extract_answer(assistant_content)
        if parsed_answer is not None:
            break

        # Handle function tool calls (preferred)
        did_call_tool = False
        tool_calls = getattr(assistant_message, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                if tool_rounds_used >= max_tool_rounds:
                    break
                try:
                    call_type = getattr(tool_call, "type", None)
                    func = getattr(tool_call, "function", None)
                    name = getattr(func, "name", None) if func else None
                    args_str = getattr(func, "arguments", None) if func else None
                    if call_type == "function" and name == "search":
                        try:
                            parsed_args = json.loads(args_str or "{}")
                        except Exception:
                            parsed_args = {}
                        query_list = parsed_args.get("query_list", [])
                        if isinstance(query_list, str):
                            query_list = [query_list]
                        if not isinstance(query_list, list):
                            query_list = []

                        # Call the executable search tool and use its string directly
                        tool_msg_content = search(query_list)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": getattr(tool_call, "id", None),
                            "name": "search",
                            "content": tool_msg_content,
                        })
                        tool_rounds_used += 1
                        did_call_tool = True
                except Exception:
                    # If any tool parsing fails, continue gracefully
                    continue

        # Termination controls
        if did_call_tool:
            consecutive_no_tool_turns = 0
            # Continue to allow model to consume tool outputs
            continue

        # No tools used this turn and no <answer> found
        consecutive_no_tool_turns += 1
        if tool_rounds_used >= max_tool_rounds and consecutive_no_tool_turns >= 2:
            break

    return {
        "messages": messages,
        "query": query,
        "answer": parsed_answer,
    }


def _predict_for_record(
    record: "OrderedDict[str, Any]",
    *,
    model: str,
    max_tool_rounds: int,
    max_response_tokens: int,
    temperature: float,
) -> "OrderedDict[str, Any]":
    """Run prediction for a single JSONL record and append predicted fields.

    This function preserves original key order by operating on an OrderedDict copy
    and appending new keys at the end: "predicted_messages" and "predicted_answer".
    """
    question = str(record.get("question", ""))
    try:
        result = run_query(
            question,
            model=model,
            max_tool_rounds=max_tool_rounds,
            max_response_tokens=max_response_tokens,
            temperature=temperature,
        )
        predicted_messages = result.get("messages", [])
        predicted_answer = result.get("answer")
    except Exception:
        logger.exception("Prediction failed for record id=%s", record.get("id"))
        predicted_messages = []
        predicted_answer = None

    out_rec: OrderedDict[str, Any] = OrderedDict(record)
    out_rec["predicted_messages"] = predicted_messages
    out_rec["predicted_answer"] = predicted_answer
    return out_rec


def process_jsonl_file(
    input_path: str,
    output_path: str,
    *,
    workers: int,
    model: str,
    max_tool_rounds: int,
    max_response_tokens: int,
    temperature: float,
) -> None:
    """Process a JSONL file concurrently and write augmented results to output_path.

    - Reads each line as a JSON object while preserving key order
    - Calls run_query(question) per line
    - Appends predicted_messages and predicted_answer to each object
    - Writes results as JSONL in the same order as input
    """
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    logger.info(
        "Starting batch processing: input=%s, output=%s, workers=%d",
        input_path,
        output_path,
        workers,
    )

    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        # Load all valid records to maintain strict input order in output
        records: list[OrderedDict[str, Any]] = []
        for line_no, line in enumerate(fin, start=1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                obj = cast(OrderedDict[str, Any], json.loads(line_stripped, object_pairs_hook=OrderedDict))
                if not isinstance(obj, dict):
                    raise ValueError("JSON object expected")
                records.append(obj)  # OrderedDict due to object_pairs_hook
            except Exception:
                logger.exception("Skipping invalid JSONL on line %d", line_no)
                continue

        if not records:
            logger.warning("No valid records found in input file.")
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _predict_for_record,
                    rec,
                    model=model,
                    max_tool_rounds=max_tool_rounds,
                    max_response_tokens=max_response_tokens,
                    temperature=temperature,
                )
                for rec in records
            ]

            # Collect results in input order by indexing over futures list
            results: list[OrderedDict[str, Any]] = [None] * len(futures)  # type: ignore[assignment]
            for idx, fut in enumerate(futures):
                try:
                    results[idx] = fut.result()
                except Exception:
                    logger.exception("Worker failed at index %d", idx)
                    # On failure, still write the original record with empty predictions
                    out_rec = OrderedDict(records[idx])
                    out_rec["predicted_messages"] = []
                    out_rec["predicted_answer"] = None
                    results[idx] = out_rec

        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False))
            fout.write("\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run query rollout or batch JSONL processing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", type=str, help="Single user query text")
    group.add_argument("--input", "-i", type=str, help="Path to input JSONL file")

    parser.add_argument("--output", "-o", type=str, help="Path to output JSONL file (required for --input)")
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 1) * 2), help="Number of ThreadPool workers for batch mode")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--max-tool-rounds", type=int, default=MAX_TOOL_CALL_ROUNDS_DEFAULT, help="Maximum tool call rounds")
    parser.add_argument("--max-response-tokens", type=int, default=MAX_RESPONSE_TOKENS_DEFAULT, help="Maximum tokens per response")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.input:
        if not args.output:
            raise SystemExit("--output is required when using --input for batch mode")
        process_jsonl_file(
            args.input,
            args.output,
            workers=args.workers,
            model=args.model,
            max_tool_rounds=args.max_tool_rounds,
            max_response_tokens=args.max_response_tokens,
            temperature=args.temperature,
        )
    else:
        # Single query mode
        result = run_query(
            args.query,
            model=args.model,
            max_tool_rounds=args.max_tool_rounds,
            max_response_tokens=args.max_response_tokens,
            temperature=args.temperature,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()