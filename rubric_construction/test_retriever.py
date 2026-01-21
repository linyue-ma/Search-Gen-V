import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import aiohttp


@dataclass
class RequestStats:
    total_requests: int = 0
    total_success: int = 0
    total_errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def add_latency(self, latency_ms: float):
        self.latencies_ms.append(latency_ms)

    def merge(self, other: "RequestStats"):
        self.total_requests += other.total_requests
        self.total_success += other.total_success
        self.total_errors += other.total_errors
        self.latencies_ms.extend(other.latencies_ms)


class AsyncStatsCollector:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._stats = RequestStats()

    async def record(self, ok: bool, latency_ms: Optional[float]):
        async with self._lock:
            self._stats.total_requests += 1
            if ok:
                self._stats.total_success += 1
            else:
                self._stats.total_errors += 1
            if latency_ms is not None:
                self._stats.latencies_ms.append(latency_ms)

    async def snapshot(self) -> RequestStats:
        async with self._lock:
            snap = RequestStats(
                total_requests=self._stats.total_requests,
                total_success=self._stats.total_success,
                total_errors=self._stats.total_errors,
                latencies_ms=list(self._stats.latencies_ms),
            )
        return snap


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def now_ms() -> float:
    return time.perf_counter() * 1000.0


async def bounded_sleep_until(start_ms: float, interval_ms: float):
    target = start_ms + interval_ms
    remaining = (target - now_ms()) / 1000.0
    if remaining > 0:
        await asyncio.sleep(remaining)


async def do_post_json(session: aiohttp.ClientSession, url: str, data: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    async with session.post(url, json=data, timeout=timeout_s) as resp:
        resp.raise_for_status()
        return await resp.json()


async def search_once(session: aiohttp.ClientSession,
                      base_url: str,
                      query: str,
                      k: int,
                      timeout_s: float,
                      stats: AsyncStatsCollector) -> List[Dict[str, Any]]:
    url = f"{base_url}/search"
    payload = {"query": query, "k": k}
    start = now_ms()
    try:
        res = await do_post_json(session, url, payload, timeout_s)
        latency = now_ms() - start
        hits = res.get("hits", [])
        await stats.record(True, latency)
        return hits
    except Exception:
        await stats.record(False, None)
        return []


async def fetch_once(session: aiohttp.ClientSession,
                     base_url: str,
                     segid: str,
                     timeout_s: float,
                     stats: AsyncStatsCollector) -> Optional[Dict[str, Any]]:
    url = f"{base_url}/fetch"
    payload = {"segid": segid}
    start = now_ms()
    try:
        res = await do_post_json(session, url, payload, timeout_s)
        latency = now_ms() - start
        await stats.record(True, latency)
        return res.get("segment")
    except Exception:
        await stats.record(False, None)
        return None


async def worker_task(
    worker_id: int,
    session: aiohttp.ClientSession,
    base_url: str,
    queries: List[str],
    search_k: int,
    do_fetch: bool,
    fetch_top_n: int,
    timeout_s: float,
    stats_search: AsyncStatsCollector,
    stats_fetch: AsyncStatsCollector,
    rate_limit_interval_ms: Optional[float],
    total_iterations: int,
):
    start_time_ms = now_ms()
    for i in range(total_iterations):
        # Simple rate limiting per worker by fixed interval pacing
        if rate_limit_interval_ms is not None:
            await bounded_sleep_until(start_time_ms, rate_limit_interval_ms * (i + 1))

        query = random.choice(queries)
        hits = await search_once(session, base_url, query, search_k, timeout_s, stats_search)

        if do_fetch and hits:
            fetch_tasks = []
            for hit in hits[:fetch_top_n]:
                segid = str(hit.get("docid"))
                fetch_tasks.append(
                    fetch_once(session, base_url, segid, timeout_s, stats_fetch)
                )
            if fetch_tasks:
                await asyncio.gather(*fetch_tasks, return_exceptions=True)


async def run_load_test(
    base_url: str,
    concurrency: int,
    total_requests: int,
    qps: Optional[float],
    timeout_s: float,
    queries: List[str],
    search_k: int,
    do_fetch: bool,
    fetch_top_n: int,
):
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None)

    stats_search = AsyncStatsCollector()
    stats_fetch = AsyncStatsCollector()

    # Work distribution
    iterations_per_worker = total_requests // concurrency
    remainder = total_requests % concurrency
    per_worker = [iterations_per_worker + (1 if i < remainder else 0) for i in range(concurrency)]

    rate_interval_ms = None
    if qps and qps > 0:
        # Total QPS is split across workers
        per_worker_qps = max(qps / concurrency, 0.0001)
        rate_interval_ms = 1000.0 / per_worker_qps

    start_wall = time.time()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for wid in range(concurrency):
            if per_worker[wid] == 0:
                continue
            tasks.append(
                worker_task(
                    worker_id=wid,
                    session=session,
                    base_url=base_url,
                    queries=queries,
                    search_k=search_k,
                    do_fetch=do_fetch,
                    fetch_top_n=fetch_top_n,
                    timeout_s=timeout_s,
                    stats_search=stats_search,
                    stats_fetch=stats_fetch,
                    rate_limit_interval_ms=rate_interval_ms,
                    total_iterations=per_worker[wid],
                )
            )
        await asyncio.gather(*tasks)

    end_wall = time.time()
    duration_s = max(end_wall - start_wall, 1e-9)

    s_snap = await stats_search.snapshot()
    f_snap = await stats_fetch.snapshot()

    def summarize(name: str, st: RequestStats):
        lat = st.latencies_ms
        summary = {
            "name": name,
            "requests": st.total_requests,
            "success": st.total_success,
            "errors": st.total_errors,
            "error_rate": round(st.error_rate * 100, 2),
            "throughput_rps": round((st.total_requests / duration_s), 2),
            "latency_ms_p50": round(percentile(lat, 50), 2),
            "latency_ms_p90": round(percentile(lat, 90), 2),
            "latency_ms_p95": round(percentile(lat, 95), 2),
            "latency_ms_p99": round(percentile(lat, 99), 2),
            "latency_ms_max": round(max(lat) if lat else 0.0, 2),
        }
        return summary

    report = {
        "duration_seconds": round(duration_s, 3),
        "base_url": base_url,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "qps": qps,
        "search_k": search_k,
        "do_fetch": do_fetch,
        "fetch_top_n": fetch_top_n,
        "search": summarize("search", s_snap),
        "fetch": summarize("fetch", f_snap) if do_fetch else None,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent load tester for retriever_server.py")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:8000", help="Base URL of retriever server")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=100, help="Total number of /search requests to send")
    parser.add_argument("--qps", type=float, default=None, help="Target total QPS (approx). If unset, run as fast as possible")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout seconds")
    parser.add_argument("--k", type=int, default=20, help="Top-k for /search")
    parser.add_argument("--fetch", action="store_true", help="After /search, call /fetch on top-N docids")
    parser.add_argument("--fetch-top-n", type=int, default=3, help="How many docids from /search to fetch")
    parser.add_argument("--queries", type=str, default=None, help="Path to a text file with one query per line")

    return parser.parse_args()


def load_queries(path: Optional[str]) -> List[str]:
    if path is None:
        # Fallback small pool to avoid empty input
        return [
            "what is information retrieval",
            "neural networks basics",
            "climate change impact on economy",
            "python asyncio tutorial",
            "large language models",
        ]
    with open(path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    if not queries:
        raise ValueError("Query list is empty")
    return queries


def main():
    args = parse_args()
    queries = load_queries(args.queries)

    asyncio.run(
        run_load_test(
            base_url=args.host,
            concurrency=max(1, args.concurrency),
            total_requests=max(1, args.requests),
            qps=args.qps if (args.qps is None or args.qps > 0) else None,
            timeout_s=max(0.1, args.timeout),
            queries=queries,
            search_k=max(1, args.k),
            do_fetch=bool(args.fetch),
            fetch_top_n=max(0, args.fetch_top_n),
        )
    )


if __name__ == "__main__":
    main()
