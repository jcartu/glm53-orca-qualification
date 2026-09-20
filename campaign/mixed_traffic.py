#!/usr/bin/env python3
"""Compare vLLM prefill-fairness engines under reproducible mixed traffic.

Each cell keeps C long decode streams active while a seeded Poisson arrival
process injects cold prefills whose lengths are sampled log-uniformly from the
configured range. The script switches fairness policy only at an idle engine
boundary and writes raw JSON, a compact CSV summary, and a dependency-free SVG.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import math
import random
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


POLICY_COLORS = {
    "off": "#7a8491",
    "compute_share_0.2": "#8cc8f2",
    "compute_share_0.4": "#119a8e",
    "compute_share_0.5": "#1683d8",
    "compute_share_0.8": "#07539a",
    "compute_share": "#1683d8",
}
FAIRNESS_CONFIG_FIELDS = ("prefill_compute_share", "prefill_compute_half_life")


def policy_label(policy: str) -> str:
    if policy.startswith("compute_share_"):
        return f"compute {policy.removeprefix('compute_share_')}"
    return policy


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def parse_prometheus(text: str) -> dict[str, float]:
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:scheduler_compute_seconds_total",
        "vllm:decode_prefill_scheduled_tokens_total",
        "vllm:decode_prefill_decode_only_steps_total",
        "vllm:decode_prefill_fairness_bypasses_total",
    }
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in wanted:
            continue
        if name == "vllm:scheduler_compute_seconds_total":
            for service_class in ("decode", "prefill"):
                if f'class="{service_class}"' in line:
                    name = f"{name}:{service_class}"
                    break
            else:
                continue
        if name in parsed:
            continue
        try:
            parsed[name] = float(line.rsplit(None, 1)[1])
        except (IndexError, ValueError):
            continue
    return parsed


async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict[str, float]:
    response = await client.get(f"{base_url}/metrics")
    response.raise_for_status()
    return parse_prometheus(response.text)


async def wait_for_idle(
    client: httpx.AsyncClient, base_url: str, timeout: float = 120.0
) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        last = await scrape_metrics(client, base_url)
        if (
            last.get("vllm:num_requests_running", 0.0) == 0.0
            and last.get("vllm:num_requests_waiting", 0.0) == 0.0
        ):
            return last
        await asyncio.sleep(0.5)
    raise RuntimeError(f"engine did not become idle: {last}")


async def tokenize_count(
    client: httpx.AsyncClient, base_url: str, model: str, content: str
) -> int:
    response = await client.post(
        f"{base_url}/tokenize",
        json={"model": model, "messages": [{"role": "user", "content": content}]},
    )
    response.raise_for_status()
    return int(response.json()["count"])


async def exact_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    target_tokens: int,
    identity: str,
) -> str:
    """Construct a prompt with an exact templated token count.

    The unique identity appears before the repeated body, preventing reuse even
    if a connector ignores cache_salt. Direct correction normally converges in
    two requests; binary search is retained as a tokenizer-independent fallback.
    """
    prefix = (
        "Mixed scheduler traffic probe. Treat this as independent filler.\n"
        f"TRAFFIC-ID: {identity}\n"
        "BEGIN FILLER\n"
    )
    suffix = "\nEND FILLER\nReply with one period."
    base_count = await tokenize_count(client, base_url, model, prefix + suffix)
    repetitions = max(target_tokens - base_count, 0)
    for _ in range(6):
        content = prefix + (" x" * repetitions) + suffix
        observed = await tokenize_count(client, base_url, model, content)
        if observed == target_tokens:
            return content
        repetitions = max(repetitions + target_tokens - observed, 0)

    low, high = 0, max(repetitions * 2 + 64, target_tokens * 2)
    while low <= high:
        middle = (low + high) // 2
        content = prefix + (" x" * middle) + suffix
        observed = await tokenize_count(client, base_url, model, content)
        if observed == target_tokens:
            return content
        if observed < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    raise RuntimeError(f"cannot construct exact {target_tokens}-token prompt")


async def iter_sse(response: httpx.Response):
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


async def decode_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    content: str,
    cache_salt: str,
    max_tokens: int,
    stop: asyncio.Event,
    state: dict[str, Any],
) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "cache_salt": cache_salt,
    }
    state.update(samples=[], chunk_times=[], completion_tokens=0)
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(f"decode HTTP {response.status_code}: {body[:1000]}")
        last_count = 0
        async for event in iter_sse(response):
            now = time.monotonic()
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                count = int(usage["completion_tokens"])
                if count > last_count:
                    state["samples"].append((now, count))
                    state["completion_tokens"] = count
                    last_count = count
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if any(
                    delta.get(name)
                    for name in ("reasoning", "reasoning_content", "content")
                ):
                    state["chunk_times"].append(now)
            if stop.is_set():
                return
    state["finished_early"] = True


async def prefill_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    content: str,
    cache_salt: str,
    offered_at: float,
) -> dict[str, Any]:
    delay = offered_at - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "cache_salt": cache_salt,
    }
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(f"prefill HTTP {response.status_code}: {body[:1000]}")
        async for event in iter_sse(response):
            now = time.monotonic()
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if first_token_at is None and any(
                    delta.get(name)
                    for name in ("reasoning", "reasoning_content", "content")
                ):
                    first_token_at = now
    finished = time.monotonic()
    first_token_at = first_token_at or finished
    return {
        "offered_at": offered_at,
        "started_at": started,
        "first_token_at": first_token_at,
        "finished_at": finished,
        "queue_to_first_token_seconds": first_token_at - offered_at,
        "request_to_first_token_seconds": first_token_at - started,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def count_at(samples: list[tuple[float, int]], timestamp: float) -> int:
    result = 0
    for sample_time, count in samples:
        if sample_time > timestamp:
            break
        result = count
    return result


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def policy_payload(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "off":
        return {"prefill_compute_share": None, "prefill_compute_half_life": None}
    if name == "compute_share" or name.startswith("compute_share_"):
        target = (
            float(name.removeprefix("compute_share_"))
            if name.startswith("compute_share_")
            else args.compute_share
        )
        if not 0.0 < target < 1.0:
            raise ValueError(f"compute-share target must be between 0 and 1: {target}")
        return {"prefill_compute_share": target, "prefill_compute_half_life": None}
    raise ValueError(f"unknown policy: {name}")


async def set_fairness_config(
    client: httpx.AsyncClient,
    base_url: str,
    requested: dict[str, Any],
) -> dict[str, Any]:
    await wait_for_idle(client, base_url)
    response = await client.post(f"{base_url}/prefill_fairness", json=requested)
    if response.status_code != 200:
        raise RuntimeError(
            f"cannot set fairness: HTTP {response.status_code}: {response.text}"
        )
    result = response.json()
    if result.get("applied") is not True:
        raise RuntimeError(f"fairness update was not applied: {result}")
    config = result["config"]
    if any(
        key not in config or config[key] != value for key, value in requested.items()
    ):
        raise RuntimeError(
            f"active fairness configuration differs from request: {config}"
        )
    return config


async def metrics_sampler(
    client: httpx.AsyncClient,
    base_url: str,
    stop: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    while not stop.is_set():
        try:
            samples.append(
                {"time": time.monotonic(), **await scrape_metrics(client, base_url)}
            )
        except Exception as error:
            samples.append({"time": time.monotonic(), "error": str(error)})
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            pass


async def wait_for_decode_warmup(
    states: list[dict[str, Any]],
    tasks: list[asyncio.Task[Any]],
    target: int,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for task in tasks:
            if task.done() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        if all(int(state.get("completion_tokens", 0)) >= target for state in states):
            return
        await asyncio.sleep(0.05)
    raise RuntimeError("decode streams did not reach warmup target")


async def run_cell(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    policy: str,
    concurrency: int,
    repeat: int,
    lengths: list[int],
    arrival_offsets: list[float],
    run_identity: str,
) -> dict[str, Any]:
    active_config = await set_fairness_config(
        client, args.base_url, policy_payload(policy, args)
    )
    cell_id = f"{run_identity}-{policy}-c{concurrency}-r{repeat}"
    decode_contents = [
        await exact_prompt(
            client,
            args.base_url,
            args.model,
            args.decode_prompt_tokens,
            f"{cell_id}-decode-{index}",
        )
        for index in range(concurrency)
    ]
    prefill_contents = [
        await exact_prompt(
            client,
            args.base_url,
            args.model,
            length,
            f"{cell_id}-prefill-{index}",
        )
        for index, length in enumerate(lengths)
    ]

    decode_stop = asyncio.Event()
    metric_stop = asyncio.Event()
    decode_states: list[dict[str, Any]] = [{} for _ in range(concurrency)]
    metric_samples: list[dict[str, Any]] = []
    decode_tasks = [
        asyncio.create_task(
            decode_stream(
                client,
                args.base_url,
                args.model,
                content,
                f"{cell_id}-decode-salt-{index}",
                args.decode_max_tokens,
                decode_stop,
                decode_states[index],
            )
        )
        for index, content in enumerate(decode_contents)
    ]
    sampler = asyncio.create_task(
        metrics_sampler(client, args.base_url, metric_stop, metric_samples)
    )
    prefill_tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        await wait_for_decode_warmup(
            decode_states, decode_tasks, args.decode_warmup_tokens, args.warmup_timeout
        )
        baseline_start = time.monotonic()
        baseline_start_counts = [
            count_at(state["samples"], baseline_start) for state in decode_states
        ]
        await asyncio.sleep(args.baseline_seconds)
        baseline_end = time.monotonic()
        baseline_end_counts = [
            count_at(state["samples"], baseline_end) for state in decode_states
        ]
        metrics_before = await scrape_metrics(client, args.base_url)

        collision_start = time.monotonic()
        collision_start_counts = [
            count_at(state["samples"], collision_start) for state in decode_states
        ]
        prefill_tasks = [
            asyncio.create_task(
                prefill_request(
                    client,
                    args.base_url,
                    args.model,
                    content,
                    f"{cell_id}-prefill-salt-{index}",
                    collision_start + arrival_offsets[index],
                )
            )
            for index, content in enumerate(prefill_contents)
        ]
        if args.window_seconds > 0.0:
            window_end = collision_start + args.window_seconds
            remaining = window_end - time.monotonic()
            if remaining > 0.0:
                await asyncio.sleep(remaining)
            collision_end = time.monotonic()
            prefill_results = []
            for task in prefill_tasks:
                if not task.done():
                    continue
                result = task.result()
                prefill_results.append(result)
        else:
            prefill_results = await asyncio.gather(*prefill_tasks)
            collision_end = max(row["first_token_at"] for row in prefill_results)
        collision_end_counts = [
            count_at(state["samples"], collision_end) for state in decode_states
        ]
        metrics_after = await scrape_metrics(client, args.base_url)
        end_running = metrics_after.get("vllm:num_requests_running", 0.0)
        end_waiting = metrics_after.get("vllm:num_requests_waiting", 0.0)
        for task in prefill_tasks:
            if not task.done():
                task.cancel()
        if prefill_tasks:
            await asyncio.gather(*prefill_tasks, return_exceptions=True)
        await asyncio.sleep(args.recovery_seconds)
    finally:
        for task in prefill_tasks:
            if not task.done():
                task.cancel()
        if prefill_tasks:
            await asyncio.gather(*prefill_tasks, return_exceptions=True)
        decode_stop.set()
        for task in decode_tasks:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except asyncio.TimeoutError:
                task.cancel()
        metric_stop.set()
        await sampler

    baseline_seconds = baseline_end - baseline_start
    collision_seconds = collision_end - collision_start
    baseline_tokens = sum(
        end - start for start, end in zip(baseline_start_counts, baseline_end_counts)
    )
    decode_tokens = sum(
        end - start for start, end in zip(collision_start_counts, collision_end_counts)
    )
    baseline_decode_tps = baseline_tokens / baseline_seconds
    collision_decode_tps = decode_tokens / collision_seconds
    per_stream_tps = [
        (end - start) / collision_seconds
        for start, end in zip(collision_start_counts, collision_end_counts)
    ]
    prefill_tokens = sum(row["prompt_tokens"] for row in prefill_results)
    prefill_tps = prefill_tokens / collision_seconds
    total_useful_tps = (prefill_tokens + decode_tokens) / collision_seconds
    gaps = [
        current - previous
        for state in decode_states
        for previous, current in zip(state["chunk_times"], state["chunk_times"][1:])
        if collision_start <= current <= collision_end
    ]
    ttfts = [row["queue_to_first_token_seconds"] for row in prefill_results]
    denominator = concurrency * sum(rate * rate for rate in per_stream_tps)
    jain = sum(per_stream_tps) ** 2 / denominator if denominator > 0.0 else None
    metric_window = [
        row for row in metric_samples if collision_start <= row["time"] <= collision_end
    ]
    max_running = max(
        (row.get("vllm:num_requests_running", 0.0) for row in metric_window),
        default=0.0,
    )
    max_waiting = max(
        (row.get("vllm:num_requests_waiting", 0.0) for row in metric_window),
        default=0.0,
    )
    decode_compute = metric_delta(
        metrics_before, metrics_after, "vllm:scheduler_compute_seconds_total:decode"
    )
    prefill_compute = metric_delta(
        metrics_before, metrics_after, "vllm:scheduler_compute_seconds_total:prefill"
    )
    compute_total = decode_compute + prefill_compute
    return {
        "policy": policy,
        "active_config": active_config,
        "concurrency": concurrency,
        "repeat": repeat,
        "cell_id": cell_id,
        "prefill_lengths": lengths,
        "arrival_offsets_seconds": arrival_offsets,
        "baseline_seconds": baseline_seconds,
        "collision_seconds": collision_seconds,
        "baseline_decode_tokens_per_second": baseline_decode_tps,
        "collision_decode_tokens_per_second": collision_decode_tps,
        "decode_retention_percent": (
            100.0 * collision_decode_tps / baseline_decode_tps
            if baseline_decode_tps > 0.0
            else None
        ),
        "decode_tokens": decode_tokens,
        "per_stream_decode_tokens_per_second": per_stream_tps,
        "decode_jain_fairness": jain,
        "decode_inter_chunk_gap_seconds": {
            "median": statistics.median(gaps) if gaps else None,
            "p95": percentile(gaps, 0.95),
            "p99": percentile(gaps, 0.99),
            "max": max(gaps) if gaps else None,
        },
        "prefill_tokens": prefill_tokens,
        "offered_prefill_tokens": sum(lengths),
        "offered_prefill_requests": len(lengths),
        "completed_prefill_requests": len(prefill_results),
        "prefill_request_completion_percent": (
            100.0 * len(prefill_results) / len(lengths) if lengths else 100.0
        ),
        "prefill_tokens_per_second": prefill_tps,
        "prefill_ttft_seconds": {
            "median": statistics.median(ttfts),
            "p95": percentile(ttfts, 0.95),
            "max": max(ttfts),
        },
        "total_useful_tokens_per_second": total_useful_tps,
        "scheduler": {
            "max_running": max_running,
            "max_waiting": max_waiting,
            "end_running_before_cancel": end_running,
            "end_waiting_before_cancel": end_waiting,
            "decode_compute_seconds": decode_compute,
            "prefill_compute_seconds": prefill_compute,
            "measured_prefill_compute_share": (
                prefill_compute / compute_total if compute_total > 0.0 else None
            ),
        },
        "prefills": prefill_results,
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["policy"], cell["concurrency"])].append(cell)
    rows: list[dict[str, Any]] = []
    for (policy, concurrency), selected in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        rows.append(
            {
                "policy": policy,
                "concurrency": concurrency,
                "runs": len(selected),
                "total_useful_tokens_per_second": mean(
                    [row["total_useful_tokens_per_second"] for row in selected]
                ),
                "decode_tokens_delivered": mean(
                    [row["decode_tokens"] for row in selected]
                ),
                "baseline_decode_tokens_per_second": mean(
                    [row["baseline_decode_tokens_per_second"] for row in selected]
                ),
                "collision_decode_tokens_per_second": mean(
                    [row["collision_decode_tokens_per_second"] for row in selected]
                ),
                "decode_retention_percent": mean(
                    [row["decode_retention_percent"] for row in selected]
                ),
                "prefill_tokens_per_second": mean(
                    [row["prefill_tokens_per_second"] for row in selected]
                ),
                "prefill_tokens_completed": mean(
                    [row["prefill_tokens"] for row in selected]
                ),
                "prefill_request_completion_percent": mean(
                    [row["prefill_request_completion_percent"] for row in selected]
                ),
                "end_waiting_requests": mean(
                    [row["scheduler"]["end_waiting_before_cancel"] for row in selected]
                ),
                "decode_gap_p95_ms": 1000.0
                * mean(
                    [
                        row["decode_inter_chunk_gap_seconds"]["p95"]
                        for row in selected
                        if row["decode_inter_chunk_gap_seconds"]["p95"] is not None
                    ]
                ),
                "decode_gap_p99_ms": 1000.0
                * mean(
                    [
                        row["decode_inter_chunk_gap_seconds"]["p99"]
                        for row in selected
                        if row["decode_inter_chunk_gap_seconds"]["p99"] is not None
                    ]
                ),
                "prefill_ttft_p95_seconds": mean(
                    [row["prefill_ttft_seconds"]["p95"] for row in selected]
                ),
                "decode_jain_fairness": mean(
                    [
                        row["decode_jain_fairness"]
                        for row in selected
                        if row["decode_jain_fairness"] is not None
                    ]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nice_ceiling(value: float) -> float:
    if value <= 0.0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    return math.ceil(value / magnitude * 1.1) * magnitude


def render_svg(
    path: Path,
    rows: list[dict[str, Any]],
    concurrencies: list[int],
    policies: list[str],
    subtitle: str,
) -> None:
    width, height = 1560, 720
    panel_width, panel_height = 450, 430
    panel_y = 150
    panel_xs = [60, 555, 1050]
    metrics = [
        (
            "collision_decode_tokens_per_second",
            "Aggregate decode rate",
            "tokens / second",
            True,
        ),
        (
            "prefill_tokens_per_second",
            "Prefill throughput",
            "tokens / second",
            True,
        ),
        ("decode_gap_p99_ms", "Decode p99 gap", "milliseconds", False),
    ]
    lookup = {(row["policy"], row["concurrency"]): row for row in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<text x="60" y="54" font-family="sans-serif" font-size="30" font-weight="700" fill="#17202a">GLM-5.3-Flash mixed decode + cold-prefill sweep</text>',
        f'<text x="60" y="86" font-family="sans-serif" font-size="16" fill="#53606d">{html.escape(subtitle)}</text>',
    ]
    legend_x = 60
    for policy in policies:
        color = POLICY_COLORS.get(policy, "#555")
        parts.append(
            f'<rect x="{legend_x}" y="105" width="16" height="16" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 23}" y="118" font-family="sans-serif" font-size="14" fill="#25313d">{html.escape(policy_label(policy))}</text>'
        )
        legend_x += 175

    for panel_x, (key, title, unit, higher) in zip(panel_xs, metrics):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        ymax = nice_ceiling(max(values, default=1.0))
        chart_x = panel_x + 58
        chart_y = panel_y + 48
        chart_w = panel_width - 78
        chart_h = panel_height - 105
        parts.append(
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" rx="14" fill="white" stroke="#dce2e8"/>'
        )
        parts.append(
            f'<text x="{panel_x + 24}" y="{panel_y + 31}" font-family="sans-serif" font-size="19" font-weight="700" fill="#24313d">{title}</text>'
        )
        parts.append(
            f'<text x="{panel_x + panel_width - 24}" y="{panel_y + 31}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#66737f">{"higher" if higher else "lower"} is better</text>'
        )
        for tick in range(5):
            value = ymax * tick / 4
            y = chart_y + chart_h - chart_h * tick / 4
            parts.append(
                f'<line x1="{chart_x}" y1="{y:.1f}" x2="{chart_x + chart_w}" y2="{y:.1f}" stroke="#e8edf2"/>'
            )
            parts.append(
                f'<text x="{chart_x - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#71808e">{value:.0f}</text>'
            )
        group_w = chart_w / len(concurrencies)
        bar_w = min(28.0, group_w / (len(policies) + 1))
        for group_index, concurrency in enumerate(concurrencies):
            center = chart_x + group_w * (group_index + 0.5)
            total_bar_w = bar_w * len(policies)
            for policy_index, policy in enumerate(policies):
                row = lookup.get((policy, concurrency))
                if row is None or row.get(key) is None:
                    continue
                value = float(row[key])
                bar_h = chart_h * value / ymax
                x = center - total_bar_w / 2 + policy_index * bar_w
                y = chart_y + chart_h - bar_h
                color = POLICY_COLORS.get(policy, "#555")
                label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
                parts.append(
                    f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>'
                )
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 6, chart_y + 10):.1f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#34414d">{label}</text>'
                )
            parts.append(
                f'<text x="{center:.1f}" y="{chart_y + chart_h + 24}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="600" fill="#34414d">C{concurrency}</text>'
            )
        parts.append(
            f'<text x="{panel_x + panel_width / 2}" y="{panel_y + panel_height - 17}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#71808e">{html.escape(unit)}</text>'
        )

    parts.append(
        '<text x="60" y="665" font-family="sans-serif" font-size="14" fill="#43515e">Rates are normalized by observed collision duration. Decode service and prefill throughput are a trade-off.</text>'
    )
    parts.append(
        '<text x="60" y="690" font-family="sans-serif" font-size="13" fill="#71808e">Read all three panels together: aggregate decode rate, useful prefill service, and user-visible decode stalls.</text>'
    )
    parts.append("</svg>")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(parts) + "\n")
    temporary.replace(path)


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    requested_policies = [
        item.strip() for item in args.policies.split(",") if item.strip()
    ]
    policies: list[str] = []
    for policy in requested_policies:
        if policy == "compute_share":
            policies.extend(
                f"compute_share_{target:g}"
                for target in (float(item) for item in args.compute_shares.split(","))
            )
        else:
            policies.append(policy)
    concurrencies = [int(item) for item in args.concurrencies.split(",")]
    invalid = {
        policy
        for policy in policies
        if policy != "off" and not policy.startswith("compute_share_")
    }
    if invalid:
        raise ValueError(f"unknown policies: {sorted(invalid)}")
    run_identity = f"mixed-{uuid.uuid4().hex[:12]}"
    timeout = httpx.Timeout(None, connect=30.0)
    cells: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        initial_response = await client.get(f"{args.base_url}/prefill_fairness")
        initial_response.raise_for_status()
        initial_config = initial_response.json()
        workload_by_repeat: dict[int, dict[str, Any]] = {}
        for repeat in range(1, args.repeats + 1):
            rng = random.Random(args.seed + repeat - 1)

            def sample_length() -> int:
                return round(
                    math.exp(
                        rng.uniform(
                            math.log(args.min_prefill_tokens),
                            math.log(args.max_prefill_tokens),
                        )
                    )
                )

            offsets: list[float] = []
            lengths: list[int] = []
            if args.window_seconds > 0.0:
                elapsed = 0.0
                while elapsed < args.window_seconds:
                    offsets.append(elapsed)
                    lengths.append(sample_length())
                    elapsed += rng.expovariate(1.0 / args.mean_arrival_seconds)
            else:
                elapsed = 0.0
                for index in range(args.prefills_per_cell):
                    if index:
                        elapsed += rng.expovariate(1.0 / args.mean_arrival_seconds)
                    offsets.append(elapsed)
                    lengths.append(sample_length())
            workload_by_repeat[repeat] = {"lengths": lengths, "offsets": offsets}

        try:
            for repeat in range(1, args.repeats + 1):
                workload = workload_by_repeat[repeat]
                for concurrency_index, concurrency in enumerate(concurrencies):
                    rotation = (repeat - 1 + concurrency_index) % len(policies)
                    ordered = policies[rotation:] + policies[:rotation]
                    for policy in ordered:
                        print(
                            f"starting policy={policy} C={concurrency} repeat={repeat} "
                            f"prefills={workload['lengths']}",
                            flush=True,
                        )
                        cell = await run_cell(
                            client,
                            args,
                            policy,
                            concurrency,
                            repeat,
                            workload["lengths"],
                            workload["offsets"],
                            run_identity,
                        )
                        cells.append(cell)
                        print(
                            f"finished policy={policy} C={concurrency} "
                            f"total={cell['total_useful_tokens_per_second']:.1f} "
                            f"decode={cell['collision_decode_tokens_per_second']:.1f} "
                            f"retained={cell['decode_retention_percent']:.1f}% "
                            f"prefill={cell['prefill_tokens_per_second']:.1f} "
                            f"gap_p95={1000 * cell['decode_inter_chunk_gap_seconds']['p95']:.1f}ms",
                            flush=True,
                        )
                        await wait_for_idle(client, args.base_url)
                        await asyncio.sleep(args.cooldown_seconds)
        finally:
            await set_fairness_config(
                client,
                args.base_url,
                {key: initial_config[key] for key in FAIRNESS_CONFIG_FIELDS},
            )

    summary = aggregate(cells)
    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_identity": run_identity,
            "base_url": args.base_url,
            "model": args.model,
            "policies": policies,
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "seed": args.seed,
            "prefills_per_cell": args.prefills_per_cell,
            "window_seconds": args.window_seconds,
            "prefill_distribution": "seeded log-uniform",
            "prefill_range_tokens": [
                args.min_prefill_tokens,
                args.max_prefill_tokens,
            ],
            "mean_arrival_seconds": args.mean_arrival_seconds,
            "decode_prompt_tokens": args.decode_prompt_tokens,
            "decode_warmup_tokens": args.decode_warmup_tokens,
            "baseline_seconds": args.baseline_seconds,
            "initial_config": initial_config,
            "workload_by_repeat": workload_by_repeat,
        },
        "summary": summary,
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--policies", default="off,compute_share")
    parser.add_argument("--concurrencies", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=53053)
    parser.add_argument("--prefills-per-cell", type=int, default=6)
    parser.add_argument("--min-prefill-tokens", type=int, default=10_000)
    parser.add_argument("--max-prefill-tokens", type=int, default=100_000)
    parser.add_argument("--mean-arrival-seconds", type=float, default=0.75)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.0,
        help="fixed acceptance window; zero runs until every offered prefill completes",
    )
    parser.add_argument("--decode-prompt-tokens", type=int, default=4094)
    parser.add_argument("--decode-warmup-tokens", type=int, default=256)
    parser.add_argument("--decode-max-tokens", type=int, default=65_536)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--recovery-seconds", type=float, default=2.0)
    parser.add_argument("--cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--warmup-timeout", type=float, default=180.0)
    parser.add_argument("--compute-share", type=float, default=0.5)
    parser.add_argument(
        "--compute-shares",
        default="0.2,0.5,0.8",
        help="targets expanded when the policy list contains compute_share",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="glm53-fairness-mixed-traffic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_prefill_tokens < 1 or args.max_prefill_tokens < args.min_prefill_tokens:
        raise SystemExit("invalid prefill token range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(async_main(args))
    json_path = args.output_dir / f"{args.name}.json"
    csv_path = args.output_dir / f"{args.name}.csv"
    svg_path = args.output_dir / f"{args.name}.svg"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(csv_path, result["summary"])
    metadata = result["metadata"]
    if metadata["window_seconds"] > 0.0:
        subtitle = (
            f"{metadata['window_seconds']:g}s fixed window, seeded Poisson cold-prefill "
            f"arrivals, {metadata['prefill_range_tokens'][0] // 1000}k–"
            f"{metadata['prefill_range_tokens'][1] // 1000}k tokens, "
            f"seed {metadata['seed']}"
        )
    else:
        subtitle = (
            f"Completion-bounded: {metadata['prefills_per_cell']} seeded log-uniform cold prefills/cell, "
            f"{metadata['prefill_range_tokens'][0] // 1000}k–"
            f"{metadata['prefill_range_tokens'][1] // 1000}k tokens, "
            f"seed {metadata['seed']}"
        )
    render_svg(
        svg_path,
        result["summary"],
        metadata["concurrencies"],
        metadata["policies"],
        subtitle,
    )
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"svg={svg_path}")


if __name__ == "__main__":
    main()
