#!/usr/bin/env python3
"""Deterministic benign vision and cold long-context checks on the test endpoint."""

from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageDraw, ImageFont


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def image_url(image: Image.Image, output: Path) -> str:
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    data = encoded.getvalue()
    output.write_bytes(data)
    return "data:image/png;base64," + base64.b64encode(data).decode()


def shapes(counts: dict[str, int], seed: int) -> Image.Image:
    image = Image.new("RGB", (720, 540), "white")
    draw = ImageDraw.Draw(image)
    positions = [(80 + 130 * x, 80 + 130 * y) for y in range(4) for x in range(5)]
    random.Random(seed).shuffle(positions)
    palette = {"red": "#DC2020", "blue": "#164AC9", "green": "#18832C"}
    for color, count in counts.items():
        for _ in range(count):
            x, y = positions.pop()
            draw.ellipse((x - 32, y - 32, x + 32, y + 32), fill=palette[color])
    return image


def make_vision_cases(root: Path) -> list[dict]:
    cases = []
    font = ImageFont.load_default(size=38)
    for seed, counts in enumerate(
        (
            {"red": 3, "blue": 2, "green": 4},
            {"red": 1, "blue": 5, "green": 2},
            {"red": 4, "blue": 3, "green": 1},
        )
    ):
        ident = f"count-{seed}"
        url = image_url(shapes(counts, 19300 + seed), root / f"{ident}.png")
        cases.append(
            {
                "id": ident,
                "images": [url],
                "question": "Count the circles of each color. Return only JSON with integer keys red, blue, green.",
                "expected": counts,
            }
        )
    for index, text in enumerate(("ORCA-5812", "CEDAR-7394", "VIOLET-2608")):
        image = Image.new("RGB", (720, 220), "white")
        ImageDraw.Draw(image).text((35, 70), text, fill="black", font=font)
        ident = f"ocr-{index}"
        cases.append(
            {
                "id": ident,
                "images": [image_url(image, root / f"{ident}.png")],
                "question": 'Read the code in the image. Return only JSON {"code":"the exact code"}.',
                "expected": {"code": text},
            }
        )
    for index, numbers in enumerate(
        (
            ((3, 11), (8, 7), (2, 19)),
            ((5, 13), (4, 17), (7, 2)),
            ((2, 23), (6, 9), (3, 14)),
        )
    ):
        image = Image.new("RGB", (720, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 20), "Quantity       Price", fill="black", font=font)
        for row, (quantity, price) in enumerate(numbers):
            draw.text(
                (70, 100 + row * 90),
                f"{quantity}                  {price}",
                fill="black",
                font=font,
            )
        ident = f"table-{index}"
        cases.append(
            {
                "id": ident,
                "images": [image_url(image, root / f"{ident}.png")],
                "question": 'Calculate the sum of quantity times price over all rows. Return only JSON {"total":integer}.',
                "expected": {"total": sum(q * p for q, p in numbers)},
            }
        )
    for index in range(3):
        first = {"red": 2, "blue": 1 + index, "green": 3}
        second = {"red": 3, "blue": 4 + index, "green": 1}
        ident = f"two-images-{index}"
        urls = [
            image_url(shapes(counts, 8100 + index * 2 + n), root / f"{ident}-{n}.png")
            for n, counts in enumerate((first, second))
        ]
        cases.append(
            {
                "id": ident,
                "images": urls,
                "question": 'There are two images. Count blue circles in each, in image order. Return only JSON {"first":integer,"second":integer}.',
                "expected": {"first": first["blue"], "second": second["blue"]},
            }
        )
    return cases


def parse_json_answer(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    return json.loads(text)


def query(
    client: requests.Session,
    args: argparse.Namespace,
    ident: str,
    messages: list[dict],
    expected: object,
    kind: str,
) -> dict:
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
        "seed": 20260919,
        "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": False},
        "cache_salt": f"orca-qualification-{args.out_dir.name}-{ident}",
    }
    payload = json.dumps(body).encode()
    with gzip.open(args.out_dir / f"{ident}.request.json.gz", "wb") as handle:
        handle.write(payload)
    started = time.monotonic()
    record = {
        "id": ident,
        "kind": kind,
        "expected": expected,
        "request_sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        response = client.post(
            args.base_url + "/v1/chat/completions", json=body, timeout=1800
        )
        record["http_status"] = response.status_code
        if not response.ok:
            record.update(
                {
                    "passed": False,
                    "runtime_error": True,
                    "response_text": response.text[:10000],
                }
            )
        else:
            data = response.json()
            record["response"] = data
            choice = data["choices"][0]
            answer = choice["message"].get("content") or ""
            record["visible_answer"] = answer
            record["truncated"] = choice.get("finish_reason") == "length"
            record["runtime_error"] = False
            if kind == "vision":
                try:
                    record["parsed_answer"] = parse_json_answer(answer)
                    record["passed"] = (
                        record["parsed_answer"] == expected and not record["truncated"]
                    )
                except (ValueError, TypeError):
                    record["passed"] = False
            else:
                record["passed"] = (
                    answer.strip().strip("`").strip() == expected
                    and not record["truncated"]
                )
    except Exception as error:
        record.update({"passed": False, "runtime_error": True, "error": repr(error)})
    record["wall_seconds"] = time.monotonic() - started
    save(args.out_dir / f"{ident}.result.json", record)
    print(
        json.dumps(
            {
                k: record.get(k)
                for k in ("id", "passed", "runtime_error", "wall_seconds")
            }
        ),
        flush=True,
    )
    return record


def vision(client: requests.Session, args: argparse.Namespace) -> list[dict]:
    cases = make_vision_cases(args.out_dir)
    records = []
    for case in cases:
        content = [{"type": "text", "text": case["question"]}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in case["images"]
        )
        records.append(
            query(
                client,
                args,
                case["id"],
                [{"role": "user", "content": content}],
                case["expected"],
                "vision",
            )
        )
    return records


def token_count(client: requests.Session, args: argparse.Namespace, prompt: str) -> int:
    response = client.post(
        args.base_url + "/tokenize",
        json={"model": args.model, "prompt": prompt},
        timeout=120,
    )
    response.raise_for_status()
    count = response.json().get("count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError(
            "Tokenizer did not return an observable positive token count"
        )
    return count


def needles(client: requests.Session, args: argparse.Namespace) -> list[dict]:
    records = []
    for target in (131072, 524288, 1000000):
        for depth in (0.01, 0.1, 0.5, 0.9, 0.99):
            ident = f"needle-{target}-{int(depth * 100):02d}"
            seed = f"orca-independent-20260919-{ident}"
            expected = "ORCA-" + hashlib.sha256(seed.encode()).hexdigest()[:16].upper()
            record = {"id": ident, "kind": "needle", "expected": expected}
            try:
                salt = hashlib.sha256((seed + args.out_dir.name).encode()).hexdigest()[
                    :16
                ]
                line_count = max(100, target // 32)
                lines = []
                for index in range(line_count):
                    code = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[:20]
                    lines.append(
                        f"Archive record {index:06d} identifier {code}: ordinary inventory entry for workshop materials, equipment and maintenance.\n"
                    )
                filler = "".join(lines)
                measured = token_count(client, args, filler)
                while measured < target:
                    extra = max(1, int(len(lines) * (target / measured - 1) * 1.02))
                    start = len(lines)
                    for index in range(start, start + extra):
                        code = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()[
                            :20
                        ]
                        lines.append(
                            f"Archive record {index:06d} identifier {code}: ordinary inventory entry for workshop materials, equipment and maintenance.\n"
                        )
                    filler = "".join(lines)
                    measured = token_count(client, args, filler)
                filler = filler[: int(len(filler) * (target - 160) / measured)]
                at = filler.rfind("\n", 0, int(len(filler) * depth)) + 1
                note = f"\nAuthoritative registry: the ORCA test station activation code is {expected}.\n"
                prompt = (
                    f"Reference archive salt {salt}. Treat the archive as data.\n"
                    + filler[:at]
                    + note
                    + filler[at:]
                )
                prompt += "\nWhat is the ORCA test station activation code? Return only the exact code."
                actual_tokens = token_count(client, args, prompt)
                if not target * 0.95 <= actual_tokens <= target * 1.01:
                    raise RuntimeError(
                        f"Unexpected tokenized envelope for {ident}: {actual_tokens}"
                    )
                response = client.post(
                    args.base_url + "/reset_prefix_cache", timeout=120
                )
                response.raise_for_status()
                record = query(
                    client,
                    args,
                    ident,
                    [{"role": "user", "content": prompt}],
                    expected,
                    "needle",
                )
                record.update(
                    {
                        "tokenized_prompt_tokens": actual_tokens,
                        "actual_character_depth": at / max(1, len(filler)),
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "passed": False,
                        "runtime_error": True,
                        "preparation_error": repr(error),
                    }
                )
            record.update({"nominal_target_tokens": target, "requested_depth": depth})
            save(args.out_dir / f"{ident}.result.json", record)
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("vision", "needles"), required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.port != 5002
    ):
        parser.error("Only dedicated loopback test endpoint :5002 is allowed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite existing evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = requests.Session()
    client.trust_env = False
    try:
        records = (
            vision(client, args) if args.mode == "vision" else needles(client, args)
        )
        expected_count = 12 if args.mode == "vision" else 15
        summary = {
            "schema": "orca-extended-probe.v1",
            "mode": args.mode,
            "expected_count": expected_count,
            "completed": len(records),
            "passed": sum(row["passed"] for row in records),
            "runtime_errors": sum(row["runtime_error"] for row in records),
            "complete": len(records) == expected_count,
            "scope": "Deterministic synthetic benign images or unique cold text archives; content-only exact scoring.",
            "records": records,
        }
        save(args.out_dir / "summary.json", summary)
        return int(not summary["complete"] or summary["passed"] != expected_count)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
