"""Opt-in checks against a private Lemonade instance; reloads the named model."""

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(base, route, payload, stream=False):
    request = urllib.request.Request(
        f"{base.rstrip('/')}/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as error:
        if route == "unload" and error.code == 404:
            return None
        raise
    with response:
        if not stream:
            return json.load(response)
        content, reasoning, first_content, timings, usage = "", "", None, None, None
        finish_reason = None
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                reasoning += choice.get("delta", {}).get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason") or finish_reason
                token = choice.get("delta", {}).get("content") or ""
                if token and first_content is None:
                    first_content = time.perf_counter() - started
                content += token
            timings = chunk.get("timings") or timings
            usage = chunk.get("usage") or usage
        return {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish_reason,
            "client_first_content_s": first_content,
            "client_total_s": time.perf_counter() - started,
            "timings": timings,
            "usage": usage,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:13315/api/v1")
    parser.add_argument("--mode", choices=["gpu", "npu", "saved", "both"], default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-format", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename to retain earlier evidence.")
    passage = (Path(__file__).resolve().parents[1] / "tests/prompts/prose.txt").read_text(
        encoding="utf-8-sig"
    )
    filler = "\n".join(f"Unrelated passage {i}:\n{passage}" for i in range(1, 9))
    cases = [
        ("early-fact", f"Project Alder's access code is ZEST-4827.\n{filler}\n"
         "What is Project Alder's access code? Return only the code.", "ZEST-4827"),
        ("late-fact", f"{filler}\nProject Birch's access code is LIME-1935.\n"
         "What is Project Birch's access code? Return only the code.", "LIME-1935"),
        ("ledger", f"{filler}\nAn unrelated warehouse started with 27 boxes, received 15, "
         "and shipped 8. How many boxes remain? Return only the integer.", "34"),
    ]
    with urllib.request.urlopen(f"{args.base_url.rstrip('/')}/health", timeout=10) as response:
        health = json.load(response)
    record = {
        "model": args.model,
        "lemonade": health,
        "checks": [],
    }
    modes = ["gpu", "npu"] if args.mode == "both" else [args.mode]
    try:
        for mode in modes:
            post(args.base_url, "unload", {"model_name": args.model})
            load = {"model_name": args.model}
            if mode != "saved":
                split = "0,1" if mode == "npu" else "1"
                load.update(
                    ctx_size=8192,
                    llamacpp_backend="vulkan",
                    llamacpp_device="XDNA0,Vulkan0" if mode == "npu" else "Vulkan0",
                    llamacpp_args=f"-ts {split} -ub 2048 -b 2048 --flash-attn on --parallel 1",
                    merge_args=False,
                )
            post(args.base_url, "load", load)
            for name, prompt, expected in cases:
                request = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                result = post(args.base_url, "chat/completions", request, stream=True)
                exact = result["content"].strip() == expected
                correct = exact or (name == "ledger" and re.fullmatch(
                    r"27\s*\+\s*15\s*-\s*8\s*=\s*34", result["content"].strip()
                ) is not None)
                check = {
                    "mode": mode, "case": name, "expected": expected,
                    "prompt_sha256": hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest(),
                    "passed": correct and (exact or not args.strict_format),
                    "answer_correct": correct, "format_correct": exact, **result,
                }
                record["checks"].append(check)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                first = result['client_first_content_s']
                first_text = f"{first:.2f}s" if first is not None else "none"
                print(f"{mode} {name}: {'PASS' if check['passed'] else 'FAIL'} "
                      f"({ascii(result['content'].strip())}; format {'PASS' if exact else 'FAIL'}; "
                      f"first content {first_text})", flush=True)
    finally:
        post(args.base_url, "unload", {"model_name": args.model})
    if not all(check["passed"] for check in record["checks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
