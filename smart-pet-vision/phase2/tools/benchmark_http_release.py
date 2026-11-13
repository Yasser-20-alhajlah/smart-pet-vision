#!/usr/bin/env python3
"""Measure the existing service over loopback HTTP; no training or metric selection.

The workload is frozen before execution. psutil samples the server's RSS, not the
client's memory. This is a sequential, one-worker benchmark, not a load test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def valid_response(response):
    if response.status_code != 200:
        return False
    try:
        body = response.json()
        return (
            body.get("verdict") in ("ready", "refer_to_staff")
            and body.get("staff_approval_required") is True
            and body.get("automatic_publication_allowed") is False
        )
    except (ValueError, TypeError, AttributeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--images",
        type=Path,
        required=True,
        help="JSON list of image paths, frozen before inference",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    if a.out.exists() or a.out.with_suffix(".server.log").exists():
        p.error("Receipt or log exists; choose a new output path")
    paths = [Path(v).resolve() for v in json.loads(a.images.read_text())]
    if not paths or a.warmup < 0 or not all(v.is_file() for v in paths):
        p.error("Provide existing images and a nonnegative warm-up count")
    import numpy as np
    import psutil
    import requests

    a.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.manifest.read_text())
    # Bind files/settings before model execution. No result-dependent selection.
    inputs = [{"image": v.name, "sha256": sha(v)} for v in paths]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES="",
        WANDB_MODE="disabled",
    )
    stop = threading.Event()
    rss = []
    errors = []
    started = time.perf_counter()
    with a.out.with_suffix(".server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "spv.serve:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
            ],
            cwd=ROOT,
            env=dict(env, SPV_MANIFEST=str(a.manifest.resolve())),
            stdout=log,
            stderr=log,
        )
        process = psutil.Process(proc.pid)

        def sample():
            while not stop.wait(0.02):
                try:
                    rss.append((time.perf_counter(), process.memory_info().rss))
                except psutil.NoSuchProcess:
                    break

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}"

        def post(path):
            with path.open("rb") as f:
                return requests.post(
                    url + "/analyze",
                    params={"images": "false"},
                    files={"file": (path.name, f, "image/jpeg")},
                    timeout=120,
                )

        try:
            while True:
                try:
                    health = requests.get(url + "/health", timeout=2)
                    if health.ok and health.json().get("status") == "ok":
                        break
                except requests.RequestException:
                    pass
                if proc.poll() is not None or time.perf_counter() - started > 180:
                    raise RuntimeError(
                        "Service failed to become healthy; inspect the server log"
                    )
                time.sleep(0.25)
            startup = time.perf_counter() - started
            for i in range(a.warmup):
                if not valid_response(post(paths[i % len(paths)])):
                    raise RuntimeError(
                        "Warm-up response violates the documented contract"
                    )
            began = time.perf_counter()
            lat = []
            codes = {}
            good = 0
            for i, path in enumerate(paths):
                t = time.perf_counter()
                response = post(path)
                lat.append((time.perf_counter() - t) * 1000)
                key = str(response.status_code)
                codes[key] = codes.get(key, 0) + 1
                passed = valid_response(response)
                good += int(passed)
                if not passed:
                    errors.append({"index": i, "status": response.status_code})
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(paths)} HTTP requests measured", flush=True)
            ended = time.perf_counter()
            invalid = requests.post(
                url + "/analyze",
                files={"file": ("x.txt", b"x", "text/plain")},
                timeout=10,
            ).status_code
            empty = requests.post(
                url + "/analyze",
                files={"file": ("e.jpg", b"", "image/jpeg")},
                timeout=10,
            ).status_code
        finally:
            stop.set()
            thread.join(timeout=2)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    measured = [b for t, b in rss if began <= t <= ended]
    if not measured:
        raise RuntimeError("No memory samples captured during measurement")
    versions = {
        k: importlib.metadata.version(k)
        for k in (
            "torch",
            "torchvision",
            "ultralytics",
            "numpy",
            "psutil",
            "fastapi",
            "uvicorn",
            "requests",
        )
    }
    result = {
        "scope": "Existing served-v1 weights; CPU loopback HTTP benchmark; no accuracy evaluation, training, GPU or threshold selection.",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "runtime_code_sha256": {
            f: sha(ROOT / "src/spv" / f)
            for f in (
                "pipeline.py",
                "serve.py",
                "classify.py",
                "instance_alignment.py",
                "multi_animal_review.py",
            )
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if sys.platform == "darwin"
        else platform.processor(),
        "versions": versions,
        "thread_environment": {
            k: env.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "manifest_settings": {
            k: v
            for k, v in manifest.items()
            if k not in ("detector", "segmenter", "classifier")
        },
        "weights_sha256": {
            k: sha(v)
            for k, v in manifest.items()
            if k in ("detector", "segmenter", "classifier")
        },
        "input_sequence": inputs,
        "distinct_images": len({v["sha256"] for v in inputs}),
        "server": "uvicorn, one worker, loopback, sequential multipart POST /analyze?images=false",
        "startup_s": startup,
        "warmup_requests": a.warmup,
        "requests": len(paths),
        "valid_responses": good,
        "status_codes": codes,
        "http_latency_ms": {
            f"p{q}": float(np.percentile(lat, q)) for q in (50, 95, 99)
        },
        "measured_wall_s": ended - began,
        "throughput_requests_s": len(paths) / (ended - began),
        "server_rss_mb": {
            "definition": "server process RSS only; decimal MB; sampled every 20 ms; sampled peak is a lower bound on instantaneous peak",
            "samples": len(measured),
            "peak_measured": max(measured) / 1e6,
            "peak_including_startup": max(b for _, b in rss) / 1e6,
        },
        "non_image_status": invalid,
        "empty_image_status": empty,
        "errors": errors,
        "passed_http_contract": good == len(paths) and invalid == 415 and empty == 400,
        "not_a_T4_A8_remeasurement": True,
    }
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "http_latency_ms",
                    "throughput_requests_s",
                    "server_rss_mb",
                    "passed_http_contract",
                )
            },
            indent=2,
        )
    )
    return 0 if result["passed_http_contract"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
