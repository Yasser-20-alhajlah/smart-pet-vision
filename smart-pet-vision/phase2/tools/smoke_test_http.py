"""Real-weight functional smoke test of the HTTP service over TCP loopback. Records every request.

Not an accuracy study and not a benchmark: one process, one image, a handful of requests.
Run from smart-pet-vision/phase2 in an environment with ../requirements.txt installed and the three
private weight files at the paths in the manifest.

  python tools/smoke_test_http.py --image cat.png --out smoke_result.json [--manifest runs/manifest.json] [--port 8765]

Requests: /health; the image with images=false and images=true (returned PNGs decoded); a synthetic
two-animal composite (the image twice, side by side; the verdict actually returned is recorded, a
wrong 'ready' included); a blank image; invalid uploads (text 415, empty 400, >15 MB 413).
Every /analyze response must carry staff_approval_required=true and automatic_publication_allowed=false.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parents[1]  # smart-pet-vision/phase2


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_revision() -> dict:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True)
    if r.returncode == 0:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                               capture_output=True, text=True).stdout.splitlines()
        return {"kind": "git", "commit": r.stdout.strip(), "local_modifications_tracked_files": dirty}
    idx = HERE.parents[1] / "CODE_ARCHIVE_INDEX.json"  # running from an extracted reviewed_source.zip
    if idx.is_file():
        d = json.loads(idx.read_text())
        return {"kind": "code archive", "commit": d["source_commit"], "tree": d["source_tree"]}
    return {"kind": "unknown"}


def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def policy_ok(body: dict) -> bool:
    return body.get("staff_approval_required") is True and body.get("automatic_publication_allowed") is False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="authorised single-animal test image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    out, img_path, url = Path(a.out).resolve(), Path(a.image).resolve(), f"http://127.0.0.1:{a.port}"
    os.chdir(HERE)
    transcript, results = [], {}

    def log(cmd, text):
        transcript.append({"t": time.time(), "cmd": cmd, "out": text})
        print("$ " + cmd + "\n" + text + "\n", flush=True)

    man = json.loads(Path(a.manifest).read_text())
    weights = {k: {"path": man[k], "sha256": sha(man[k]), "bytes": Path(man[k]).stat().st_size}
               for k in ("detector", "segmenter", "classifier")}
    env = {**os.environ, "PYTHONPATH": "src"}
    pre = subprocess.run([sys.executable, "tools/preflight_release.py", "--manifest", a.manifest,
                          "--requirements", "../requirements.txt", "--strict-versions"],
                         capture_output=True, text=True, env=env)
    log(f"python tools/preflight_release.py --manifest {a.manifest} --strict-versions",
        (pre.stdout.strip() or pre.stderr.strip())[:900] + f"\n(exit {pre.returncode})")

    srv_env = {**env, "SPV_MANIFEST": a.manifest, "CUDA_VISIBLE_DEVICES": ""}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "spv.serve:app", "--host", "127.0.0.1",
                            "--port", str(a.port), "--workers", "1"], env=srv_env,
                           stdout=open(out.parent / "uvicorn.log", "w"), stderr=subprocess.STDOUT)
    log(f"SPV_MANIFEST={a.manifest} uvicorn spv.serve:app --host 127.0.0.1 --port {a.port} --workers 1",
        f"started pid {srv.pid}")
    try:
        t0, h = time.time(), None
        while time.time() - t0 < 300:
            try:
                h = requests.get(url + "/health", timeout=120)
                break
            except requests.ConnectionError:
                time.sleep(1)
        if h is None:
            raise SystemExit("service did not answer /health within 300 s")
        results["health"] = {"status": h.status_code, "body": h.json(), "first_response_s": round(time.time() - t0, 2)}
        log(f"curl -s {url}/health", json.dumps(h.json()) + f"   [HTTP {h.status_code}]")

        def analyze(name, data, images, key):
            t = time.perf_counter()
            r = requests.post(url + "/analyze", params={"images": str(images).lower()},
                              files={"file": (name, data, "image/png")}, timeout=300)
            wall = (time.perf_counter() - t) * 1e3
            body = r.json()
            keep = {k: body.get(k) for k in ("verdict", "species", "confidence", "reasons", "staff_approval_required",
                                             "automatic_publication_allowed", "segmentation_matched", "mask_fraction",
                                             "timings", "image_size")}
            keep["n_detections"] = len(body.get("detections", []))
            keep["policy_flags_ok"] = policy_ok(body)
            if images:
                dec = {}
                for k in ("annotated_png", "cutout_png", "mask_png"):
                    im = Image.open(io.BytesIO(base64.b64decode(body[k])))
                    im.load()
                    dec[k] = {"size": list(im.size), "mode": im.mode}
                keep["decoded_images"] = dec
            keep["http_status"], keep["client_wall_ms"] = r.status_code, round(wall, 1)
            results[key] = keep
            show = {k: keep[k] for k in ("verdict", "species", "n_detections", "policy_flags_ok")}
            show["reasons"] = [x.get("code") for x in (keep["reasons"] or [])]
            log(f"curl -s -F file=@{name} '{url}/analyze?images={str(images).lower()}'",
                json.dumps(show) + f"   [HTTP {r.status_code}, {wall:.0f} ms]")

        img_bytes = img_path.read_bytes()
        analyze(img_path.name, img_bytes, False, "single_images_false")
        analyze(img_path.name, img_bytes, True, "single_images_true")

        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        two = Image.new("RGB", (im.width * 2, im.height))
        two.paste(im, (0, 0))
        two.paste(im.transpose(Image.FLIP_LEFT_RIGHT), (im.width, 0))
        analyze("two_animals_composite.png", png_bytes(two), False, "two_animal_composite")
        results["two_animal_composite"]["note"] = ("Synthetic: the same image twice (second copy mirrored). "
                                                   "Shows the result actually returned; not an evaluation.")

        analyze("blank.png", png_bytes(Image.new("RGB", (640, 480), (128, 128, 128))), False, "blank_image")

        for name, data, ctype, expect in (("notes.txt", b"not an image", "text/plain", 415),
                                          ("empty.jpg", b"", "image/jpeg", 400),
                                          ("big.jpg", b"0" * (15 * 1024 * 1024 + 1), "image/jpeg", 413)):
            r = requests.post(url + "/analyze", files={"file": (name, data, ctype)}, timeout=120)
            results[f"invalid_{name}"] = {"http_status": r.status_code, "expected": expect, "ok": r.status_code == expect,
                                          "body": r.text[:120]}
            log(f"curl -s -o /dev/null -w '%{{http_code}}' -F file=@{name} {url}/analyze",
                f"{r.status_code} (expected {expect})")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log("stop uvicorn", f"exit {srv.returncode}")

    import fastapi, numpy, PIL, torch, ultralytics, uvicorn  # noqa: E401  (versions for the record)
    analyses = [v for k, v in results.items() if k.startswith(("single", "two", "blank"))]
    rec = {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "Functional real-weight smoke test over TCP loopback. Not an accuracy evaluation, not a benchmark.",
        "source": source_revision(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count(),
                        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                        "ultralytics": ultralytics.__version__, "fastapi": fastapi.__version__,
                        "uvicorn": uvicorn.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__},
        "weights_recomputed": weights,
        "preflight": {"returncode": pre.returncode, "stdout": pre.stdout[-1500:], "stderr": pre.stderr[-800:]},
        "test_image": {"name": img_path.name, "sha256": sha(img_path)},
        "results": results,
        "summary": {"health_200": results["health"]["status"] == 200,
                    "all_analyze_200": all(v["http_status"] == 200 for v in analyses),
                    "policy_flags_ok_on_every_analyze": all(v["policy_flags_ok"] for v in analyses),
                    "invalid_inputs_as_expected": all(v["ok"] for k, v in results.items() if k.startswith("invalid_")),
                    "decoded_png_count": len(results["single_images_true"].get("decoded_images", {}))},
    }
    out.write_text(json.dumps(rec, indent=1) + "\n")
    (out.parent / (out.stem + "_transcript.json")).write_text(json.dumps(transcript, indent=1))
    print(json.dumps(rec["summary"]))


if __name__ == "__main__":
    main()
