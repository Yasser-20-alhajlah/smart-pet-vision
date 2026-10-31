"""M8 support: ONNX export with parity checks, latency benchmark, team-set evaluation."""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import save_json


# -------------------------------------------------------------------- ONNX
def export_classifier_onnx(model: torch.nn.Module, path: Path, size: int = 224) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve existing export; choose a new path: {path}")
    import torch
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return path


def classifier_parity(model: torch.nn.Module, onnx_path: Path, size: int = 224, n: int = 8,
                      seed: int = 0) -> dict:
    import torch
    import onnxruntime as ort

    torch.manual_seed(seed)
    x = torch.randn(n, 3, size, size)
    with torch.no_grad():
        ref = model.cpu().eval()(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    if out.shape != ref.shape:
        raise ValueError(f"Classifier parity output shapes differ: {out.shape} != {ref.shape}.")
    diff = float(np.abs(out - ref).max())
    return {"model": "classifier", "max_abs_diff": diff, "argmax_agreement":
            float((out.argmax(1) == ref.argmax(1)).mean()), "passed_A9": diff <= 1e-3}


def export_yolo_onnx(weights: Path, imgsz: int, output_path: Path | None = None) -> Path:
    """Export an additional candidate without replacing the source weights/export."""
    import shutil
    import tempfile
    from ultralytics import YOLO

    weights = Path(weights)
    destination = Path(output_path) if output_path is not None else weights.with_name(weights.stem + "_candidate.onnx")
    if destination.exists():
        raise FileExistsError(f"Preserve existing export; choose a new path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spv_export_") as folder:
        source = Path(folder) / weights.name
        shutil.copy2(weights, source)
        exported = Path(YOLO(str(source)).export(format="onnx", imgsz=imgsz, opset=17,
                                                simplify=False, dynamic=False))
        # Exclusive creation also protects against a concurrent writer.
        with exported.open("rb") as inp, destination.open("xb") as out:
            shutil.copyfileobj(inp, out)
    return destination


def yolo_parity(weights: Path, onnx_path: Path, imgsz: int, n: int = 4, seed: int = 0) -> dict:
    """Raw-head parity: same input tensor through PyTorch and ONNX Runtime."""
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    if n < 1:
        raise ValueError("Parity requires at least one input.")

    torch.manual_seed(seed)
    net = YOLO(str(weights)).model.float().cpu().eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    diffs, scales = [], []
    for _ in range(n):
        x = torch.rand(1, 3, imgsz, imgsz)
        with torch.no_grad():
            ref = net(x)
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
        out = sess.run(None, {name: x.numpy()})[0]
        if out.shape != ref.shape:
            raise ValueError(f"Detector parity output shapes differ: {out.shape} != {ref.shape}.")
        diffs.append(float(np.abs(out - ref).max()))
        scales.append(float(np.abs(ref).max()))
    # Box outputs are in pixels (up to imgsz), so the relative difference is reported too.
    return {"model": Path(weights).stem, "max_abs_diff": max(diffs),
            "seed": seed, "inputs": n, "imgsz": imgsz, "tolerance": 1e-3,
            "max_rel_diff": max(d / max(s, 1e-9) for d, s in zip(diffs, scales)),
            "passed_A9": max(diffs) <= 1e-3}


# --------------------------------------------------------------- benchmark
def _peak_rss_mb() -> float:
    # macOS reports bytes; Linux reports KiB. Both return MiB here.
    divisor = 2 ** 20 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def benchmark(pipe, image_paths: list[Path], warmup: int = 10, runs: int = 200) -> dict:
    """End-to-end latency per image at batch size 1, after warm-up, with per-stage split."""
    if not image_paths or runs < 1 or warmup < 0:
        raise ValueError("Provide images, at least one measured run and nonnegative warm-up.")
    import torch
    from .runtime import environment_record
    images = [Image.open(p).convert("RGB") for p in image_paths]
    for i in range(warmup):
        pipe.analyze(images[i % len(images)], with_images=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rows = []
    t0 = time.perf_counter()
    for i in range(runs):
        t = time.perf_counter()
        res = pipe.analyze(images[i % len(images)], with_images=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rows.append({**res["timings"], "wall_ms": (time.perf_counter() - t) * 1e3})
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    q = lambda c, p: float(np.percentile(df[c], p))
    return {
        "environment": environment_record(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runs": runs, "warmup": warmup, "batch_size": 1,
        "latency_ms": {"p50": q("wall_ms", 50), "p95": q("wall_ms", 95), "p99": q("wall_ms", 99),
                       "mean": float(df.wall_ms.mean())},
        "stage_mean_ms": {c: float(df[c].mean()) for c in ("detect_ms", "classify_ms", "segment_ms")},
        "throughput_img_per_s": runs / elapsed,
        "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                            if torch.cuda.is_available() else None),
        "peak_process_rss_mb": _peak_rss_mb(),
        "A8_applicable": bool(torch.cuda.is_available() and
                               "T4" in torch.cuda.get_device_name(0)),
        "passed_A8": (q("wall_ms", 95) <= 2000 if torch.cuda.is_available() and
                      "T4" in torch.cuda.get_device_name(0) else None),
    }


def service_success_rate(client, image_paths: list[Path]) -> dict:
    """Criterion A7: fraction of valid requests that return HTTP 200 with a verdict."""
    ok = 0
    for p in image_paths:
        with open(p, "rb") as f:
            r = client.post("/analyze?images=false", files={"file": (p.name, f, "image/jpeg")})
        ok += int(r.status_code == 200 and "verdict" in r.json())
    rate = ok / max(len(image_paths), 1)
    return {"requests": len(image_paths), "succeeded": ok, "success_rate": rate,
            "passed_A7": rate >= 0.99}


# ---------------------------------------------------------------- team set
def prepare_team(src: Path, dst: Path, public_meta: pd.DataFrame, max_side: int = 2048,
                 phash_threshold: int = 4) -> pd.DataFrame:
    """Strip EXIF/GPS, convert to RGB JPEG, and check for overlap with the public data."""
    import imagehash
    from PIL import ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    dst.mkdir(parents=True, exist_ok=True)
    pub = np.array([np.frombuffer(bytes.fromhex(h), np.uint8) for h in public_meta.phash])
    table = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    rows = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic", ".webp"}:
            continue
        im = ImageOps.exif_transpose(Image.open(p))      # apply phone rotation first
        im = Image.fromarray(np.array(im.convert("RGB")))  # drops EXIF, incl. GPS
        im.thumbnail((max_side, max_side))
        out = dst / f"{p.stem}.jpg"
        im.save(out, quality=95)
        h = np.frombuffer(bytes.fromhex(str(imagehash.phash(im))), np.uint8)
        nearest = int(table[np.bitwise_xor(pub, h)].sum(1).min()) if len(pub) else 64
        rows.append({"image": p.stem, "width": im.width, "height": im.height,
                     "nearest_public_hamming": nearest,
                     "possible_public_duplicate": nearest <= phash_threshold})
    return pd.DataFrame(rows)


def evaluate_team(pipe, team_dir: Path, labels_csv: Path, masks_dir: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """External test: run the full system on team photos. labels.csv: image,species[,n_animals]."""
    from sklearn.metrics import accuracy_score, f1_score

    from .segment import mask_metrics

    labels = pd.read_csv(labels_csv)
    rows = []
    for r in labels.itertuples():
        im = Image.open(team_dir / f"{r.image}.jpg")
        res = pipe.analyze(im, with_images=masks_dir is not None)
        row = {"image": r.image, "true": r.species, "pred": res["species"],
               "confidence": res["confidence"], "verdict": res["verdict"],
               "reasons": ";".join(x["code"] for x in res["reasons"]),
               "n_detected": len(res["detections"])}
        if masks_dir is not None and (masks_dir / f"{r.image}.png").exists():
            import base64
            import io

            pred = np.array(Image.open(io.BytesIO(base64.b64decode(res["mask_png"])))) > 0
            gt = np.array(Image.open(masks_dir / f"{r.image}.png").convert("L")) > 0
            row.update(mask_metrics(pred, gt))
        rows.append(row)
    df = pd.DataFrame(rows)
    valid = df[df.true.isin(["cat", "dog"])]
    accepted = valid[valid.verdict == "ready"]
    summary = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(valid.true, valid.pred)) if len(valid) else None,
        "macro_f1": float(f1_score(valid.true, valid.pred, average="macro")) if len(valid) else None,
        "referral_rate": float((df.verdict != "ready").mean()),
        "errors_escaping_review": int((accepted.true != accepted.pred).sum()),
        "accuracy_on_accepted": float((accepted.true == accepted.pred).mean()) if len(accepted) else None,
    }
    if "iou" in df:
        summary.update({"mIoU": float(df.iou.mean()), "mean_dice": float(df.dice.mean())})
    return summary, df


# ------------------------------------------------ deployment evidence (added 24 Oct 2026)
def http_service_test(repo: Path, manifest: Path, image_paths: list[Path], port: int = 8765,
                      warmup: int = 5, startup_timeout: float = 300.0,
                      log_path: Path | None = None) -> dict:
    """Start the FastAPI app with uvicorn in a separate process and call it over real HTTP."""
    import os
    import subprocess
    import sys
    from collections import Counter

    import requests

    if not image_paths or warmup < 0 or startup_timeout <= 0:
        raise ValueError("HTTP checks require images, nonnegative warm-up and a positive timeout.")

    python_path = str(Path(repo) / "src")
    if os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + os.pathsep.join(
            str(Path(p).resolve()) for p in os.environ["PYTHONPATH"].split(os.pathsep) if p)
    env = {**os.environ, "SPV_MANIFEST": str(manifest), "PYTHONPATH": python_path}
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "spv.serve:app", "--host", "127.0.0.1",
                             "--port", str(port), "--workers", "1"], cwd=str(repo), env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        health = None
        while time.time() - t0 < startup_timeout:
            try:
                response = requests.get(f"{url}/health", timeout=min(5, startup_timeout))
                response.raise_for_status()
                health = response.json()
                if health.get("status") != "ok":
                    health = None
                    continue
                break
            except (requests.exceptions.RequestException, ValueError):
                if proc.poll() is not None:
                    raise RuntimeError("Service exited before becoming healthy.")
                time.sleep(1)
        if health is None:
            raise RuntimeError("service did not start")
        startup_s = time.time() - t0

        def post(p):
            with open(p, "rb") as f:
                return requests.post(f"{url}/analyze", params={"images": "false"},
                                     files={"file": (p.name, f, "image/jpeg")}, timeout=120)

        for i in range(warmup):
            post(image_paths[i % len(image_paths)])
        codes, lat, ok = Counter(), [], 0
        for p in image_paths:
            t = time.perf_counter()
            r = post(p)
            lat.append((time.perf_counter() - t) * 1e3)
            codes[r.status_code] += 1
            ok += int(r.status_code == 200 and "verdict" in r.json())
        bad_text = requests.post(f"{url}/analyze", files={"file": ("x.txt", b"x", "text/plain")},
                                 timeout=60).status_code
        bad_empty = requests.post(f"{url}/analyze", files={"file": ("e.jpg", b"", "image/jpeg")},
                                  timeout=60).status_code
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path:
            log.close()
    rate = ok / max(len(image_paths), 1)
    q = lambda v: float(np.percentile(lat, v))
    return {"server": "uvicorn, separate process, 127.0.0.1, 1 worker", "manifest": Path(manifest).name,
            "health": health, "startup_s": startup_s, "warmup_requests": warmup,
            "requests": len(image_paths), "succeeded": ok, "success_rate": rate,
            "status_codes": {str(k): v for k, v in codes.items()},
            "http_latency_ms": {"p50": q(50), "p95": q(95), "p99": q(99), "mean": float(np.mean(lat))},
            "non_image_upload_status": bad_text, "empty_upload_status": bad_empty,
            "passed_A7": rate >= 0.99}


def _bench_main(repo: str, manifest: str, images_txt: str, out_json: str, warmup: int, runs: int) -> None:
    """Entry point for `benchmark_isolated`: one backend per Python process."""
    import json
    import os
    import torch

    os.chdir(repo)
    from .common import load_config
    from .pipeline import Manifest, PetPipeline

    cfg = load_config()
    pipe = PetPipeline(Manifest.load(manifest), cfg)
    if str(pipe.m.detector).endswith(".onnx"):
        # The CUDA execution provider failed on this runtime (cuDNN frontend error), so the ONNX
        # detector is benchmarked on ONNX Runtime's CPU provider and recorded as such.
        import functools
        pipe.detector.predict = functools.partial(pipe.detector.predict, device="cpu")
    paths = [Path(p) for p in Path(images_txt).read_text().split("\n") if p]
    res = benchmark(pipe, paths, warmup=warmup, runs=runs)
    det = str(pipe.m.detector)
    if det.endswith(".onnx"):
        try:
            sess = pipe.detector.predictor.model.session
            res["detector_execution_provider"] = sess.get_providers()[0]
        except Exception as exc:  # report rather than guess
            res["detector_execution_provider"] = f"unknown ({type(exc).__name__})"
    else:
        res["detector_execution_provider"] = "PyTorch CUDA" if torch.cuda.is_available() else "PyTorch CPU"
    res["detector_file"] = Path(det).name
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2, default=float)


def benchmark_isolated(repo: Path, manifest: Path, image_paths: list[Path], out_json: Path,
                       warmup: int = 10, runs: int = 200) -> dict:
    """Run `benchmark` for one backend in a fresh process, so peak RSS belongs to that backend."""
    import json
    import os
    import subprocess
    import sys

    lst = Path(out_json).with_suffix(".images.txt")
    lst.write_text("\n".join(str(p) for p in image_paths))
    code = (f"from spv import deploy; deploy._bench_main({str(repo)!r}, {str(manifest)!r}, "
            f"{str(lst)!r}, {str(out_json)!r}, {warmup}, {runs})")
    python_path = str(Path(repo) / "src")
    if os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + os.pathsep.join(
            str(Path(p).resolve()) for p in os.environ["PYTHONPATH"].split(os.pathsep) if p)
    env = {**os.environ, "PYTHONPATH": python_path}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(repo), env=env, capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    lst.unlink()
    return json.loads(Path(out_json).read_text())


def yolo_parity_real(weights: Path, onnx_path: Path, image_paths: list[Path], imgsz: int,
                     conf: float, iou: float) -> dict:
    """Parity on real test images: raw head outputs and final detections (PyTorch vs ONNX, CPU)."""
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from .instance_alignment import pair_detection_boxes

    if not image_paths:
        raise ValueError("Real-image parity requires at least one image.")

    net = YOLO(str(weights)).model.float().cpu().eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    lb = LetterBox((imgsz, imgsz), auto=False)
    raw_diff = []
    for p in image_paths:
        arr = lb(image=np.array(Image.open(p).convert("RGB")))
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
        with torch.no_grad():
            ref = net(x)
        ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
        out = sess.run(None, {name: x.numpy()})[0]
        if out.shape != ref.shape:
            raise ValueError(f"Detector parity output shapes differ: {out.shape} != {ref.shape}.")
        raw_diff.append(float(np.abs(out - ref).max()))
    pt, ox = YOLO(str(weights)), YOLO(str(onnx_path), task="detect")
    same_count = same_cls = 0
    box_diff = []
    for p in image_paths:
        # PyTorch defaults to a minimal rectangle, while a static ONNX export
        # uses a square. Compare the same square input on both backends.
        a = pt.predict(str(p), conf=conf, iou=iou, imgsz=imgsz, rect=False, device="cpu", verbose=False)[0].boxes
        b = ox.predict(str(p), conf=conf, iou=iou, imgsz=imgsz, rect=False, device="cpu", verbose=False)[0].boxes
        if len(a) == len(b):
            same_count += 1
            if len(a):
                j = pair_detection_boxes(a.xyxy.numpy(), b.xyxy.numpy())
                if (a.cls.numpy() == b.cls.numpy()[j]).all():
                    same_cls += 1
                box_diff.append(float(np.abs(a.xyxy.numpy() - b.xyxy.numpy()[j]).max()))
            else:
                same_cls += 1
    n = len(image_paths)
    return {"images": n, "input": f"letterboxed {imgsz} px, CPU for both",
            "final_detection_matching": "greedy one-to-one IoU, class-agnostic",
            "raw_head_max_abs_diff": max(raw_diff),
            "same_number_of_detections": same_count / n, "same_classes": same_cls / n,
            "final_box_max_abs_diff_px": max(box_diff) if box_diff else 0.0,
            "conf": conf, "nms_iou": iou}


def system_failure_gallery(pipe, team_dir: Path, cases: pd.DataFrame, path: Path, cols: int = 4,
                           tile: int = 300) -> None:
    """Annotated pipeline outputs with one explicit failure statement per tile.

    `cases` needs the columns image and statement.
    """
    import base64
    import io
    import textwrap

    from PIL import ImageDraw

    rows = (len(cases) + cols - 1) // cols
    cap_h = 64
    sheet = Image.new("RGB", (cols * tile, rows * (tile + cap_h)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(cases.itertuples()):
        res = pipe.analyze(Image.open(team_dir / f"{r.image}.jpg"), with_images=True)
        im = Image.open(io.BytesIO(base64.b64decode(res["annotated_png"]))).convert("RGB")
        im.thumbnail((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * (tile + cap_h)
        sheet.paste(im, (x, y))
        text = f"{r.image}: {r.statement}"
        for j, line in enumerate(textwrap.wrap(text, 48)[:4]):
            d.text((x + 3, y + tile + 2 + 14 * j), line, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)
