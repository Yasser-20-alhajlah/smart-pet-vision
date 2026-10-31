"""Split the detector ONNX parity gap into box-coordinate and class-score channels.

Hypothesis under test (stated before running): the raw-head maximum difference of about 0.002
comes from the four box-coordinate channels, which are in pixels (up to the input size), while
the class-score channels (0..1) differ far less. The A9b criterion (max abs diff <= 0.001 over the
whole head) is NOT changed by this diagnosis; any per-channel figure is reported separately.

Protocol: identical to spv.deploy.yolo_parity (seed 0, n random inputs, CPU, float32, eval mode),
plus a float64 PyTorch reference for each input. CPU only, a few minutes, no training.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


def group_stats(a, b, n_box=4):
    """Max abs diff of (a - b) on box channels and on score channels. Shapes (1, 4+nc, anchors)."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] <= n_box:
        raise ValueError(f"Unexpected head shape {a.shape} vs {b.shape}")
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    box, score = d[:, :n_box], d[:, n_box:]
    i = np.unravel_index(int(d.argmax()), d.shape)
    return {"max_abs_all": float(d.max()), "max_abs_box": float(box.max()), "max_abs_score": float(score.max()),
            "argmax_channel": int(i[1]), "argmax_group": "box" if i[1] < n_box else "score",
            "box_value_scale": float(np.abs(b[:, :n_box]).max())}


def run(weights, onnx_path, imgsz=640, n=4, seed=0):
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    net32 = YOLO(str(weights)).model.float().cpu().eval()
    net64 = YOLO(str(weights)).model.double().cpu().eval()
    if net32.training or net64.training:
        raise RuntimeError("Model is not in eval mode.")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if inp.type != "tensor(float)":
        raise RuntimeError(f"ONNX input dtype is {inp.type}, expected float32.")
    first = lambda r: r[0] if isinstance(r, (list, tuple)) else r
    rows = []
    for k in range(n):
        x = torch.rand(1, 3, imgsz, imgsz)          # same draw order as spv.deploy.yolo_parity
        with torch.no_grad():
            p32 = first(net32(x)).numpy()
            p64 = first(net64(x.double())).numpy()
        ox = sess.run(None, {inp.name: x.numpy()})[0]
        rows.append({"input": k,
                     "onnx_vs_torch32": group_stats(ox, p32),
                     "torch32_vs_torch64": group_stats(p32, p64),
                     "onnx_vs_torch64": group_stats(ox, p64)})
    worst = lambda key, g: max(r[key][g] for r in rows)
    summary = {pair: {g: worst(pair, g) for g in ("max_abs_all", "max_abs_box", "max_abs_score")}
               for pair in ("onnx_vs_torch32", "torch32_vs_torch64", "onnx_vs_torch64")}
    a9b_value = summary["onnx_vs_torch32"]["max_abs_all"]
    return {"criterion": "A9b detector ONNX max abs diff <= 0.001 over the whole raw head (unchanged)",
            "A9b_value_this_run": a9b_value, "A9b_met": a9b_value <= 1e-3,
            "hypothesis": "gap is concentrated in box-coordinate channels (pixel units)",
            "hypothesis_supported": summary["onnx_vs_torch32"]["max_abs_box"] > 1e-3 >= summary["onnx_vs_torch32"]["max_abs_score"],
            "summary": summary, "per_input": rows,
            "protocol": {"seed": seed, "inputs": n, "imgsz": imgsz, "device": "CPU", "dtype": "float32 (+float64 reference)"},
            "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest(),
            "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "python": platform.python_version()},
            "note": "Per-channel figures are diagnostic only and are reported separately from A9b."}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--inputs", type=int, default=4)
    ap.add_argument("--out", required=True, help="New JSON file; an existing file is not overwritten")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        ap.error(f"{out} exists; choose a new file name.")
    rec = run(a.weights, a.onnx, a.imgsz, a.inputs)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("A9b_value_this_run", "A9b_met", "hypothesis_supported", "summary")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
