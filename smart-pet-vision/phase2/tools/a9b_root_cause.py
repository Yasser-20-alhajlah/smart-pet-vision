"""Re-run the A9b detector ONNX parity gate and locate where the difference arises.

A9b (unchanged): maximum absolute difference between the PyTorch and ONNX Runtime raw detector head
<= 0.001, seed 0, 4 random 640 x 640 inputs, CPU, float32 (same steps as spv.deploy.yolo_parity).
This tool does not change the criterion, the weights, the export settings or the service. It adds
diagnostics that are reported separately from the gate:

1. float64 PyTorch reference: is the ONNX output further from exact arithmetic than PyTorch float32?
2. Stage-by-stage comparison inside the detection head (neck features, box-distribution logits,
   DFL distances in grid units, final boxes in pixels), using intermediate outputs added to an
   in-memory copy of the ONNX graph; the final output of that copy is checked against the original.
3. Location of the largest difference: channel group, feature level (stride 8/16/32), value size,
   and size in float32 units in the last place (ulp).
4. PyTorch against PyTorch, both float32, same inputs: the default CPU backend against the same
   model with oneDNN (mkldnn) disabled, and against one CPU thread (Ultralytics' ONNX export sets
   torch to one thread, so a gate run in the same process as an export uses one thread).

CPU only, a few minutes, no training and no evaluation on images.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HEAD = "/model.23"
NECK = ["/model.16/cv2/act/Mul_output_0", "/model.19/cv2/act/Mul_output_0", "/model.22/cv2/act/Mul_output_0"]
LOGITS = f"{HEAD}/Concat_output_0"            # (1, 64, A) box-distribution logits before DFL
DFL = f"{HEAD}/dfl/Reshape_1_output_0"        # (1, 4, A) distances in grid units


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _maxdiff(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def instrumented_session(onnx_path):
    import onnx
    import onnxruntime as ort

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    known = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output)}
    for name in NECK + [LOGITS, DFL]:
        if name not in known:
            raise RuntimeError(f"Intermediate tensor not found in the ONNX graph: {name}")
        model.graph.output.append(known[name])
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


class Capture:
    """Forward hooks on the Detect head: neck features in, box logits and DFL distances."""

    def __init__(self, net):
        det = net.model[-1]
        self.data = {}
        det.register_forward_pre_hook(self._neck)
        det.dfl.register_forward_hook(self._dfl)
        self.strides = det.stride.numpy().astype(int).tolist()

    # Hooks must return None, otherwise PyTorch replaces the module input/output.
    def _neck(self, module, inp):
        self.data["neck"] = [t.detach().numpy().copy() for t in inp[0]]

    def _dfl(self, module, inp, out):
        self.data["logits"] = inp[0].detach().numpy().copy()
        self.data["dfl"] = out.detach().numpy().copy()


def level_of(anchor, sizes):
    start = 0
    for lvl, n in enumerate(sizes):
        if anchor < start + n:
            return lvl
        start += n
    raise IndexError(anchor)


def run(weights, onnx_path, imgsz=640, n=4, seed=0):
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    net32 = YOLO(str(weights)).model.float().cpu().eval()
    net64 = YOLO(str(weights)).model.double().cpu().eval()
    cap32, cap64 = Capture(net32), Capture(net64)
    strides = cap32.strides
    sizes = [(imgsz // s) ** 2 for s in strides]
    stride_of_anchor = np.concatenate([np.full(k, s, np.float64) for k, s in zip(sizes, strides)])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inst = instrumented_session(onnx_path)
    name = sess.get_inputs()[0].name
    def first(r):
        return r[0] if isinstance(r, (list, tuple)) else r

    rows = []
    for k in range(n):
        x = torch.rand(1, 3, imgsz, imgsz)               # same draw order as spv.deploy.yolo_parity
        with torch.no_grad():
            p32 = first(net32(x)).numpy()
            c32 = {"neck": cap32.data["neck"], "logits": cap32.data["logits"], "dfl": cap32.data["dfl"]}
            p64 = first(net64(x.double())).numpy()
            with torch.backends.mkldnn.flags(enabled=False):
                p32_nomkl = first(net32(x)).numpy()
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                p32_1t = first(net32(x)).numpy()
            finally:
                torch.set_num_threads(threads)
        o = sess.run(None, {name: x.numpy()})[0]
        io = inst.run(None, {name: x.numpy()})
        o_inst, neck_o, logits_o, dfl_o = io[0], io[1:4], io[4], io[5]

        d = np.abs(o.astype(np.float64) - p32.astype(np.float64))
        c, a = np.unravel_index(int(d[0].argmax()), d[0].shape)
        ref_val = float(p32[0, c, a])
        ulp = float(np.spacing(np.float32(abs(ref_val))))
        box = d[0, :4]
        box_grid = box / stride_of_anchor                 # same difference expressed in grid cells
        per_level = {}
        start = 0
        for s, m in zip(strides, sizes):
            sl = slice(start, start + m)
            per_level[f"stride_{s}"] = {
                "box_px_max_abs": float(box[:, sl].max()),
                "box_grid_max_abs": float(box_grid[:, sl].max()),
                "dfl_grid_max_abs_onnx_vs_torch32": _maxdiff(dfl_o[:, :, sl], c32["dfl"][:, :, sl]),
                "box_px_value_max": float(np.abs(p32[0, :4, sl]).max()),
            }
            start += m
        rows.append({
            "input": k,
            "A9b_value": float(d.max()),
            "argmax": {"channel": int(c), "group": "box" if c < 4 else "score",
                       "channel_name": ["x_center", "y_center", "width", "height"][c] if c < 4 else f"class_{c - 4}",
                       "anchor": int(a), "stride": strides[level_of(a, sizes)], "torch32_value_px": ref_val,
                       "float32_ulp_at_value": ulp, "diff_in_ulp": float(d.max() / ulp)},
            "score_max_abs": float(d[0, 4:].max()),
            "relative_max": float(d.max() / np.abs(p32).max()),
            "per_level": per_level,
            "stages_onnx_vs_torch32": {
                "neck_features_max_abs": [_maxdiff(a_, b_) for a_, b_ in zip(neck_o, c32["neck"])],
                "box_logits_max_abs": _maxdiff(logits_o, c32["logits"]),
                "dfl_grid_max_abs": _maxdiff(dfl_o, c32["dfl"]),
                "final_head_max_abs": float(d.max()),
            },
            "stages_torch32_vs_torch64": {
                "neck_features_max_abs": [_maxdiff(a_, b_) for a_, b_ in zip(c32["neck"], cap64.data["neck"])],
                "box_logits_max_abs": _maxdiff(c32["logits"], cap64.data["logits"]),
                "dfl_grid_max_abs": _maxdiff(c32["dfl"], cap64.data["dfl"]),
                "final_head_max_abs": _maxdiff(p32, p64),
            },
            "onnx_vs_torch64_final_max_abs": _maxdiff(o, p64),
            "torch32_default_vs_torch32_mkldnn_off_final_max_abs": _maxdiff(p32, p32_nomkl),
            "torch32_default_vs_torch32_one_thread_final_max_abs": _maxdiff(p32, p32_1t),
            "onnx_vs_torch32_one_thread_final_max_abs": _maxdiff(o, p32_1t),
            "instrumented_graph_output_equals_original": bool(np.array_equal(o, o_inst)),
            "instrumented_graph_output_max_abs_vs_original": _maxdiff(o, o_inst),
        })

    def worst(f):
        return max(f(r) for r in rows)
    value = worst(lambda r: r["A9b_value"])
    return {
        "criterion": "A9b detector ONNX raw-head max abs diff <= 0.001 (unchanged)",
        "protocol": {"seed": seed, "inputs": n, "imgsz": imgsz, "device": "CPU", "dtype": "float32",
                     "same_steps_as": "spv.deploy.yolo_parity",
                     "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count()},
        "A9b_value": value, "A9b_met": value <= 1e-3,
        "summary": {
            "onnx_vs_torch32_final": value,
            "onnx_vs_torch32_score_channels": worst(lambda r: r["score_max_abs"]),
            "onnx_vs_torch32_relative": worst(lambda r: r["relative_max"]),
            "torch32_vs_torch64_final": worst(lambda r: r["stages_torch32_vs_torch64"]["final_head_max_abs"]),
            "onnx_vs_torch64_final": worst(lambda r: r["onnx_vs_torch64_final_max_abs"]),
            "torch32_default_vs_torch32_mkldnn_off_final": worst(lambda r: r["torch32_default_vs_torch32_mkldnn_off_final_max_abs"]),
            "torch32_default_vs_torch32_one_thread_final": worst(lambda r: r["torch32_default_vs_torch32_one_thread_final_max_abs"]),
            "onnx_vs_torch32_one_thread_final": worst(lambda r: r["onnx_vs_torch32_one_thread_final_max_abs"]),
            "argmax_strides": sorted({r["argmax"]["stride"] for r in rows}),
            "argmax_diff_in_ulp_max": worst(lambda r: r["argmax"]["diff_in_ulp"]),
            "dfl_grid_onnx_vs_torch32": worst(lambda r: r["stages_onnx_vs_torch32"]["dfl_grid_max_abs"]),
            "box_logits_onnx_vs_torch32": worst(lambda r: r["stages_onnx_vs_torch32"]["box_logits_max_abs"]),
            "instrumented_output_equal_in_all_inputs": all(r["instrumented_graph_output_equals_original"] for r in rows),
        },
        "per_input": rows,
        "strides": strides, "anchors_per_level": sizes,
        "weights_sha256": _sha(weights), "onnx_sha256": _sha(onnx_path),
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "python": platform.python_version(),
                     "numpy": np.__version__},
        "note": "Only A9b_value is the gate. All other figures are diagnostics, reported separately; "
                "none of them re-scores or relaxes A9b.",
    }


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
    with out.open("x") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps({"A9b_value": rec["A9b_value"], "A9b_met": rec["A9b_met"], "summary": rec["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
