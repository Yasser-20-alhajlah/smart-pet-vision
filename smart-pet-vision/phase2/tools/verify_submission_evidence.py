"""Check that numbers stated in the Phase 2 report match the saved evidence files.

Standard library only. Exit code 0 when every check passes, 1 otherwise.
Usage: python tools/verify_submission_evidence.py [--root PHASE2_DIR]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

A9_TOLERANCE = 0.001  # fixed before the Phase 2 runs; must never change


def load(root, name):
    return json.loads((root / "report" / "evidence" / name).read_text())


def checks(root):
    tex = (root / "report" / "main.tex").read_text()
    ev = lambda n: load(root, n)
    out = []

    def check(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    # ONNX: tolerance unchanged, every variant failed, range quoted correctly
    var = ev("M8_onnx_parity_recheck_variants.json")
    diffs = [r["max_abs_diff"] for r in var["results"] if "max_abs_diff" in r]
    check("A9b tolerance is 0.001 in every variant",
          all(r.get("tolerance") == A9_TOLERANCE for r in var["results"] if "max_abs_diff" in r))
    check("no ONNX variant within tolerance", all(d > A9_TOLERANCE for d in diffs), f"min={min(diffs):.6f}")
    check("report quotes the variant range",
          f"{min(diffs):.5f}"[:7] in tex and f"{max(diffs):.5f}"[:7] in tex,
          f"{min(diffs):.5f}--{max(diffs):.5f}")
    acc = {r["id"]: r for r in csv.DictReader(open(root / "report" / "evidence" / "M8_acceptance.csv"))}
    check("A9b recorded as not met", acc["A9b"]["met"] == "False")
    forbidden = [r"A9b\W{0,3}(is\s+)?met\b", r"A9b[^.\n]{0,20}\bpass(ed|es)?\b", r"ONNX detector (passed|meets)"]
    hits = [m.group(0) for p in forbidden for m in re.finditer(p, tex, flags=re.I)
            if "not" not in tex[max(0, m.start() - 12):m.end()].lower()]
    check("report never states the ONNX detector passed", not hits, "; ".join(hits))

    # CPU release validation (W&B f8manl0i)
    s = ev("P2_cpu_release_validation_f8manl0i.json")["summary"]
    correct = round(s["accuracy"] * s["classification_n"])
    check("f8manl0i counts", (s["n"], s["classification_n"], s["multi_animal_ready"], s["referrals"]) == (401, 399, 17, 81))
    check("report shows f8manl0i accuracy", f"{correct}/{s['classification_n']}" in tex, f"{correct}/{s['classification_n']}")
    check("report shows referrals", f"{s['referrals']}/{s['n']}" in tex)
    check("f8manl0i is not labelled a new test", s["independent_new_test"] is False)

    # CPU HTTP validation (W&B mu2tolkg)
    h = ev("P2_cpu_http_validation_mu2tolkg.json")
    lat = h["http_latency_ms"]
    stated = "{:.1f}/{:.1f}/{:.1f}".format(lat["p50"], lat["p95"], lat["p99"])
    check("report shows CPU HTTP percentiles", stated in tex, stated)
    check("HTTP 300/300 and guards 300/300",
          h["requests"] == h["succeeded"] == h["publication_guards_passed_requests"] == 300)
    check("demo requests excluded from timing", h["timing_includes_demo"] is False)
    check("HTTP run is CPU", h["device"] == "CPU")
    check("CPU latency not used for A8", "A8 & p95 latency, T4" in tex)

    # Every W&B run linked in the report is a well-formed id
    runs = set(re.findall(r"smart-pet-vision/runs/([a-z0-9]+)", tex))
    check("report links f8manl0i and mu2tolkg", {"f8manl0i", "mu2tolkg"} <= runs, ",".join(sorted(runs)))

    # Every figure referenced by the report source is listed (figures live in the source ZIP)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    check("figure references found", len(figs) >= 1, f"{len(figs)} figures")
    return out, figs


def submission_checks(root):
    """Checks on the instructor-template report (submission/main.tex) against the saved evidence."""
    path = root / "submission" / "main.tex"
    if not path.is_file():
        return []
    tex = path.read_text()
    out = []

    def check(name, ok, detail=""):
        out.append(("submission: " + name, bool(ok), detail))

    def has(value, fmt="{:.3f}"):
        return fmt.format(value) in tex

    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing = [f for f in figs if not (path.parent / f).is_file()]
    check("all figures present", figs and not missing, ", ".join(missing) or f"{len(figs)} figures")
    forbidden = [r"A9b\W{0,3}(is\s+)?met\b", r"A9b[^.\n]{0,20}\bpass(ed|es)?\b", r"ONNX detector (passed|meets)"]
    hits = [m.group(0) for p in forbidden for m in re.finditer(p, tex, flags=re.I)
            if "not" not in tex[max(0, m.start() - 12):m.end()].lower()]
    check("never states the ONNX detector passed", not hits, "; ".join(hits))
    check("A9b row is Fail", re.search(r"A9b ONNX parity, detector &[^\n]*\\textbf\{Fail\}", tex) is not None)
    check("28.7 px shift marked as confounded", "28.7" not in tex or "confounded" in tex)
    body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    check("no prompt left from the template", "\\prompt{" not in body.replace("\\newcommand{\\prompt}", ""))

    g = load(root, "P2_gap_experiments_20261102.json")
    a = g["A_detector_coco_baseline"]
    check("COCO detector baseline (val, test, op)",
          has(a["val"]["box_mAP50"]) and has(a["val"]["box_mAP50_95"]) and has(a["test"]["box_mAP50"])
          and has(a["test"]["box_mAP50_95"]) and has(a["test_at_operating_point"]["f1"]))
    b = g["B_segmentation_baselines"]["yolo11s_seg_coco_as_is_conf0.25"]
    check("COCO segmenter baseline", has(b["val"]["mask_mAP50_95"]) and has(b["test"]["mIoU"]))
    check("SAM baseline on validation", has(g["B_segmentation_baselines"]["sam2.1_b_oracle_box_val300"]["mIoU"]))
    c = g["C_segmenter_conf_on_validation"]
    check("mask confidence sweep and choice", all(has(r["mIoU"], "{:.4f}") for r in c["val_sweep"])
          and has(c["chosen_conf"], "{:.2f}"))
    d = g["D_pipeline_validation"]
    st = {r["stage"]: r for r in d["stage_percentiles_T4"]}
    check("per-stage p50/p95", all(has(st[k]["p50_ms"], "{:.1f}") and has(st[k]["p95_ms"], "{:.1f}")
                                   for k in ("detect_ms", "classify_ms", "segment_ms", "total_ms")))
    e = g["E_onnx"]
    check("ONNX task-level mAP", has(e["onnx_test"]["box_mAP50"]) and has(e["onnx_test"]["box_mAP50_95"]))
    cd = e["channel_diagnosis"]["summary"]
    check("ONNX channel diagnosis", has(cd["onnx_vs_torch64"]["max_abs_all"], "{:.5f}")
          and cd["onnx_vs_torch32"]["max_abs_all"] > A9_TOLERANCE)
    check("gap experiments not described as A9b met", e["channel_diagnosis"]["A9b_met"] is False)

    opt = load(root, "P2_training_optimizer_provenance_20261104.json")
    lr_fit = opt["conclusion"]["lr0"]
    check("optimiser stated as recorded (AdamW, lr0 0.001667), never SGD",
          "SGD" not in body and "AdamW" in body and f"{lr_fit:.6f}" in body
          and opt["sources"]["wandb_logged_lr"]["max_relative_error_epochs_4_40"] < 1e-4)

    ann = load(root, "M4_training_annotation_manifest_20261103.json")
    check("annotation correction: 4 train labels, val/test unchanged, not retrained, no gain claimed",
          ann["yolo_detection_labels_changed"] == 4 and ann["val_test_unchanged"]
          and ann["training_or_evaluation_not_performed"] and "not retrained" in body)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--figures", type=Path, help="Folder holding the report figures (optional)")
    a = ap.parse_args()
    results, figs = checks(a.root)
    if a.figures:
        missing = [f for f in figs if not (a.figures.parent / f).is_file() and not (a.figures / Path(f).name).is_file()]
        results.append(("all report figures present", not missing, ", ".join(missing)))
    results += submission_checks(a.root)
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
