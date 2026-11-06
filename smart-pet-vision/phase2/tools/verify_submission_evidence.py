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
    def ev(n):
        return load(root, n)
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
    body = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%"))
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
    check("mask confidence sweep and choice", all(has(r["mIoU"], "{:.5f}") for r in c["val_sweep"])
          and has(c["chosen_conf"], "{:.2f}"))
    ms = g["D_pipeline_validation"]["mask_match_sweep"]
    check("match-IoU sweep quoted as counts, with the per-image limitation",
          all(f"{r['unmatched']} of 1,106" in tex or f"{r['unmatched']} (" in tex for r in ms)
          and f"{ms[1]['referred']} referred" in tex and "only counts were saved" in tex
          and "were not evaluated through the pipeline" in tex)
    d = g["D_pipeline_validation"]
    st = {r["stage"]: r for r in d["stage_percentiles_T4"]}
    check("per-stage p50/p95", all(has(st[k]["p50_ms"], "{:.1f}") and has(st[k]["p95_ms"], "{:.1f}")
                                   for k in ("detect_ms", "classify_ms", "segment_ms", "total_ms")))
    e = g["E_onnx"]
    check("ONNX task-level mAP, unrounded and called close (not identical)",
          all(has(e[k][m], "{:.5f}") for k in ("onnx_test", "pytorch_test") for m in ("box_mAP50", "box_mAP50_95"))
          and "not identical" in tex and "mAP equal" not in tex)
    check("channel diagnosis: box vs score channels quoted from saved analysis",
          has(e["channel_diagnosis"]["summary"]["onnx_vs_torch32"]["max_abs_box"], "{:.7f}") and e["channel_diagnosis"]["summary"]["onnx_vs_torch32"]["max_abs_score"] < 1e-6
          and has(e["channel_diagnosis"]["summary"]["onnx_vs_torch64"]["max_abs_box"], "{:.7f}") and "four random" in tex)
    cd = e["channel_diagnosis"]["summary"]
    check("ONNX channel diagnosis", has(cd["onnx_vs_torch64"]["max_abs_all"], "{:.5f}")
          and cd["onnx_vs_torch32"]["max_abs_all"] > A9_TOLERANCE)
    check("gap experiments not described as A9b met", e["channel_diagnosis"]["A9b_met"] is False)

    opt = load(root, "P2_training_optimizer_provenance_20261104.json")
    lr_fit = opt["conclusion"]["lr0"]
    check("optimiser stated as recorded (AdamW, lr0 0.001667), never SGD",
          "SGD" not in body and "AdamW" in body and f"{lr_fit:.6f}" in body
          and opt["sources"]["wandb_logged_lr"]["max_relative_error_epochs_4_40"] < 1e-4)
    dl = load(root, "P2_training_optimizer_direct_log_20261104.json")["notebooks"]
    check("optimiser direct log in both executed notebooks; W&B link stated as inferred",
          len(dl) == 2 and all(any(f"AdamW(lr={lr_fit:.6f}" in entry["text"] for entry in n["log_lines"]) for n in dl.values())
          and "P2\\_training\\_optimizer\\_direct\\_log" in body and "inferred" in body)
    check("code row does not claim later commits changed only the report",
          "later commits change only this report" not in body and "CODE\\_ARCHIVE\\_INDEX.json" in body)

    ann = load(root, "M4_training_annotation_manifest_20261103.json")
    check("annotation correction: 4 train labels, val/test unchanged, not retrained, no gain claimed",
          ann["yolo_detection_labels_changed"] == 4 and ann["val_test_unchanged"]
          and ann["training_or_evaluation_not_performed"] and "not retrained" in body)

    fin = load(root, "M8_onnx_parity_recheck_A9b_20261103.json")
    r50 = fin["results"][0]["real_images_50"]
    check("matched-preprocessing final-detection agreement quoted, A9b still failed",
          r50["same_classes"] == 1.0 and f"{r50['final_box_max_abs_diff_px']:.5f}" in body
          and not fin["any_variant_passed"])

    re5 = load(root, "M8_onnx_parity_recheck_A9b_20261105.json")
    rc5 = re5["root_cause_diagnosis"]["summary"]
    runs5 = re5["official_gate_runs"]
    check("5 Nov A9b re-run quoted as recorded (0.0018 one thread, 0.0023 two threads), still failed",
          not re5["A9b_met"] and not any(r["passed"] for r in runs5)
          and f"{runs5[0]['max_abs_diff']:.4f}" == "0.0018" and f"{runs5[1]['max_abs_diff'][0]:.4f}" == "0.0023"
          and "0.0018 with one PyTorch thread and 0.0023 with two" in body
          and f"{rc5['torch32_default_vs_torch32_one_thread_final']:.4f}" == "0.0019" and "differs from itself by 0.0019" in body
          and rc5["argmax_strides"] == [32] and "\\ev{M8_onnx_parity_recheck_A9b_20261105}" in body)

    m5k = load(root, "P2_m5_kaggle_regression_20261105.json")
    s3 = m5k["same_300_ids"]
    check("Kaggle M5 regression quoted as recorded (same-300 YOLO/SAM IoU with and without border; full-test mAP50-95)",
          f"{s3['yolo']['iou']:.3f}" == "0.925" and f"{s3['sam_oracle_box']['iou']:.3f}" == "0.812"
          and f"{s3['yolo']['iou_no_border']:.3f}" == "0.934" and f"{s3['sam_oracle_box']['iou_no_border']:.3f}" == "0.948"
          and f"{m5k['yolo_full_test']['mask_mAP50_95']:.3f}" == "0.875" and f"{m5k['yolo_full_test']['iou_no_border']:.3f}" == "0.930"
          and "& 0.925 & 0.960 & 0.934 & 0.964" in body and "& 0.812 & 0.883 & 0.948 & 0.956" in body)
    m4k = load(root, "P2_m4_corrected_candidate_kaggle_20261105.json")
    cv = m4k["curve_metrics_conf0.001_nms0.70"]
    op = m4k["operating_point_conf0.40_nms0.50_iou0.50_test"]
    check("Kaggle M4 candidate quoted as recorded and not promoted (val mAP50-95 0.839 < 0.844; op. point 1,051/49/55)",
          f"{cv['candidate']['val']['mAP50_95']:.3f}" == "0.839" and f"{cv['historical']['val']['mAP50_95']:.3f}" == "0.844"
          and op["candidate"]["tp"] == 1051 and op["candidate"]["fp"] == 49 and op["candidate"]["fn"] == 55
          and "Candidate & val & 0.979 & 0.839" in body and "1,051/49/55" in body and "so the served detector is kept" in body)
    sz = load(root, "P2_size_table_erratum_20261105.json")["recount_relative_bins"]
    check("Recall-by-size Oxford column equals the relative-bin recount (4 medium, 1,102 large; found sum = 1,061 TP)",
          sz["medium"]["n"] == 4 and sz["large"]["n"] == 1102 and sz["medium"]["found"] + sz["large"]["found"] == 1061
          and "Medium (1--10\\%) & 4 & 1.000" in body and "Large ($\\geq10\\%$) & 1,102 & 0.959" in body
          and "1,102 of 1,106 test boxes" in body)

    rep = load(root, "P2_closure_v3_20261103_extract.json")["gpu_benchmark"]
    check("3 Nov T4 repeat quoted as recorded and kept separate",
          "{:.1f}/{:.1f}/{:.1f}".format(rep["p50_ms"], rep["p95_ms"], rep["p99_ms"]) in body
          and "not combined" in body and rep["measurement"].startswith("synchronized in-process"))
    sg = load(root, "P2_same_animal_grouping_run_20261106.json")
    cp = sg["cross_partition_check_split_v1"]
    check("same-animal grouping run quoted as recorded (groups, crossing groups, images, per partition) and not called confirmed leakage",
          f'{sg["grouping_params"]["identity_groups"]:,} similarity groups' in body
          and f'{cp["groups_crossing_partitions"]} groups ({cp["images_in_crossing_groups"]} images: {cp["images_in_crossing_groups_by_partition"]["test"]} test, '
              f'{cp["images_in_crossing_groups_by_partition"]["val"]} validation, {cp["images_in_crossing_groups_by_partition"]["train"]} train)' in body
          and "human identity review of the linked pairs has not been done" in body
          and "not approved" in body and "confirmed identity leakage" not in body.replace("not confirmed", ""))
    rr = load(root, "reproduction/reproduction_eval_mohammed_20261106.json")
    rc = rr["comparison"]
    keys = [k for k in rc if k != "all_within_tolerance"]
    rec = load(root, "P2_reproduction_run_20261106.json")
    check("6 Nov reproduction quoted as recorded (6 metrics, all identical) and operator confirmation stated as pending",
          len(keys) == 6 and rc["all_within_tolerance"] is True and all(rc[k]["difference"] == 0.0 for k in keys)
          and rec["comparison"]["metrics_identical"] == 6
          and "6 of 6 metrics identical" in body and "written statement in the log is pending" in body
          and all(f'{rr["reproduced"][k]:.3f}' in body for k in ("mAP50", "mAP50_95", "mask_mAP50", "mask_mAP50_95", "mIoU", "mean_dice"))
          and "not yet claimed as a confirmed independent reproduction" in body)
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
    n_old = len(results)
    results += submission_checks(a.root)
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    failed = [r for r in results if not r[1]]
    # The two scopes are reported separately so the total is not read as checks of the submitted report alone.
    print(f"{len(results) - len(failed)}/{len(results)} checks passed "
          f"({n_old} on the superseded free-form report report/main.tex, "
          f"{len(results) - n_old} on the submission report submission/main.tex)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
