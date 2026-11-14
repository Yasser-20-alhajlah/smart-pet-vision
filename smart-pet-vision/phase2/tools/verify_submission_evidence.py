"""Check that numbers stated in the Phase 2 report match the saved evidence files.

Standard library only. Exit code 0 when every check passes, 1 otherwise.
Usage: python tools/verify_submission_evidence.py [--root PHASE2_DIR]
"""
import argparse
import csv
import hashlib
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
    with (root / 'report/evidence/M8_acceptance.csv').open() as stream:
        acc = {r['id']: r for r in csv.DictReader(stream)}
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
    orv = load(root, "P2_same_animal_review_owner_20261107.json")
    check("same-animal grouping run and the owner's review quoted as recorded (groups, crossing groups, images, verdicts, test share)",
          f'{sg["grouping_params"]["identity_groups"]:,} similarity groups' in body
          and f'{cp["groups_crossing_partitions"]} groups ({cp["images_in_crossing_groups"]} images: {cp["images_in_crossing_groups_by_partition"]["test"]} test, '
              f'{cp["images_in_crossing_groups_by_partition"]["val"]} validation, {cp["images_in_crossing_groups_by_partition"]["train"]} train)' in body
          and orv["verdicts"]["clear duplicate"] == cp["groups_crossing_partitions"]
          and "recorded every group as a clear duplicate" in body
          and f'{orv["groups_touching_test"]["test_images"]} of the 1,106 test images' in body
          and "not approved" in body)
    rr = load(root, "reproduction/reproduction_eval_mohammed_20261106.json")
    rc = rr["comparison"]
    keys = [k for k in rc if k != "all_within_tolerance"]
    rec = load(root, "P2_reproduction_run_20261106.json")
    check("6 Nov reproduction quoted as recorded (6 metrics, all identical) and the operator attributed to the project owner's statement",
          len(keys) == 6 and rc["all_within_tolerance"] is True and all(rc[k]["difference"] == 0.0 for k in keys)
          and rec["comparison"]["metrics_identical"] == 6
          and "6 of 6 metrics identical" in body and "this is the statement of the project owner" in body
          and all(f'{rr["reproduced"][k]:.3f}' in body for k in ("mAP50", "mAP50_95", "mask_mAP50", "mask_mAP50_95", "mIoU", "mean_dice"))
          and "the Colab record shows the owner's computer and account" in body and "signed by Mohammed" not in body)
    sd = load(root, "P2_supplementary_D_multi_animal_review_20261106.json")["summary"]
    base, cand = sd["baseline"], sd["candidate"]
    check("Supplementary D quoted as recorded (17 -> 10 of 52; 45 -> 127 of 349) and not presented as a holdout",
          (base["multi_animal_ready"], cand["multi_animal_ready"], base["multi_animal_images"]) == (17, 10, 52)
          and (base["single_animal_referrals"], cand["single_animal_referrals"], base["single_animal_images"]) == (45, 127, 349)
          and sd["independent_holdout"] is False and sd["thresholds_selected_using_this_set"] is False
          and "Multi-animal photos marked ready (of 52) & 17 & 10" in body
          and "45 (12.9\\%) & 127 (36.4\\%)" in body and "not an independent holdout" in body)
    se = load(root, "P2_supplementary_E_class_balance_val_20261106.json")
    d, b2 = se["results"]["delivered"], se["results"]["balanced"]

    def f3(x):
        return f"{x:.3f}"

    check("Supplementary E quoted as recorded (validation only, test split unused, not promoted)",
          se["test_split_used"] is False
          and f"{f3(d['val_ap']['mAP50'])} / {f3(d['val_ap']['mAP50_95'])} & {f3(b2['val_ap']['mAP50'])} / {f3(b2['val_ap']['mAP50_95'])}" in body
          and f"{f3(b2['operating_point']['val_precision'])} / {f3(b2['operating_point']['val_recall'])} / {f3(b2['operating_point']['val_f1'])}" in body
          and f"{f3(d['val_per_class_at_op']['dog']['recall'])} & {f3(b2['val_per_class_at_op']['cat']['recall'])} / {f3(b2['val_per_class_at_op']['dog']['recall'])}" in body
          and "The test split was not used" in body and "It is not promoted" in body)
    sf = load(root, "P2_supplementary_F_split_v2_retrain_20261107.json")
    sb, dv, ox = sf["side_by_side"], sf["detector_v2"], sf["onnx_v2_detector"]
    op = dv["test_at_operating_point"]
    check("Supplementary F quoted as recorded (split v2 retrain: not promoted, test sets differ, ONNX not claimed as parity)",
          sf["split_v2"]["crossing_groups"] == 0 and sf["test_images"]["same_images"] is False
          and sum(sf["split_v2"]["counts"].values()) == 7335
          and abs(op["tp"] / (op["tp"] + op["fp"]) - op["precision"]) < 1e-9 and abs(op["tp"] / (op["tp"] + op["fn"]) - op["recall"]) < 1e-9
          and f"{f3(sb['detector_mAP50']['v1'])} / {f3(sb['detector_mAP50_95']['v1'])} & {f3(sb['detector_mAP50']['v2'])} / {f3(sb['detector_mAP50_95']['v2'])}" in body
          and f"{f3(op['precision'])} / {f3(op['recall'])} / {f3(op['f1'])}" in body
          and f"{f3(sb['mask_mAP50_95']['v1'])} & {f3(sb['mask_mAP50_95']['v2'])}" in body
          and f"{f3(sb['mIoU']['v1'])} / {f3(sb['dice']['v1'])} & {f3(sb['mIoU']['v2'])} / {f3(sb['dice']['v2'])}" in body
          and ox["a9b_protocol"]["within_bound"] is True and ox["real_50_images"]["within_bound"] is False
          and f"{ox['a9b_protocol']['max_abs_diff']:.5f}" in body and f"{ox['real_50_images']['raw_head_max_abs_diff']:.4f}" in body
          and "is not treated as parity" in body and "They are therefore not promoted" in body
          and "not the same images" in body)
    fr = load(root, "P2_supplementary_F_onnx_v2_rerun_20261111.json")
    check("Supplementary F ONNX re-run quoted as recorded (same value as the run record; range over seeds and threads)",
          fr["protocol_run"]["max_abs_diff"] == ox["a9b_protocol"]["max_abs_diff"] and all(x["within_bound"] for x in fr["runs"])
          and fr["weights_sha256"] == sf["v2_weights_sha256_recorded_by_run"]["detector"]
          and fr["onnx_sha256"] == sf["v2_weights_sha256_recorded_by_run"]["detector_onnx"]
          and f"{fr['min_max_abs_diff']:.5f} to {fr['max_max_abs_diff']:.5f}" in body)
    sg = load(root, "P2_supplementary_G_second_recipe_val_20261111.json")
    gs, gn = sg["results"]["delivered_yolo11s"], sg["results"]["second_recipe_yolo11n"]
    check("Supplementary G quoted as recorded (YOLO11n vs delivered YOLO11s, validation only, test split unused)",
          sg["test_split_used"] is False
          and abs(gs["operating_point"]["val_f1"] - d["operating_point"]["val_f1"]) < 1e-9
          and f"{f3(gs['val_ap']['mAP50'])} / {f3(gs['val_ap']['mAP50_95'])} & {f3(gn['val_ap']['mAP50'])} / {f3(gn['val_ap']['mAP50_95'])}" in body
          and f"{f3(gn['operating_point']['val_precision'])} / {f3(gn['operating_point']['val_recall'])} / {f3(gn['operating_point']['val_f1'])}" in body
          and f"{f3(gs['val_recall_at_op']['cat'])} / {f3(gs['val_recall_at_op']['dog'])} & {f3(gn['val_recall_at_op']['cat'])} / {f3(gn['val_recall_at_op']['dog'])}" in body
          and f"{gs['detector_latency_ms_T4']['p50']:.1f} / {gs['detector_latency_ms_T4']['p95']:.1f} & {gn['detector_latency_ms_T4']['p50']:.1f} / {gn['detector_latency_ms_T4']['p95']:.1f}" in body
          and f"{gs['parameters'] / gn['parameters']:.1f} times fewer parameters" in body)
    ss, sn = sg["segmenter"]["results"]["delivered_yolo11s_seg"], sg["segmenter"]["results"]["second_recipe_yolo11n_seg"]
    check("Supplementary G segmenter quoted as recorded (YOLO11n-seg vs delivered YOLO11s-seg, validation only)",
          sg["segmenter"]["test_split_used"] is False and ss["val_masks"]["n"] == sn["val_masks"]["n"] == 1106
          and f"{f3(ss['val_coco']['mask_mAP50'])} / {f3(ss['val_coco']['mask_mAP50_95'])} & {f3(sn['val_coco']['mask_mAP50'])} / {f3(sn['val_coco']['mask_mAP50_95'])}" in body
          and f"{f3(ss['val_masks']['mIoU'])} / {f3(ss['val_masks']['mean_dice'])} & {f3(sn['val_masks']['mIoU'])} / {f3(sn['val_masks']['mean_dice'])}" in body
          and f"{f3(ss['val_masks']['mIoU_no_border'])} / {f3(ss['val_masks']['mean_dice_no_border'])} & {f3(sn['val_masks']['mIoU_no_border'])} / {f3(sn['val_masks']['mean_dice_no_border'])}" in body
          and f"{ss['segmenter_latency_ms_T4']['p50']:.1f} / {ss['segmenter_latency_ms_T4']['p95']:.1f} & {sn['segmenter_latency_ms_T4']['p50']:.1f} / {sn['segmenter_latency_ms_T4']['p95']:.1f}" in body
          and f"{ss['parameters'] / sn['parameters']:.1f} times fewer parameters" in body)
    v2 = load(root, "P2_v2_headline_three_seeds_20261113.json")
    vs, va = v2["seeds"], v2["summary_over_seeds"]
    import statistics as _st
    def _row(get, key):
        vals = [get(vs[k]) for k in ("42", "43", "44")]
        return (abs(_st.mean(vals) - va[key]["mean"]) < 1e-12 and abs(_st.stdev(vals) - va[key]["sd"]) < 1e-12
                and " & ".join(f3(x) for x in vals) + f" & {f3(va[key]['mean'])} $\\pm$ {f3(va[key]['sd'])}" in body)
    check("Split-v2 three-seed run quoted as recorded (per-seed values, mean and sd, checkpoint rule, ONNX not claimed as parity, nothing served)",
          sum(v2["frozen_before_the_run"]["split_v2_counts"].values()) == 7335
          and all(abs(s["detector"]["test_at_operating_point"]["tp"] / (s["detector"]["test_at_operating_point"]["tp"] + s["detector"]["test_at_operating_point"]["fp"])
                      - s["detector"]["test_at_operating_point"]["precision"]) < 1e-9 for s in vs.values())
          and _row(lambda s: s["classifier"]["accuracy"], "classifier_accuracy")
          and _row(lambda s: s["detector"]["mAP50"], "detector_mAP50") and _row(lambda s: s["detector"]["mAP50_95"], "detector_mAP50_95")
          and _row(lambda s: s["detector"]["test_at_operating_point"]["f1"], "detector_f1_at_operating_point")
          and _row(lambda s: s["segmenter"]["mask_mAP50_95"], "mask_mAP50_95") and _row(lambda s: s["segmenter"]["mIoU"], "mIoU")
          and _row(lambda s: s["segmenter"]["dice"], "dice")
          and " & ".join(f"{vs[k]['onnx']['a9b_max_abs_diff']:.5f}" for k in ("42", "43", "44")) in body
          and " & ".join(f"{vs[k]['onnx']['real_50_raw_head_max_abs_diff']:.4f}" for k in ("42", "43", "44")) in body
          and [vs[k]["onnx"]["a9b_met"] for k in ("42", "43", "44")] == [True, False, False]
          and all(s["onnx"]["same_count_fraction"] == 1.0 and s["onnx"]["same_class_fraction"] == 1.0 for s in vs.values())
          and v2["served_checkpoint_by_rule"]["seed"] == int(max(vs, key=lambda k: vs[k]["detector"]["best_val_mAP50_95"]))
          and "points to seed 44" in body and "is not a parity claim" in body and "nothing is served from them yet" in body
          and "had already been evaluated once on 7 November" in body)
    sv = load(root, "P2_v2_system_eval_seed44_20261113.json")
    sb, sh, st_, sd, sr = (sv["benchmark_pytorch_T4"], sv["http_service_pytorch"], sv["team_photos_system"],
                           sv["team_photos_detection"], sv["team_photos_system"]["referral_analysis"])
    f1_ = lambda x: f"{x:.1f}"
    acc = {r["id"]: r for r in sv["acceptance_table"]}
    check("Split-v2 system evaluation of seed 44 quoted as recorded (acceptance rows, T4 latency, HTTP, team photos, what did not improve)",
          sv["weights_sha256"]["detector"] == v2["seeds"]["44"]["detector"]["weights_sha256"]
          and sv["manifest"]["detector_conf"] == vs["44"]["detector"]["operating_point"]["conf"]
          and [acc[i]["met"] for i in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9a", "A9b")] == [True] * 9 + [False]
          and abs(acc["A3"]["value"] - vs["44"]["detector"]["mAP50"]) < 1e-12 and abs(acc["A5"]["value"] - vs["44"]["segmenter"]["mIoU"]) < 1e-12
          and abs(acc["A9b"]["value"] - vs["44"]["onnx"]["a9b_max_abs_diff"]) < 1e-12
          and f"{f3(acc['A1']['value'])} / {f3(acc['A2']['value'])} & Pass" in body and f"{f3(acc['A4']['value'])} & Pass" in body
          and f"{sh['succeeded']}/{sh['requests']} & Pass" in body and sh["succeeded"] == sh["requests"] == 300
          and f"{f1_(sb['latency_ms']['p95'])}\\,ms & Pass" in body and "0.0038 & Not met" in body
          and f"{f1_(sb['latency_ms']['p50'])} / {f1_(sb['latency_ms']['p95'])} / {f1_(sb['latency_ms']['p99'])}" in body
          and f"{f1_(sh['http_latency_ms']['p50'])} / {f1_(sh['http_latency_ms']['p95'])} / {f1_(sh['http_latency_ms']['p99'])}" in body
          and sr["single_species_correct"] + sr["classification_errors"] == sr["single_species_photos"]
          and f"{sr['single_species_correct']} of {sr['single_species_photos']} single-species photos" in body
          and st_["errors_escaping_review"] == 0 and "lets no wrong label through as ready" in body
          and abs(sd["at_operating_point"]["tp"] / (sd["at_operating_point"]["tp"] + sd["at_operating_point"]["fn"]) - sd["at_operating_point"]["recall"]) < 1e-12
          and f"recall {f3(sd['at_operating_point']['recall'])}" in body
          and f"{sr['multi_animal_accepted_ready']} of {sr['multi_animal_photos']} multi-animal photos are marked ready, against 17 of 52" in body
          and sv["figures_check"]["detector_counts_equal_record"] and sv["figures_check"]["segmenter_mIoU_equals_record"])
    eb = load(root, "P2_v2_error_breakdown_seed44_20261113.json")
    ev_dir = root / "report" / "evidence"
    def rows(name):
        with open(ev_dir / name, newline="") as fh:
            return list(csv.DictReader(fh))
    ebfp, ebfn = rows("P2_v2_seed44_fp_breakdown_20261113.csv"), rows("P2_v2_seed44_fn_breakdown_20261113.csv")
    kinds = lambda rs: {k: sum(r["kind"] == k for r in rs) for k in {r["kind"] for r in rs}}
    nz = lambda d: {k: v for k, v in d.items() if v}
    fpk, fnk, dl = eb["fp_by_kind"], eb["fn_by_kind"], eb["delivered_build_for_comparison"]
    pct = lambda n, d: f"{100 * n / d:.1f}"
    check("Split-v2 error breakdown of seed 44 quoted as recorded (counts by cause from the per-error CSV files, delivered-build column, served-build decision)",
          hashlib.sha256((ev_dir / "P2_v2_seed44_fp_breakdown_20261113.csv").read_bytes()).hexdigest() == eb["fp_csv_sha256"]
          and hashlib.sha256((ev_dir / "P2_v2_seed44_fn_breakdown_20261113.csv").read_bytes()).hexdigest() == eb["fn_csv_sha256"]
          and kinds(ebfp) == nz(fpk) and kinds(ebfn) == nz(fnk)
          and len(ebfp) == eb["counts"]["fp"] == sv["figures_check"]["detector_counts"]["fp"]
          and len(ebfn) == eb["counts"]["fn"] == sv["figures_check"]["detector_counts"]["fn"]
          and eb["detector_sha256"] == sv["weights_sha256"]["detector"]
          and sum(r["image_also_has_fp"] == "True" for r in ebfn) == eb["fn_in_images_with_fp"]
          and nz(dl["fp_by_kind"]) == kinds(rows("M4_fp_breakdown.csv")) and nz(dl["fn_by_kind"]) == kinds(rows("M4_fn_breakdown.csv"))
          and f"& {fpk['class_confusion']} & {pct(fpk['class_confusion'], len(ebfp))} & {dl['fp_by_kind']['class_confusion']} \\\\" in body
          and f"& {fpk['localisation']} & {pct(fpk['localisation'], len(ebfp))} & {dl['fp_by_kind']['localisation']} \\\\" in body
          and f"& {fpk['background']} & {pct(fpk['background'], len(ebfp))} & {dl['fp_by_kind']['background']} \\\\" in body
          and f"& {fnk['localisation']} & {pct(fnk['localisation'], len(ebfn))} & {dl['fn_by_kind']['localisation']} \\\\" in body
          and f"& {fnk['class_confusion']} & {pct(fnk['class_confusion'], len(ebfn))} & {dl['fn_by_kind']['class_confusion']} \\\\" in body
          and f"& {fnk['missed']} & {pct(fnk['missed'], len(ebfn))} & {dl['fn_by_kind']['missed']} \\\\" in body
          and f"False negatives & & {len(ebfn)} & 100.0 & {dl['counts']['fn']} \\\\" in body
          and f"{eb['fn_in_images_with_fp']} of the {len(ebfn)} false negatives are in images that also have a false positive" in body
          and sum(r["kind"] == "class_confusion" and r["pred"] == "dog" for r in ebfp) == 9 and "nine are cats detected as dogs" in body
          and "are not a paired comparison" in body
          and eb["owner_rule_20261113"]["outcome"] == "not adopted" and "keeps the delivered split-v1 weights" in body)
    pm = lambda key: f"{f3(va[key]['mean'])} $\\pm$ {f3(va[key]['sd'])}"
    card = (Path(__file__).resolve().parent.parent / "docs" / "model_card.md").read_text()
    pmc = lambda key: f"{f3(va[key]['mean'])} ± {f3(va[key]['sd'])}"
    check("Headline front matter and model card quote the split-v2 means as recorded and say which build is served",
          f"detection mAP50 {pm('detector_mAP50')} (mAP50--95 {pm('detector_mAP50_95')}) and F1 {pm('detector_f1_at_operating_point')}" in body
          and f"segmentation mean IoU {pm('mIoU')}, classification accuracy {pm('classifier_accuracy')}" in body
          and f"{pm('detector_mAP50')} / {pm('detector_mAP50_95')}" in body and f"{pm('mIoU')} / {pm('mask_mAP50_95')}" in body
          and f"({f3(sd['at_operating_point']['recall'])} for the split-v2 checkpoints)" in body
          and f"({sr['multi_animal_accepted_ready']} of {sr['multi_animal_photos']})" in body
          and "the packaged service and demonstration use this build" in body
          and "nine are met and A9b is not" in body
          and f"{pmc('detector_mAP50')} / {pmc('detector_mAP50_95')}" in card and pmc("classifier_accuracy") in card
          and f"{pmc('mask_mAP50_95')} / {pmc('mIoU')} / {pmc('dice')}" in card
          and "These checkpoints are not the served weights" in card
          and f"{sr['multi_animal_accepted_ready']} of {sr['multi_animal_photos']} multi-animal photos wrongly marked ready" in card)
    cpu = load(root, 'cpu_http_release_20261114/measurement.json')
    from preflight_release import EXPECTED_SHA256
    receipt_path = root/'report/evidence/cpu_http_release_20261114/measurement.json'
    provenance = load(root, 'cpu_http_release_20261114/PROVENANCE.json')
    check('Current CPU HTTP receipt is bound to unchanged served weights and runtime',
          hashlib.sha256(receipt_path.read_bytes()).hexdigest() == provenance['original_measurement_sha256']
          and provenance['runtime_files_match_source_commit']
          and cpu['weights_sha256'] == EXPECTED_SHA256
          and all(hashlib.sha256((root/'src/spv'/name).read_bytes()).hexdigest()==value
                  for name,value in cpu['runtime_code_sha256'].items()))
    check('Current CPU HTTP contract and 300-input sequence',
          cpu['passed_http_contract'] and cpu['requests']==cpu['valid_responses']==len(cpu['input_sequence'])==300
          and cpu['distinct_images']==50 and cpu['warmup_requests']==5
          and cpu['non_image_status']==415 and cpu['empty_image_status']==400)
    latency='/'.join(f"{cpu['http_latency_ms'][f'p{q}']:.1f}" for q in (50,95,99))
    check('Current CPU HTTP percentiles throughput and RSS quoted without a T4 claim',
          latency in body and f"{cpu['throughput_requests_s']:.2f} requests/s" in body
          and f"{cpu['server_rss_mb']['peak_measured']:.1f}" in body
          and cpu['cpu'] in body and 'does not remeasure the T4-specific A8' in body)
    ref = load(root, 'v2_cpu_reference_20261114/measurement.json')
    meta = load(root, 'v2_cpu_reference_20261114/PROVENANCE.json')
    base = load(root, 'v2_saved_review_20261114/seed44_record.json')
    folder = root/'report/evidence/v2_cpu_reference_20261114'
    check('V2 CPU receipt identity and platform limits are explicit',
          hashlib.sha256((folder/'measurement.json').read_bytes()).hexdigest()==meta['receipt_sha256']
          and ref['verification_kind']=='automated_cpu_reference'
          and ref['human_second_member_claimed'] is False and ref['provenance']['device']=='cpu'
          and ref['provenance']['dirty'] is False and ref['seed']==44
          and ref['metadata_sha256']==base['split_v2_metadata_sha256']
          and ref['weights_sha256']==base['weights_sha256']
          and ref['environment_differences']['numpy']['actual']=='2.3.5')
    expected={'mAP50':base['detector']['coco']['mAP50'],'mAP50_95':base['detector']['coco']['mAP50_95'],
              'mask_mAP50':base['segmenter']['coco']['mask_mAP50'],'mask_mAP50_95':base['segmenter']['coco']['mask_mAP50_95'],
              'mIoU':base['segmenter']['test_metrics']['mIoU'],'mean_dice':base['segmenter']['test_metrics']['mean_dice']}
    check('V2 CPU six comparisons recomputed against frozen originals',
          all(ref['originals'][k]==v and ref['comparison'][k]['tolerance']==.005
              and abs(ref['reproduced'][k]-v)<=.005
              and abs(ref['reproduced'][k]-v-ref['comparison'][k]['difference'])<1e-12
              and f"{ref['reproduced'][k]:.6f}" in body for k,v in expected.items())
          and 'automated CPU verification' in body and 'not a second team member' in body)
    with (folder/'per_image.csv').open() as stream:
        mask_rows=list(csv.DictReader(stream))
    import statistics
    check('V2 CPU overlap scores recomputed from 1050 unique images',
          hashlib.sha256((folder/'per_image.csv').read_bytes()).hexdigest()==meta['per_image_sha256']
          and len(mask_rows)==len({v['image'] for v in mask_rows})==1050
          and all(abs(statistics.mean(float(v[field]) for v in mask_rows)-ref['reproduced'][metric])<1e-12
                  for field,metric in [('iou','mIoU'),('dice','mean_dice')]))
    by=ref['per_class_from_this_evaluation'];extra=ref['additional_metrics_no_new_acceptance_thresholds']
    check('V2 CPU per-species AP and counts reconcile; additional classifier values unchanged',
          all(abs(sum(by[c][field] for c in ('cat','dog'))/2-ref['reproduced'][metric])<1e-12
              for field,metric in [('box_AP50_95','mAP50_95'),('mask_AP50_95','mask_mAP50_95')])
          and all(sum(by[c]['detector_at_operating_point'][k] for c in ('cat','dog'))==extra['detector_operating_point'][k]['reproduced']
                  for k in ('tp','fp','fn'))
          and all(v['difference']==0 for v in extra['classifier'].values())
          and all(f"{by[c][k]:.6f}" in body for c in ('cat','dog') for k in ('box_AP50_95','mask_AP50_95')))
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
