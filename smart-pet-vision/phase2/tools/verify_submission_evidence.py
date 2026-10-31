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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--figures", type=Path, help="Folder holding the report figures (optional)")
    a = ap.parse_args()
    results, figs = checks(a.root)
    if a.figures:
        missing = [f for f in figs if not (a.figures.parent / f).is_file() and not (a.figures / Path(f).name).is_file()]
        results.append(("all report figures present", not missing, ", ".join(missing)))
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
