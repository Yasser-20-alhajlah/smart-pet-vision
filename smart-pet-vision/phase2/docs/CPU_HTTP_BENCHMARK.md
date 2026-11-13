# Served-package HTTP benchmark

The 14 November receipt in `report/evidence/cpu_http_release_20261114/`
measures the unchanged served-v1 system over actual loopback HTTP. It does not
evaluate model accuracy, use a GPU, train, select thresholds, or remeasure the
T4-specific A8 criterion. The full original receipt is retained unchanged.

On Apple M5 CPU, one worker and one sequential client, five warm-up requests
preceded 300 measured requests (50 Oxford v1 test images, cycled six times).
All 300 responses passed the HTTP and staff-approval contract. Non-image and
empty inputs returned 415 and 400. HTTP p50/p95/p99 were
124.9/135.3/138.4 ms, throughput 7.96 requests/s over 37.67 seconds, and
peak sampled server RSS 919.4 MB. RSS is server-only decimal MB sampled every
20 ms, so it may miss instantaneous peaks. It is not in-process GPU allocation.
This is a latency benchmark, not a concurrent load or model-quality test.

The public Oxford archive MD5 and all 50 image MD5 values match the frozen
split-v1 metadata. Selection occurred before inference. No private team image
was used or published. All three served weight hashes match
`tools/preflight_release.py`; the five recorded runtime files match source
`6f97aa2cb91a65077573ec8a6bf80d633b02f7bb`. The receipt truthfully records
the dirty worktree because report and verification tools were being improved.
It does not identify this as a clean checkout of a later documentation commit.

## Repeating this measurement

Use an isolated CPU environment and the three authorised served-v1 weights.
Keep the original Linux/Colab `../requirements.txt` for its recorded platform.
For this macOS ARM64 probe, install those direct pins with NumPy **2.3.5**
instead of **2.1.3**, plus `psutil==7.2.2`. Ultralytics 8.4.182 explicitly
excludes NumPy 2.1.x on Darwin. Python 3.12.14 and the exact installed versions
are retained in `measurement.json`; this is a platform-specific environment,
not the original Colab environment. Do not claim that the original environment
preflight passed after substituting a package.

Prepare a local manifest with only the three weight paths resolved to existing
files; retain every other value from `runs/manifest.json`. Freeze a JSON array
of 300 image paths before execution (50 selected public images repeated six
times) and retain its selection and image hashes. Then, from `phase2/`:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python tools/benchmark_http_release.py --manifest /path/to/served_manifest.json \
  --images /path/to/frozen_workload.json --warmup 5 --out /new/path/http_receipt.json
```

The tool disables CUDA and W&B, launches one uvicorn worker, times the full
multipart request/response, samples server-process RSS, validates every staff
guard, and records input, weight and runtime-source hashes. It refuses to
overwrite an existing receipt or server log. Framework thread counts were not
separately captured in the retained run; the environment variables record the
requested counts. Return the JSON, server log, selection and environment.
Use new output paths. Performance depends on the host and workload; do not
combine this CPU run with the historical T4 figures.
