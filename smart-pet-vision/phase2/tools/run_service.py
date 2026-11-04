"""Check the prerequisites, then start the service on this computer only (loopback address).

  python tools/run_service.py                 # checks, then serves http://127.0.0.1:8000/demo
  python tools/run_service.py --check-only    # checks only
  python tools/run_service.py --port 8010 --allow-version-drift

Run from smart-pet-vision/phase2 in the environment made from ../requirements.txt.
What it does: checks the Python version, that the needed packages import, that the three private
weight files are at the manifest paths with the recorded SHA-256 and that package versions match
../requirements.txt (tools/preflight_release.py; version drift is allowed only with
--allow-version-drift), and that the port is free; then runs uvicorn with one worker.
What it never does: install packages, download weights or data, bind to a non-loopback address,
read or send credentials. Ultralytics online checks (YOLO_OFFLINE=1) and W&B (WANDB_MODE=disabled)
are switched off for the service process. Stop the service with Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # smart-pet-vision/phase2
LOOPBACK = ("127.0.0.1", "localhost", "::1")
CHECKED_PYTHON = (3, 13)


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(("  ok    " if ok else "  FAIL  ") + label + (f": {detail}" if detail else ""))
    return ok


def port_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind(("::1" if host == "::1" else "127.0.0.1", port))
            return True
        except OSError:
            return False


def prerequisites(manifest: str, host: str, port: int, allow_drift: bool) -> bool:
    print("Prerequisites")
    v = sys.version_info
    if (v.major, v.minor) == CHECKED_PYTHON:
        check(True, "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        print(f"  note  Python {v.major}.{v.minor}.{v.micro}: the checked environments used Python 3.13; this version is untested")
    ok = check(host in LOOPBACK, "loopback address only", host)
    ok &= check((HERE / manifest).is_file(), "manifest present", manifest)
    if not ok:  # stop before the slower checks
        return False
    missing = []
    for mod in ("fastapi", "uvicorn", "multipart", "PIL", "numpy", "torch", "ultralytics"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    ok &= check(not missing, "packages importable",
                (", ".join(missing) + " missing; install ../requirements.txt yourself") if missing else "")
    if ok:
        cmd = [sys.executable, "tools/preflight_release.py", "--manifest", manifest,
               "--requirements", str(HERE.parent / "requirements.txt")]
        if not allow_drift:
            cmd.append("--strict-versions")
        pre = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": str(HERE / "src")})
        ok &= check(pre.returncode == 0,
                    "preflight: weight files, SHA-256, operating point" + ("" if allow_drift else ", pinned versions"),
                    "" if pre.returncode == 0 else (pre.stderr.strip() or pre.stdout.strip())[-600:])
    free = port_free(host, port)
    ok &= check(free, f"port {port} free", "" if free else "choose another --port or stop the process using it")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="runs/manifest.json")
    ap.add_argument("--host", default="127.0.0.1", help="loopback only: 127.0.0.1, localhost or ::1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-version-drift", action="store_true",
                    help="accept package versions that differ from ../requirements.txt (reported, not fixed)")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not prerequisites(a.manifest, a.host, a.port, a.allow_version_drift):
        print("\nNot started. See docs/TROUBLESHOOTING.md.")
        return 1
    if a.check_only:
        print("\nAll checks passed (service not started: --check-only).")
        return 0
    return serve(a.manifest, a.host, a.port)


def serve(manifest: str, host: str, port: int) -> int:
    env = {**os.environ, "SPV_MANIFEST": manifest, "PYTHONPATH": str(HERE / "src"),
           "YOLO_OFFLINE": "1", "WANDB_MODE": "disabled"}
    url = f"http://{'[::1]' if host == '::1' else host}:{port}"
    print(f"\nStarting the service. Open {url}/demo   (health: {url}/health)   Stop: Ctrl+C\n"
          "The first analysis loads the three models and takes longer than the next ones.\n", flush=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "spv.serve:app", "--host", host,
                             "--port", str(port), "--workers", "1"], cwd=HERE, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\nService stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
