# Smart Pet Vision Phase 2 reviewed source snapshot

The ZIP in this folder contains the reviewed local code snapshot based on commit 54a78bb2816a82d69a70c8414da1f434d15f3970, with packaging and preflight fixes. Extract it and work in code/. The archive is the source artifact; the historical repository directories are retained separately.

Validation: 108 local unit tests passed. Three fingerprint-matched weights were loaded in a real CPU TCP-loopback HTTP check, including one previously inspected team cat photograph, images on/off, a blank image, and invalid inputs. This is not a fresh test-set accuracy evaluation, GPU benchmark, or presentation-host rehearsal. No training or hosted deployment is claimed.

Restore authorized weights at the paths in code/runs/manifest.json; follow code/docs/RUN_SERVICE.md. Weights, raw photographs, and student IDs are excluded. Some archived evidence paths are redacted. Figures and the private report/source are delivered through the authorized team Drive folder; this source archive is not the Overleaf submission ZIP.

Historical test metrics remain tied to their original experiment records. The ONNX detector remains outside the accepted serving path. Training-optimizer provenance and some historical mask settings remain unresolved and must not be guessed.

Archive SHA256: 3fe103ef7bde93aebb02e9aa736b2743a4271aebc80311dfc244864073216fde
