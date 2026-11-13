# Split-v2 checkpoint verification by a second team member

This is the team's additional verification of the new headline results, not a new signature or instructor submission condition. The recorded split-v1 six-metric reproduction remains separate. Do not train, refit temperatures, change thresholds or choose a seed from the test results. Use the already selected seed 44 and a free T4 runtime only. Stop if free T4 is unavailable; do not buy compute.

## Frozen inputs

- [Three-seed run folder](https://drive.google.com/drive/folders/1cE_Le0cnmtE1ILXIY5LWht00c9RD1aLJ)
- [Seed-44 folder](https://drive.google.com/drive/folders/1IDOBgZ6B9qwun9snrIBVnawWCBsI1yxc)
- [Original seed-44 record](https://drive.google.com/file/d/1uqYaG5PNQzunjUc5teauGvV2hrrpp6QQ/view)
- [Frozen full metadata](https://drive.google.com/file/d/1HOU3b76UQwwumYNY3Z2s7p2Bxq8i05xj/view), SHA-256 `c06f4d9e5dd4e333b4d5e17f1b07c115209dd5b9da8a6e99b4d48db8316db13f`
- [Classifier](https://drive.google.com/file/d/1f9MdMJhMmb8K6aAChsrDkOXecctuwAgJ/view), SHA-256 `98eb8956c27ffa1d439aaa98ec3a4d8156489e5b1541ee1debe26b75c633040c`
- [Detector](https://drive.google.com/file/d/1pY-fLj7OqE95eLHeBvPHecfVu1guYi9U/view), SHA-256 `92548b78d3d04491d3ba8573c8139bfb04b587b96aed5bd0ed3e2923bdb9ac08`
- [Segmenter](https://drive.google.com/file/d/13vETSe7psY6j3c0HjngQ56RH3b7fWLVi/view), SHA-256 `47f2e2412d8e3109bd2bcd4fcb5bdbb3a5bb250e78a8d112a22f720381ae7a08`
- [Code](https://github.com/Yasser-20-alhajlah/smart-pet-vision); use the delivery commit printed in the final report. Original training source remains `f62ded8fc190820842f5d57d87d12556bd67d4b3`.

The full metadata was reconstructed with the original merge and train-only box corrections using NumPy 2.1.3 and pandas 2.2.3. Its bytes match all three original run records; no split was regenerated. The original v1 and v2 CSV hashes are retained in `report/evidence/v2_saved_review_20261114/SOURCE_REVIEW.json`.

## Prepare a clean Colab runtime

Select the free T4, then mount your team's Drive using the normal Google account flow. Never put passwords or W&B keys in a cell. Download the five linked input files to a private `inputs` folder in Drive, using these names: `metadata.csv`, `P2_v2_headline_record.json`, `finetune.pt`, `detector.pt`, `segmenter.pt`. Do not commit them. Run `nvidia-smi` and retain its output.

Clone the code and pin the exact delivered revision:

```python
import subprocess
subprocess.run(['git', 'clone', 'https://github.com/Yasser-20-alhajlah/smart-pet-vision', '/content/spv-review'], check=True)
# Read the delivery SHA from the report. Paste that SHA below; do not use an unrelated checkout.
DELIVERY_SHA = 'PASTE_DELIVERED_COMMIT_FROM_REPORT'
subprocess.run(['git', '-C', '/content/spv-review', 'checkout', DELIVERY_SHA], check=True)
subprocess.run(['python', '-m', 'pip', 'install', '-r', '/content/spv-review/smart-pet-vision/requirements.txt'], check=True)
```

Set `INPUTS` to the actual private folder you saved, not a guessed Drive path. Prepare data from the frozen metadata; this downloads Oxford from its official source if absent. It does not make a new split:

```python
from pathlib import Path
import sys, shutil, hashlib, json, os
P2 = Path('/content/spv-review/smart-pet-vision/phase2')
sys.path.insert(0, str(P2/'src'))
from spv.common import load_config
from spv import data
INPUTS = Path('/content/drive/MyDrive/SET_YOUR_REAL_PRIVATE_INPUT_FOLDER')
WORK = Path('/content/spv-v2-reproduction')
assert not WORK.exists(), 'Use a fresh work directory'
record = json.loads((INPUTS/'P2_v2_headline_record.json').read_text())
assert record['seed'] == 44
assert hashlib.sha256((INPUTS/'metadata.csv').read_bytes()).hexdigest() == record['split_v2_metadata_sha256']
cfg = load_config()
cfg['paths']['raw'] = WORK/'raw'
cfg['paths']['processed'] = WORK/'processed'
cfg['paths']['runs'] = WORK/'runs'
cfg['wandb']['enabled'] = False
saved = cfg['paths']['runs']/'data'
saved.mkdir(parents=True)
shutil.copy2(INPUTS/'metadata.csv', saved/'metadata.csv')
os.environ['WANDB_MODE'] = 'disabled'
frame = data.ensure_data(cfg)
assert frame.split.value_counts().to_dict() == {'train':4903, 'val':1382, 'test':1050}
IMAGE_DIR = data.dataset_dir(cfg)/'images'
```

Run the dedicated verification tool. Enter your own name for `OPERATOR`; do not attribute automated work to another person. This freezes the original temperature, thresholds, weights and test cohort and refuses wrong hashes, library versions or an existing output receipt:

```python
OPERATOR = 'ENTER_YOUR_OWN_FULL_NAME'
OUTPUT = INPUTS/'second_member_v2_seed44.json'
command = [
 'python', str(P2/'tools/reproduce_v2_eval.py'),
 '--operator', OPERATOR,
 '--record', str(INPUTS/'P2_v2_headline_record.json'),
 '--metadata', str(INPUTS/'metadata.csv'),
 '--cls-weights', str(INPUTS/'finetune.pt'),
 '--det-weights', str(INPUTS/'detector.pt'),
 '--seg-weights', str(INPUTS/'segmenter.pt'),
 '--det-yaml', str(WORK/'processed/yolo_detect/data.yaml'),
 '--seg-yaml', str(WORK/'processed/yolo_segment/data.yaml'),
 '--processed', str(WORK/'processed'), '--image-dir', str(IMAGE_DIR),
 '--out', str(OUTPUT)]
completed = subprocess.run(command, cwd=P2, text=True, capture_output=True)
(INPUTS/'second_member_v2_seed44.log').write_text(completed.stdout+completed.stderr)
print(completed.stdout[-5000:]); print('Exit code:', completed.returncode)
```

## Return these outputs

1. The saved executed `.ipynb`, with all cell outputs, actual GPU information and any errors retained.
2. `second_member_v2_seed44.json`, its `.per_image.csv` and the complete `.log`.
3. `pip_freeze.txt` (`python -m pip freeze`) and `git rev-parse HEAD` output.
4. A short message with the actual operator, date, GPU, whether the whole evaluation completed, and all differences. No signature form is required.

The tool evaluates the classifier and the detector's frozen operating point as well as six main detection/segmentation metrics. Only the existing six-metric 0.005 reproduction tolerance is used; extra values are reported without invented acceptance thresholds. Repeated evaluation of an already inspected test set is labelled reproduction, never a new untouched final test. If a metric differs, preserve the receipt and investigate the environment or inputs; do not tune on test or silently replace the original headline numbers.

## What was already verified without GPU

On 14 November all nine weights matched their source SHA-256 values and ZIP CRC checks, all thirty classifier epoch rows matched live W&B histories including epoch times, and 57 saved-output arithmetic checks passed. Those checks are not a second person's checkpoint inference. `tools/reproduce_v2_eval.py` has been reviewed and input guards tested; its full model evaluation must be executed by the teammate before that distinct claim is made.
