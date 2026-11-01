# Model card: Smart Pet Vision (Phase 2 release)

## Intended use
Assist shelter and adoption-platform staff who review one candidate listing photo of a cat or a dog. The service returns the detected animal(s), the species with a temperature-scaled confidence, a body mask, and a technical verdict: `ready` or `refer_to_staff` with reasons. **Every photo, ready or referred, requires staff approval; automatic publication is disabled** (`staff_approval_required: true`, `automatic_publication_allowed: false` in every response).

## Out of scope
Automatic publication; identifying individual animals or people; tracking; veterinary or medical use; species other than cats and dogs; unsupervised public-facing decisions.

## Models (release manifest `runs/manifest.json`)
| Component | Model | Input | Weights SHA-256 |
|---|---|---|---|
| Detector | YOLO11s, fine-tuned 40 epochs (W&B 2seovilv) | 640 px letterbox | f48e4d8ea5c925c18a4b43744c2f697bfc01a0a473492275f000848125da1cd4 |
| Segmenter | YOLO11s-seg, fine-tuned 40 epochs (W&B 3ztpz2tc) | 640 px letterbox | 868f3842bf0e05c8b936d3e51e6e81b7f44fdcf12f7b666e5041bf1d47a5d8ee |
| Classifier | ResNet-50, ImageNet weights, fine-tuned (Phase 1, W&B 8nvxbsn2) | 224 px crop | 22dd58d471f2dbb39eef10331f208d668a569b1feae378171df8074fbe2081ba |

Operating point (chosen on the Oxford validation split): detector confidence 0.40, NMS IoU 0.50; classifier temperature 0.05, verification threshold 0.90; minimum matched-mask share 0.05.

## Data
Oxford-IIIT Pet (CC BY-SA 4.0), 7,335 images after cleaning, group-aware split 5,123 / 1,106 / 1,106 (seed 42). Body boxes and masks derived from the trimaps. A private set of 401 team photos (owner consent; metadata removed) is used only for supplementary evaluation and is not committed to the repository.

## Performance (Oxford test split unless stated)
| Measure | Value | Conditions |
|---|---|---|
| Detection mAP50 / mAP50-95 | 0.984 / 0.850 | 640 px, conf 0.001 evaluation |
| Detection P / R / F1 | 0.958 / 0.959 / 0.959 | conf 0.40, NMS 0.50 |
| Mask mAP50-95; mean IoU; mean Dice | 0.875; 0.923; 0.959 | 640 px, conf 0.25 |
| Classification accuracy / macro-F1 | 0.999 / 0.999 | 224 px (Phase 1) |
| Team photos: detection recall | 0.732 (dog 0.651) | 401 photos, 503 animals, conf 0.40 |
| Team photos: single-species species accuracy | 386/399 | CPU run W&B f8manl0i |
| End-to-end p95 latency | 61 ms (T4, in-process); 431 ms (CPU HTTP, W&B mu2tolkg) | batch 1 |

## Known failure modes
- Missed secondary or small animals: on team photos 17 of 52 multi-animal photos were marked technically ready because only one animal was detected.
- Lower recall on non-Oxford photos (team set recall 0.732 vs 0.959 on Oxford).
- Dark dogs in clothing detected as cats (detector/classifier disagreement then triggers referral).
- Masks bleed into similar-coloured blankets or clothing; occlusion truncates masks.
- The ONNX detector export does not meet the numerical parity criterion (max abs diff 0.0018-0.0020 vs 0.001); the service uses PyTorch.

## Ethical considerations
Photos may show people or homes; the service returns no identity information and keeps no uploaded image. Team photos are kept in private storage only.
