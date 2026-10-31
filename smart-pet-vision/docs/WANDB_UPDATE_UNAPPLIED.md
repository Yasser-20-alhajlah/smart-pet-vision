# W&B documentation update prepared but not applied

Authenticated W&B project access was unavailable. This text was not published to W&B and no historical run was edited or synchronized in this review.

Target project: https://wandb.ai/yasser-alhajlah-kaust/smart-pet-vision
Runs referenced by source records: 2seovilv (detector), 3ztpz2tc (YOLO segmentation), 1jb5y453 (SAM), fqow5ly7 (deployment). Their cloud access was not verified.

Suggested project note after authenticated review: Current source corrections and 41 discovered tests are documented in the execution review folder. 40 software/ASGI/privacy tests passed and one PyTorch test was skipped. Saved-record arithmetic and five binary hashes were verified. No current trained-weight inference, parity or T4 measurement was performed; detector A9b remains failed. Historical and prior CPU results are retained separately. Every output requires staff approval and prohibits automatic publication.

Apply only after access is available: verify the correct entity/project and run IDs, append a clearly dated project note linking the execution review folder, retain all historical logs and failed runs, then reopen the note and verify its content. Do not backdate runs or mark local offline folders as synchronized without a cloud record.

Evidence: https://drive.google.com/drive/folders/1eiq8c7EKQaUCrnxk4Y1E0-SacJ67cSF3


Latest execution update, not published to W&B: 49 tests passed; real-weight CPU regression completed on 401 inspected photos, 386/399 correct species predictions, 81 referrals, 320 ready photos, 17/52 multi-animal ready cases and 0/2 mixed-species ready cases. Actual HTTP served 300/300 requests. Fresh classifier ONNX parity passed at 2.9802322e-7; raw detector parity still fails at 0.00201416 > 0.001. See https://drive.google.com/drive/folders/1KE1ZgeYBzT4Rv05FJk4ufrkNNx6aeWfk .
