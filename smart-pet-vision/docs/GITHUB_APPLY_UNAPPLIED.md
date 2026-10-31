# GitHub changes prepared but not applied

Authenticated GitHub account: mosaed2022-ai. The connected installation returned zero accessible repositories. A search for smart-pet-vision could not resolve an installed project repository. No repository URL was found in the inspected project documents. No remote branch, pull request or merge was performed.

The local review branch is review/phase2-20261030. Its baseline is a synthetic snapshot of the reviewed source archive, not a claimed GitHub commit. The release includes git_review.patch and the exact base/final local commit IDs. First expose the existing project repository through the GitHub installation, then verify owner/name and the current main branch. Create a new review branch there. Back up and compare every affected source file against the source_inventory hashes before applying the patch or copying files from this release. If upstream is newer or the patch cannot be applied cleanly, reconcile those changes; do not overwrite them with this snapshot. Run the documented software tests and trained-weight checks, push the review branch and open a PR with the actual test results and remaining failures. Do not merge without explicit authorization.

Existing installation settings: https://github.com/settings/installations/176946083
