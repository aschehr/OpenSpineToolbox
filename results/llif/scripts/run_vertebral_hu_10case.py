"""Runner: trabecular HU (bone density) at L4 and L5 for the same 10
manually-validated cases used throughout this project. Separate from
`run_llif_report_10case.py` because this is the one parameter needing the
CT intensity volume (not just labels) -- kept as its own small run rather
than re-running the whole heavier report, and because CT files (~230MB
each) are only worth loading once per case here, not bundled into every
other run that doesn't need them.

This data is also the reference set for the point-sampling MANUAL
validation of vertebral_hu (see results/llif/README.md): for each case/
level below, a human clicks 3-5 points in the central trabecular region
in ITK-SNAP, records the HU value the cursor shows at each point, and the
average of those points is compared against this run's mean_hu -- a
"same ballpark" region-vs-points check, not a point-landmark ICC/Pearson
comparison (there's no single correct point for a region mean)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

from ostk.io import load_label, load_ct  # noqa: E402
from ostk.crest_height import load_label_ids  # noqa: E402
from ostk.vertebral_hu import vertebral_hu_from_label  # noqa: E402

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
CASE_IDS = ["0441", "0049", "0231", "0680", "0508", "0368", "0184", "0910", "0655", "1058"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vertebral_hu_10case.jsonl")


def verify_revision(retries: int = 3) -> None:
    assert os.path.isdir(SNAPSHOT), f"pinned snapshot missing on disk: {SNAPSHOT}"
    last_err = None
    for attempt in range(retries):
        try:
            remote = subprocess.run(
                ["git", "ls-remote", "https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K"],
                capture_output=True, text=True, timeout=30,
            )
            lines = [l for l in remote.stdout.splitlines() if "refs/heads/v4" in l]
            if not lines:
                raise RuntimeError(f"could not read refs/heads/v4 from the remote "
                                   f"(attempt {attempt+1}/{retries}); stderr={remote.stderr[:200]!r}")
            remote_hash = lines[0].split()[0]
            assert remote_hash == PINNED_REVISION, (
                f"pinned revision {PINNED_REVISION} does not match remote v4 HEAD {remote_hash}"
            )
            print(f"revision verified: {PINNED_REVISION}", flush=True)
            return
        except (RuntimeError, subprocess.SubprocessError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(3)
    raise RuntimeError(f"revision verification failed after {retries} attempts: {last_err}")


def main() -> None:
    verify_revision()
    label_ids = load_label_ids(os.path.join(SNAPSHOT, "dataset_labels.json"))

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["case_id"])
    todo = [c for c in CASE_IDS if c not in done]
    if len(sys.argv) > 1:
        todo = [c for c in todo if c == sys.argv[1]]
    print(f"CHECKPOINT: {len(done)} already done, {len(todo)} to do this invocation", flush=True)

    with open(OUT_PATH, "a", encoding="utf-8") as out_fh:
        for case_id in todo:
            t0 = time.time()
            label_path = os.path.join(SNAPSHOT, "labels", f"{case_id}_label.nii.gz")
            ct_path = os.path.join(SNAPSHOT, "ct", f"{case_id}_ct.nii.gz")
            lab, aff = load_label(label_path)
            ct, ct_aff = load_ct(ct_path)
            assert (ct_aff == aff).all(), f"{case_id}: CT/label affine mismatch"
            rec = {
                "case_id": case_id,
                "L4": vertebral_hu_from_label(lab, ct, aff, "L4", label_ids, case_id=case_id),
                "L5": vertebral_hu_from_label(lab, ct, aff, "L5", label_ids, case_id=case_id),
            }
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            print(f"{case_id} ({time.time()-t0:.1f}s): "
                 f"L4 mean={rec['L4']['mean_hu']}  L5 mean={rec['L5']['mean_hu']}", flush=True)

    print(f"\nDONE -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
