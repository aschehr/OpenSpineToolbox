"""One-off runner: crest-height classification over the 10 manually-validated
cases, pinned to a verified HF snapshot revision.

Revision provenance (checked this run, not asserted):
  `git ls-remote https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K`
  shows refs/heads/v4 -> 9f69480a572624029a927bc6661309a068f1f2c3, which matches
  the locally cached snapshot directory of the same name. Both are printed
  again below at run time so the log is self-verifying.

Not part of the ostk package (it hard-codes a local HF cache path and a
10-case allowlist) — the reusable classification code is ostk/crest_height.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

from ostk.io import load_label  # noqa: E402
from ostk.crest_height import crest_height_from_label, load_label_ids  # noqa: E402

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
CASE_IDS = ["0441", "0049", "0231", "0680", "0508", "0368", "0184", "0910", "0655", "1058"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "crest_height_10case.jsonl")


def verify_revision() -> None:
    assert os.path.isdir(SNAPSHOT), f"pinned snapshot missing on disk: {SNAPSHOT}"
    remote = subprocess.run(
        ["git", "ls-remote", "https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K"],
        capture_output=True, text=True, timeout=30,
    )
    lines = [l for l in remote.stdout.splitlines() if "refs/heads/v4" in l]
    assert lines, "could not read refs/heads/v4 from the remote"
    remote_hash = lines[0].split()[0]
    assert remote_hash == PINNED_REVISION, (
        f"pinned revision {PINNED_REVISION} does not match remote v4 HEAD {remote_hash}"
    )
    print(f"revision verified: local snapshot dir + remote refs/heads/v4 both = {PINNED_REVISION}")


def main() -> None:
    verify_revision()
    label_ids = load_label_ids(os.path.join(SNAPSHOT, "dataset_labels.json"))

    records = []
    for case_id in CASE_IDS:
        label_path = os.path.join(SNAPSHOT, "labels", f"{case_id}_label.nii.gz")
        lab, aff = load_label(label_path)
        rec = crest_height_from_label(lab, aff, label_ids, case_id=case_id)
        records.append(rec)
        print(f"{case_id}: R={rec['right']['level']}({rec['right']['height_mm']}mm) "
              f"L={rec['left']['level']}({rec['left']['height_mm']}mm) "
              f"obliquity={rec['obliquity_mm']}mm  flags={rec['qc_flags']}")

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(records)} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
