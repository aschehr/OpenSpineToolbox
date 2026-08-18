"""Runner: full LLIF L4-L5 report over the same 10 validated cases, pinned
to the same verified v4 revision. Checkpointed (flush + resume-on-restart)
since this is heavier per-case work than any earlier single-parameter run
(each case does multiple femoral-head fits + multiple body-isolation
erosions)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

from ostk.io import load_label  # noqa: E402
from ostk.crest_height import load_label_ids  # noqa: E402
from ostk.llif import llif_level_report_from_label  # noqa: E402

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
CASE_IDS = ["0441", "0049", "0231", "0680", "0508", "0368", "0184", "0910", "0655", "1058"]
OUT_PATH = os.path.join(os.path.dirname(__file__), "llif_report_10case.jsonl")


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
    print(f"revision verified: {PINNED_REVISION}", flush=True)


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
            label_path = os.path.join(SNAPSHOT, "labels", f"{case_id}_label.nii.gz")
            lab, aff = load_label(label_path)
            rec = llif_level_report_from_label(lab, aff, "L4", "L5", label_ids, case_id=case_id)
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            w = rec["vertebral_wedging"]
            print(f"{case_id}: listhesis={rec['lateral_listhesis']['value']}mm  "
                 f"coronal_angle={rec['coronal_disc_angle']['value']}deg  "
                 f"seg_LL={rec['segmental_lordosis']['value']}deg  "
                 f"wedge_L4={w['L4']['wedge_ratio']}  wedge_L5={w['L5']['wedge_ratio']}", flush=True)

    print(f"\nDONE -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
