"""Phase 1 of the full-dataset LLIF run: real llif_level_report_from_label +
adjacent_disc_height_ratio_from_label output over ALL 802 cases in the
pinned v4 snapshot (label-only, no CT -- see run_vertebral_hu_full802_streaming.py
for the CT-dependent HU/BMD parameters, run separately/afterward to avoid
compounding memory+disk pressure on an 8GB machine).

Same pattern as run_llif_cohort_calibration.py (its 100-case predecessor),
just the full case list. WORKERS=3 kept from that script on purpose: 6
workers caused near-instant kills previously, attributed to memory
pressure -- confirmed again here (only ~60MB free RAM observed before this
run started, on an 8GB machine). Measured real timing before starting this
run: ~330s/case average (0184=310.6s, 0910=290.8s, 1058=400.8s) -- at
WORKERS=3 that's roughly 802*330/3 =~ 88,000s (~24.5 hours) for the full
run. Checkpointed (resume from whatever's already in the output file) since
a job this long WILL be interrupted by memory-pressure kills; intended to
be run under run_full802_pipeline.sh's restart-on-exit loop, not invoked
bare for the full run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "llif_report_full802.jsonl")
WORKERS = 3


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


def _worker(case_id: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")
    from ostk.io import load_label
    from ostk.crest_height import load_label_ids
    from ostk.disc_height import adjacent_disc_height_ratio_from_label
    from ostk.llif import llif_level_report_from_label

    label_ids = load_label_ids(os.path.join(SNAPSHOT, "dataset_labels.json"))
    lab, aff = load_label(os.path.join(SNAPSHOT, "labels", f"{case_id}_label.nii.gz"))
    report = llif_level_report_from_label(lab, aff, "L4", "L5", label_ids, case_id=case_id)
    report["adjacent_disc_height_ratio"] = adjacent_disc_height_ratio_from_label(
        lab, aff, "L4", "L5", label_ids, case_id=case_id)
    return report


def main() -> None:
    verify_revision()
    labels_dir = os.path.join(SNAPSHOT, "labels")
    all_ids = sorted(f.split("_")[0] for f in os.listdir(labels_dir) if f.endswith("_label.nii.gz"))

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["case_id"])
    todo = [c for c in all_ids if c not in done]
    print(f"CHECKPOINT: {len(all_ids)} total, {len(done)} already done, {len(todo)} remaining",
         flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    n_done = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out_fh:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001 -- never drop a case silently
                    rec = {"case_id": cid, "error": str(e), "qc_flags": [f"exception:{type(e).__name__}"]}
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                n_done += 1
                print(f"PROGRESS: {n_done}/{len(todo)} this run ({len(done)+n_done}/{len(all_ids)} total)",
                     flush=True)

    print(f"DONE -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
