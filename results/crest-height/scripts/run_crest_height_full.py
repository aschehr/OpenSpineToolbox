"""Full-cohort crest-height run (all 802 cases in the pinned v4 snapshot).

Checkpointed: writes each case's result to OUT_PATH as soon as it completes
(flushed immediately), and on startup skips any case_id already present in
that file. Re-running this script after an interruption resumes rather than
restarting -- it does not recompute cases already on disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "crest_height_full802.jsonl")
WORKERS = 6


def _worker(case_id: str) -> dict:
    sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")
    from ostk.io import load_label
    from ostk.crest_height import crest_height_from_label, load_label_ids
    label_ids = load_label_ids(os.path.join(SNAPSHOT, "dataset_labels.json"))
    label_path = os.path.join(SNAPSHOT, "labels", f"{case_id}_label.nii.gz")
    lab, aff = load_label(label_path)
    return crest_height_from_label(lab, aff, label_ids, case_id=case_id)


def main() -> None:
    labels_dir = os.path.join(SNAPSHOT, "labels")
    all_ids = sorted(
        f.split("_")[0] for f in os.listdir(labels_dir) if f.endswith("_label.nii.gz")
    )

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["case_id"])

    todo = [c for c in all_ids if c not in done]
    print(f"CHECKPOINT: {len(all_ids)} total cases, {len(done)} already done in "
          f"{OUT_PATH}, {len(todo)} remaining", flush=True)
    if not todo:
        print("nothing to do -- already complete", flush=True)
        return

    t0 = time.time()
    n_done = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out_fh:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(_worker, c): c for c in todo}
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001 -- never drop a case silently
                    rec = {"case_id": cid, "parameter": "crest_height", "value": None,
                           "qc_flags": [f"exception:{type(e).__name__}"], "error": str(e)}
                out_fh.write(json.dumps(rec) + "\n")
                out_fh.flush()
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else float("nan")
                    print(f"PROGRESS: {n_done}/{len(todo)} this run "
                          f"({len(done) + n_done}/{len(all_ids)} total)  "
                          f"elapsed={elapsed/60:.1f}min  eta={eta_min:.1f}min", flush=True)

    print(f"DONE: wrote {n_done} new records -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
