"""Phase 2 of the full-dataset LLIF run: vertebral_hu (L4, L5) + the
Zhou-et-al.-protocol-matched zhou_l1_l4_bmd, over ALL 802 cases -- the
two parameters needing the CT intensity volume, not just labels.

Disk-constrained by design: only 14GB free vs. ~194GB (802 * ~242MB) if
every case's CT were kept at once. So this downloads ONE case's CT (via
huggingface_hub, same pinned revision as everything else), computes both
parameters, deletes that CT file's blob immediately, and moves to the
next case -- disk usage stays roughly flat throughout instead of growing
without bound. Serial by design (not parallelized): parallel downloads
would make disk usage unpredictable, which defeats the point.

Run AFTER run_llif_report_full802.py finishes (see run_full802_pipeline.sh)
-- not concurrently -- to avoid compounding memory+disk pressure on an
8GB machine that already showed instability at higher parallelism.

Checkpointed (resume from whatever's already in the output file): given
the per-case cost here is a ~242MB download PLUS ~35s of computation,
resuming without re-downloading/re-computing already-done cases matters
even more than in the label-only run."""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, "/Users/ashleyschehr/OpenSpineToolbox")

PINNED_REVISION = "9f69480a572624029a927bc6661309a068f1f2c3"
REPO_ID = "anonymous-mlhc/CTSpinoPelvic1K"
SNAPSHOT = (
    "/Users/ashleyschehr/.cache/huggingface/hub/"
    "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/" + PINNED_REVISION
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vertebral_hu_full802.jsonl")
MIN_FREE_GB = 5.0  # abort rather than risk filling the disk


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


def _free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def _download_ct(case_id: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=PINNED_REVISION,
                           filename=f"ct/{case_id}_ct.nii.gz")


def _delete_ct(local_path: str) -> None:
    """Remove both the snapshot symlink AND the blob it points to, so disk
    is actually reclaimed (a bare os.remove on the symlink alone would
    leave the ~242MB blob behind in the hub cache's blobs/ dir)."""
    real = os.path.realpath(local_path)
    for p in {local_path, real}:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def main() -> None:
    verify_revision()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from ostk.io import load_label, load_ct
    from ostk.crest_height import load_label_ids
    from ostk.endplate_footprint import isolate_vertebral_body
    from ostk.masks import binary_mask, largest_component
    from ostk.vertebral_hu import vertebral_hu_from_label
    from ostk.vertebral_hu_zhou import ZHOU_LEVELS, zhou_l1_l4_bmd_from_label

    label_ids = load_label_ids(os.path.join(SNAPSHOT, "dataset_labels.json"))
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
    if len(sys.argv) > 1:
        todo = [c for c in todo if c == sys.argv[1]]
    print(f"CHECKPOINT: {len(all_ids)} total, {len(done)} already done, {len(todo)} remaining",
         flush=True)

    needed_levels = set(ZHOU_LEVELS) | {"L5"}  # L1-L4 for zhou, L4+L5 for vertebral_hu

    with open(OUT_PATH, "a", encoding="utf-8") as out_fh:
        for i, case_id in enumerate(todo, 1):
            free_gb = _free_gb(SNAPSHOT)
            if free_gb < MIN_FREE_GB:
                print(f"ABORT: only {free_gb:.1f}GB free (< {MIN_FREE_GB}GB floor) "
                     f"before case {case_id} -- stopping rather than risk filling the disk",
                     flush=True)
                sys.exit(1)

            try:
                ct_path = _download_ct(case_id)
                lab, aff = load_label(os.path.join(labels_dir, f"{case_id}_label.nii.gz"))
                ct, ct_aff = load_ct(ct_path)
                assert (ct_aff == aff).all(), "CT/label affine mismatch"

                body_masks = {}
                for lv in needed_levels:
                    m = largest_component(binary_mask(lab, label_ids[lv])) if lv in label_ids else None
                    body_masks[lv] = (isolate_vertebral_body(m, aff, erosion_mm=10.0)
                                      if m is not None and m.any() else None)

                rec = {
                    "case_id": case_id,
                    "vertebral_hu": {
                        lv: vertebral_hu_from_label(lab, ct, aff, lv, label_ids, case_id=case_id,
                                                    body_mask=body_masks.get(lv))
                        for lv in ("L4", "L5")
                    },
                    "zhou_l1_l4_bmd": zhou_l1_l4_bmd_from_label(
                        lab, ct, aff, label_ids, case_id=case_id,
                        body_masks={lv: body_masks.get(lv) for lv in ZHOU_LEVELS}),
                }
            except Exception as e:  # noqa: BLE001 -- never drop a case silently
                rec = {"case_id": case_id, "error": str(e), "qc_flags": [f"exception:{type(e).__name__}"]}
            finally:
                try:
                    _delete_ct(ct_path)
                except NameError:
                    pass  # download itself failed -- nothing to delete

            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            print(f"PROGRESS: {i}/{len(todo)} this run ({len(done)+i}/{len(all_ids)} total) "
                 f"case={case_id} free_disk={_free_gb(SNAPSHOT):.1f}GB", flush=True)

    print(f"DONE -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
