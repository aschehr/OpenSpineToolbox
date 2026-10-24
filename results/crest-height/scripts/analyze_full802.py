"""Analysis over the real full-cohort output file: level distribution,
comparison against the 10-case validated subset, and identification of
missing-L5 cases for the LSTV sub-analysis. Reads only what the pipeline
already wrote to disk -- no recomputation."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
FULL_PATH = os.path.join(HERE, "crest_height_full802.jsonl")
TEN_CASE_IDS = ["0441", "0049", "0231", "0680", "0508", "0368", "0184", "0910", "0655", "1058"]

recs = []
with open(FULL_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print(f"loaded {len(recs)} records from {FULL_PATH}")

# --- QC flag breakdown --------------------------------------------------
flag_counts = {}
for r in recs:
    for f in r.get("qc_flags", []):
        flag_counts[f] = flag_counts.get(f, 0) + 1
print("\n=== qc_flags counts across all 802 ===")
for f, c in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {f}: {c}")

# --- level distribution (only clean cases: qc_flags == ['ok']) ----------
clean = [r for r in recs if r.get("qc_flags") == ["ok"]]
print(f"\nclean cases (qc_flags exactly ['ok']): {len(clean)} / {len(recs)}")

level_counts = {"right": {}, "left": {}}
for r in clean:
    for side in ("right", "left"):
        lv = r[side]["level"]
        level_counts[side][lv] = level_counts[side].get(lv, 0) + 1

print("\n=== level distribution, RIGHT side (clean cases only) ===")
n_clean = len(clean)
for lv, c in sorted(level_counts["right"].items(), key=lambda kv: -kv[1]):
    print(f"  {lv}: {c}  ({100*c/n_clean:.1f}%)")

print("\n=== level distribution, LEFT side (clean cases only) ===")
for lv, c in sorted(level_counts["left"].items(), key=lambda kv: -kv[1]):
    print(f"  {lv}: {c}  ({100*c/n_clean:.1f}%)")

# --- combined (both sides pooled, matches the R+L pooled framing used for
#     the 10-case kappa validation) -----------------------------------------
pooled = {}
for r in clean:
    for side in ("right", "left"):
        lv = r[side]["level"]
        pooled[lv] = pooled.get(lv, 0) + 1
n_pooled = sum(pooled.values())
print(f"\n=== pooled R+L level distribution (n={n_pooled} side-observations from {n_clean} cases) ===")
for lv, c in sorted(pooled.items(), key=lambda kv: -kv[1]):
    print(f"  {lv}: {c}  ({100*c/n_pooled:.1f}%)")

# --- cross-check against the 10-case validated subset --------------------
by_id = {r["case_id"]: r for r in recs}
print("\n=== 10-case validated subset, re-read from the SAME full-cohort file ===")
sub_pooled = {}
for cid in TEN_CASE_IDS:
    r = by_id[cid]
    for side in ("right", "left"):
        lv = r[side]["level"]
        sub_pooled[lv] = sub_pooled.get(lv, 0) + 1
        print(f"  {cid} {side}: {lv}")
print("subset pooled counts:", sub_pooled)

# --- LSTV sub-analysis input: cases missing L5 ----------------------------
missing_l5 = [r for r in recs if "missing_label:L5" in r.get("qc_flags", [])]
print(f"\n=== cases with missing_label:L5 flag: {len(missing_l5)} ===")
for r in missing_l5:
    print(f"  {r['case_id']}: qc_flags={r['qc_flags']}")

missing_l4 = [r for r in recs if "missing_label:L4" in r.get("qc_flags", [])]
print(f"\n=== cases with missing_label:L4 flag: {len(missing_l4)} ===")
for r in missing_l4:
    print(f"  {r['case_id']}: qc_flags={r['qc_flags']}")

exc = [r for r in recs if any(f.startswith("exception:") for f in r.get("qc_flags", []))]
print(f"\n=== cases that raised an exception: {len(exc)} ===")
for r in exc:
    print(f"  {r['case_id']}: {r.get('qc_flags')}  error={r.get('error')}")
