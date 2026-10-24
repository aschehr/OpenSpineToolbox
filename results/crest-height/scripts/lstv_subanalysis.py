"""LSTV / transitional-anatomy sub-analysis, reading the real full-cohort
output file. Two groups, kept clean of cross-contamination:

  A) missing-L5 cases (n=12): L5 unfittable in the mask -- obliquity only.
  B) the full manifest-flagged LSTV cohort (SACRALIZATION + LUMBARIZATION +
     SEMI_SACRALIZATION, n=33) vs a general-sample group that EXCLUDES any
     LSTV-labeled case (even the ones that still fit successfully) so the
     comparison group is genuinely non-LSTV.
"""
import json
import os
import statistics as stats

from scipy import stats as sstats

HERE = "/private/tmp/claude-501/-Users-ashleyschehr-OpenSpineToolbox/09fd3c32-c4a9-4dc4-ba7f-3bf96385d2ee/scratchpad"
FULL_PATH = os.path.join(HERE, "crest_height_full802.jsonl")
SNAP = ("/Users/ashleyschehr/.cache/huggingface/hub/"
        "datasets--anonymous-mlhc--CTSpinoPelvic1K/snapshots/"
        "9f69480a572624029a927bc6661309a068f1f2c3")

manifest = {c["volume_id"]: c for c in json.load(open(os.path.join(SNAP, "manifest.json")))}
recs = [json.loads(l) for l in open(FULL_PATH) if l.strip()]
by_id = {r["case_id"]: r for r in recs}

LSTV_LABELS = {"SACRALIZATION", "LUMBARIZATION", "SEMI_SACRALIZATION"}
lstv_ids = {cid for cid, m in manifest.items() if m.get("lstv_label") in LSTV_LABELS and cid in by_id}
missing_l5_ids = {r["case_id"] for r in recs if "missing_label:L5" in r.get("qc_flags", [])}

print(f"manifest LSTV-labeled cases in cohort: {len(lstv_ids)} "
      f"(SACRALIZATION/LUMBARIZATION/SEMI_SACRALIZATION)")
print(f"missing_label:L5 cases (mask has no separate L5): {len(missing_l5_ids)}")
print(f"overlap (LSTV-labeled AND missing L5): {len(lstv_ids & missing_l5_ids)}")
print(f"LSTV-labeled but L5 still fit successfully: {sorted(lstv_ids - missing_l5_ids)}")

def obl(ids):
    return [by_id[c]["obliquity_mm"] for c in ids if by_id[c]["obliquity_mm"] is not None]

def report(name, ids):
    vals = obl(ids)
    print(f"\n--- {name}: n={len(vals)} ---")
    print(f"  values: {vals}")
    if vals:
        print(f"  mean={stats.mean(vals):.2f}  sd={(stats.stdev(vals) if len(vals)>1 else 0):.2f}  "
              f"median={stats.median(vals):.2f}  min={min(vals):.2f}  max={max(vals):.2f}")
    return vals

group_a = report("Group A: missing-L5 cases (n=12)", missing_l5_ids)
group_b = report("Group B: all manifest-LSTV-labeled cases (n=33)", lstv_ids)

# clean general/comparison group: qc ok AND not LSTV-labeled
general_ids = {r["case_id"] for r in recs if r.get("qc_flags") == ["ok"]} - lstv_ids
group_general = report(f"General comparison sample (qc ok, non-LSTV, n={len(general_ids)})", general_ids)

print("\n=== statistical comparisons vs the clean general sample ===")
for name, grp in [("missing-L5 (Group A)", group_a), ("all LSTV-labeled (Group B)", group_b)]:
    if len(grp) < 2:
        print(f"{name}: n too small for a t-test")
        continue
    t, p = sstats.ttest_ind(grp, group_general, equal_var=False)
    u, pu = sstats.mannwhitneyu(grp, group_general, alternative="two-sided")
    print(f"{name} vs general: Welch t={t:.3f} p={p:.4f}   Mann-Whitney U={u:.1f} p={pu:.4f}")

out = {
    "group_a_missing_L5": {"n": len(group_a), "case_ids": sorted(missing_l5_ids), "obliquity_mm": group_a},
    "group_b_all_lstv_labeled": {"n": len(group_b), "case_ids": sorted(lstv_ids), "obliquity_mm": group_b},
    "general_non_lstv": {
        "n": len(group_general),
        "mean_obliquity_mm": round(stats.mean(group_general), 3),
        "sd_obliquity_mm": round(stats.stdev(group_general), 3),
        "median_obliquity_mm": round(stats.median(group_general), 3),
    },
}
out_path = os.path.join(HERE, "lstv_subanalysis_result.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote -> {out_path}")
