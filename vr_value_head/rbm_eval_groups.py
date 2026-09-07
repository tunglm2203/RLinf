"""The 3 standard robometer eval groups, sourced from robometer's dataset_category.py (read-only).

  RBM-EVAL-ID   = DATASET_MAP["rbm-1m-id"]["eval"]    (in-distribution held-out: oxe/racer/mw/libero)
  RBM-EVAL-OOD  = DATASET_MAP["rbm-1m-ood"]["eval"]   (unseen robots: trossen/mit_franka/so101/koch/xarm)
  LIBERO-mixed  = DATASET_MAP["libero_pi0"]["eval"]   (libero success + failure combined -> quality metrics)

We import the lists by FILE PATH (not `import robometer…`) so we don't trigger robometer's package
init / heavy deps — dataset_category.py only needs `typing`. Nested eval entries (paired success+
failure) are flattened + de-duplicated. Add more groups later by adding keys here.
"""
import importlib.util
from pathlib import Path

_DC = Path("/mnt/workspace/vla_bench/robometer/robometer/data/dataset_category.py")
_spec = importlib.util.spec_from_file_location("robometer_dataset_category", _DC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DATASET_MAP = _mod.DATASET_MAP


def _flatten(entries):
    out = []
    for e in entries:
        out.extend(e) if isinstance(e, list) else out.append(e)
    return list(dict.fromkeys(out))                       # de-dup, preserve order


GROUPS = {
    "RBM-EVAL-ID": _flatten(DATASET_MAP["rbm-1m-id"]["eval"]),
    "RBM-EVAL-OOD": _flatten(DATASET_MAP["rbm-1m-ood"]["eval"]),
    "LIBERO-mixed": _flatten(DATASET_MAP["libero_pi0"]["eval"]),
}
ALL_DATASETS = list(dict.fromkeys(d for g in GROUPS.values() for d in g))


if __name__ == "__main__":
    for g, ds in GROUPS.items():
        print(f"\n{g}  ({len(ds)} datasets):")
        for d in ds:
            print(f"  {d}")
    print(f"\nunion of all groups: {len(ALL_DATASETS)} unique datasets")
