#!/usr/bin/env python3
import re
import sys
import argparse
import subprocess
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict, OrderedDict

SYSTEM_FILENAME = "test.current.test.pred.test_all"

# Parse lines like: "ACC: 0.79 (99/126)"
ACC_PAIR_RE = re.compile(r"\((\d+)\s*/\s*(\d+)\)")

# Header that precedes the per-tag table in accuracy.py
PRF_HEADER_RE = re.compile(r"^\s*Per-gold-tag Precision\s*/\s*Recall\s*/\s*F1", re.I)

# Parse per-tag table rows like: "ASP   1.76  2.02  1.88"
TAG_ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_\-]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*$"
)

# Macro extraction
MACRO_P_RE = re.compile(r"Macro\s+Precision:\s*([\d.]+)%")
MACRO_R_RE = re.compile(r"Macro\s+Recall:\s*([\d.]+)%")
MACRO_F1_RE = re.compile(r"Macro\s+F1:\s*([\d.]+)%")

# Mapping line: "  2 → AUX"
MAPPING_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*→\s*([A-Za-z0-9_\-]+)\s*$")


# ----------------------------------------------------------------------
# Run accuracy.py and extract:
# - accuracy
# - per-tag precision/recall/F1
# - mapping (system→gold)
# ----------------------------------------------------------------------
def run_accuracy(py_path: Path, system_file: Path, gold_file: Path, mode: str):
    cmd = [
        sys.executable, str(py_path),
        str(system_file), str(gold_file),
        "--mode", mode, "--prf"
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"accuracy.py failed ({mode})\n"
            f"STDERR:\n{proc.stderr}\n"
            f"STDOUT:\n{proc.stdout}\n"
        )

    stdout = proc.stdout
    lines = stdout.splitlines()

    # --- Accuracy ---
    m = ACC_PAIR_RE.search(stdout)
    if not m:
        raise ValueError("Cannot parse accuracy from accuracy.py output.")
    correct, total = int(m.group(1)), int(m.group(2))
    acc = correct / total if total else 0.0

    # --- Per-tag PRF ---
    per_tag = {}
    start_idx = None
    for i, line in enumerate(lines):
        if PRF_HEADER_RE.search(line):
            start_idx = i + 1
            break

    if start_idx is not None:
        if lines[start_idx].strip().lower().startswith("tag"):
            start_idx += 1
        i = start_idx
        while i < len(lines):
            row = lines[i].strip()
            if not row:
                break
            mrow = TAG_ROW_RE.match(row)
            if not mrow:
                break
            tag = mrow.group(1)
            prec = float(mrow.group(2))
            rec = float(mrow.group(3))
            f1 = float(mrow.group(4))
            per_tag[tag] = (prec, rec, f1)
            i += 1

    # --- Macros ---
    mp = MACRO_P_RE.search(stdout)
    mr = MACRO_R_RE.search(stdout)
    mf = MACRO_F1_RE.search(stdout)
    macros = None
    if mp and mr and mf:
        macros = (
            float(mp.group(1)),
            float(mr.group(1)),
            float(mf.group(1))
        )

    # --- Mapping ---
    mapping = {}
    for line in lines:
        mline = MAPPING_RE.match(line)
        if mline:
            src = mline.group(1)
            tgt = mline.group(2)
            mapping[src] = tgt

    return acc, per_tag, macros, mapping


# ----------------------------------------------------------------------
def find_experiments(root: Path):
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / SYSTEM_FILENAME).is_file():
            yield sub


# ----------------------------------------------------------------------
def average_per_tag(list_of_dicts, preferred_order=None):
    acc_p = defaultdict(list)
    acc_r = defaultdict(list)
    acc_f = defaultdict(list)

    for d in list_of_dicts:
        for tag, (p, r, f) in d.items():
            acc_p[tag].append(p)
            acc_r[tag].append(r)
            acc_f[tag].append(f)

    tags = sorted(set(acc_p) | set(acc_r) | set(acc_f))
    if preferred_order:
        order = [t for t in preferred_order if t in tags]
        order += [t for t in tags if t not in order]
    else:
        order = tags

    out = OrderedDict()
    for tag in order:
        mp = mean(acc_p[tag]) if acc_p[tag] else 0.0
        mr = mean(acc_r[tag]) if acc_r[tag] else 0.0
        mf = mean(acc_f[tag]) if acc_f[tag] else 0.0
        out[tag] = (mp, mr, mf)
    return out


# ----------------------------------------------------------------------
# NEW: Compute majority label for each cluster ID
# ----------------------------------------------------------------------
def compute_src_majority(mapping_counts):
    """
    Given mapping_counts[src][tgt] = occurrences,
    return src_majority[src] = the gold tag most frequently mapped to.
    """
    out = {}
    for src, tgt_dict in mapping_counts.items():
        if tgt_dict:
            best_tgt = max(tgt_dict.items(), key=lambda x: x[1])[0]
            out[src] = best_tgt
    return out


# ----------------------------------------------------------------------
# NEW: Aggregate mapping by majority label → gold label
# ----------------------------------------------------------------------
def aggregate_by_src_label(mapping_counts, src_majority):
    agg = defaultdict(lambda: defaultdict(int))
    for src, tgt_dict in mapping_counts.items():
        lbl = src_majority.get(src)
        if lbl is None:
            continue
        for tgt, cnt in tgt_dict.items():
            agg[lbl][tgt] += cnt
    return agg


# ----------------------------------------------------------------------
# NEW: Pretty print mappings (numeric + label)
# ----------------------------------------------------------------------
def print_mapping_counts(title, mapping_counts, src_majority=None):
    print(f"\n{title}")
    header = f"{'SrcID':<8}"
    if src_majority:
        header += f"{'SrcLbl':<8}"
    header += f"{'Tgt':<8} {'Count':>8}"
    print(header)

    rows = []
    for src, tgt_dict in mapping_counts.items():
        for tgt, count in tgt_dict.items():
            rows.append((src, tgt, count))
    rows.sort(key=lambda x: (x[0], -x[2], x[1]))

    for src, tgt, count in rows:
        line = f"{src:<8}"
        if src_majority:
            line += f"{src_majority.get(src,''):<8}"
        line += f"{tgt:<8}{count:8d}"
        print(line)


# ----------------------------------------------------------------------
def print_label_level_counts(title, agg_counts):
    print(f"\n{title}")
    print(f"{'SrcLbl':<8} {'Tgt':<8} {'Count':>8}")

    rows = []
    for src_lbl, tgt_dict in agg_counts.items():
        for tgt, cnt in tgt_dict.items():
            rows.append((src_lbl, tgt, cnt))

    rows.sort(key=lambda x: (x[0], -x[2], x[1]))

    for src_lbl, tgt, cnt in rows:
        print(f"{src_lbl:<8} {tgt:<8} {cnt:8d}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments_root")
    ap.add_argument("--gold", default="/Users/milamarcheva/Downloads/all_tags.txt")
    ap.add_argument("--accuracy-script",
                    help="Path to accuracy.py (default: alongside this script)")
    ap.add_argument("--tag-order",
                    default="ASP,AUX,C,DET,DIV,N,P,T,V,X")
    args = ap.parse_args()

    root = Path(args.experiments_root)
    gold = Path(args.gold)

    acc_py = Path(args.accuracy_script) if args.accuracy_script else \
             (Path(__file__).resolve().parent / "accuracy.py")

    preferred_order = [t.strip() for t in args.tag_order.split(",")]

    # Results
    acc_m21, acc_o2o = [], []
    prf_m21_list, prf_o2o_list = [], []

    # NEW: mapping counts
    mapping_counts_m21 = defaultdict(lambda: defaultdict(int))
    mapping_counts_o2o = defaultdict(lambda: defaultdict(int))

    # Process experiments
    for exp_dir in find_experiments(root):
        sys_file = exp_dir / SYSTEM_FILENAME

        try:
            a21, p21, _, map21 = run_accuracy(acc_py, sys_file, gold, "many2one")
            aoo, poo, _, mapoo = run_accuracy(acc_py, sys_file, gold, "one2one")
        except Exception as e:
            print(f"[WARN] Skipping {exp_dir.name}: {e}")
            continue

        acc_m21.append(a21)
        acc_o2o.append(aoo)
        prf_m21_list.append(p21)
        prf_o2o_list.append(poo)

        for src, tgt in map21.items():
            mapping_counts_m21[src][tgt] += 1
        for src, tgt in mapoo.items():
            mapping_counts_o2o[src][tgt] += 1

    # ------------------------------------------------------------------
    # Print accuracy summary
    # ------------------------------------------------------------------
    print("\n=== Overall Summary ===")
    print(f"Average many-to-one accuracy: {mean(acc_m21):.3f} (SD={stdev(acc_m21) if len(acc_m21)>1 else 0:.3f})")
    print(f"Average one-to-one accuracy:  {mean(acc_o2o):.3f} (SD={stdev(acc_o2o) if len(acc_o2o)>1 else 0:.3f})")

    # ------------------------------------------------------------------
    # Print PRF summary
    # ------------------------------------------------------------------
    avg_prf_m21 = average_per_tag(prf_m21_list, preferred_order)
    avg_prf_o2o = average_per_tag(prf_o2o_list, preferred_order)

    print("\nPer-gold-tag Precision/Recall/F1 — MANY2ONE")
    print(f"{'Tag':<6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for tag, (p, r, f) in avg_prf_m21.items():
        print(f"{tag:<6} {p:8.2f} {r:8.2f} {f:8.2f}")

    print("\nPer-gold-tag Precision/Recall/F1 — ONE2ONE")
    print(f"{'Tag':<6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for tag, (p, r, f) in avg_prf_o2o.items():
        print(f"{tag:<6} {p:8.2f} {r:8.2f} {f:8.2f}")

    # ------------------------------------------------------------------
    # NEW: Mapping Statistics
    # ------------------------------------------------------------------
    # Majority label per cluster ID
    src_majority_m21 = compute_src_majority(mapping_counts_m21)
    src_majority_o2o = compute_src_majority(mapping_counts_o2o)

    # Numeric → gold mapping tables
    print_mapping_counts("Mapping frequencies — MANY2ONE", mapping_counts_m21, src_majority_m21)
    print_mapping_counts("Mapping frequencies — ONE2ONE",  mapping_counts_o2o, src_majority_o2o)

    # Collapse to label-level mapping (ASP→ASP, DET→AUX, etc.)
    agg_m21 = aggregate_by_src_label(mapping_counts_m21, src_majority_m21)
    agg_o2o = aggregate_by_src_label(mapping_counts_o2o, src_majority_o2o)

    print_label_level_counts(
        "Label-level mappings — MANY2ONE (SrcLbl → Tgt)",
        agg_m21
    )
    print_label_level_counts(
        "Label-level mappings — ONE2ONE (SrcLbl → Tgt)",
        agg_o2o
    )


if __name__ == "__main__":
    main()
