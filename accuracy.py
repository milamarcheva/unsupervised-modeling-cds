#!/usr/bin/env python3
import os
import sys
import argparse
from collections import defaultdict

def read_aligned(system_file, gold_file):
    with open(system_file, "r") as f:
        s_lines = [ln.strip() for ln in f if ln.strip()]
    with open(gold_file, "r") as f:
        g_lines = [ln.strip() for ln in f if ln.strip()]
    if len(s_lines) != len(g_lines):
        print(f"Warning: {len(s_lines)} system lines vs {len(g_lines)} gold lines. Truncating to shortest.", file=sys.stderr)
    n = min(len(s_lines), len(g_lines))
    s_lines = s_lines[:n]
    g_lines = g_lines[:n]
    for i, (s, g) in enumerate(zip(s_lines, g_lines), start=1):
        st, gt = s.split(), g.split()
        if len(st) != len(gt):
            raise ValueError(f"Numbers of tokens don't match at line {i}")
    return s_lines, g_lines

def counts(s_lines, g_lines):
    A = defaultdict(lambda: defaultdict(int))  # co-occurrence counts: sys -> gold -> count
    S_tags, G_tags = set(), set()
    for s, g in zip(s_lines, g_lines):
        for st, gt in zip(s.split(), g.split()):
            A[st][gt] += 1
            S_tags.add(st); G_tags.add(gt)
    return A, sorted(S_tags), sorted(G_tags)

def learn_many_to_one(A, s_list):
    M = {}
    for s in s_list:
        if A[s]:
            M[s] = max(A[s], key=A[s].get)
        else:
            M[s] = "UNK"
    return M

def learn_one_to_one(A, s_list, g_list):
    # Build (possibly rectangular) matrix of matches
    import numpy as np
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise RuntimeError(
            "One-to-one mode requires SciPy. Install it with: pip install scipy"
        ) from e

    ns, ng = len(s_list), len(g_list)
    mat = np.zeros((ns, ng), dtype=int)
    for i, s in enumerate(s_list):
        for j, g in enumerate(g_list):
            mat[i, j] = A[s][g]

    # Pad to square so every system tag is assigned (extra columns = dummy golds)
    if ns == ng:
        cost = -mat
        row_ind, col_ind = linear_sum_assignment(cost)
    elif ns < ng:
        cost = -mat  # each system row gets a unique real gold col
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # ns > ng: pad with zero columns (dummy golds)
        pad = np.zeros((ns, ns - ng), dtype=int)
        mat_sq = np.concatenate([mat, pad], axis=1)
        cost = -mat_sq
        row_ind, col_ind = linear_sum_assignment(cost)

    M = {}
    for i, j in zip(row_ind, col_ind):
        s = s_list[i]
        if j < ng:
            M[s] = g_list[j]
        else:
            # assigned to a dummy column → no real gold available
            M[s] = "UNK"
    # Any unassigned s (shouldn't happen) → UNK
    for s in s_list:
        M.setdefault(s, "UNK")
    return M

def evaluate_and_write(s_lines, g_lines, M, out_path, tokens_lines=None):
    total = correct = 0
    with open(out_path, "w") as out:
        for idx, (s, g) in enumerate(zip(s_lines, g_lines)):
            stags = s.split()   # system tags (numeric or cluster IDs)
            gtags = g.split()   # gold tags
            mapped = [M.get(t, "UNK") for t in stags]  # predicted tags after mapping

            # Decide what "token" is: either real tokens (if provided) or system tags
            if tokens_lines is not None:
                toks = tokens_lines[idx].split()
                if len(toks) != len(gtags):
                    raise ValueError(f"Token count mismatch at line {idx+1} between tokens-file and gold-file")
                tokens_for_output = toks
            else:
                tokens_for_output = stags  # fall back to system tags as "tokens"

            # Build: token1 pred1 gold1 token2 pred2 gold2 ...
            parts = []
            for tok, pred_tag, gold_tag in zip(tokens_for_output, mapped, gtags):
                parts.extend([tok, pred_tag, gold_tag])
                total += 1
                if pred_tag == gold_tag:
                    correct += 1

            out.write(" ".join(parts) + "\n")

    acc = (correct / total) if total else 0.0
    return acc, correct, total

def compute_prf(s_lines, g_lines, mapping):
    # collect predictions paired with gold
    pairs = []
    for s, g in zip(s_lines, g_lines):
        for st, gt in zip(s.split(), g.split()):
            pred = mapping.get(st, "UNK")
            pairs.append((pred, gt))

    gold_tags = sorted({gt for _, gt in pairs})
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for pred, gold in pairs:
        if pred == gold:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    per_tag = {}
    sum_p = sum_r = sum_f1 = 0.0
    for tag in gold_tags:
        p = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0.0
        r = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0.0
        f1 = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
        per_tag[tag] = (p, r, f1)
        sum_p += p; sum_r += r; sum_f1 += f1

    n = len(gold_tags) if gold_tags else 1
    return per_tag, (sum_p/n, sum_r/n, sum_f1/n)

def main():
    p = argparse.ArgumentParser(description="Evaluate tagging with many-to-one or one-to-one mapping and write mapped predictions.")
    p.add_argument("system_file", help="System output (numeric tags), one sentence per line")
    p.add_argument("gold_file", help="Gold tags, one sentence per line")
    p.add_argument("--mode", choices=["many2one", "one2one"], default="many2one",
                   help="Mapping mode (default: many2one)")
    p.add_argument("--output-filename", default="test.current.test.pred.test_all_postags",
                   help="Output filename (default matches original script)")
    p.add_argument("--prf", action="store_true",
                   help="Also print per-tag precision/recall/F1 and macro averages")
    p.add_argument("--tokens-file", help="Optional: test sentences (tokens), one sentence per line, aligned with gold_file")
    args = p.parse_args()

    s_lines, g_lines = read_aligned(args.system_file, args.gold_file)
    tokens_lines = None
    if args.tokens_file:
        with open(args.tokens_file, "r") as f:
            tokens_lines = [ln.strip() for ln in f if ln.strip()]
        # Align number of lines with gold/system
        if len(tokens_lines) != len(g_lines):
            print(f"Warning: {len(tokens_lines)} token lines vs {len(g_lines)} gold lines. Truncating to shortest.", file=sys.stderr)
            n = min(len(tokens_lines), len(g_lines))
            tokens_lines = tokens_lines[:n]
            s_lines = s_lines[:n]
            g_lines = g_lines[:n]
        # Sanity check: same #tokens per sentence
        for i, (tln, gln) in enumerate(zip(tokens_lines, g_lines), start=1):
            toks = tln.split()
            gtags = gln.split()
            if len(toks) != len(gtags):
                raise ValueError(f"Numbers of tokens don't match at line {i} between tokens-file and gold-file")

    A, s_list, g_list = counts(s_lines, g_lines)

    if args.mode == "many2one":
        M = learn_many_to_one(A, s_list)
    else:
        M = learn_one_to_one(A, s_list, g_list)

    out_dir = os.path.dirname(os.path.abspath(args.system_file))
    # out_path = os.path.join(out_dir, args.output_filename)
    base, ext = os.path.splitext(args.output_filename)
    out_filename = f"{base}_{args.mode}{ext}"
    out_path = os.path.join(out_dir, out_filename)

    acc, correct, total = evaluate_and_write(s_lines, g_lines, M, out_path, tokens_lines=tokens_lines)

    print(f"Mode: {args.mode}")
    print(f"ACC: {acc:.2f} ({correct}/{total})")
    print("Mapping (system → gold):")
    for s in sorted(M):
        print(f"  {s} → {M[s]}")
    print(f"Predicted tags written to: {out_path}")

    if args.prf:
        per_tag, (macro_p, macro_r, macro_f1) = compute_prf(s_lines, g_lines, M)
        print("\nPer-gold-tag Precision / Recall / F1")
        print(f"{'Tag':<10} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        for tag in sorted(per_tag):
            p_, r_, f_ = per_tag[tag]
            print(f"{tag:<10} {p_*100:8.2f} {r_*100:8.2f} {f_*100:8.2f}")
        # These three lines are easy for other scripts to parse:
        print(f"\nMacro Precision: {macro_p*100:.2f}%")
        print(f"Macro Recall:    {macro_r*100:.2f}%")
        print(f"Macro F1:        {macro_f1*100:.2f}%")

if __name__ == "__main__":
    main()
