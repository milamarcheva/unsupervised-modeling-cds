#!/usr/bin/env python3
import argparse, random
from collections import Counter, defaultdict
from statistics import mean, stdev

# ---------- helpers ----------
def read_gold(gold_file):
    with open(gold_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    tags_per_sent = [ln.split() for ln in lines]
    flat = [t for sent in tags_per_sent for t in sent]
    return tags_per_sent, flat

def accuracy(pred_seq, gold_seq):
    correct = sum(p == g for p, g in zip(pred_seq, gold_seq))
    return correct / len(gold_seq) if gold_seq else 0.0

def uniform_random_preds(n_tokens, tagset):
    return [random.choice(tagset) for _ in range(n_tokens)]

def prior_weighted_random_preds(n_tokens, tag_priors):
    # tag_priors: list of (tag, prob)
    tags, probs = zip(*tag_priors)
    # cumulative sample
    import bisect
    cdf = []
    total = 0.0
    for p in probs:
        total += p
        cdf.append(total)
    preds = []
    for _ in range(n_tokens):
        u = random.random()
        i = bisect.bisect_left(cdf, u)
        preds.append(tags[i])
    return preds

def permuted_preds(gold_seq):
    # exact-count shuffle
    preds = gold_seq[:]
    random.shuffle(preds)
    return preds

# ---------- many-to-one & one-to-one mapping ----------
def many_to_one_map(sys_seq, gold_seq):
    # learn system_tag -> gold_tag with max co-occurrence
    co = defaultdict(lambda: defaultdict(int))
    for s, g in zip(sys_seq, gold_seq):
        co[s][g] += 1
    mapping = {}
    for s in co:
        mapping[s] = max(co[s], key=co[s].get)
    return mapping

def apply_map(sys_seq, mapping):
    return [mapping.get(s, "UNK") for s in sys_seq]

def one_to_one_map(sys_seq, gold_seq):
    # Hungarian with pure-python fallback (greedy) to avoid SciPy dependency
    # Build co-occurrence
    co = defaultdict(lambda: defaultdict(int))
    for s, g in zip(sys_seq, gold_seq):
        co[s][g] += 1
    s_list = sorted(co.keys())
    g_list = sorted(set(gold_seq))
    # build matrix
    mat = [[co[s][g] for g in g_list] for s in s_list]
    # Greedy assignment as a simple proxy (not optimal like Hungarian, but OK for baseline)
    assigned_g = set()
    mapping = {}
    # repeatedly choose the (s,g) with current max count
    pairs = []
    for i, s in enumerate(s_list):
        for j, g in enumerate(g_list):
            pairs.append((mat[i][j], s, g))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for cnt, s, g in pairs:
        if s not in mapping and g not in assigned_g:
            mapping[s] = g
            assigned_g.add(g)
    # any leftover sys tags map to some remaining gold (or UNK)
    remaining_g = [g for g in g_list if g not in assigned_g]
    for s in s_list:
        mapping.setdefault(s, remaining_g[0] if remaining_g else "UNK")
    return mapping

# ---------- random clustering baseline ----------
def random_cluster_sys_tags(n_tokens, K):
    # assign each token a random system id in {0..K-1}
    return [f"S{random.randrange(K)}" for _ in range(n_tokens)]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Random baselines for tag accuracy.")
    ap.add_argument("gold_file", help="Gold tag file (one sentence per line)")
    ap.add_argument("--trials", type=int, default=200, help="Monte Carlo trials per random baseline (default: 200)")
    ap.add_argument("--K", type=int, default=10, help="Number of system tags for random clustering (default: 10)")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed (default: 13)")
    args = ap.parse_args()

    # gold_file = "/Users/milamarcheva/Downloads/all_tags.txt"
    random.seed(args.seed)
    sents, flat_gold = read_gold(args.gold_file)
    n_tokens = len(flat_gold)
    tag_counts = Counter(flat_gold)
    tagset = sorted(tag_counts.keys())
    priors = [(t, tag_counts[t] / n_tokens) for t in tagset]

    # Majority baseline
    maj_tag, maj_count = tag_counts.most_common(1)[0]
    maj_preds = [maj_tag] * n_tokens
    acc_majority = accuracy(maj_preds, flat_gold)

    # Uniform random baseline
    acc_uniform = []
    for _ in range(args.trials):
        preds = uniform_random_preds(n_tokens, tagset)
        acc_uniform.append(accuracy(preds, flat_gold))

    # Prior-weighted random baseline
    acc_prior = []
    for _ in range(args.trials):
        preds = prior_weighted_random_preds(n_tokens, priors)
        acc_prior.append(accuracy(preds, flat_gold))

    # Permutation baseline
    acc_perm = []
    for _ in range(args.trials):
        preds = permuted_preds(flat_gold)
        acc_perm.append(accuracy(preds, flat_gold))

    # Random clustering + many-to-one / one-to-one mapping (matches your eval style)
    acc_rand_cluster_m21 = []
    acc_rand_cluster_o2o = []
    for _ in range(args.trials):
        sys_tags = random_cluster_sys_tags(n_tokens, args.K)
        # many-to-one
        m21 = many_to_one_map(sys_tags, flat_gold)
        preds_m21 = apply_map(sys_tags, m21)
        acc_rand_cluster_m21.append(accuracy(preds_m21, flat_gold))
        # one-to-one (greedy)
        o2o = one_to_one_map(sys_tags, flat_gold)
        preds_o2o = apply_map(sys_tags, o2o)
        acc_rand_cluster_o2o.append(accuracy(preds_o2o, flat_gold))

    # Print results
    def fmt(xs):
        return (mean(xs), stdev(xs) if len(xs) > 1 else 0.0)

    mu_u, sd_u = fmt(acc_uniform)
    mu_p, sd_p = fmt(acc_prior)
    mu_perm, sd_perm = fmt(acc_perm)
    mu_rc_m21, sd_rc_m21 = fmt(acc_rand_cluster_m21)
    mu_rc_o2o, sd_rc_o2o = fmt(acc_rand_cluster_o2o)

    print(f"Tokens: {n_tokens}, Sentences: {len(sents)}, Tags: {len(tagset)}")
    print("\n=== Simple baselines ===")
    print(f"Majority-tag accuracy:         {acc_majority:.4f}  (tag='{maj_tag}', freq={maj_count/n_tokens:.3f})")
    print(f"Uniform random accuracy:       {mu_u:.4f}  (SD={sd_u:.4f}, trials={args.trials})")
    print(f"Prior-weighted random acc:     {mu_p:.4f}  (SD={sd_p:.4f}, trials={args.trials})")
    print(f"Permutation (counts preserved):{mu_perm:.4f}  (SD={sd_perm:.4f}, trials={args.trials})")

    print("\n=== Random clustering scored like your system ===")
    print(f"Random clusters + many-to-one: {mu_rc_m21:.4f}  (SD={sd_rc_m21:.4f}, K={args.K}, trials={args.trials})")
    print(f"Random clusters + one-to-one : {mu_rc_o2o:.4f}  (SD={sd_rc_o2o:.4f}, K={args.K}, trials={args.trials})")

if __name__ == "__main__":
    main()
