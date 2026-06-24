#!/usr/bin/env python3
import argparse
import math
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUTS = REPO_ROOT / "outputs" / "bg" / "20260622"
DEFAULT_GOLD = REPO_ROOT / "sample-data" / "bg_categoryinduction_gold" / "all_tags.txt"
NUMERIC_METRICS = [
    "vi",
    "vi_norm",
    "acc_many2one",
    "acc_one2one",
    "train_logZ",
]


def read_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def validate_alignment(system_lines, gold_lines, path: Path):
    if len(system_lines) != len(gold_lines):
        raise ValueError(
            f"{path} has {len(system_lines)} lines, but gold has {len(gold_lines)}"
        )
    for index, (system_line, gold_line) in enumerate(zip(system_lines, gold_lines), start=1):
        if len(system_line.split()) != len(gold_line.split()):
            raise ValueError(
                f"{path} token count mismatch on line {index}: "
                f"{len(system_line.split())} vs {len(gold_line.split())}"
            )


def choose_prediction_file(run_dir: Path, gold_lines):
    explicit_candidates = [
        run_dir / "test.current.test.pred.test_all",
        run_dir / "stage2.current.test.pred.0",
        run_dir / "stage2.test.pred.0",
    ]
    wildcard_candidates = sorted(run_dir.glob("*.current.test.pred.0"))
    wildcard_candidates += sorted(run_dir.glob("*.test.pred.0"))
    wildcard_candidates += sorted(run_dir.glob("*.current.test.pred.test_all"))
    wildcard_candidates += sorted(run_dir.glob("*.test_all"))

    seen = set()
    for candidate in explicit_candidates + wildcard_candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        lines = read_lines(candidate)
        if not lines:
            continue
        try:
            validate_alignment(lines, gold_lines, candidate)
        except ValueError:
            continue
        return candidate, lines
    raise FileNotFoundError(f"No aligned prediction file found in {run_dir}")


def flatten(lines):
    return [token for line in lines for token in line.split()]


def build_counts(system_lines, gold_lines):
    matrix = {}
    system_tags = set()
    gold_tags = set()
    for system_line, gold_line in zip(system_lines, gold_lines):
        for system_tag, gold_tag in zip(system_line.split(), gold_line.split()):
            matrix.setdefault(system_tag, {})
            matrix[system_tag][gold_tag] = matrix[system_tag].get(gold_tag, 0) + 1
            system_tags.add(system_tag)
            gold_tags.add(gold_tag)
    return matrix, sorted(system_tags), sorted(gold_tags)


def learn_many_to_one(matrix, system_tags):
    mapping = {}
    for system_tag in system_tags:
        targets = matrix.get(system_tag, {})
        mapping[system_tag] = max(targets, key=targets.get) if targets else "UNK"
    return mapping


def learn_one_to_one(matrix, system_tags, gold_tags):
    columns = list(gold_tags)
    if len(system_tags) > len(columns):
        columns.extend([None] * (len(system_tags) - len(columns)))

    @lru_cache(maxsize=None)
    def best_score(index, used_mask):
        if index == len(system_tags):
            return 0
        score = -1
        system_tag = system_tags[index]
        for column_index, column_tag in enumerate(columns):
            if used_mask & (1 << column_index):
                continue
            pair_score = matrix.get(system_tag, {}).get(column_tag, 0) if column_tag is not None else 0
            candidate = pair_score + best_score(index + 1, used_mask | (1 << column_index))
            if candidate > score:
                score = candidate
        return score

    mapping = {}
    used_mask = 0
    for index, system_tag in enumerate(system_tags):
        best_column = None
        best_total = -1
        for column_index, column_tag in enumerate(columns):
            if used_mask & (1 << column_index):
                continue
            pair_score = matrix.get(system_tag, {}).get(column_tag, 0) if column_tag is not None else 0
            candidate = pair_score + best_score(index + 1, used_mask | (1 << column_index))
            if candidate > best_total:
                best_total = candidate
                best_column = column_index
        used_mask |= 1 << best_column
        mapping[system_tag] = columns[best_column] if columns[best_column] is not None else "UNK"
    return mapping


def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mutual_information(labels_a, labels_b):
    total = len(labels_a)
    if total == 0:
        return 0.0
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    joint = Counter(zip(labels_a, labels_b))
    total_float = float(total)
    mi = 0.0
    for (label_a, label_b), joint_count in joint.items():
        p_ab = joint_count / total_float
        p_a = counts_a[label_a] / total_float
        p_b = counts_b[label_b] / total_float
        mi += p_ab * math.log2(p_ab / (p_a * p_b))
    return mi


def variation_of_information(gold_labels, system_labels):
    h_gold = entropy(gold_labels)
    h_system = entropy(system_labels)
    mi = mutual_information(gold_labels, system_labels)
    vi = h_gold + h_system - 2 * mi
    denom = h_gold + h_system
    vi_norm = vi / denom if denom else 0.0
    return vi, vi_norm


def mapping_accuracy(system_lines, gold_lines, mode):
    matrix, system_tags, gold_tags = build_counts(system_lines, gold_lines)
    if mode == "many2one":
        mapping = learn_many_to_one(matrix, system_tags)
    elif mode == "one2one":
        mapping = learn_one_to_one(matrix, system_tags, gold_tags)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    correct = 0
    total = 0
    for system_line, gold_line in zip(system_lines, gold_lines):
        for system_tag, gold_tag in zip(system_line.split(), gold_line.split()):
            total += 1
            if mapping.get(system_tag, "UNK") == gold_tag:
                correct += 1
    return correct / total if total else 0.0


def parse_output_map(path: Path):
    metrics = {}
    if not path.is_file():
        return metrics
    for line in read_lines(path):
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        metrics[key] = value
    return metrics


def format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6f}"
    return str(value)


def to_float(value):
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def quantile(values, probability):
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def sample_sd(values, mean):
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def sample_skew(values, mean, sd):
    n = len(values)
    if n <= 2 or sd == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * sum(((value - mean) / sd) ** 3 for value in values)


def distribution_stats(values):
    if not values:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "sem": None,
            "min": None,
            "p05": None,
            "q1": None,
            "median": None,
            "q3": None,
            "p95": None,
            "max": None,
            "iqr": None,
            "skew": None,
        }
    mean = sum(values) / len(values)
    sd = sample_sd(values, mean)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "sem": sd / math.sqrt(len(values)),
        "min": min(values),
        "p05": quantile(values, 0.05),
        "q1": q1,
        "median": quantile(values, 0.50),
        "q3": q3,
        "p95": quantile(values, 0.95),
        "max": max(values),
        "iqr": q3 - q1,
        "skew": sample_skew(values, mean, sd),
    }


def betacf(a, b, x):
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def student_t_cdf(t_value, df):
    if df <= 0:
        return None
    if t_value == 0:
        return 0.5
    x = df / (df + t_value * t_value)
    ibeta = regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t_value > 0:
        return 1.0 - 0.5 * ibeta
    return 0.5 * ibeta


def paired_t_test(diffs):
    stats = distribution_stats(diffs)
    n = stats["n"]
    if n <= 1 or stats["sd"] == 0:
        return None, n - 1, None
    t_stat = stats["mean"] / (stats["sd"] / math.sqrt(n))
    cdf = student_t_cdf(abs(t_stat), n - 1)
    p_value = 2.0 * (1.0 - cdf) if cdf is not None else None
    return t_stat, n - 1, p_value


def extract_combo(run_name):
    match = re.search(r"_(\d+_\d+_\d+_\d+_\d+_\d+)\.out$", run_name)
    return match.group(1) if match else None


def evaluate_run(run_dir: Path, gold_lines, line_name=None):
    pred_path, system_lines = choose_prediction_file(run_dir, gold_lines)
    gold_labels = flatten(gold_lines)
    system_labels = flatten(system_lines)
    vi, vi_norm = variation_of_information(gold_labels, system_labels)
    m2o = mapping_accuracy(system_lines, gold_lines, "many2one")
    o2o = mapping_accuracy(system_lines, gold_lines, "one2one")
    output_metrics = parse_output_map(run_dir / "output.map")
    return {
        "line": line_name,
        "run": run_dir.name,
        "combo": extract_combo(run_dir.name),
        "pred_file": pred_path.name,
        "vi": vi,
        "vi_norm": vi_norm,
        "acc_many2one": m2o,
        "acc_one2one": o2o,
        "train_logZ": output_metrics.get("train.logZ"),
    }


def summarize_by_line(rows):
    summary_rows = []
    line_names = sorted({row["line"] for row in rows if row.get("line") is not None})
    for line_name in line_names:
        line_rows = [row for row in rows if row.get("line") == line_name]
        summary = {"line": line_name, "n_runs": len(line_rows)}
        for metric in NUMERIC_METRICS:
            values = [to_float(row.get(metric)) for row in line_rows]
            values = [value for value in values if value is not None]
            stats = distribution_stats(values)
            for stat_name, stat_value in stats.items():
                summary[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(summary)
    return summary_rows


def sign_flip_pvalue(diffs, samples, seed):
    nonzero_diffs = [diff for diff in diffs if diff != 0]
    if not nonzero_diffs:
        return 1.0
    observed = abs(sum(nonzero_diffs) / len(nonzero_diffs))
    rng = random.Random(seed)
    more_extreme = 0
    for _ in range(samples):
        flipped_sum = 0.0
        for diff in nonzero_diffs:
            flipped_sum += diff if rng.getrandbits(1) else -diff
        if abs(flipped_sum / len(nonzero_diffs)) >= observed:
            more_extreme += 1
    return (more_extreme + 1) / (samples + 1)


def paired_line_tests(rows, samples=10000, seed=1):
    by_line_combo = {}
    for row in rows:
        line_name = row.get("line")
        combo = row.get("combo")
        if line_name is None or combo is None:
            continue
        by_line_combo[(line_name, combo)] = row

    line_names = sorted({row.get("line") for row in rows if row.get("line") is not None})
    paired_rows = []
    for line_a, line_b in combinations(line_names, 2):
        combos_a = {combo for (line, combo) in by_line_combo if line == line_a}
        combos_b = {combo for (line, combo) in by_line_combo if line == line_b}
        shared_combos = sorted(combos_a & combos_b)
        for metric in NUMERIC_METRICS:
            diffs = []
            paired_values = []
            for combo in shared_combos:
                value_a = to_float(by_line_combo[(line_a, combo)].get(metric))
                value_b = to_float(by_line_combo[(line_b, combo)].get(metric))
                if value_a is None or value_b is None:
                    continue
                diffs.append(value_a - value_b)
                paired_values.append((value_a, value_b))
            n = len(diffs)
            if n:
                stats_a = distribution_stats([value_a for value_a, _ in paired_values])
                stats_b = distribution_stats([value_b for _, value_b in paired_values])
                diff_stats = distribution_stats(diffs)
                signflip_p = sign_flip_pvalue(diffs, samples=samples, seed=seed)
                t_stat, t_df, t_p = paired_t_test(diffs)
            else:
                stats_a = distribution_stats([])
                stats_b = distribution_stats([])
                diff_stats = distribution_stats([])
                signflip_p = t_stat = t_df = t_p = None
            paired_rows.append(
                {
                    "line_a": line_a,
                    "line_b": line_b,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_a": stats_a["mean"],
                    "mean_b": stats_b["mean"],
                    "mean_diff_a_minus_b": diff_stats["mean"],
                    "diff_sd": diff_stats["sd"],
                    "diff_median": diff_stats["median"],
                    "diff_q1": diff_stats["q1"],
                    "diff_q3": diff_stats["q3"],
                    "diff_skew": diff_stats["skew"],
                    "t_stat": t_stat,
                    "t_df": t_df,
                    "p_two_sided_paired_t": t_p,
                    "p_two_sided_signflip": signflip_p,
                    "paired_on": "combo",
                    "samples": samples if n else None,
                }
            )
    return paired_rows


def default_sidecar_path(output_path, suffix):
    if not output_path:
        return None
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix or '.tsv'}"))


def parse_lines_arg(lines_arg):
    if lines_arg is None:
        return None
    line_names = []
    for raw_item in lines_arg.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"line{item}"
        line_names.append(item)
    return line_names


def iter_run_dirs(outputs_root: Path, line_names):
    if line_names is None:
        for run_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
            yield None, run_dir
        return

    for line_name in line_names:
        line_dir = outputs_root / line_name
        if not line_dir.is_dir():
            print(f"warning: missing line directory: {line_dir}", file=sys.stderr)
            continue
        for run_dir in sorted(path for path in line_dir.iterdir() if path.is_dir()):
            yield line_name, run_dir


def write_rows(rows, header, output_path=None):
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                format_value(row[column])
                for column in header
            )
        )
    text = "\n".join(lines) + "\n"
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate output runs against a gold tag file."
    )
    parser.add_argument("outputs_root", nargs="?", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument(
        "--lines",
        nargs="?",
        const="line4,line5,line7,line8",
        default=None,
        help=(
            "Evaluate nested runs under comma-separated line directories. "
            "Pass without a value for line4,line5,line7,line8; numbers like 4,5 are accepted."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first missing or unaligned run instead of skipping with a warning.",
    )
    parser.add_argument("--out", help="Write TSV results to this path instead of stdout.")
    parser.add_argument(
        "--summary-out",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write per-line mean/sd TSV. In --lines mode with --out, defaults to "
            "<out>_summary.tsv. Pass an explicit path to override."
        ),
    )
    parser.add_argument(
        "--paired-out",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write paired line-comparison TSV. In --lines mode with --out, defaults to "
            "<out>_paired.tsv. Pass an explicit path to override."
        ),
    )
    parser.add_argument(
        "--paired-samples",
        type=int,
        default=10000,
        help="Number of sign-flip samples for paired p-values.",
    )
    parser.add_argument("--paired-seed", type=int, default=1)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    gold_path = Path(args.gold)
    gold_lines = read_lines(gold_path)
    line_names = parse_lines_arg(args.lines)

    rows = []
    skipped = 0
    for line_name, run_dir in iter_run_dirs(outputs_root, line_names):
        try:
            rows.append(evaluate_run(run_dir, gold_lines, line_name=line_name))
        except (FileNotFoundError, ValueError) as error:
            if args.strict:
                raise
            skipped += 1
            print(f"warning: skipped {run_dir}: {error}", file=sys.stderr)

    header = [
        "line",
        "run",
        "combo",
        "pred_file",
        "vi",
        "vi_norm",
        "acc_many2one",
        "acc_one2one",
        "train_logZ",
    ]
    if line_names is None:
        header = [column for column in header if column not in {"line", "combo"}]

    write_rows(rows, header, args.out)
    if line_names is not None:
        summary_out = args.summary_out
        if summary_out is None and args.out:
            summary_out = default_sidecar_path(args.out, "summary")
        paired_out = args.paired_out
        if paired_out is None and args.out:
            paired_out = default_sidecar_path(args.out, "paired")

        summary_rows = summarize_by_line(rows)
        summary_header = ["line", "n_runs"]
        for metric in NUMERIC_METRICS:
            summary_header.extend(
                [
                    f"{metric}_n",
                    f"{metric}_mean",
                    f"{metric}_sd",
                    f"{metric}_sem",
                    f"{metric}_min",
                    f"{metric}_p05",
                    f"{metric}_q1",
                    f"{metric}_median",
                    f"{metric}_q3",
                    f"{metric}_p95",
                    f"{metric}_max",
                    f"{metric}_iqr",
                    f"{metric}_skew",
                ]
            )
        if summary_out != "":
            write_rows(summary_rows, summary_header, summary_out)

        paired_rows = paired_line_tests(
            rows,
            samples=args.paired_samples,
            seed=args.paired_seed,
        )
        paired_header = [
            "line_a",
            "line_b",
            "metric",
            "n_pairs",
            "mean_a",
            "mean_b",
            "mean_diff_a_minus_b",
            "diff_sd",
            "diff_median",
            "diff_q1",
            "diff_q3",
            "diff_skew",
            "t_stat",
            "t_df",
            "p_two_sided_paired_t",
            "p_two_sided_signflip",
            "paired_on",
            "samples",
        ]
        if paired_out != "":
            write_rows(paired_rows, paired_header, paired_out)

    print(f"evaluated {len(rows)} runs; skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
