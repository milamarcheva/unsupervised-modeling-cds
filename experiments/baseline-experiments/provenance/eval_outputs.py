#!/usr/bin/env python3
import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from html import escape
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


class AlignmentError(ValueError):
    pass


def read_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def validate_alignment(system_lines, gold_lines, path: Path):
    if len(system_lines) != len(gold_lines):
        raise AlignmentError(
            f"{path} has {len(system_lines)} lines, but gold has {len(gold_lines)}"
        )
    for index, (system_line, gold_line) in enumerate(zip(system_lines, gold_lines), start=1):
        if len(system_line.split()) != len(gold_line.split()):
            raise AlignmentError(
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
    alignment_errors = []
    for candidate in explicit_candidates + wildcard_candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        lines = read_lines(candidate)
        if not lines:
            continue
        try:
            validate_alignment(lines, gold_lines, candidate)
        except AlignmentError as error:
            alignment_errors.append(str(error))
            continue
        return candidate, lines
    if alignment_errors:
        raise AlignmentError(
            "No aligned prediction file found in "
            f"{run_dir}; first misalignment: {alignment_errors[0]}"
        )
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


def accuracy_from_mapping(system_lines, gold_lines, mapping):
    correct = 0
    total = 0
    for system_line, gold_line in zip(system_lines, gold_lines):
        for system_tag, gold_tag in zip(system_line.split(), gold_line.split()):
            total += 1
            if mapping.get(system_tag, "UNK") == gold_tag:
                correct += 1
    return correct, total, correct / total if total else 0.0


def compute_prf(system_lines, gold_lines, mapping):
    pairs = []
    for system_line, gold_line in zip(system_lines, gold_lines):
        for system_tag, gold_tag in zip(system_line.split(), gold_line.split()):
            pairs.append((mapping.get(system_tag, "UNK"), gold_tag))

    gold_tags = sorted({gold_tag for _, gold_tag in pairs})
    true_positive = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)
    for predicted_tag, gold_tag in pairs:
        if predicted_tag == gold_tag:
            true_positive[gold_tag] += 1
        else:
            false_positive[predicted_tag] += 1
            false_negative[gold_tag] += 1

    per_tag = {}
    for tag in gold_tags:
        precision = (
            true_positive[tag] / (true_positive[tag] + false_positive[tag])
            if true_positive[tag] + false_positive[tag] > 0
            else 0.0
        )
        recall = (
            true_positive[tag] / (true_positive[tag] + false_negative[tag])
            if true_positive[tag] + false_negative[tag] > 0
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_tag[tag] = (precision * 100.0, recall * 100.0, f1 * 100.0)

    tag_count = len(gold_tags) if gold_tags else 1
    macro_precision = sum(values[0] for values in per_tag.values()) / tag_count
    macro_recall = sum(values[1] for values in per_tag.values()) / tag_count
    macro_f1 = sum(values[2] for values in per_tag.values()) / tag_count
    return per_tag, (macro_precision, macro_recall, macro_f1)


def accuracy_details(system_lines, gold_lines, mapping):
    correct, total, accuracy = accuracy_from_mapping(system_lines, gold_lines, mapping)
    per_tag, macro = compute_prf(system_lines, gold_lines, mapping)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_tag": per_tag,
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "mapping": mapping,
    }


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
        if value != 0.0 and abs(value) < 1.0e-6:
            return f"{value:.6e}"
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
    match = re.search(
        r"_(\d+(?:_\d+){5,6})(?:_[A-Za-z]+Stages_order\d+)?\.out$",
        run_name,
    )
    return match.group(1) if match else None


def extract_order(run_name):
    match = re.search(r"_order(\d+)\.out$", run_name)
    return match.group(1) if match else None


def evaluate_run(run_dir: Path, gold_lines, line_name=None, group_name=None):
    pred_path, system_lines = choose_prediction_file(run_dir, gold_lines)
    gold_labels = flatten(gold_lines)
    system_labels = flatten(system_lines)
    vi, vi_norm = variation_of_information(gold_labels, system_labels)
    matrix, system_tags, gold_tags = build_counts(system_lines, gold_lines)
    m2o_mapping = learn_many_to_one(matrix, system_tags)
    o2o_mapping = learn_one_to_one(matrix, system_tags, gold_tags)
    m2o_details = accuracy_details(system_lines, gold_lines, m2o_mapping)
    o2o_details = accuracy_details(system_lines, gold_lines, o2o_mapping)
    output_metrics = parse_output_map(run_dir / "output.map")
    return {
        "line": line_name,
        "group": group_name,
        "run": run_dir.name,
        "combo": extract_combo(run_dir.name),
        "order": extract_order(run_dir.name),
        "pred_file": pred_path.name,
        "vi": vi,
        "vi_norm": vi_norm,
        "acc_many2one": m2o_details["accuracy"],
        "acc_one2one": o2o_details["accuracy"],
        "train_logZ": output_metrics.get("train.logZ"),
        "_accuracy_analysis": {
            "many2one": m2o_details,
            "one2one": o2o_details,
        },
    }


def summarize_by(rows, group_column):
    summary_rows = []
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )
    for group_name in group_names:
        group_rows = [row for row in rows if row.get(group_column) == group_name]
        summary = {group_column: group_name, "n_runs": len(group_rows)}
        for metric in NUMERIC_METRICS:
            values = [to_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if value is not None]
            stats = distribution_stats(values)
            for stat_name, stat_value in stats.items():
                summary[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(summary)
    return summary_rows


def summarize_by_line(rows):
    return summarize_by(rows, "line")


def sort_tag_key(tag):
    try:
        return (0, int(tag))
    except (TypeError, ValueError):
        return (1, str(tag))


def ordered_tags(tags, preferred_order=None):
    tag_set = set(tags)
    if not preferred_order:
        return sorted(tag_set, key=sort_tag_key)
    ordered = [tag for tag in preferred_order if tag in tag_set]
    ordered.extend(tag for tag in sorted(tag_set, key=sort_tag_key) if tag not in ordered)
    return ordered


def compute_src_majority(mapping_counts):
    src_majority = {}
    for src, tgt_counts in mapping_counts.items():
        if tgt_counts:
            src_majority[src] = max(
                tgt_counts.items(),
                key=lambda item: (item[1], str(item[0])),
            )[0]
    return src_majority


def aggregate_by_src_label(mapping_counts, src_majority):
    aggregated = defaultdict(lambda: defaultdict(int))
    for src, tgt_counts in mapping_counts.items():
        src_label = src_majority.get(src)
        if src_label is None:
            continue
        for tgt, count in tgt_counts.items():
            aggregated[src_label][tgt] += count
    return aggregated


def accuracy_analysis_tables(rows, group_column, preferred_tag_order=None):
    macro_rows = []
    per_tag_rows = []
    mapping_rows = []
    label_mapping_rows = []
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )

    for group_name in group_names:
        group_rows = [row for row in rows if row.get(group_column) == group_name]
        for mode in ("many2one", "one2one"):
            analyses = [
                row["_accuracy_analysis"][mode]
                for row in group_rows
                if row.get("_accuracy_analysis", {}).get(mode)
            ]
            if not analyses:
                continue

            accuracy_stats = distribution_stats(
                [analysis["accuracy"] for analysis in analyses]
            )
            macro_precision_stats = distribution_stats(
                [analysis["macro_precision"] for analysis in analyses]
            )
            macro_recall_stats = distribution_stats(
                [analysis["macro_recall"] for analysis in analyses]
            )
            macro_f1_stats = distribution_stats(
                [analysis["macro_f1"] for analysis in analyses]
            )
            total_correct = sum(analysis["correct"] for analysis in analyses)
            total_tokens = sum(analysis["total"] for analysis in analyses)
            macro_rows.append(
                {
                    group_column: group_name,
                    "mode": mode,
                    "n_runs": len(analyses),
                    "total_correct": total_correct,
                    "total_tokens": total_tokens,
                    "micro_accuracy": total_correct / total_tokens if total_tokens else None,
                    "accuracy_mean": accuracy_stats["mean"],
                    "accuracy_sd": accuracy_stats["sd"],
                    "accuracy_median": accuracy_stats["median"],
                    "macro_precision_mean": macro_precision_stats["mean"],
                    "macro_precision_sd": macro_precision_stats["sd"],
                    "macro_precision_median": macro_precision_stats["median"],
                    "macro_recall_mean": macro_recall_stats["mean"],
                    "macro_recall_sd": macro_recall_stats["sd"],
                    "macro_recall_median": macro_recall_stats["median"],
                    "macro_f1_mean": macro_f1_stats["mean"],
                    "macro_f1_sd": macro_f1_stats["sd"],
                    "macro_f1_median": macro_f1_stats["median"],
                }
            )

            tag_values = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})
            for analysis in analyses:
                for tag, (precision, recall, f1) in analysis["per_tag"].items():
                    tag_values[tag]["precision"].append(precision)
                    tag_values[tag]["recall"].append(recall)
                    tag_values[tag]["f1"].append(f1)
            for tag in ordered_tags(tag_values, preferred_tag_order):
                precision_stats = distribution_stats(tag_values[tag]["precision"])
                recall_stats = distribution_stats(tag_values[tag]["recall"])
                f1_stats = distribution_stats(tag_values[tag]["f1"])
                per_tag_rows.append(
                    {
                        group_column: group_name,
                        "mode": mode,
                        "tag": tag,
                        "n_runs": precision_stats["n"],
                        "precision_mean": precision_stats["mean"],
                        "precision_sd": precision_stats["sd"],
                        "precision_median": precision_stats["median"],
                        "recall_mean": recall_stats["mean"],
                        "recall_sd": recall_stats["sd"],
                        "recall_median": recall_stats["median"],
                        "f1_mean": f1_stats["mean"],
                        "f1_sd": f1_stats["sd"],
                        "f1_median": f1_stats["median"],
                    }
                )

            mapping_counts = defaultdict(lambda: defaultdict(int))
            for analysis in analyses:
                for src, tgt in analysis["mapping"].items():
                    mapping_counts[src][tgt] += 1
            src_majority = compute_src_majority(mapping_counts)
            for src in sorted(mapping_counts, key=sort_tag_key):
                tgt_counts = mapping_counts[src]
                for tgt, count in sorted(
                    tgt_counts.items(),
                    key=lambda item: (-item[1], sort_tag_key(item[0])),
                ):
                    mapping_rows.append(
                        {
                            group_column: group_name,
                            "mode": mode,
                            "system_tag": src,
                            "system_majority_label": src_majority.get(src),
                            "gold_tag": tgt,
                            "count": count,
                        }
                    )

            label_counts = aggregate_by_src_label(mapping_counts, src_majority)
            for src_label in ordered_tags(label_counts, preferred_tag_order):
                tgt_counts = label_counts[src_label]
                for tgt, count in sorted(
                    tgt_counts.items(),
                    key=lambda item: (-item[1], sort_tag_key(item[0])),
                ):
                    label_mapping_rows.append(
                        {
                            group_column: group_name,
                            "mode": mode,
                            "system_majority_label": src_label,
                            "gold_tag": tgt,
                            "count": count,
                        }
                    )

    return macro_rows, per_tag_rows, mapping_rows, label_mapping_rows


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    tie_counts = []
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = average_rank
        if end - index > 1:
            tie_counts.append(end - index)
        index = end
    return ranks, tie_counts


def mann_whitney_u_test(values_a, values_b):
    values_a = [value for value in values_a if value is not None]
    values_b = [value for value in values_b if value is not None]
    n_a = len(values_a)
    n_b = len(values_b)
    if not values_a or not values_b:
        return {
            "n_a": n_a,
            "n_b": n_b,
            "u_a": None,
            "u_b": None,
            "u_min": None,
            "z": None,
            "p_two_sided_mann_whitney": None,
        }

    pooled_values = values_a + values_b
    ranks, tie_counts = average_ranks(pooled_values)
    rank_sum_a = sum(ranks[:n_a])
    u_a = rank_sum_a - (n_a * (n_a + 1) / 2.0)
    u_b = n_a * n_b - u_a
    u_min = min(u_a, u_b)

    total_n = n_a + n_b
    mean_u = n_a * n_b / 2.0
    tie_sum = sum(tie_count ** 3 - tie_count for tie_count in tie_counts)
    variance = (n_a * n_b / 12.0) * (
        (total_n + 1) - (tie_sum / (total_n * (total_n - 1)))
    )
    if variance <= 0:
        z_value = None
        p_value = None
    else:
        z_value = (u_a - mean_u) / math.sqrt(variance)
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))

    return {
        "n_a": n_a,
        "n_b": n_b,
        "u_a": u_a,
        "u_b": u_b,
        "u_min": u_min,
        "z": z_value,
        "p_two_sided_mann_whitney": p_value,
    }


def mann_whitney_group_tests(rows, group_column):
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )
    test_rows = []
    for group_a, group_b in combinations(group_names, 2):
        rows_a = [row for row in rows if row.get(group_column) == group_a]
        rows_b = [row for row in rows if row.get(group_column) == group_b]
        for metric in NUMERIC_METRICS:
            values_a = [to_float(row.get(metric)) for row in rows_a]
            values_a = [value for value in values_a if value is not None]
            values_b = [to_float(row.get(metric)) for row in rows_b]
            values_b = [value for value in values_b if value is not None]
            stats_a = distribution_stats(values_a)
            stats_b = distribution_stats(values_b)
            test = mann_whitney_u_test(values_a, values_b)
            test_rows.append(
                {
                    f"{group_column}_a": group_a,
                    f"{group_column}_b": group_b,
                    "metric": metric,
                    "n_a": test["n_a"],
                    "n_b": test["n_b"],
                    "mean_a": stats_a["mean"],
                    "mean_b": stats_b["mean"],
                    "median_a": stats_a["median"],
                    "median_b": stats_b["median"],
                    "u_a": test["u_a"],
                    "u_b": test["u_b"],
                    "u_min": test["u_min"],
                    "z": test["z"],
                    "p_two_sided_mann_whitney": test["p_two_sided_mann_whitney"],
                    "paired_on": "none",
                    "test": "Mann-Whitney U; normal approximation with tie correction",
                }
            )
    return test_rows


def histogram_rows(rows, group_column, bins):
    if bins <= 0:
        raise ValueError("--hist-bins must be greater than 0")
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )
    result_rows = []
    for metric in NUMERIC_METRICS:
        metric_values = []
        by_group = {}
        for group_name in group_names:
            values = [
                to_float(row.get(metric))
                for row in rows
                if row.get(group_column) == group_name
            ]
            values = [value for value in values if value is not None]
            by_group[group_name] = values
            metric_values.extend(values)
        if not metric_values:
            continue

        min_value = min(metric_values)
        max_value = max(metric_values)
        if min_value == max_value:
            start_value = min_value - 0.5
            width = 1.0 / bins
        else:
            start_value = min_value
            width = (max_value - min_value) / bins

        for group_name, values in by_group.items():
            counts = [0] * bins
            for value in values:
                if min_value == max_value:
                    bin_index = bins // 2
                elif value == max_value:
                    bin_index = bins - 1
                else:
                    bin_index = int((value - start_value) / width)
                    bin_index = min(max(bin_index, 0), bins - 1)
                counts[bin_index] += 1
            for bin_index, count in enumerate(counts):
                bin_start = start_value + bin_index * width
                result_rows.append(
                    {
                        "metric": metric,
                        group_column: group_name,
                        "bin_index": bin_index,
                        "bin_start": bin_start,
                        "bin_end": bin_start + width,
                        "count": count,
                    }
                )
    return result_rows


def read_tsv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def histogram_counts_by_group(rows, group_column, metric, bins):
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )
    values_by_group = {}
    metric_values = []
    for group_name in group_names:
        values = [
            to_float(row.get(metric))
            for row in rows
            if row.get(group_column) == group_name
        ]
        values = [value for value in values if value is not None]
        if values:
            values_by_group[group_name] = values
            metric_values.extend(values)
    if len(values_by_group) < 2:
        return None

    min_value = min(metric_values)
    max_value = max(metric_values)
    if min_value == max_value:
        start_value = min_value - 0.5
        width = 1.0 / bins
    else:
        start_value = min_value
        width = (max_value - min_value) / bins

    counts_by_group = {}
    for group_name, values in values_by_group.items():
        counts = [0] * bins
        for value in values:
            if min_value == max_value:
                bin_index = bins // 2
            elif value == max_value:
                bin_index = bins - 1
            else:
                bin_index = int((value - start_value) / width)
                bin_index = min(max(bin_index, 0), bins - 1)
            counts[bin_index] += 1
        counts_by_group[group_name] = counts

    return {
        "counts_by_group": counts_by_group,
        "start_value": start_value,
        "width": width,
        "min_value": min_value,
        "max_value": max_value,
        "max_count": max(max(counts) for counts in counts_by_group.values()),
    }


def histogram_x_label(metric):
    if metric == "train_logZ":
        return "Final LL"
    return f"Final {metric}"


def format_histogram_group_label(group_name, all_group_names):
    all_group_names = set(all_group_names)
    if group_name == "MLUcompliant":
        return "Brown-compliant"
    if group_name == "noncompliant":
        if "MLUcompliant" in all_group_names:
            return "non-Brown-compliant"
        return "non-compliant"
    return group_name


def histogram_group_style(group_name, all_group_names, group_index):
    all_group_names = set(all_group_names)
    if group_name in {"compliant", "MLUcompliant"}:
        return {
            "facecolor": "#1f77b4",
            "edgecolor": "blue",
            "alpha": 0.5,
        }
    if group_name == "noncompliant":
        return {
            "facecolor": "#ff7f0e",
            "edgecolor": "red",
            "alpha": 0.3,
        }

    fallback_styles = [
        {"facecolor": "#1f77b4", "edgecolor": "blue", "alpha": 0.5},
        {"facecolor": "#ff7f0e", "edgecolor": "red", "alpha": 0.3},
        {"facecolor": "#2ca02c", "edgecolor": "#1b7f1b", "alpha": 0.35},
        {"facecolor": "#9467bd", "edgecolor": "#6d3f9c", "alpha": 0.35},
        {"facecolor": "#17becf", "edgecolor": "#0f8d99", "alpha": 0.35},
        {"facecolor": "#8c564b", "edgecolor": "#6f4038", "alpha": 0.35},
    ]
    return fallback_styles[group_index % len(fallback_styles)]


def write_histogram_svg_plots(rows, group_column, bins, hist_dir, metrics=None):
    output_dir = Path(hist_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas_width = 900
    canvas_height = 540
    margin_left = 78
    margin_right = 24
    margin_top = 54
    margin_bottom = 78
    plot_width = canvas_width - margin_left - margin_right
    plot_height = canvas_height - margin_top - margin_bottom

    metrics = metrics or NUMERIC_METRICS
    for metric in metrics:
        histogram = histogram_counts_by_group(rows, group_column, metric, bins)
        if histogram is None:
            continue

        counts_by_group = histogram["counts_by_group"]
        max_count = histogram["max_count"] or 1
        bin_width = plot_width / bins
        group_names = list(counts_by_group)
        elements = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{canvas_width}" height="{canvas_height}" '
                f'viewBox="0 0 {canvas_width} {canvas_height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                f'<text x="{canvas_width / 2}" y="28" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="20" '
                f'font-weight="700">{escape(metric)} histogram</text>'
            ),
            (
                f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
                f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
                f'stroke="#111827" stroke-width="1.5"/>'
            ),
            (
                f'<line x1="{margin_left}" y1="{margin_top}" '
                f'x2="{margin_left}" y2="{margin_top + plot_height}" '
                f'stroke="#111827" stroke-width="1.5"/>'
            ),
        ]

        for tick in range(6):
            y_value = max_count * tick / 5.0
            y_pos = margin_top + plot_height - (y_value / max_count) * plot_height
            elements.extend(
                [
                    (
                        f'<line x1="{margin_left - 5}" y1="{y_pos:.2f}" '
                        f'x2="{margin_left}" y2="{y_pos:.2f}" stroke="#111827"/>'
                    ),
                    (
                        f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" '
                        f'text-anchor="end" font-family="Helvetica, Arial, sans-serif" '
                        f'font-size="11">{y_value:.0f}</text>'
                    ),
                    (
                        f'<line x1="{margin_left}" y1="{y_pos:.2f}" '
                        f'x2="{margin_left + plot_width}" y2="{y_pos:.2f}" '
                        f'stroke="#e5e7eb" stroke-width="1"/>'
                    ),
                ]
            )

        for tick in range(6):
            x_value = histogram["start_value"] + histogram["width"] * bins * tick / 5.0
            x_pos = margin_left + plot_width * tick / 5.0
            elements.extend(
                [
                    (
                        f'<line x1="{x_pos:.2f}" y1="{margin_top + plot_height}" '
                        f'x2="{x_pos:.2f}" y2="{margin_top + plot_height + 5}" '
                        f'stroke="#111827"/>'
                    ),
                    (
                        f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 22}" '
                        f'text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
                        f'font-size="11">{x_value:.3f}</text>'
                    ),
                ]
            )

        for group_index, (group_name, counts) in enumerate(counts_by_group.items()):
            style = histogram_group_style(group_name, group_names, group_index)
            for bin_index, count in enumerate(counts):
                if count == 0:
                    continue
                bar_height = (count / max_count) * plot_height
                x_pos = margin_left + bin_index * bin_width
                y_pos = margin_top + plot_height - bar_height
                elements.append(
                    (
                        f'<rect x="{x_pos:.2f}" y="{y_pos:.2f}" '
                        f'width="{bin_width:.2f}" height="{bar_height:.2f}" '
                        f'fill="{style["facecolor"]}" stroke="{style["edgecolor"]}" '
                        f'stroke-width="1.2" opacity="{style["alpha"]}"/>'
                    )
                )

        legend_x = margin_left + plot_width - 170
        legend_y = margin_top + 8
        for group_index, group_name in enumerate(counts_by_group):
            style = histogram_group_style(group_name, group_names, group_index)
            y_pos = legend_y + group_index * 22
            elements.extend(
                [
                    (
                        f'<rect x="{legend_x}" y="{y_pos}" width="14" height="14" '
                        f'fill="{style["facecolor"]}" stroke="{style["edgecolor"]}" '
                        f'stroke-width="1.2" opacity="{style["alpha"]}"/>'
                    ),
                    (
                        f'<text x="{legend_x + 20}" y="{y_pos + 12}" '
                        f'font-family="Helvetica, Arial, sans-serif" font-size="12">'
                        f'{escape(format_histogram_group_label(group_name, group_names))}</text>'
                    ),
                ]
            )

        elements.extend(
            [
                (
                    f'<text x="{margin_left + plot_width / 2}" '
                    f'y="{canvas_height - 18}" text-anchor="middle" '
                    f'font-family="Helvetica, Arial, sans-serif" font-size="13">'
                    f'{escape(histogram_x_label(metric))}</text>'
                ),
                "</svg>",
            ]
        )
        (output_dir / f"{metric}_hist.svg").write_text(
            "\n".join(elements) + "\n",
            encoding="utf-8",
        )


def write_histogram_plots(rows, group_column, bins, hist_dir, metrics=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "warning: matplotlib is not installed; wrote SVG histograms instead",
            file=sys.stderr,
        )
        write_histogram_svg_plots(rows, group_column, bins, hist_dir, metrics=metrics)
        return

    output_dir = Path(hist_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_names = sorted(
        {row[group_column] for row in rows if row.get(group_column) is not None}
    )
    metrics = metrics or NUMERIC_METRICS
    for metric in metrics:
        values_by_group = {}
        for group_name in group_names:
            values = [
                to_float(row.get(metric))
                for row in rows
                if row.get(group_column) == group_name
            ]
            values = [value for value in values if value is not None]
            if values:
                values_by_group[group_name] = values
        if len(values_by_group) < 2:
            continue

        plt.figure(figsize=(8, 5))
        present_group_names = list(values_by_group)
        for group_index, (group_name, values) in enumerate(values_by_group.items()):
            style = histogram_group_style(group_name, present_group_names, group_index)
            plt.hist(
                values,
                bins=bins,
                color=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=1.2,
                alpha=style["alpha"],
                label=format_histogram_group_label(group_name, present_group_names),
            )
        plt.xlabel(histogram_x_label(metric))
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_hist.png", dpi=160)
        plt.close()


def default_hist_dir_from_tsv(tsv_path):
    path = Path(tsv_path)
    return str(path.with_name(f"{path.stem}_histograms"))



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


def default_sidecar_prefix(output_path, suffix):
    if not output_path:
        return None
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{suffix}"))


def sidecar_from_prefix(prefix, suffix):
    path = Path(prefix)
    return str(path.with_name(f"{path.name}_{suffix}.tsv"))


def write_accuracy_analysis_outputs(rows, group_column, prefix, preferred_tag_order=None):
    macro_rows, per_tag_rows, mapping_rows, label_mapping_rows = accuracy_analysis_tables(
        rows,
        group_column,
        preferred_tag_order=preferred_tag_order,
    )
    group_header = [group_column]
    write_rows(
        macro_rows,
        group_header
        + [
            "mode",
            "n_runs",
            "total_correct",
            "total_tokens",
            "micro_accuracy",
            "accuracy_mean",
            "accuracy_sd",
            "accuracy_median",
            "macro_precision_mean",
            "macro_precision_sd",
            "macro_precision_median",
            "macro_recall_mean",
            "macro_recall_sd",
            "macro_recall_median",
            "macro_f1_mean",
            "macro_f1_sd",
            "macro_f1_median",
        ],
        sidecar_from_prefix(prefix, "macro"),
    )
    write_rows(
        per_tag_rows,
        group_header
        + [
            "mode",
            "tag",
            "n_runs",
            "precision_mean",
            "precision_sd",
            "precision_median",
            "recall_mean",
            "recall_sd",
            "recall_median",
            "f1_mean",
            "f1_sd",
            "f1_median",
        ],
        sidecar_from_prefix(prefix, "per_tag_prf"),
    )
    write_rows(
        mapping_rows,
        group_header
        + [
            "mode",
            "system_tag",
            "system_majority_label",
            "gold_tag",
            "count",
        ],
        sidecar_from_prefix(prefix, "mapping_counts"),
    )
    write_rows(
        label_mapping_rows,
        group_header
        + [
            "mode",
            "system_majority_label",
            "gold_tag",
            "count",
        ],
        sidecar_from_prefix(prefix, "label_mapping_counts"),
    )


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


def parse_groups_arg(groups_arg):
    if groups_arg is None:
        return None
    group_names = []
    for raw_item in groups_arg.split(","):
        item = raw_item.strip()
        if item:
            group_names.append(item)
    return group_names


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


def iter_group_run_dirs(outputs_root: Path, group_names):
    for group_name in group_names:
        group_dir = outputs_root / group_name
        if not group_dir.is_dir():
            print(f"warning: missing group directory: {group_dir}", file=sys.stderr)
            continue
        for run_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            yield group_name, run_dir


def group_pair_key(run_name):
    combo = extract_combo(run_name)
    order = extract_order(run_name)
    if combo is None or order is None:
        return run_name
    return f"{combo}_order{order}"


def collect_group_run_dirs(outputs_root: Path, group_names):
    run_dirs_by_group = {}
    for group_name in group_names:
        group_dir = outputs_root / group_name
        if not group_dir.is_dir():
            print(f"warning: missing group directory: {group_dir}", file=sys.stderr)
            run_dirs_by_group[group_name] = {}
            continue
        group_runs = {}
        for run_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            key = group_pair_key(run_dir.name)
            if key in group_runs:
                raise ValueError(
                    f"Duplicate paired run key {key} in {group_dir}: "
                    f"{group_runs[key].name} and {run_dir.name}"
                )
            group_runs[key] = run_dir
        run_dirs_by_group[group_name] = group_runs
    return run_dirs_by_group


def evaluate_group_runs(outputs_root: Path, group_names, gold_lines, strict=False):
    rows = []
    skipped = 0
    run_dirs_by_group = collect_group_run_dirs(outputs_root, group_names)
    all_keys = sorted(
        {
            key
            for group_runs in run_dirs_by_group.values()
            for key in group_runs
        }
    )

    for key in all_keys:
        present_groups = [group_name for group_name in group_names if key in run_dirs_by_group[group_name]]
        missing_groups = [group_name for group_name in group_names if key not in run_dirs_by_group[group_name]]
        if missing_groups:
            skipped += len(group_names)
            present_dirs = [str(run_dirs_by_group[group_name][key]) for group_name in present_groups]
            message = (
                f"paired run {key} is missing in groups {', '.join(missing_groups)}"
            )
            if present_dirs:
                message += f"; dropped present counterparts: {', '.join(present_dirs)}"
            if strict:
                raise FileNotFoundError(message)
            print(f"warning: skipped {message}", file=sys.stderr)
            continue

        pair_rows = []
        pair_error = None
        for group_name in group_names:
            run_dir = run_dirs_by_group[group_name][key]
            try:
                pair_rows.append(
                    evaluate_run(run_dir, gold_lines, group_name=group_name)
                )
            except (FileNotFoundError, ValueError) as error:
                pair_error = (group_name, run_dir, error)
                break

        if pair_error is not None:
            skipped += len(group_names)
            group_name, run_dir, error = pair_error
            counterpart_groups = [name for name in group_names if name != group_name]
            message = (
                f"{run_dir}: {error}; dropped corresponding runs in groups "
                f"{', '.join(counterpart_groups)} to keep group counts aligned"
            )
            if strict:
                raise type(error)(str(error))
            print(f"warning: skipped {message}", file=sys.stderr)
            continue

        rows.extend(pair_rows)

    return rows, skipped


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
        "--groups",
        nargs="?",
        const="compliant,noncompliant",
        default=None,
        help=(
            "Evaluate nested runs under comma-separated group directories. "
            "Pass without a value for compliant,noncompliant."
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
            "Write per-line or per-group summary TSV. In --lines/--groups mode "
            "with --out, defaults to <out>_summary.tsv. Pass an explicit path to override."
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
        "--mann-whitney-out",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write Mann-Whitney U group-comparison TSV. In --groups mode with --out, "
            "defaults to <out>_mann_whitney.tsv. Pass an explicit path to override."
        ),
    )
    parser.add_argument(
        "--hist-out",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write histogram bin-count TSV. In --groups mode with --out, defaults to "
            "<out>_hist.tsv. Pass an explicit path to override."
        ),
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Number of bins for histogram TSVs and optional plots.",
    )
    parser.add_argument(
        "--hist-dir",
        help=(
            "Optionally write overlay histogram plots to this directory. "
            "Writes PNGs with matplotlib, otherwise SVGs."
        ),
    )
    parser.add_argument(
        "--hist-tsv",
        help=(
            "Read rows from an existing evaluation TSV and regenerate histogram plots "
            "without rerunning evaluation."
        ),
    )
    parser.add_argument(
        "--hist-metric",
        choices=NUMERIC_METRICS,
        help=(
            "When used with --hist-tsv, regenerate only this histogram metric "
            "instead of all numeric metrics."
        ),
    )
    parser.add_argument(
        "--hist-group-column",
        default="group",
        help=(
            "Grouping column to use with --hist-tsv. Defaults to 'group'."
        ),
    )
    parser.add_argument(
        "--accuracy-analysis-out",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write script_accuracy_python-style macro, per-tag PRF, mapping-count, "
            "and label-mapping TSVs using this output prefix. Pass without a value "
            "with --out to default to <out>_accuracy."
        ),
    )
    parser.add_argument(
        "--tag-order",
        help="Optional comma-separated display order for per-tag PRF and label mappings.",
    )
    parser.add_argument(
        "--paired-samples",
        type=int,
        default=10000,
        help="Number of sign-flip samples for paired p-values.",
    )
    parser.add_argument("--paired-seed", type=int, default=1)
    args = parser.parse_args()
    if args.lines is not None and args.groups is not None:
        parser.error("--lines and --groups cannot be used together")

    if args.hist_tsv:
        if args.hist_out not in (None, ""):
            parser.error("--hist-out is not used with --hist-tsv")
        rows = read_tsv_rows(args.hist_tsv)
        if not rows:
            parser.error(f"--hist-tsv has no data rows: {args.hist_tsv}")
        if args.hist_group_column not in rows[0]:
            parser.error(
                f"--hist-group-column {args.hist_group_column!r} is not a column in "
                f"{args.hist_tsv}"
            )
        if args.hist_metric and args.hist_metric not in rows[0]:
            parser.error(
                f"--hist-metric {args.hist_metric!r} is not a column in {args.hist_tsv}"
            )
        metrics = [args.hist_metric] if args.hist_metric else None
        hist_dir = args.hist_dir or default_hist_dir_from_tsv(args.hist_tsv)
        write_histogram_plots(
            rows,
            args.hist_group_column,
            args.hist_bins,
            hist_dir,
            metrics=metrics,
        )
        print(f"wrote histogram plots to {hist_dir}", file=sys.stderr)
        return

    outputs_root = Path(args.outputs_root)
    gold_path = Path(args.gold)
    gold_lines = read_lines(gold_path)
    line_names = parse_lines_arg(args.lines)
    group_names = parse_groups_arg(args.groups)

    if group_names is not None:
        rows, skipped = evaluate_group_runs(
            outputs_root, group_names, gold_lines, strict=args.strict
        )
    else:
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
        "group",
        "run",
        "combo",
        "order",
        "pred_file",
        "vi",
        "vi_norm",
        "acc_many2one",
        "acc_one2one",
        "train_logZ",
    ]
    if line_names is None and group_names is None:
        header = [
            column
            for column in header
            if column not in {"line", "group", "combo", "order"}
        ]
    elif line_names is not None:
        header = [column for column in header if column not in {"group", "order"}]
    elif group_names is not None:
        header = [column for column in header if column != "line"]

    write_rows(rows, header, args.out)
    if line_names is not None or group_names is not None:
        summary_out = args.summary_out
        if summary_out is None and args.out:
            summary_out = default_sidecar_path(args.out, "summary")

        group_column = "line" if line_names is not None else "group"
        summary_rows = summarize_by(rows, group_column)
        summary_header = [group_column, "n_runs"]
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

    if line_names is not None:
        paired_out = args.paired_out
        if paired_out is None and args.out:
            paired_out = default_sidecar_path(args.out, "paired")

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

    if group_names is not None:
        mann_whitney_out = args.mann_whitney_out
        if mann_whitney_out is None and args.out:
            mann_whitney_out = default_sidecar_path(args.out, "mann_whitney")
        if mann_whitney_out != "":
            mann_whitney_rows = mann_whitney_group_tests(rows, "group")
            mann_whitney_header = [
                "group_a",
                "group_b",
                "metric",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "median_a",
                "median_b",
                "u_a",
                "u_b",
                "u_min",
                "z",
                "p_two_sided_mann_whitney",
                "paired_on",
                "test",
            ]
            write_rows(mann_whitney_rows, mann_whitney_header, mann_whitney_out)

        hist_out = args.hist_out
        if hist_out is None and args.out:
            hist_out = default_sidecar_path(args.out, "hist")
        if hist_out != "":
            hist_header = ["metric", "group", "bin_index", "bin_start", "bin_end", "count"]
            write_rows(
                histogram_rows(rows, "group", args.hist_bins),
                hist_header,
                hist_out,
            )
        if args.hist_dir:
            write_histogram_plots(rows, "group", args.hist_bins, args.hist_dir)

    if args.accuracy_analysis_out is not None:
        if line_names is None and group_names is None:
            parser.error("--accuracy-analysis-out requires --lines or --groups")
        accuracy_prefix = args.accuracy_analysis_out
        if accuracy_prefix == "":
            if not args.out:
                parser.error("--accuracy-analysis-out needs a path when --out is not set")
            accuracy_prefix = default_sidecar_prefix(args.out, "accuracy")
        group_column = "line" if line_names is not None else "group"
        preferred_tag_order = (
            [tag.strip() for tag in args.tag_order.split(",") if tag.strip()]
            if args.tag_order
            else None
        )
        write_accuracy_analysis_outputs(
            rows,
            group_column,
            accuracy_prefix,
            preferred_tag_order=preferred_tag_order,
        )

    print(f"evaluated {len(rows)} runs; skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
