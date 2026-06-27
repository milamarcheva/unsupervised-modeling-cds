#!/usr/bin/env python3
import argparse
import random
import re
import shlex
import subprocess
import sys
from itertools import permutations as permut
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = REPO_ROOT / "outputs" / "out2025" / "MLUcompliant"
DIR_PREFIX = (
    "20250503_MorphBrownStagesOrderedNoUnlocking_dilute2_order5555_"
    "grouping5_PREPAnchorIndex0_n10_v10"
)
NAME_RE = re.compile(
    r"^20250503_MorphBrownStagesOrderedNoUnlocking_dilute2_order5555_"
    r"grouping5_PREPAnchorIndex0_n10_v10_(?P<combo>\d+_\d+_\d+_\d+_\d+_\d+_\d+)"
    r"_MLUStages_order(?P<order>\d+)\.out$"
)


def build_selected_lists10():
    random.seed(10)

    all_lists = []
    for unc_cop_anchor_index in range(0, 5):
        for unc_aux_anchor_index in range(0, 5):
            for c_aux_anchor_index in range(0, 3):
                for c_cop_anchor_index in range(0, 3):
                    for art_anchor_index in range(0, 2):
                        for ir_past_anchor_index in range(0, 13):
                            for ir3_anchor_index in range(0, 3):
                                all_lists.append(
                                    [
                                        unc_cop_anchor_index,
                                        unc_aux_anchor_index,
                                        c_aux_anchor_index,
                                        c_cop_anchor_index,
                                        art_anchor_index,
                                        ir_past_anchor_index,
                                        ir3_anchor_index,
                                    ]
                                )

    selected_lists = random.sample(all_lists, 500)
    selected_lists10 = random.sample(selected_lists, 10)
    return selected_lists10


def build_random_subset_orders_mlu():
    stages_mlu = {
        2: list(permut(["ING", "PREP", "PLU"])),
        3: list(permut(["IRPAST", "POS", "UNCCOP"])),
        4: list(permut(["ART", "RPAST", "R3"])),
        5: list(permut(["IR3", "UNCAUX", "CCOP", "CAUX"])),
    }

    orders_mlu = []
    for order2 in stages_mlu[2]:
        for order3 in stages_mlu[3]:
            for order4 in stages_mlu[4]:
                for order5 in stages_mlu[5]:
                    orders_mlu.append(",".join(order2 + order3 + order4 + order5))

    return random.sample(orders_mlu, 200)


def build_expected_runs():
    selected_lists10 = build_selected_lists10()
    random_subset_orders_mlu = build_random_subset_orders_mlu()

    expected = {}
    for anchor_list in selected_lists10:
        combo = "_".join(str(value) for value in anchor_list)
        expected[combo] = {}
        for order_index, order_str in enumerate(random_subset_orders_mlu):
            run_dir_name = f"{DIR_PREFIX}_{combo}_MLUStages_order{order_index}.out"
            expected[combo][order_index] = {
                "dir_name": run_dir_name,
                "combo": combo,
                "order_index": order_index,
                "order_str": order_str,
                "anchors": anchor_list,
            }
    return expected


def list_empty_runs(expected_runs):
    empty_runs = []
    unexpected_empty = []
    for path in sorted(OUTPUTS_DIR.glob("*.out")):
        if not path.is_dir():
            continue
        if any(path.iterdir()):
            continue

        match = NAME_RE.match(path.name)
        if not match:
            unexpected_empty.append(path)
            continue

        combo = match.group("combo")
        order_index = int(match.group("order"))
        run_info = expected_runs.get(combo, {}).get(order_index)
        if run_info is None:
            unexpected_empty.append(path)
            continue
        empty_runs.append(run_info)
    return empty_runs, unexpected_empty


def build_command(run_info):
    unc_cop, unc_aux, c_aux, c_cop, art, ir_past, ir3 = run_info["anchors"]
    exec_dir = f"outputs/out2025/MLUcompliant/{run_info['dir_name']}"

    command = [
        "scala",
        "-cp",
        "induction.jar",
        "induction.Induction",
        "-create",
        "-modelType",
        "hmm",
        "-Options.K",
        "10",
        "-Options.stage2.numIters",
        "1",
        "-Options.stage2.online",
        "True",
        "-Options.onlinePerm",
        "False",
        "-Options.stage2.miniBatches",
        "True",
        "-Options.stage2.miniBatchSize",
        "1",
        "-inputFormat",
        "raw",
        "-inputPaths",
        "sample-data/sents_Brown_shuffled_plusTest.raw",
        "-Options.outputExampleFreq",
        "1000",
        "-log.msPerLine",
        "100",
        "-log.maxIndLevel",
        "100",
        "-overwrite",
        "True",
        "-Options.gradualUnlocking",
        "False",
        "-trainEnd",
        "295245",
        "-testStart",
        "295245",
        "-testEnd",
        "296695",
        "-outputCurrentState",
        "True",
        "-Options.inductionType",
        "morph",
        "-Options.diluteValue",
        "2",
        "-Options.orderStr",
        run_info["order_str"],
        "-Options.grouping",
        "5",
        "-Options.anchor1",
        "True",
        "-Options.PREPAnchorIndex",
        "0",
        "-Options.UNCCOPAnchorIndex",
        str(unc_cop),
        "-Options.UNCAUXAnchorIndex",
        str(unc_aux),
        "-Options.CAUXAnchorIndex",
        str(c_aux),
        "-Options.CCOPAnchorIndex",
        str(c_cop),
        "-Options.ARTAnchorIndex",
        str(art),
        "-Options.IRPASTAnchorIndex",
        str(ir_past),
        "-Options.IR3AnchorIndex",
        str(ir3),
        "-execDir",
        exec_dir,
    ]
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Print or rerun empty outputs/out2025/MLUcompliant trainings."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the missing trainings instead of only printing commands.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit how many missing runs to print or execute.",
    )
    args = parser.parse_args()

    if not OUTPUTS_DIR.is_dir():
        print(f"Missing directory: {OUTPUTS_DIR}", file=sys.stderr)
        return 1

    expected_runs = build_expected_runs()
    empty_runs, unexpected_empty = list_empty_runs(expected_runs)

    print(f"Empty expected runs: {len(empty_runs)}")
    print(f"Unexpected empty directories: {len(unexpected_empty)}")
    if unexpected_empty:
        for path in unexpected_empty:
            print(f"  unexpected: {path.name}")

    if args.max_runs is not None:
        empty_runs = empty_runs[: args.max_runs]

    for run_info in empty_runs:
        combo = run_info["combo"]
        order_index = run_info["order_index"]
        print(f"{run_info['dir_name']}  combo={combo} order={order_index}")

    if not args.run:
        for run_info in empty_runs:
            command = build_command(run_info)
            print(shlex.join(command))
        return 0

    for index, run_info in enumerate(empty_runs, start=1):
        command = build_command(run_info)
        print(f"[{index}/{len(empty_runs)}] {run_info['dir_name']}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
