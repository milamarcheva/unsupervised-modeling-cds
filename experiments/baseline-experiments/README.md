# Baseline experiments

This directory runs the existing `unsupervised-modeling` toolkit without editing its source, saved runs, or the thesis chapters. The original one-pass results are preserved. A sequential queue trains English and Bulgarian for 20 passes each and scores passes 5, 10, and 20. The first-pass gold-category/cluster contingency counts and all three evaluation scores must reproduce the original baseline before further results are published. Any individual token differences are recorded in the verification report.

## Files

- `hmm_iterations_table.tex`: table for insertion into the thesis; pending cells are replaced automatically as checkpoints finish.
- `hmm_iterations_setup.tex`: experimental-setup and initialisation description, with the existing bibliography key for Liang and Klein (2009).
- `status.json` and `queue.log`: progress and errors.
- `results_with_logz.tsv` and `results_with_logz.json`: table data, including primary metric training logZ and source paths. The toolkit records logZ to three decimal places.
- `results.tsv` and `results.json`: original queue exports of the category-recovery scores.
- `publish_table.py`: adds the saved checkpoint training logZ values and publishes the table; `--watch` follows the active queue without restarting training.
- `training_logz_verification.json` and `table_status.json`: training-score reproduction checks and table publication status.
- `manifest.json` and `commands.txt`: configuration, checksums, and exact training commands.
- `provenance/`: snapshots of the existing toolkit, Scala runtime library, evaluator, and original baseline options/predictions.
- `runs/`: new training outputs; existing run directories are never overwritten.

The input corpora and gold labels are read from `/Users/milamarcheva/Desktop/unsupervised-modeling`. Outputs are written only here. Training uses the same settings as each original run, apart from the number of iterations and logging/output paths. One JVM runs at a time, with a 4 GB heap limit and two available JVM processors. Minibatch size remains 1 and the existing seed remains 1; no parameter grid or additional random restarts are run.

## Initialisation source

Liang and Klein (2009), Section 4 and footnote 3, motivate neutral initialisation with noise to break symmetry: https://aclanthology.org/N09-1069.pdf . The repository's `AProblem.scala` uses `(1 + U)^0.001`, then adds the saved initialisation smoothing of `0.01` and normalises. This is the toolkit's near-uniform random initialisation, not the exact exponential expression printed in the paper. The setup text states this distinction explicitly.

## Evaluation

The scorer is the repository's existing `eval_outputs.py`, snapshotted before launch. It verifies sentence/token alignment, uses optimal one-to-one mapping, and computes VI in bits. Iteration filenames are zero-based: passes 1, 5, 10, and 20 correspond to suffixes 0, 4, 9, and 19. Original one-pass results are never replaced by rerun values.

The initial English attempt was stopped by an unnecessarily strict check requiring identical token assignments. Two adjacent tokens with gold category P swapped cluster labels 3 and 5; all gold-category/cluster contingency counts and all evaluation scores were identical. The displayed initial parameters also matched the saved original. This does not establish bitwise reproduction of training, but it does reproduce the baseline at the level of every category-recovery measure reported here. The stopped attempt and its logs are preserved for audit. The validation now requires identical contingency counts and scores, rather than identical token assignments; the model, seed, and training settings are unchanged.

Training logZ is the primary table metric. Original one-pass values come from the preserved `output.map` field `train.logZ`. New values come from `stage2.train.performance.4`, `.9`, and `.19`, field `logZ`, for passes 5, 10, and 20. These are the same metric: the token-normalised training score accumulated during the indicated online-EM pass. They are not a fresh evaluation of the complete training corpus under the final fixed model. The caption and setup text state this definition. No convergence or improvement claim is assumed in advance.

Training was already running when logZ was requested. A separate `publish_table.py --watch` process follows the existing result files and publishes the expanded table without altering the training process. Its writes are atomic and it restores the expanded table if the already-loaded older queue writer temporarily emits its original layout. Future queue launches use the updated reporting function directly. The completion monitor also runs the publisher before reporting results, ensuring that all completed rows contain logZ.

## Execution

Use the bundled Python runtime to run `run_queue.py`. `--prepare-only` creates the provenance snapshots and initial table without starting training. A file lock prevents duplicate queue processes. If a run already exists but is incomplete, the queue stops rather than overwriting it; diagnose the recorded error before retrying.

The original model and dataset are assumed to remain unchanged while the queue runs; corpus checksums are verified. Do not edit the generated table during training, because checkpoint processing regenerates it. Copy the completed table into Overleaf when the status is `complete`.

## Location after completion

The completed experiments now live in `/Users/milamarcheva/Desktop/unsupervised-modeling/experiments/baseline-experiments`. They were originally run in `/Users/milamarcheva/Downloads/PhDThesis_Mila_01102026/experiments/hmm_iterations_20261004` using the code and datasets in the Desktop repository. The original location is a symbolic link to this directory so earlier file links still work. All original files were verified by SHA-256 immediately after the move. The working manifest and commands now use the current location; the exact original manifest and commands are preserved in `provenance/original_execution_manifest.json` and `provenance/original_execution_commands.txt`. Historical run logs and options retain their original recorded paths. `relocation.json` records the move and original file hashes.
