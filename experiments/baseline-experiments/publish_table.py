#!/usr/bin/env python3
"""Publish the requested checkpoint metrics, including the toolkit's training logZ.

The running training queue is left untouched. This reporter also repairs the table
if the already-loaded older queue writer publishes its previous column layout.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parent
LANGUAGES = {'en': ('English', 10), 'bg': ('Bulgarian', 8)}
PASSES = (1, 5, 10, 20)


def atomic_if_changed(path, text):
    if path.exists() and path.read_text() == text:
        return False
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', encoding='utf-8', delete=False) as f:
        f.write(text)
        temporary = Path(f.name)
    temporary.replace(path)
    return True


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def logz(path, key):
    fields = dict(line.split('\t', 1) for line in path.read_text().splitlines() if '\t' in line)
    value = float(fields[key])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {key}: {path}')
    return value


def enrich(results):
    enriched = json.loads(json.dumps(results))
    verification = {}
    for lang, rows in enriched.items():
        original_path = ROOT / 'provenance' / f'{lang}_original_output.map'
        original_logz = logz(original_path, 'train.logZ')
        rerun_path = ROOT / 'runs' / f'{lang}_20passes.out' / 'stage2.train.performance.0'
        # A scored checkpoint implies the completed first-pass performance file.
        if (ROOT / f'{lang}_one_pass_verification.json').exists() and rerun_path.exists():
            rerun_logz = logz(rerun_path, 'logZ')
            verification[lang] = {'original_train_logz': original_logz, 'rerun_train_logz': rerun_logz,
                                  'matching_at_saved_precision': original_logz == rerun_logz}
            if original_logz != rerun_logz:
                raise ValueError(f'{lang}: first-pass training logZ differs from the original baseline')
        for count, row in rows.items():
            source = (original_path if count == '1' else
                      ROOT / 'runs' / f'{lang}_20passes.out' / f'stage2.train.performance.{int(count)-1}')
            row['train_logz'] = logz(source, 'train.logZ' if count == '1' else 'logZ')
            row['train_logz_source'] = str(source.relative_to(ROOT))
            for key in ('many_to_one', 'one_to_one', 'vi', 'train_logz'):
                if not math.isfinite(row[key]):
                    raise ValueError(f'Nonfinite metric: {lang}, {count}, {key}')
    return enriched, verification


def latex(results):
    complete = all(str(n) in results[l] for l in LANGUAGES for n in PASSES)
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\begin{tabular}{llrrrrr}', r'\hline',
             r'Language & Passes & States & Train $\log Z$ & M2O (\%) & O2O (\%) & VI \\', r'\hline']
    for lang, (name, states) in LANGUAGES.items():
        for count in PASSES:
            row = results[lang].get(str(count))
            values = (f"{row['train_logz']:.3f} & {100*row['many_to_one']:.2f} & "
                      f"{100*row['one_to_one']:.2f} & {row['vi']:.3f}"
                      if row else r'\multicolumn{4}{c}{Pending}')
            lines.append(f'{name} & {count} & {states} & {values} ' + r'\\')
        lines.append(r'\hline')
    caption = (r'HMM baseline performance after repeated passes through the same morphemically tokenised training corpus. '
               r'Minibatch size is one utterance and initialisation seed is 1 throughout. '
               r'The one-pass rows retain the original baseline results; subsequent rows use checkpoints from a matched 20-pass run for each language. '
               r'Training $\log Z$, the primary metric, is the token-normalised score recorded by the toolkit during the last reported online-EM pass; '
               r'it is not a separate rescore of the training corpus with parameters fixed at the checkpoint. '
               r'M2O and O2O denote many-to-one and one-to-one accuracy on the annotated evaluation data. '
               r'Higher training $\log Z$ and accuracies, and lower variation of information (VI, in bits), are better.')
    if not complete:
        caption += ' Pending denotes an experiment that has not yet been scored.'
    lines += [r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{tab:hmm_iteration_baselines}', r'\end{table}', '']
    return '\n'.join(lines)


def publish(results=None):
    with (ROOT / 'table_publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if results is None:
            results = json.loads((ROOT / 'results.json').read_text())
        enriched, verification = enrich(results)
        changed = atomic_if_changed(ROOT / 'results_with_logz.json', json_text(enriched))
        atomic_if_changed(ROOT / 'training_logz_verification.json', json_text(verification))
        stream = io.StringIO()
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        metrics = ('train_logz', 'many_to_one', 'one_to_one', 'vi', 'vi_norm')
        writer.writerow(['language', 'passes', 'states', *metrics, 'source', 'train_logz_source'])
        for lang, (_, states) in LANGUAGES.items():
            for count in PASSES:
                row = enriched[lang].get(str(count))
                if row:
                    writer.writerow([lang, count, states, *[row[k] for k in metrics],
                                     row['source'], row['train_logz_source']])
        atomic_if_changed(ROOT / 'results_with_logz.tsv', stream.getvalue())
        atomic_if_changed(ROOT / 'hmm_iterations_table.tex', latex(enriched))
        complete = all(str(n) in enriched[l] for l in LANGUAGES for n in PASSES)
        atomic_if_changed(ROOT / 'table_status.json', json_text({
            'status': 'complete' if complete else 'waiting_for_checkpoints',
            'completed_rows': sum(len(rows) for rows in enriched.values()),
            'required_rows': 8, 'primary_metric': 'train_logz'}))
        if changed:
            print(f'Published {sum(len(rows) for rows in enriched.values())}/8 rows with training logZ.', flush=True)
        return complete


def watch():
    with (ROOT / 'table_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'Table reporter started, PID {os.getpid()}.', flush=True)
        while True:
            try:
                complete = publish()
                state = json.loads((ROOT / 'status.json').read_text())
                if state['status'] in ('complete', 'failed'):
                    # The queue writes its final status after its final table write.
                    publish()
                    print(f"Queue {state['status']}; table complete={complete}.", flush=True)
                    return
            except Exception as error:
                atomic_if_changed(ROOT / 'table_status.json', json_text({'status': 'failed', 'error': str(error)}))
                raise
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    watch() if args.watch else publish()
