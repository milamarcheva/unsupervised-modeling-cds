#!/usr/bin/env python3
"""Run the existing HMM at 20 passes; score checkpoints without changing its code."""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/milamarcheva/Desktop/unsupervised-modeling')
JAVA = Path('/opt/homebrew/Cellar/openjdk/21.0.2/libexec/openjdk.jdk/Contents/Home/bin/java')
SCALA = Path('/Users/milamarcheva/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala-library/2.10.2/scala-library-2.10.2.jar')
PASSES = (1, 5, 10, 20)
LANGUAGES = {
    'en': {'name': 'English', 'K': 10, 'data': 'sample-data/sents_Brown_shuffled_plusTest.raw',
           'gold': 'sample-data/en_categoryinduction_gold/all_tags.txt', 'train': 295245, 'test': 1450},
    'bg': {'name': 'Bulgarian', 'K': 8, 'data': 'sample-data/bg/bg_hmm_tokenised_morphLemma_duplicates_shuffled_seed42_plusTest.txt',
           'gold': 'sample-data/bg_categoryinduction_gold/all_tags.txt', 'train': 239244, 'test': 350},
}

def atomic(path, text):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)

def save_json(path, data):
    atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def parse_options(text):
    return dict(line.split('\t', 1) for line in text.splitlines() if '\t' in line)

def scorer():
    spec = importlib.util.spec_from_file_location('hmm_evaluation_snapshot', ROOT / 'provenance/eval_outputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def scores(module, pred, gold):
    module.validate_alignment(pred, gold, Path('prediction'))
    vi, vi_norm = module.variation_of_information(module.flatten(gold), module.flatten(pred))
    return {'many_to_one': module.mapping_accuracy(pred, gold, 'many2one'),
            'one_to_one': module.mapping_accuracy(pred, gold, 'one2one'),
            'vi': vi, 'vi_norm': vi_norm}

def baseline(lang):
    if lang == 'en':
        archive = REPO / 'outputs/23042024.zip'
        prefix = '23042024/shuffledNormalTest.out/'
        with zipfile.ZipFile(archive) as z:
            return {name: z.read(prefix + name).decode('utf-8') for name in
                    ('options.map', 'output.map', 'stage2.current.test.pred.0')}
    path = REPO / 'outputs/bg/20260622/ShuffledNoStagesNoGradualUnlocking.out'
    return {name: (path / name).read_text() for name in
            ('options.map', 'output.map', 'stage2.current.test.pred.0')}

def make_command(lang, options):
    info = LANGUAGES[lang]
    keys = ['modelType', 'K', 'inductionType', 'inputFormat', 'trainStart', 'trainEnd',
            'testStart', 'testEnd', 'initType', 'initNoise', 'initSmoothing', 'initRandom',
            'onlinePerm', 'onlinePermRandom', 'numThreads', 'outputExampleFreq']
    values = {'Options.' + key: options['Options.' + key] for key in keys}
    for key, value in options.items():
        if key.startswith(('Options.stage1.', 'Options.stage2.')):
            values[key] = value
    values.update({'Options.stage2.numIters': '20', 'Options.inputPaths': str(REPO / info['data']),
                   'Options.outputIterFreq': '1', 'Options.outputCurrentState': 'true',
                   'Options.outputFullPred': 'false', 'Options.numOutputParams': '10',
                   'Options.useBG': str(lang == 'bg').lower(),
                   'log.maxIndLevel': '2', 'log.msPerLine': '10000',
                   'exec.execDir': str(ROOT / 'runs' / (lang + '_20passes.out'))})
    cmd = [str(JAVA), '-Xmx4g', '-XX:ActiveProcessorCount=2', '-Dfile.encoding=UTF-8',
           '-cp', str(ROOT / 'provenance/induction.jar') + os.pathsep + str(ROOT / 'provenance/scala-library-2.10.2.jar'),
           'induction.Induction', '-create']
    for key, value in values.items():
        cmd.extend(['-' + key, value])
    return cmd

def write_table(results):
    from publish_table import publish
    publish(results)

def prepare():
    if (ROOT / 'manifest.json').exists():
        return json.loads((ROOT / 'manifest.json').read_text())
    provenance = ROOT / 'provenance'
    provenance.mkdir(exist_ok=True)
    (ROOT / 'runs').mkdir(exist_ok=True)
    for source, name in [(REPO / 'induction.jar', 'induction.jar'),
                         (REPO / 'eval_outputs.py', 'eval_outputs.py'), (SCALA, SCALA.name)]:
        shutil.copy2(source, provenance / name)
    module = scorer()
    manifest = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'repo': str(REPO),
                'jar_sha256': digest(provenance / 'induction.jar'), 'languages': {}}
    results = {}
    for lang, info in LANGUAGES.items():
        files = baseline(lang)
        for name, contents in files.items():
            atomic(provenance / (lang + '_original_' + name), contents)
        opts = parse_options(files['options.map'])
        assert opts['Options.stage2.numIters'] == '1'
        assert opts['Options.stage2.miniBatchSize'] == '1'
        assert opts['Options.inductionType'] == 'normal'
        assert int(opts['Options.trainEnd']) == info['train']
        gold = module.read_lines(REPO / info['gold'])
        assert len(gold) == info['test']
        with (REPO / info['data']).open() as f:
            n = sum(1 for line in f if line.strip())
        assert n == info['train'] + info['test'], (lang, n)
        pred = [line.strip() for line in files['stage2.current.test.pred.0'].splitlines() if line.strip()]
        row = scores(module, pred, gold)
        row['source'] = 'original_saved_baseline'
        results[lang] = {'1': row}
        manifest['languages'][lang] = {**info, 'data_sha256': digest(REPO / info['data']),
                                     'gold_sha256': digest(REPO / info['gold']),
                                     'command': make_command(lang, opts)}
    save_json(ROOT / 'manifest.json', manifest)
    save_json(ROOT / 'results.json', results)
    write_table(results)
    atomic(ROOT / 'commands.txt', '\n\n'.join(shlex.join(v['command']) for v in manifest['languages'].values()) + '\n')
    save_json(ROOT / 'status.json', {'status': 'prepared', 'languages': {lang: 'queued' for lang in LANGUAGES}})
    return manifest

def run():
    lock = (ROOT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare()
    module = scorer()
    results = json.loads((ROOT / 'results.json').read_text())
    state = {'status': 'running', 'pid': os.getpid(), 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'languages': {lang: 'queued' for lang in LANGUAGES}, 'checkpoints': {lang: [] for lang in LANGUAGES}}
    child = None
    try:
        for lang, config in manifest['languages'].items():
            if all(str(n) in results[lang] for n in PASSES):
                state['languages'][lang] = 'complete'
                continue
            assert digest(REPO / config['data']) == config['data_sha256'], 'Training data changed'
            assert digest(REPO / config['gold']) == config['gold_sha256'], 'Gold data changed'
            out = ROOT / 'runs' / (lang + '_20passes.out')
            if out.exists():
                raise RuntimeError(f'Refusing to overwrite an existing run: {out}')
            gold = module.read_lines(REPO / config['gold'])
            original = module.read_lines(ROOT / 'provenance' / (lang + '_original_stage2.current.test.pred.0'))
            state['languages'][lang] = 'running'
            state['active_language'] = lang
            save_json(ROOT / 'status.json', state)
            print(f'Starting {lang}: 20 passes, minibatch 1, seed 1', flush=True)
            with (ROOT / (lang + '_console.log')).open('w') as console:
                child = subprocess.Popen(config['command'], cwd=REPO, stdout=console, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save_json(ROOT / 'status.json', state)
                processed = set()
                while True:
                    code = child.poll()
                    for count in PASSES:
                        if count in processed:
                            continue
                        marker = out / f'stage2.current.test.performance.{count-1}'
                        path = out / f'stage2.current.test.pred.{count-1}'
                        if not marker.exists() or not marker.stat().st_size or not path.exists():
                            continue
                        pred = module.read_lines(path)
                        row = scores(module, pred, gold)
                        if count == 1:
                            module.validate_alignment(original, gold, Path('original prediction'))
                            pred_tokens, original_tokens = module.flatten(pred), module.flatten(original)
                            gold_tokens = module.flatten(gold)
                            mismatches = sum(a != b for a, b in zip(pred_tokens, original_tokens))
                            matching_contingency = (Counter(zip(gold_tokens, pred_tokens)) ==
                                                    Counter(zip(gold_tokens, original_tokens)))
                            verification = {'token_mismatches': mismatches, 'token_count': len(gold_tokens),
                                            'matching_contingency': matching_contingency,
                                            'criterion': 'Exact gold-category/cluster contingency counts and all three scores must match the saved baseline.',
                                            'original': results[lang]['1'], 'rerun': row,
                                            'matching_metrics': all(abs(row[k]-results[lang]['1'][k]) < 1e-10
                                                                    for k in ('many_to_one', 'one_to_one', 'vi'))}
                            verification['accepted'] = matching_contingency and verification['matching_metrics']
                            save_json(ROOT / (lang + '_one_pass_verification.json'), verification)
                            if not verification['accepted']:
                                raise RuntimeError(f'{lang}: first pass does not reproduce the original baseline; comparison paused')
                            print(f'{lang}: first-pass contingency counts and scores reproduce the saved baseline exactly; '
                                  f'{mismatches} token assignments differ', flush=True)
                        else:
                            row['source'] = str(path.relative_to(ROOT))
                            results[lang][str(count)] = row
                            save_json(ROOT / 'results.json', results)
                            write_table(results)
                            print(f'{lang}: {count} passes scored: {row}', flush=True)
                        processed.add(count)
                        state['checkpoints'][lang] = sorted(processed)
                        save_json(ROOT / 'status.json', state)
                    if code is not None:
                        if code != 0 or processed != set(PASSES):
                            raise RuntimeError(f'{lang}: training exited {code}; scored checkpoints {sorted(processed)}. See {lang}_console.log')
                        break
                    time.sleep(5)
            assert digest(REPO / config['data']) == config['data_sha256'], 'Training data changed during the run'
            state['languages'][lang] = 'complete'
            save_json(ROOT / 'status.json', state)
        state['status'] = 'complete'
        state['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        print('Queue complete; LaTeX table contains all requested results.', flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
        state['status'] = 'failed'
        state['error'] = str(error)
        print(f'Queue stopped: {error}', flush=True)
        raise
    finally:
        save_json(ROOT / 'status.json', state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        print('Prepared commands, preserved original one-pass results, and created the LaTeX table.')
    else:
        run()
