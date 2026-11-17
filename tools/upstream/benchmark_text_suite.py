"""End-to-end public CLI text suite: full outputs and separately scoped timings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import statistics
import sys

from benchmark_common import CONTROL_PROMPT, assert_same_output, header_fields, run_saved, sha256, validate_native, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from liquid_inference.cli import MODELS,DEFAULTS


def cli_prefix(executable):
    """Pin the source package via PYTHONPATH, without cwd package shadowing."""
    return [str(executable), '-P', '-m', 'liquid_inference']


def text_diagnostics(generations):
    """Descriptive flags only: repetition is not a coherence/quality score."""
    fractions = []
    for row in generations:
        ids = row['token_ids']
        grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
        fractions.append((len(grams)-len(set(grams)))/len(grams) if grams else 0.)
    return dict(unique_token_streams=len({tuple(g['token_ids']) for g in generations}),
                empty_visible_rows=sum(not g['text'].strip() for g in generations),
                replacement_character_rows=sum('\ufffd' in g['text'] for g in generations),
                repeated_token_4gram_fraction_median=statistics.median(fractions),
                repeated_token_4gram_fraction_by_row=fractions)


def validate_cli(directory, vocabulary, cap=512):
    response = json.loads((directory/'stdout.txt').read_text())
    rows = response['generations']
    if len(rows) != 32:
        raise ValueError('Expected all 32 public CLI generations')
    # --verbose retains the actual native metadata in stderr. Reconstruct token
    # files from the public JSON, then check native FNV/counts/rate independently.
    native = directory/'native-validation'; native.mkdir(exist_ok=False)
    (native/'stdout.txt').write_text((directory/'stderr.txt').read_text())
    for i, row in enumerate(rows):
        (native/f'tokens.{i}.txt').write_text('\n'.join(map(str, row['token_ids']))+'\n')
    result = validate_native(native, 32, vocabulary, cap)
    if (response['tokens_per_second'] != result['rate'] or
            [g['finish_reason'] for g in rows] != result['reasons'] or
            [g['generated_tokens'] for g in rows] != result['counts']):
        raise ValueError('CLI JSON and native metadata disagree')
    fields = header_fields((directory/'stderr.txt').read_text())
    result['prompt_tokens'] = int(fields['batch_prompt_tokens'])
    return response, result


def summarize(runs):
    return {key: dict(median=statistics.median(r[key] for r in runs),
                      minimum=min(r[key] for r in runs), maximum=max(r[key] for r in runs))
            for key in ['rate', 'wall_seconds', 'end_to_end_tokens_per_second']}


def assert_prefix(reference, response):
    if len(reference['generations']) != len(response['generations']):
        raise ValueError('Prior campaign row coverage')
    for old,new in zip(reference['generations'],response['generations']):
        ids=old['token_ids']
        if new['token_ids'][:len(ids)] != ids:
            raise ValueError('Prior 512-cap token prefix changed')
        if old['finish_reason']=='stop_token' and (new['token_ids'] != ids or new['finish_reason']!='stop_token'):
            raise ValueError('Prior natural stop changed')


def collect(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime.resolve()
    uncapped = getattr(args, 'no_token_limit', False)
    report = dict(complete=False, scope='Public CLI end-to-end unprofiled B32 suite; no paired optimization comparison.',
                  platform=platform.platform(), runtime_sha256=sha256(runtime),
                  frontend_cli_sha256=sha256(ROOT/'liquid_inference/cli.py'),
                  settings=dict(prompt=CONTROL_PROMPT, batch_size=32, temperature=.528, seed=42,
                                context_length=8192, max_new_tokens=None if uncapped else 512, top_k=0, top_p=1., min_p=0.,
                                repetition_penalty=1., warmups=1, measured_runs=5,
                                reference_math=DEFAULTS['reference_math'],
                                graph_policy='normal CLI default; actual mode/setup reported per invocation'),
                  models=[])
    def save(): write_json(out/'campaign.json', report)
    def idle(label):
        run_saved(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'], out/label, 30)
        if (out/label/'stdout.txt').read_text().strip():
            raise ValueError('Existing GPU compute process; refusing overlapping benchmark')
    try:
        environment={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT)}
        preflight='import hashlib,json;from pathlib import Path;import liquid_inference.cli as c;print(json.dumps(dict(cli_file=c.__file__,cli_sha256=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest())))'
        run_saved([sys.executable,'-P','-c',preflight],out/'cli-import-preflight',60,env=environment)
        binding=json.loads((out/'cli-import-preflight/stdout.txt').read_text())
        if Path(binding['cli_file']).resolve()!=(ROOT/'liquid_inference/cli.py').resolve() or binding['cli_sha256']!=report['frontend_cli_sha256']:
            raise ValueError('Benchmark child CLI is not the selected source package')
        report['child_cli_binding']=binding
        idle('idle-start')
        run_saved(['nvidia-smi','-q'], out/'hardware', 30)
        run_saved([sys.executable,'-m','pip','freeze'], out/'python-packages', 60)
        for alias in MODELS:
            name = 'LFM2.5-'+alias; model = args.models.resolve()/name
            row = dict(model=name, complete=False, runs=[], warmups=[])
            report['models'].append(row); save()
            try:
                idle('idle-'+name)
                # Identity work is outside every measured invocation.
                paths = sorted(p for p in model.iterdir() if p.is_file() and
                               (p.suffix in ('.json','.jinja','.safetensors')))
                if not any(p.suffix == '.safetensors' for p in paths):
                    raise ValueError('Missing checkpoint weights')
                row['checkpoint_files'] = {p.name:dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in paths}
                config = json.loads((model/'config.json').read_text())
                prior=None
                if getattr(args,'prefix_reference',None):
                    reference_path=args.prefix_reference/name/'r1/stdout.txt'
                    prior=json.loads(reference_path.read_text())
                    parent=out/name;parent.mkdir(exist_ok=True)
                    write_json(parent/'prefix-reference.json',prior)
                    row['prefix_reference_sha256']=sha256(parent/'prefix-reference.json')
                reference = None; response = None
                for repeat in range(6):
                    label = 'warmup' if repeat == 0 else f'r{repeat}'
                    parent=out/name; parent.mkdir(exist_ok=True)
                    directory = parent/label
                    command = [*cli_prefix(sys.executable),'run',str(model),CONTROL_PROMPT,
                               '--runtime',str(runtime),'--batch-size','32','--temperature','0.528',
                               '--seed','42','--context-length','8192',
                               '--top-k','0','--top-p','1','--min-p','0','--repetition-penalty','1',
                               '--json','--verbose']
                    if uncapped: command += ['--no-token-limit']
                    else: command += ['--max-tokens','512','--timeout','1800']
                    process = run_saved(command, directory, None if uncapped else 1860, env=environment)
                    response, result = validate_cli(directory, config['vocab_size'], 8192 if uncapped else 512)
                    if uncapped and 'token_limit' in result['reasons']:
                        raise ValueError('Token cap encountered in uncapped run')
                    if prior is not None: assert_prefix(prior,response)
                    expected = report['settings']
                    for key in ['batch_size','temperature','seed','context_length','max_new_tokens','top_k','top_p','min_p','repetition_penalty','reference_math']:
                        if response['settings'][key] != expected[key]:
                            raise ValueError('Public CLI settings mismatch: '+key)
                    if reference is None: reference = result
                    assert_same_output(reference, result)
                    result.update(label=label, wall_seconds=process['wall_seconds'],
                                  end_to_end_tokens_per_second=result['total_tokens']/process['wall_seconds'])
                    row['warmups' if repeat == 0 else 'runs'].append(result)
                    row.update(prompt_mode=response['prompt_mode'], prompt_tokens=result['prompt_tokens'])
                    save()
                    print(name,label,f"{result['rate']:.3f} decode tok/s; {result['wall_seconds']:.3f}s full CLI; exact rows",flush=True)
                row.update(complete=True, metrics=summarize(row['runs']),
                           diagnostics=text_diagnostics(response['generations']))
            except Exception as error:
                row['error']=repr(error)
                print(name,'FAILED',repr(error),flush=True)
            save()
        idle('idle-end')
        report['complete'] = all(row['complete'] for row in report['models'])
        save()
        return 0 if report['complete'] else 1
    except BaseException as error:
        report['error']=repr(error); raise
    finally: save()


def fence(text):
    import re
    length = max([len(x) for x in re.findall(r'`+', text)]+[2])+1
    mark='`'*length
    return mark+'text\n'+text+'\n'+mark+'\n'


def audit(root):
    """Read-only replay of saved public/native evidence before publication."""
    report=json.loads((root/'campaign.json').read_text())
    uncapped=report.get('settings',{}).get('max_new_tokens',512) is None
    if (not report['complete'] or [r['model'] for r in report['models']] !=
            ['LFM2.5-'+alias for alias in MODELS]):
        raise ValueError('Incomplete text checkpoint census')
    invocations=streams=tokens=0
    for row in report['models']:
        config_path=ROOT/'models'/row['model']/'config.json'
        if sha256(config_path) != row['checkpoint_files']['config.json']['sha256']:
            raise ValueError('Independent local config identity mismatch')
        vocabulary=json.loads(config_path.read_text())['vocab_size']
        prior=None
        if 'prefix_reference_sha256' in row:
            reference_path=root/row['model']/'prefix-reference.json'
            if sha256(reference_path) != row['prefix_reference_sha256']:
                raise ValueError('Prior campaign identity mismatch')
            prior=json.loads(reference_path.read_text())
        if not row['complete'] or len(row['warmups']) != 1 or len(row['runs']) != 5:
            raise ValueError('Incomplete warmup/repetition coverage')
        reference=None
        for index,saved in enumerate(row['warmups']+row['runs']):
            label='warmup' if index==0 else f'r{index}'
            if saved['label'] != label: raise ValueError('Repetition order')
            directory=root/row['model']/label
            process=json.loads((directory/'process.json').read_text())
            if not process['complete'] or process['returncode'] != 0 or process['timed_out']:
                raise ValueError('Failed process in completed campaign')
            response=json.loads((directory/'stdout.txt').read_text())
            if prior is not None: assert_prefix(prior,response)
            native=directory/'native-validation'
            if (native/'stdout.txt').read_bytes() != (directory/'stderr.txt').read_bytes():
                raise ValueError('Native validation log is not original stderr')
            result=validate_native(native,32,vocabulary,8192 if uncapped else 512)
            if uncapped:
                if response['settings']['max_new_tokens'] is not None or 'token_limit' in result['reasons']:
                    raise ValueError('Unexpected token cap in uncapped evidence')
                command=json.loads((directory/'command.json').read_text())
                if '--no-token-limit' not in command or '--max-tokens' in command or '--max-new-tokens' in command:
                    raise ValueError('Uncapped CLI invocation arguments')
            if len(response['generations']) != 32: raise ValueError('Missing public outputs')
            for i,g in enumerate(response['generations']):
                ids=list(map(int,(native/f'tokens.{i}.txt').read_text().split()))
                if (ids != g['token_ids'] or len(ids) != g['generated_tokens'] or
                        g['finish_reason'] != result['reasons'][i]):
                    raise ValueError('Public JSON/native token disagreement')
            for key in result:
                if result[key] != saved[key]: raise ValueError('Summary/native disagreement: '+key)
            if response['tokens_per_second'] != result['rate']: raise ValueError('Public rate mismatch')
            if (saved['wall_seconds'] != process['wall_seconds'] or
                    saved['end_to_end_tokens_per_second'] != result['total_tokens']/process['wall_seconds']):
                raise ValueError('Full-process timing mismatch')
            if reference is None: reference=result
            assert_same_output(reference,result)
            invocations+=1;streams+=32;tokens+=result['total_tokens']
        if summarize(row['runs']) != row['metrics']: raise ValueError('Median/range mismatch')
        first=json.loads((root/row['model']/'r1/stdout.txt').read_text())
        if text_diagnostics(first['generations']) != row['diagnostics']:
            raise ValueError('Text diagnostics mismatch')
    return dict(complete=True,models=len(report['models']),invocations=invocations,
                streams=streams,generated_tokens_including_warmups=tokens,
                campaign_sha256=sha256(root/'campaign.json'))


def render(root, output, review=None):
    checked=audit(root)
    report=json.loads((root/'campaign.json').read_text())
    uncapped=report.get('settings',{}).get('max_new_tokens',512) is None
    cap_description='**no new-token cap** (EOS/stop-token or context exhaustion only)' if uncapped else 'cap 512 new tokens per row'
    hardware_path=root/'hardware/stdout.txt'
    hardware={line.partition(':')[0].strip():line.partition(':')[2].strip()
              for line in hardware_path.read_text().splitlines() if ':' in line} if hardware_path.is_file() else {}
    lines=['# End-to-end text generation review\n',
           f"Campaign complete: **{report['complete']}**. All 15 README text checkpoints; public CLI, normal default execution.\n",
           f"Hardware: {hardware.get('Product Name','unrecorded')}; driver {hardware.get('Driver Version','unrecorded')}; CUDA {hardware.get('CUDA Version','unrecorded')}. Sequential runs; idle-device checks before each model. No clock/power controls changed. The report date is the review date; Thor's wall clock is skewed, so cross-host timestamps are not used for elapsed timing.\n",
           '> '+CONTROL_PROMPT+'\n',
           f'Settings: BF16 weights/activations/KV, B32, temperature 0.528, seed 42, context capacity 8192, {cap_description}; top-k 0, top-p 1, min-p 0, repetition penalty 1. Official chat templates, raw text for known base checkpoints. One full-length warmup, five measured fresh CLI processes per model. No profiler or S1 observer.\n',
           '**Termination scope:** no token cap does not guarantee a completed answer. `stop_token` means a configured EOS/stop token was generated; `context_limit` means available context was exhausted, not natural termination. Context capacity is unchanged; repetitive models may use all remaining positions.\n' if uncapped else '',
           '**Metric boundaries:** decode tok/s is native aggregate throughput (not per-request speed); excludes loading/prefill. Full CLI wall time includes Python startup, local tokenizer/template, model loading, prefill, graph setup, generation, text decoding and JSON serialization. E2E tok/s = actual aggregate generated tokens / full CLI process wall time; not a persistent-service rate. OS file caches are not flushed. Warmups excluded from summaries. No paired baseline or speedup claim.\n',
           '**Unavailable:** streaming TTFT/inter-token latency, isolated prefill/load durations, peak GPU allocation and hardware-counter utilization are not exposed by this path and are not inferred. Coherence is qualitative; exact repeatability and repetition flags are not correctness/quality scores. A cap stop is not a completed answer.\n',
           f"Runtime SHA-256: `{report['runtime_sha256']}`. Raw evidence: `{root}`. Read-only audit: {checked['invocations']} invocations / {checked['streams']} streams / {checked['generated_tokens_including_warmups']:,} tokens including warmups. All 32 first-measured outputs per model are included below, without cherry-picking. Token IDs, native logs, argv, checkpoint hashes and all six invocations remain in the raw evidence.\n",
           '## Review findings and limitations\n',
           review.read_text() if review else 'No qualitative review notes supplied; inspect the full outputs below.\n',
           '## Measured performance\n',
           '| Model | Decode tok/s median [min–max] | CLI wall s median [min–max] | E2E tok/s median [min–max] | Tokens/run | Stop reasons (rows) | Exact 6/6 |',
           '|---|---:|---:|---:|---:|---|---|']
    for row in report['models']:
        if not row['complete']:
            lines.append(f"| {row['model']} | FAILED | — | — | — | {row.get('error','incomplete')} | no |")
            continue
        m=row['metrics']; r=row['runs'][0]
        def span(key):
            v=m[key]; return f"{v['median']:.2f} [{v['minimum']:.2f}–{v['maximum']:.2f}]"
        reasons=', '.join(f'{k}: {v}' for k,v in sorted(Counter(r['reasons']).items()))
        lines.append(f"| {row['model']} | {span('rate')} | {span('wall_seconds')} | {span('end_to_end_tokens_per_second')} | {r['total_tokens']} | {reasons} | yes |")
    lines += ['\n## Full outputs and descriptive flags\n',
              'Rows are zero-based logical request IDs. Each row has its own deterministic RNG stream. Visible text skips special tokens, but token counts include them. Base outputs are continuations, not necessarily answers; Japanese/thinking variants may differ in language or expose reasoning. Repeated token 4-gram fraction counts duplicate windows / all windows; it is descriptive, not a coherence threshold.\n']
    for row in report['models']:
        lines.append('### '+row['model']+'\n')
        if not row['complete']:
            lines.append('Failure retained: '+row.get('error','incomplete')); continue
        response=json.loads((root/row['model']/'r1/stdout.txt').read_text());d=row['diagnostics']
        lines.append(f"Prompt mode: **{row['prompt_mode']}**, {row['prompt_tokens']} tokens. Attention: `{row['runs'][0]['attention_mode']}`; graph: `{row['runs'][0].get('batch_graph_mode','unreported')}`. Unique streams: {d['unique_token_streams']}/32; empty visible rows: {d['empty_visible_rows']}; U+FFFD rows: {d['replacement_character_rows']}; median repeated 4-gram fraction: {d['repeated_token_4gram_fraction_median']:.3f}.\n")
        counts=row['runs'][0]['counts'];setups=[r['batch_graph_setup_seconds'] for r in row['runs'] if 'batch_graph_setup_seconds' in r]
        lines.append(f"Generated tokens per row min/median/max: {min(counts)}/{statistics.median(counts):g}/{max(counts)}. Reported graph setup median: {statistics.median(setups):.6f} s (included in full CLI wall time).\n" if setups else
                     f"Generated tokens per row min/median/max: {min(counts)}/{statistics.median(counts):g}/{max(counts)}. Graph setup duration not reported.\n")
        for i,g in enumerate(response['generations']):
            label=f"Row {i} — {g['generated_tokens']} tokens; {g['finish_reason']}; repeated 4-grams {d['repeated_token_4gram_fraction_by_row'][i]:.3f}"
            if i: lines.append('<details>\n<summary>'+label+'</summary>\n')
            else: lines.append('**'+label+'**\n')
            lines.append(fence(g['text']))
            if i: lines.append('</details>\n')
    output.write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--runtime',type=Path,required=True);c.add_argument('--models',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c.add_argument('--no-token-limit',action='store_true',help='Omit token and wall-time caps; retain context/EOS stopping.')
    c.add_argument('--prefix-reference',type=Path,help='Require exact saved token prefixes from the previous capped campaign.')
    r=sub.add_parser('render');r.add_argument('--root',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--review-notes',type=Path,help='Bounded qualitative observations, separate from measured metrics.')
    args=p.parse_args()
    if args.command=='collect': return collect(args)
    render(args.root,args.output,args.review_notes);return 0


if __name__=='__main__': raise SystemExit(main())
