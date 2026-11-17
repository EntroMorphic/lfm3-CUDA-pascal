"""Balanced warmed scalar baseline/candidate measurements; not release acceptance."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from check_lfm3_cublas_modes import run_owned,digest,environment
from check_lfm3_sanitizer import native_result
from lfm3_owned_process import owned_main


def require_eager(evidence,binary_hash):
    cases=evidence.get('cases',[])
    if (evidence.get('binary_sha256')!=binary_hash or len(cases)!=4 or
        {c.get('case'):c.get('activation_checks') for c in cases}!={'baseline':4240,'arithmetic':530,'window257':1060,'single':1060} or
        any(c.get('max_relative_rms')!=0 or c.get('max_logit_abs')!=0 or
            any(c.get(k) is not True for k in ('teacher_argmax_exact','greedy_exact','reset_exact_logits')) for c in cases)):
        raise ValueError('need unchanged6890 eager checks bound to this binary')


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','baseline-evidence','candidate-evidence','checkpoint','request','expected-result','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,default=20)
    a=p.parse_args()
    if not 4<=a.samples<=20 or a.samples%2:p.error('even samples in4..20 required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    for name in ('baseline','candidate','baseline_evidence','candidate_evidence','checkpoint','request','expected_result','library_dir','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=False);pins={};report=dict(status='incomplete',full_acceptance=False,records=[],warmup_runs=1,order='AB/BA alternating',samples_per_binary=a.samples)
    def pin(path):pins[str(path)]=digest(path)
    def save():(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (a.baseline,a.candidate,a.baseline_evidence,a.candidate_evidence,a.request,a.expected_result,Path(__file__),Path(__file__).with_name('lfm3_owned_process.py'),Path(__file__).with_name('check_lfm3_sanitizer.py')):pin(path)
        for binary,evidence in ((a.baseline,a.baseline_evidence),(a.candidate,a.candidate_evidence)):require_eager(json.loads(evidence.read_text()),digest(binary))
        original=json.loads(a.request.read_text());request=dict(original,warmup_runs=1,check_reset=False)
        if original.get('trace_directory') or original.get('teacher_ids') or original.get('do_sample'):raise ValueError('untraced greedy generation fixture required')
        native_result(a.expected_result,request)
        rp=a.output/'benchmark.request.json';rp.write_text(json.dumps(request,indent=2)+'\n');pin(rp)
        manifest_path=a.candidate.parent.parent/'manifest.json';manifest=json.loads(manifest_path.read_text());pin(manifest_path)
        for name,r in manifest['files'].items():
            path=manifest_path.parent/name
            if digest(path)!=r['candidate_sha256']:raise ValueError('candidate source identity differs')
            pin(path)
        generated=manifest_path.parent/manifest.get('generated_header','native/tests/lfm3_wmma_runtime_generated.cuh')
        if digest(generated)!=manifest['generated_header_sha256']:raise ValueError('generated kernel header differs')
        pin(generated)
        for path in a.library_dir.glob('*.so*'):
            if path.is_file():pin(path.resolve())
        env=environment(a.library_dir,a.library_dir);report['start_sha256']=pins.copy();save()
        for round in range(a.samples):
            for label in (('baseline','candidate') if round%2==0 else ('candidate','baseline')):
                d=a.output/f'{round:02d}-{label}';d.mkdir();binary=getattr(a,label)
                cmd=[str(binary),'--experimental',str(a.checkpoint),str(rp),str(d/'response.json')]
                r=dict(round=round,label=label,command=cmd,status='incomplete');report['records'].append(r);save();start=time.monotonic()
                code,timed=run_owned(cmd,env,d/'application.log',60)
                r.update(returncode=code,timed_out=timed,wall_seconds=time.monotonic()-start,group_cleanup_complete=True)
                if timed or code:raise ValueError('bounded benchmark application failed')
                native_result(d/'response.json',request,json.loads(a.expected_result.read_text()))
                result=json.loads((d/'response.json').read_text());r['application_exact']=True
                if result.get('warmup_runs')!=1 or result.get('tracing') is not False:raise ValueError('warmup/trace policy differs')
                for field in ('prefill_seconds','decode_seconds'):
                    value=result[field]
                    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('invalid timing')
                    r[field]=value
                if type(result.get('decode_forward_calls')) is not int or result['decode_forward_calls']!=len(result['generated_ids'])-1:raise ValueError('decode workload differs')
                r['decode_forward_calls']=result['decode_forward_calls'];r['status']='measured';save()
        report['metrics']={}
        for label in ('baseline','candidate'):
            rows=[r for r in report['records'] if r['label']==label]
            report['metrics'][label]=dict(median_prefill_seconds=statistics.median(r['prefill_seconds'] for r in rows),median_decode_seconds=statistics.median(r['decode_seconds'] for r in rows),samples=len(rows))
        report['paired_candidate_over_baseline_decode']=statistics.median(next(r['decode_seconds'] for r in report['records'] if r['round']==i and r['label']=='candidate')/next(r['decode_seconds'] for r in report['records'] if r['round']==i and r['label']=='baseline') for i in range(a.samples))
        report['status']='measurement-completed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],metrics=report['metrics'],paired_candidate_over_baseline_decode=report['paired_candidate_over_baseline_decode'],full_acceptance=False)))
    return 0 if report['status']=='measurement-completed' else 86


if __name__=='__main__':raise SystemExit(main())
