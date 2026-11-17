"""Screen explicit cuBLASLt algorithms on real layer-0 inputs; no production adoption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time

from check_lfm3_cublas_modes import run_owned,digest,environment
from lfm3_owned_process import owned_main


def tensor(checkpoint,suffix):
    with checkpoint.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if size>100000000:raise ValueError('unbounded safetensors header')
        raw=f.read(size);header=json.loads(raw)
        names=[n for n in header if n.endswith(suffix)]
        if len(names)!=1:raise ValueError('ambiguous tensor name')
        entry=header[names[0]];lo,hi=entry['data_offsets']
        if entry['dtype']!='BF16' or hi-lo>100000000 or lo<0 or hi<=lo:raise ValueError('unexpected weight tensor')
        f.seek(8+size+lo);data=f.read(hi-lo)
        if len(data)!=hi-lo:raise ValueError('truncated weights')
    return data,dict(name=names[0],shape=entry['shape'],header_sha256=hashlib.sha256(raw).hexdigest(),tensor_sha256=hashlib.sha256(data).hexdigest())


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference','binary','sanitizer','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-candidates',type=int,default=8)
    p.add_argument('--scan-only',action='store_true',help='numerical screening only; no sanitizer credit')
    p.add_argument('--step',type=int,choices=(0,1),help='explicit fixture subset, still not acceptance')
    p.add_argument('--candidate-index',type=int,action='append',help='test named exact scan indices, not kernel filtering')
    args=p.parse_args()
    if not 1<=args.max_candidates<=16:p.error('bounded candidate count required')
    if args.candidate_index and (len(args.candidate_index)>16 or any(i<0 for i in args.candidate_index)):p.error('bounded candidate indices required')
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    for name in ('checkpoint','reference','binary','sanitizer','library_dir','output'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False);pins={}
    report=dict(status='incomplete',full_acceptance=False,scan_only=args.scan_only,records=[],scope='layer0 projection only; not full eager acceptance')
    def pin(path):pins[str(path)]=digest(path)
    def save():(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    env=environment(args.library_dir,args.library_dir)
    def run(cmd,directory,seconds=60):
        start=time.monotonic();code,timed=run_owned(cmd,env,directory/'application.log',seconds)
        result=dict(command=cmd,returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
        if timed or code not in (0,86):
            report['failed_execution']=result;save();raise ValueError('bounded tool/application did not complete')
        return result
    try:
        for path in (args.binary,args.sanitizer,Path(__file__),Path(__file__).with_name('lfm3_owned_process.py'),Path(__file__).resolve().parents[1]/'native/probes/lfm3_lt_policy_probe.cu'):pin(path)
        ldd=shutil.which('ldd')
        if not ldd:raise ValueError('ldd unavailable')
        code,timed=run_owned([ldd,str(args.binary)],env,args.output/'ldd.log',10)
        text=(args.output/'ldd.log').read_text()
        if code or timed or 'not found' in text:raise ValueError('dependencies unresolved')
        deps={m[1]:Path(m[2]).resolve() for m in re.finditer(r'(\S+) => (/\S+) \(',text)}
        if not Path(deps['libcublasLt.so.13']).is_relative_to(args.library_dir):raise ValueError('wrong cuBLASLt')
        for path in deps.values():pin(path)
        report['dependencies']={k:str(v) for k,v in deps.items()}
        weight,meta=tensor(args.checkpoint,'model.language_model.layers.0.conv.in_proj.weight')
        if meta['shape']!=[6144,2048]:raise ValueError('wrong projection shape')
        report['weight']=meta;(args.output/'weights.bf16').write_bytes(weight);pin(args.output/'weights.bf16')
        for step in ((args.step,) if args.step is not None else (1,0)):
            d=args.output/f'step-{step}';d.mkdir()
            x=args.reference/f'{step}.layers.0.operator_norm.bf16';expected=args.reference/f'{step}.layers.0.conv.in_proj.bf16'
            pin(x);pin(expected);rows=x.stat().st_size//4096
            if x.stat().st_size!=rows*4096 or expected.stat().st_size!=rows*6144*2:raise ValueError('trace shape mismatch')
            args_base=[str(args.binary),str(rows),'2048','6144',str(args.output/'weights.bf16'),str(x),str(expected)]
            scan=d/'scan';scan.mkdir();record=dict(step=step,rows=rows,candidates=[]);report['records'].append(record);report['start_sha256']=pins.copy();save()
            record['scan']=run([*args_base,str(scan/'output')],scan)
            events=[json.loads(x) for x in (scan/'application.log').read_text().splitlines() if x.startswith('{')]
            if len([x for x in events if x.get('complete')])!=1:raise ValueError('missing scan completion')
            record['scan_events']=events
            exact=[x['candidate'] for x in events if x.get('supported') and x.get('mismatches')==0 and x.get('consumer_mismatches')==0]
            record['exact_candidate_indices']=exact
            if args.scan_only:save();continue
            selected=exact[:args.max_candidates] if not args.candidate_index else args.candidate_index
            if any(index not in exact for index in selected):raise ValueError('selected algorithm is not numerically exact')
            for index in selected:
                algo=scan/f'output.{index}.algo.bin';pin(algo);r=dict(index=index,tools={});record['candidates'].append(r);report['start_sha256']=pins.copy();save()
                for tool in ('initcheck','synccheck'):
                    case=d/f'candidate-{index}-{tool}';case.mkdir()
                    cmd=[str(args.sanitizer),'--tool',tool,'--error-exitcode','86','--print-limit','4','--log-file',str(case/'sanitizer.log'),*args_base,str(case/'output'),str(algo)]
                    result=run(cmd,case);r['tools'][tool]=result
                    log=(case/'sanitizer.log').read_text();totals=re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$',log)
                    result['errors']=int(totals[0]) if len(totals)==1 else None;result['internal_error']='Internal Sanitizer Error' in log
                    for kind in ('producer','consumer'):
                        actual=case/f'output.0.{kind}.bf16'
                        if actual.read_bytes()!=expected.read_bytes():raise ValueError('instrumented candidate output changed')
                    result['whole_output_exact']=True
                    result['clean']=result['returncode']==0 and result['errors']==0 and not result['internal_error']
                    save()
                r['small_case_clean']=all(t['clean'] for t in r['tools'].values());save()
        report['status']='screening-completed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False)))


if __name__=='__main__':main()
