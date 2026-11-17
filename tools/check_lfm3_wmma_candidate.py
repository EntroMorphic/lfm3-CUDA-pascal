"""Screen native WMMA split orders against real outputs, then sanitize exact cases."""
import argparse
import json
import os
from pathlib import Path
import re
import time
from check_lfm3_lt_candidates import tensor
from check_lfm3_cublas_modes import run_owned,digest,environment
from lfm3_owned_process import owned_main


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference','binary','sanitizer','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('checkpoint','reference','binary','sanitizer','library_dir','output'):setattr(a,name,getattr(a,name).resolve())
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    a.output.mkdir(parents=True,exist_ok=False);pins={};report=dict(status='incomplete',full_acceptance=False,records=[])
    def pin(path):pins[str(path)]=digest(path)
    def save():(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (a.binary,a.sanitizer,Path(__file__),Path(__file__).with_name('check_lfm3_lt_candidates.py'),Path(__file__).with_name('lfm3_owned_process.py'),Path(__file__).resolve().parents[1]/'native/probes/lfm3_wmma_policy_probe.cu'):pin(path)
        weight,meta=tensor(a.checkpoint,'model.language_model.layers.0.conv.in_proj.weight')
        if meta['shape']!=[6144,2048]:raise ValueError('weight shape differs')
        report['weight']=meta;wp=a.output/'weights.bf16';wp.write_bytes(weight);pin(wp)
        env=environment(a.library_dir,a.library_dir)
        for step in (1,0):
            x=a.reference/f'{step}.layers.0.operator_norm.bf16';expected=a.reference/f'{step}.layers.0.conv.in_proj.bf16'
            pin(x);pin(expected);rows=x.stat().st_size//4096
            if expected.stat().st_size!=rows*6144*2 or x.stat().st_size!=rows*4096:raise ValueError('trace shape differs')
            for splits in (1,2,4):
                for tool in ('none','initcheck','memcheck','synccheck'):
                    if tool!='none' and not numerical_exact:break
                    d=a.output/f'step-{step}-split-{splits}-{tool}';d.mkdir()
                    command=[str(a.binary),str(rows),'2048','6144',str(splits),str(wp),str(x),str(expected),str(d/'output')]
                    if tool!='none':command=[str(a.sanitizer),'--tool',tool,'--error-exitcode','86','--print-limit','4','--log-file',str(d/'sanitizer.log'),*command]
                    r=dict(step=step,rows=rows,splits=splits,tool=tool,command=command,status='incomplete');report['records'].append(r);report['start_sha256']=pins.copy();save();start=time.monotonic()
                    code,timed=run_owned(command,env,d/'application.log',60)
                    r.update(returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
                    if timed or code not in (0,86):raise ValueError('bounded execution incomplete')
                    events=[json.loads(line) for line in (d/'application.log').read_text().splitlines() if line.startswith('{')]
                    complete=[e for e in events if e.get('complete')]
                    if len(complete)!=1 or complete[0]['checked']!=rows*6144:raise ValueError('incomplete output')
                    actual=(d/'output.producer.bf16').read_bytes();consumer=(d/'output.consumer.bf16').read_bytes();ref=expected.read_bytes()
                    mismatches=sum(actual[i:i+2]!=ref[i:i+2] for i in range(0,len(ref),2))
                    if len(actual)!=len(ref) or actual!=consumer or mismatches!=complete[0]['mismatches']:raise ValueError('output verification differs')
                    r['comparison']=complete[0];r['whole_output_exact']=actual==ref
                    if tool=='none':numerical_exact=r['whole_output_exact']
                    else:
                        if not r['whole_output_exact']:raise ValueError('instrumented result differs')
                        log=(d/'sanitizer.log').read_text();totals=re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$',log)
                        r['errors']=int(totals[0]) if len(totals)==1 else None;r['internal_error']='Internal Sanitizer Error' in log
                        r['clean']=code==0 and r['errors']==0 and not r['internal_error']
                    r['status']='measured';save()
        report['status']='screening-completed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False)))


if __name__=='__main__':main()
