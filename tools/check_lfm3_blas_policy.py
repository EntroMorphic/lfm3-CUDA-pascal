"""Screen isolated policies on the measured small GEMM; never full acceptance."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time

from check_lfm3_projection_bedrock import make_inputs,cpu_oracle,compare
from check_lfm3_cublas_modes import run_owned,digest,libraries,environment
from lfm3_owned_process import owned_main
from prepare_lfm3_blas_policy import POLICIES


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mirrors','sanitizer','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ('mirrors','sanitizer','library_dir','output'):setattr(args,name,getattr(args,name).resolve())
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='incomplete',full_acceptance=False,records=[]);pins={}
    def pin(path):pins[str(path)]=digest(path)
    def save():(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        env=environment(args.library_dir,args.library_dir)
        for path in (Path(__file__),args.sanitizer,Path(__file__).with_name('prepare_lfm3_blas_policy.py'),Path(__file__).with_name('lfm3_owned_process.py')):pin(path)
        a,b=make_inputs('asymmetric',1);expected=cpu_oracle(a,b,1)
        for name,data in (('a.bf16',a),('b.bf16',b),('expected.bf16',expected)):
            path=args.output/name;path.write_bytes(data);pin(path)
        for policy,(budget,disallow) in POLICIES.items():
            mirror=args.mirrors/policy;binary=mirror/'build/lfm3-policy-probe';pin(binary)
            ldd=shutil.which('ldd')
            if not ldd:raise ValueError('ldd unavailable')
            code,timed=run_owned([ldd,str(binary)],env,args.output/(policy+'.ldd.log'),10)
            if timed or code!=0:raise ValueError('dependency inspection failed')
            deps=libraries((args.output/(policy+'.ldd.log')).read_text(),args.library_dir)
            for entry in deps.values():pin(Path(entry['path']))
            report.setdefault('dependencies',{})[policy]=deps
            manifest=json.loads((mirror/'manifest.json').read_text());pin(mirror/'manifest.json')
            for name,record in manifest['files'].items():
                path=mirror/name
                if digest(path)!=record['candidate_sha256']:raise ValueError('candidate source identity changed')
                pin(path)
            for tool,producer in (('initcheck','kernel'),('initcheck','missing'),('initcheck','gemm'),('synccheck','gemm')):
                directory=args.output/(policy+'-'+tool+'-'+producer);directory.mkdir()
                command=[str(args.sanitizer),'--tool',tool,'--error-exitcode','86','--print-limit','4',
                         '--log-file',str(directory/'sanitizer.log'),str(binary),producer,'1','2048','default','tensor','default',
                         str(args.output/'a.bf16'),str(args.output/'b.bf16'),str(directory/'output'),'none']
                r=dict(policy=policy,tool=tool,producer=producer,command=command,status='incomplete')
                report['records'].append(r);report['start_sha256']=pins.copy();save();start=time.monotonic()
                code,timed=run_owned(command,env,directory/'application.log',60)
                r.update(returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
                if timed or code not in (0,86):raise ValueError('application/tool did not complete')
                text=(directory/'application.log').read_text();events=[json.loads(x) for x in text.splitlines() if x.startswith('{')]
                config=[x for x in events if x.get('configuration')]
                if len(config)!=1 or config[0]['compute_enum']!=68 or config[0]['actual_math_enum']!=(16 if disallow else 0):raise ValueError('math policy differs')
                if dict(candidate_policy=policy,workspace_budget_bytes=budget,full_acceptance=False) not in events:raise ValueError('workspace policy differs')
                if dict(complete=True,outputs=6144,consumer_elements=6144) not in events:raise ValueError('incomplete application')
                log=(directory/'sanitizer.log').read_text();totals=re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$',log)
                r['internal_error']='Internal Sanitizer Error' in log
                r['reported_errors']=int(totals[0]) if len(totals)==1 else None
                r['canonical_footer']=len(totals)==1
                if producer=='missing':
                    if code!=86 or not r['reported_errors'] or 'Uninitialized __global__ memory read' not in log:raise ValueError('negative control failed')
                    r['status']='negative-control-detected'
                else:
                    if producer=='gemm':
                        for kind in ('producer','consumer'):r[kind+'_comparison']=compare((directory/('output.'+kind+'.bf16')).read_bytes(),expected)
                        if any(r[k+'_comparison']['mismatch_count'] for k in ('producer','consumer')):raise ValueError('whole-output mismatch')
                    if producer=='kernel' and (code!=0 or r['reported_errors']!=0 or r['internal_error']):raise ValueError('positive control failed')
                    r['status']='measured'
                save()
        report['status']='screening-completed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False)))


if __name__=='__main__':main()
