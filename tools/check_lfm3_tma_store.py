"""Bounded no-cuBLAS TMA/ordinary-store sanitizer diagnostic; never acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time

from lfm3_owned_process import owned_main
from check_lfm3_cublas_modes import digest,run_owned


def compare(raw,expected):
    if len(raw)!=len(expected) or len(raw)!=49152:raise ValueError('incomplete FP32 output')
    mismatches=[i for i,(a,b) in enumerate(zip(struct.unpack('<12288I',raw),struct.unpack('<12288I',expected))) if a!=b]
    return dict(checked=12288,mismatches=len(mismatches),first_mismatches=mismatches[:16],sha256=hashlib.sha256(raw).hexdigest())


def cases_for(tool,repeats):
    if tool not in ('initcheck','memcheck','synccheck'):raise ValueError('unsupported diagnostic tool')
    return ([('missing',0)] if tool=='initcheck' else [])+[(method,i) for method in ('ordinary','tma4d') for i in range(repeats)]


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','sanitizer','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--timeout',type=float,default=45)
    p.add_argument('--preimage',action='store_true',help='intentional uninitialized preimage read; diagnostic totals cannot be clean acceptance')
    p.add_argument('--tool',choices=('initcheck','memcheck','synccheck'),default='initcheck')
    args=p.parse_args()
    if not 1<=args.repeats<=3 or not 0<args.timeout<=90:p.error('bounded repeats/timeout required')
    if args.preimage and args.tool!='initcheck':p.error('preimage experiment is initcheck-only')
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):p.error('external injection forbidden')
    for name in ('binary','sanitizer','library_dir','output'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='failed',full_acceptance=False,tool=args.tool,records=[],preimage_diagnostic=args.preimage,scope='minimal TMA4D vs ordinary stores; no cuBLAS, model or custom SDK observer')
    pins={}
    def pin(path):path=Path(path).resolve();pins[str(path)]=digest(path)
    def save():(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (args.binary,args.sanitizer,Path(__file__),Path(__file__).resolve().parents[1]/'native/probes/lfm3_tma_store_probe.cu',Path(__file__).with_name('lfm3_owned_process.py'),Path(__file__).with_name('check_lfm3_cublas_modes.py')):pin(path)
        env=dict(os.environ);env['LD_LIBRARY_PATH']=str(args.library_dir)+':'+env.get('LD_LIBRARY_PATH','')
        ldd=shutil.which('ldd')
        if not ldd:raise ValueError('ldd unavailable')
        code,timed=run_owned([ldd,str(args.binary)],env,args.output/'ldd.log',10)
        text=(args.output/'ldd.log').read_text()
        if code!=0 or timed or 'not found' in text or 'libcublas' in text:raise ValueError('dependencies unresolved or cuBLAS present')
        resolved={}
        for match in re.finditer(r'(\S+) => (/\S+) \(',text):
            path=Path(match[2]).resolve();pin(path);resolved[match[1]]=str(path)
        if not Path(resolved.get('libcudart.so.13','missing')).is_relative_to(args.library_dir):raise ValueError('wrong cudart')
        report.update(start_sha256=pins.copy(),dependencies=resolved);save()
        cases=cases_for(args.tool,args.repeats)
        for case_index,(method,i) in enumerate(cases):
            d=args.output/f'{method}-{i}';d.mkdir();prefix=d/'output'
            bias=1024+case_index
            expected=b''.join(struct.pack('<f',(j%37-18)*0.25+bias) for j in range(12288))
            (d/'expected.f32').write_bytes(expected);pin(d/'expected.f32');report['start_sha256']=pins.copy()
            cmd=[str(args.sanitizer),'--tool',args.tool,'--error-exitcode','86','--print-limit','0' if args.preimage else '4','--log-file',str(d/'sanitizer.log'),str(args.binary),method,str(prefix),str(bias)]
            if args.preimage:cmd.append('preimage')
            r=dict(method=method,repeat=i,bias=bias,command=cmd,status='incomplete');report['records'].append(r);save();start=time.monotonic()
            code,timed=run_owned(cmd,env,d/'application.log',args.timeout);r.update(returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
            if timed or code not in (0,86):raise ValueError('tool/application incomplete')
            app=[json.loads(x) for x in (d/'application.log').read_text().splitlines() if x.startswith('{')]
            if len(app)!=2 or app[0].get('method')!=method or app[0].get('dimensions')!=[128,1,1,96] or app[0].get('output_initialized') is not False or app[0].get('bias')!=bias or app[0].get('preimage') is not args.preimage or app[1]!=dict(complete=True,elements=12288,method=method):raise ValueError('application identity/completion mismatch')
            log=(d/'sanitizer.log').read_text();totals=re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$',log)
            if len(totals)!=1 or 'Internal Sanitizer Error' in log:raise ValueError('incomplete/internal-error checker result')
            r['reported_errors']=errors=int(totals[0]);r['consumer_read_report']='consume_store(' in log and 'Uninitialized __global__ memory read' in log
            if args.preimage:
                before=(d/'output.before.bin').read_bytes();after=Path(str(prefix)+'.producer.f32').read_bytes()
                if len(before)!=49152 or len(after)!=49152:raise ValueError('pre/post snapshot incomplete')
                r['preimage']=dict(all_zero=not any(before),sha256=hashlib.sha256(before).hexdigest(),changed_bytes=sum(a!=b for a,b in zip(before,after)),changed_fp32_words=sum(a!=b for a,b in zip(struct.unpack('<12288I',before),struct.unpack('<12288I',after))))
            if method=='missing':
                if code!=86 or not errors or not r['consumer_read_report']:raise ValueError('missing-write control not detected')
                r['status']='negative-control-detected'
            else:
                for kind in ('producer','consumer'):r[kind+'_comparison']=compare(Path(str(prefix)+'.'+kind+'.f32').read_bytes(),expected)
                if any(r[k+'_comparison']['mismatches'] for k in ('producer','consumer')):raise ValueError('complete output comparison failed')
                if (code==0)!=(errors==0):raise ValueError('tool returncode/total disagree')
                if method=='ordinary' and (any(r[k+'_comparison']['mismatches'] for k in ('producer','consumer')) or r['consumer_read_report'] or (not args.preimage and code!=0)):raise ValueError('ordinary-store control failed')
                r['status']='measured'
            save()
        report['status']='measurement-completed'
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        report['end_sha256']={p:digest(p) if Path(p).is_file() else None for p in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False)))
    return 0 if report['status']=='measurement-completed' else 86


if __name__=='__main__':raise SystemExit(main())
