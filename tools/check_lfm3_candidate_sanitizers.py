"""Unexcluded integrated candidate checks with exact application-output validation.

This bounded request does not replace upper-context, worker, or full racecheck gates.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from check_lfm3_cublas_modes import run_owned,digest,environment,libraries
from check_lfm3_sanitizer import native_result
from lfm3_owned_process import owned_main


def checker_total(log,tool):
    errors=re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$',log)
    if tool!='racecheck':return int(errors[0]) if len(errors)==1 else None
    races=re.findall(r'(?m)^=+\s+RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)\s*$',log)
    if len(races)!=1 or len(errors)>1 or (errors and int(errors[0])):return None
    hazards,errors_count,warnings=map(int,races[0])
    return max(hazards,errors_count+warnings)


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','sanitizer','checkpoint','request','expected-result','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tools',nargs='+',choices=('initcheck','memcheck','synccheck','racecheck'),default=['initcheck','synccheck'])
    p.add_argument('--timeout',type=float,default=300)
    a=p.parse_args()
    if not 0<a.timeout<=900:p.error('bounded timeout required')
    for name in ('binary','sanitizer','checkpoint','request','expected_result','library_dir','output'):setattr(a,name,getattr(a,name).resolve())
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    a.output.mkdir(parents=True,exist_ok=False);pins={};report=dict(status='incomplete',full_acceptance=False,kernel_exclusions=[],records=[])
    def pin(path):pins[str(path)]=digest(path)
    def save():(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (a.binary,a.sanitizer,a.request,a.expected_result,Path(__file__),Path(__file__).with_name('check_lfm3_sanitizer.py'),Path(__file__).with_name('lfm3_owned_process.py')):pin(path)
        request=json.loads(a.request.read_text());expected=json.loads(a.expected_result.read_text());native_result(a.expected_result,request)
        if request.get('trace_directory'):raise ValueError('traced runs are not this integrated screen')
        env=environment(a.library_dir,a.library_dir);ldd=shutil.which('ldd')
        if not ldd:raise ValueError('ldd unavailable')
        code,timed=run_owned([ldd,str(a.binary)],env,a.output/'ldd.log',10)
        if timed or code:raise ValueError('dependency inspection incomplete')
        report['dependencies']=deps=libraries((a.output/'ldd.log').read_text(),a.library_dir)
        for entry in deps.values():pin(Path(entry['path']))
        report['start_sha256']=pins.copy();save()
        for tool in dict.fromkeys(a.tools):
            cmd=[str(a.sanitizer),'--tool',tool,'--error-exitcode','86','--print-limit','4','--log-file',str(a.output/(tool+'.log')),str(a.binary),'--experimental',str(a.checkpoint),str(a.request),str(a.output/(tool+'.json'))]
            r=dict(tool=tool,command=cmd,status='incomplete');report['records'].append(r);save();start=time.monotonic()
            code,timed=run_owned(cmd,env,a.output/(tool+'.stdout.log'),a.timeout)
            r.update(returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
            log=(a.output/(tool+'.log')).read_text() if (a.output/(tool+'.log')).exists() else ''
            r['errors']=checker_total(log,tool);r['internal_error']='Internal Sanitizer Error' in log
            try:r['application']=native_result(a.output/(tool+'.json'),request,expected);r['application_ok']=True
            except (OSError,ValueError,KeyError,TypeError) as e:r['application_ok']=False;r['application_error']=str(e)
            r['clean']=not timed and code==0 and r['errors']==0 and not r['internal_error'] and r['application_ok']
            r['status']='measured' if not timed else 'incomplete-timeout';save()
            if timed:raise ValueError('bounded sanitizer timed out; remaining tools not run')
        report['status']='scoped-passed' if all(r['clean'] for r in report['records']) else 'scoped-failed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False)))
    return 0 if report['status']=='scoped-passed' else 86


if __name__=='__main__':raise SystemExit(main())
