"""Exact bounded tail/warp-retirement tests for the native GEMM candidate."""
import argparse
import json
import os
from pathlib import Path
import struct
from check_lfm3_projection_bedrock import bf16,raw
from check_lfm3_cublas_modes import digest,environment,run_owned
from check_lfm3_candidate_sanitizers import checker_total
from lfm3_owned_process import owned_main

SHAPES=((1,1,16,1),(17,2,32,2),(63,7,48,3),(65,8,64,4),(16,9,32,2),(17,17,16,1))


def fixture(m,n,k):
    a=raw(bf16((((row+1)*(column+3))%31-15)/8) for row in range(m) for column in range(k))
    b=raw(bf16((((row+2)*(column+5))%29-14)/8) for row in range(n) for column in range(k))
    def units(data):
        result=[]
        for (word,) in struct.iter_unpack('<H',data):
            value=struct.unpack('<f',struct.pack('<I',word<<16))[0]*8
            if not value.is_integer():raise ValueError('nonintegral measured input')
            result.append(int(value))
        return result
    au,bu=units(a),units(b);values=[]
    for col in range(n):
        for row in range(m):
            terms=[au[row*k+i]*bu[col*k+i] for i in range(k)]
            if sum(abs(t) for t in terms)>=2**24:raise ValueError('exact FP32 integer bound exceeded')
            values.append(bf16(sum(terms)/64))
    return a,b,raw(values)


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','sanitizer','library-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('binary','sanitizer','library_dir','output'):setattr(a,name,getattr(a,name).resolve())
    if any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')):raise ValueError('external injection forbidden')
    a.output.mkdir(parents=True,exist_ok=False);pins={};report=dict(status='incomplete',full_acceptance=False,records=[])
    def pin(path):pins[str(path)]=digest(path)
    def save():(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (a.binary,a.sanitizer,Path(__file__),Path(__file__).with_name('check_lfm3_candidate_sanitizers.py'),Path(__file__).with_name('lfm3_owned_process.py'),Path(__file__).resolve().parents[1]/'native/probes/lfm3_wmma_policy_probe.cu'):pin(path)
        env=environment(a.library_dir,a.library_dir)
        for index,(m,n,k,splits) in enumerate(SHAPES):
            d=a.output/f'case-{index}';d.mkdir();left,right,expected=fixture(m,n,k)
            for name,data in (('weights.bf16',left),('input.bf16',right),('expected.bf16',expected)):
                path=d/name;path.write_bytes(data);pin(path)
            report['start_sha256']=pins.copy();save()
            for tool in ('initcheck','memcheck','synccheck','racecheck'):
                case=d/tool;case.mkdir()
                cmd=[str(a.sanitizer),'--tool',tool,'--error-exitcode','86','--print-limit','4','--log-file',str(case/'sanitizer.log'),str(a.binary),str(n),str(k),str(m),str(splits),str(d/'weights.bf16'),str(d/'input.bf16'),str(d/'expected.bf16'),str(case/'output')]
                record=dict(m=m,n=n,k=k,splits=splits,tool=tool,command=cmd,status='incomplete');report['records'].append(record);save()
                code,timed=run_owned(cmd,env,case/'application.log',30)
                record.update(returncode=code,timed_out=timed,group_cleanup_complete=True)
                if timed or code:raise ValueError('tail fixture tool/application failed')
                for kind in ('producer','consumer'):
                    if (case/('output.'+kind+'.bf16')).read_bytes()!=expected:raise ValueError('full tail output differs')
                log=(case/'sanitizer.log').read_text();record['errors']=checker_total(log,tool)
                if record['errors']!=0 or 'Internal Sanitizer Error' in log:raise ValueError('tail fixture checker incomplete/unclean')
                record.update(status='passed',checked=m*n,whole_output_exact=True);save()
        report['status']='scoped-passed'
    except BaseException as e:report['error']=str(e);raise
    finally:
        report['end_sha256']={k:digest(Path(k)) for k in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False,cases=len(SHAPES),tool_observations=len(report['records']))))
    return 0 if report['status']=='scoped-passed' else 86


if __name__=='__main__':raise SystemExit(main())
