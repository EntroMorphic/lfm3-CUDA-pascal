"""Calibrate and measure synchronized, read-only Sanitizer SDK launch snapshots.

This is a custom observer, not simultaneous initcheck and not acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

from check_lfm3_cublas_modes import digest,environment,libraries,run_owned
from check_lfm3_projection_bedrock import make_inputs,cpu_oracle,periodic_units,compare,metadata
from lfm3_owned_process import owned_main

N=12288
BYTES=N*4
STAGES=('before_producer','before_reducer','before_consumer')


def need(condition,message):
    if not condition:raise ValueError(message)


def observed_command(observer,directory,mode,command):
    # OwnedJob first executes a Python signal-unmask launcher. Do not preload it.
    launcher=shutil.which('env')
    need(launcher is not None and Path(launcher).is_absolute(),'absolute env executable required')
    need(mode in ('control','gemm'),'invalid observer mode')
    return [launcher,'--','LD_PRELOAD='+str(observer),'LFM_BOUNDARY_OUTPUT='+str(directory),
            'LFM_BOUNDARY_MODE='+mode,*command]


def records(directory):
    return [json.loads(line) for line in (directory/'observer.jsonl').read_text().splitlines()]


def validate_observer(directory,mode,negative=False):
    events=records(directory)
    starts=[e for e in events if 'subscribe_result' in e]
    ends=[e for e in events if e.get('observer_complete') is True]
    need(len(starts)==len(ends)==1 and starts[0]['subscribe_result']==0,'observer subscription/completion missing or failed')
    end=ends[0];need(end['initialized'] is True and end['workspace_candidates']==1,'observer identity not established')
    launches=[e for e in events if 'launch_begin' in e]
    if negative:
        need(end['failed'] is True and end['captures']==0 and len(launches)==2,'negative missing-producer control not rejected')
        need(not (directory/'before_producer.bin').exists(),'negative control synthesized a producer observation')
        return dict(negative_control_detected=True,summary=end)
    need(end['failed'] is False and end['captures']==3 and end['launches']==len(launches)==3,'observer path incomplete or failed')
    need(not any('error' in e for e in events),'observer reported an error')
    need(len({e['grid_id'] for e in launches})==3 and all(e['stream']==0 for e in launches),'launch identity/stream mismatch')
    if mode=='control':
        need(end['control_buffer_count']==2,'calibration destination identity missing')
        need(all(token in e['launch_begin'] for token,e in zip(('boundary_control_fill','boundary_control_reduce','boundary_control_consume'),launches)),'calibration kernel path changed')
        need(all(e['grid']==[48,1,1] and e['block']==[256,1,1] for e in launches),'calibration geometry changed')
    else:
        need(launches[0]['launch_begin']=='nvjet_sm110_tst_128x8_64x12_2x1_v_bz_splitK_TNT' and 'splitKreduce_kernel' in launches[1]['launch_begin'] and 'consume_all' in launches[2]['launch_begin'],'different GEMM backend path')
        need([e['grid'] for e in launches]==[[96,1,1],[192,1,1],[24,1,1]] and [e['block'] for e in launches]==[[256,1,1],[32,16,1],[256,1,1]],'GEMM launch geometry changed')
        need(launches[0]['cluster']==[2,1,1],'NVJET cluster geometry changed')
    snapshots=[e for e in events if 'snapshot_complete' in e]
    for stage,launch in zip(STAGES,launches):
        matches=[e for e in snapshots if e['snapshot_complete']==stage]
        need(len(matches)==1 and matches[0]['grid_id']==launch['grid_id'] and matches[0]['bytes']==BYTES,'snapshot identity/coverage mismatch')
        need((directory/(stage+'.bin')).stat().st_size==BYTES,'snapshot truncated')
    modules={e['module']:e for e in events if 'module_image' in e}
    module_bindings=[]
    for launch in launches:
        record=dict(function=launch['launch_begin'],module=launch['module'],available=launch['module'] in modules)
        if record['available']:
            entry=modules[launch['module']];p=directory/entry['module_image'];data=p.read_bytes()
            need(data.startswith(b'\x7fELF') and len(data)==entry['bytes'],'invalid captured CUBIN image')
            record.update(file=entry['module_image'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        module_bindings.append(record)
    return dict(summary=end,launches=launches,module_bindings=module_bindings,module_capture_complete=all(x['available'] for x in module_bindings))


def calibration_values():
    workspace=b''.join(struct.pack('<f',(i%37-18)*0.25) for i in range(N))
    output=b''.join(struct.pack('<f',(i%37-18)*0.5) for i in range(N))
    return workspace,output


def validate_calibration(directory):
    result=validate_observer(directory,'control');workspace,output=calibration_values()
    expected={'before_producer':bytes([0xa5])*BYTES,'before_reducer':workspace,'before_consumer':workspace,
              'before_reducer_output':bytes([0x5a])*BYTES,'before_consumer_output':output,'before_consumer_sink':bytes([0x3c])*BYTES}
    for name,data in expected.items():need((directory/(name+'.bin')).read_bytes()==data,'calibration snapshot does not match known producer/reducer boundary: '+name)
    result['all_calibration_bytes_match']=True
    return result


@owned_main
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('observer','control','probe','sdk','library-dir','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--timeout',type=float,default=45)
    args=p.parse_args()
    if not 1<=args.repeats<=3 or not 0<args.timeout<=90:p.error('bounded repeats/timeout required')
    for name in ('observer','control','probe','sdk','library_dir','output'):setattr(args,name,getattr(args,name).resolve())
    need(not any(os.environ.get(k) for k in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH')),'external injection is forbidden')
    need(not any(c in str(args.observer) for c in ': \t\n'),'observer path is not a single preload library')
    args.output.mkdir(parents=True,exist_ok=False)
    pins={};report=dict(status='failed',full_acceptance=False,not_initcheck=True,read_only_observer=True,boundary_synchronization=True,records=[])
    def pin(path):path=Path(path).resolve();pins[str(path)]=digest(path)
    def save():(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,command,mode):
        d=args.output/name;d.mkdir();command=observed_command(args.observer,d,mode,command)
        record=dict(name=name,command=command,mode=mode,status='incomplete');report['records'].append(record);save();t=time.monotonic()
        code,timed=run_owned(command,base,d/'application.log',args.timeout)
        record.update(returncode=code,timed_out=timed,seconds=time.monotonic()-t,group_cleanup_complete=True)
        need(not timed,'bounded observer timeout; no complete observation')
        return d,record
    try:
        root=Path(__file__).resolve().parents[1]
        for path in (args.observer,args.control,args.probe,Path(__file__),root/'tools/check_lfm3_projection_bedrock.py',root/'tools/check_lfm3_cublas_modes.py',root/'tools/lfm3_owned_process.py',root/'native/probes/lfm3_boundary_observer.cpp',root/'native/probes/lfm3_boundary_control.cu'):pin(path)
        for path in args.sdk.glob('*.so'):pin(path)
        for path in (args.sdk/'include').glob('*.h'):pin(path)
        base=environment(args.library_dir,args.library_dir)
        base['LD_LIBRARY_PATH']=str(args.library_dir)+':'+str(args.sdk)+':'+base.get('LD_LIBRARY_PATH','')
        ldd=shutil.which('ldd');need(ldd is not None,'ldd unavailable')
        code,timed=run_owned([ldd,str(args.probe)],base,args.output/'probe.ldd.log',10)
        need(code==0 and not timed,'probe dependencies unresolved')
        deps=libraries((args.output/'probe.ldd.log').read_text(),args.library_dir)
        for entry in deps.values():pin(entry['path'])
        code,timed=run_owned([ldd,str(args.observer)],base,args.output/'observer.ldd.log',10)
        need(code==0 and not timed,'observer dependencies unresolved')
        text=(args.output/'observer.ldd.log').read_text();need('not found' not in text and str(args.sdk/'libsanitizer-public.so') in text,'wrong Sanitizer API library resolution')
        report.update(dependencies=deps,start_sha256=pins.copy(),observer_preload=str(args.observer),ld_library_path=base['LD_LIBRARY_PATH']);save()
        d,r=run('calibration-positive',[str(args.control),'positive'],'control')
        need(r['returncode']==0,'positive calibration application failed')
        app=[json.loads(x) for x in (d/'application.log').read_text().splitlines() if x.startswith('{')]
        need(app==[dict(control='positive',elements=N,mismatches=0)],'positive calibration output not verified')
        r['observation']=validate_calibration(d);r['status']='measured';save()
        d,r=run('calibration-missing',[str(args.control),'missing'],'control')
        need(r['returncode']==2,'negative calibration application did not detect wrong data')
        app=[json.loads(x) for x in (d/'application.log').read_text().splitlines() if x.startswith('{')]
        need(app==[dict(control='missing',elements=N,mismatches=N)],'negative calibration data failure missing')
        r['observation']=validate_observer(d,'control',negative=True);r['status']='negative-control-detected';save()
        fixtures=args.output/'fixture';fixtures.mkdir();a,b=make_inputs('asymmetric',1);expected=cpu_oracle(a,b,1)
        for name,data in (('a.bf16',a),('b.bf16',b),('expected.bf16',expected)):
            path=fixtures/name;path.write_bytes(data);pin(path)
        left=periodic_units(a,6144,2048);right=periodic_units(b,1,2048)[0]
        partial=b''.join(struct.pack('<f',sum(x*y for x,y in zip(weights,right))*512/2048) for weights in left)*2
        report['start_sha256']=pins.copy();save()
        for i in range(args.repeats):
            name=f'gemm-{i}';prefix=args.output/name/'output'
            command=[str(args.probe),'gemm','1','2048','default','tensor','default',str(fixtures/'a.bf16'),str(fixtures/'b.bf16'),str(prefix),'none']
            d,r=run(name,command,'gemm');need(r['returncode']==0,'observed GEMM application failed')
            r['observation']=observation=validate_observer(d,'gemm')
            r['application']=app=metadata((d/'application.log').read_text(),'gemm',1,dict(compute='default',algo='tensor',math='default',sync='none'))
            workspace=[x for x in app['allocations'] if x['allocation']=='workspace'];need(len(workspace)==1,'probe workspace metadata missing')
            resources=[x for x in records(d) if 'workspace_allocation' in x]
            need(len(resources)==1 and resources[0]['workspace_allocation']==workspace[0]['base'],'observer/probe allocation identities differ')
            data={stage:(d/(stage+'.bin')).read_bytes() for stage in STAGES}
            r['before_reducer_partial_words_mismatched']=sum(x!=y for x,y in zip(struct.unpack('<12288I',data['before_reducer']),struct.unpack('<12288I',partial)))
            r['workspace_changed_bytes_before_reducer']=sum(x!=y for x,y in zip(data['before_producer'],data['before_reducer']))
            r['reducer_changed_workspace_bytes']=sum(x!=y for x,y in zip(data['before_reducer'],data['before_consumer']))
            r['snapshots_sha256']={stage:hashlib.sha256(raw).hexdigest() for stage,raw in data.items()}
            for kind in ('producer','consumer'):r[kind+'_comparison']=compare(Path(str(prefix)+'.'+kind+'.bf16').read_bytes(),expected)
            r['status']='measured';save()
        report['status']='observation-completed'
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        report['end_sha256']={p:digest(p) if Path(p).is_file() else None for p in pins};report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],full_acceptance=False,not_initcheck=True)))
    return 0 if report['status']=='observation-completed' else 86


if __name__=='__main__':raise SystemExit(main())
