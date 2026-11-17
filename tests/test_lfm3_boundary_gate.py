"""Host-only rejection tests for the synchronized-boundary measurement gate."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import check_lfm3_launch_boundary as g


def fixture(d):
    workspace,output=g.calibration_values()
    data={'before_producer':bytes([0xa5])*g.BYTES,'before_reducer':workspace,'before_consumer':workspace,
          'before_reducer_output':bytes([0x5a])*g.BYTES,'before_consumer_output':output,'before_consumer_sink':bytes([0x3c])*g.BYTES}
    for name,raw in data.items():(d/(name+'.bin')).write_bytes(raw)
    events=[dict(subscribe_result=0),dict(module_image='module-0.cubin',module=1,bytes=4)]
    (d/'module-0.cubin').write_bytes(b'\x7fELF')
    for i,(stage,name) in enumerate(zip(g.STAGES,('boundary_control_fill','boundary_control_reduce','boundary_control_consume')),1):
        events += [dict(launch_begin=name,grid_id=i,stream=0,module=1,grid=[48,1,1],block=[256,1,1],cluster=[1,1,1]),
                   dict(snapshot_complete=stage,grid_id=i,bytes=g.BYTES)]
    events.append(dict(observer_complete=True,initialized=True,failed=False,workspace_candidates=1,control_buffer_count=2,captures=3,launches=3,full_acceptance=False))
    return events,data


@pytest.mark.parametrize('damage',[None,'late-reducer','late-consumer','wrong-scratch','truncated','captures','stream','kernel','module','subscription','identity'])
def test_calibration_must_observe_before_not_after_operations(tmp_path,damage):
    events,data=fixture(tmp_path)
    if damage=='late-reducer':(tmp_path/'before_reducer_output.bin').write_bytes(data['before_consumer_output'])
    elif damage=='late-consumer':(tmp_path/'before_consumer_sink.bin').write_bytes(data['before_consumer_output'])
    elif damage=='wrong-scratch':(tmp_path/'before_reducer.bin').write_bytes(bytes(g.BYTES))
    elif damage=='truncated':(tmp_path/'before_reducer.bin').write_bytes(data['before_reducer'][:-4])
    elif damage=='captures':events[-1]['captures']=2
    elif damage=='stream':events[2]['stream']=1
    elif damage=='kernel':events[2]['launch_begin']='different_kernel'
    elif damage=='module':(tmp_path/'module-0.cubin').write_bytes(b'nope')
    elif damage=='subscription':events[0]['subscribe_result']=1
    elif damage=='identity':events[3]['grid_id']=99
    (tmp_path/'observer.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    if damage:
        with pytest.raises(ValueError):g.validate_calibration(tmp_path)
    else:
        r=g.validate_calibration(tmp_path)
        assert r['all_calibration_bytes_match'] and r['module_capture_complete']


def test_no_observer_completion_is_not_a_measurement(tmp_path):
    events,_=fixture(tmp_path)
    (tmp_path/'observer.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events[:-1]))
    with pytest.raises(ValueError):g.validate_calibration(tmp_path)


def test_preload_reaches_only_final_target_not_owned_python_launcher(tmp_path):
    import os
    import shutil
    import subprocess
    compiler=shutil.which('gcc');assert compiler
    source=tmp_path/'marker.c';library=tmp_path/'marker.so'
    source.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
__attribute__((constructor)) static void loaded(void) {
  char command[1024]={0};FILE* p=fopen("/proc/self/cmdline","r");
  if(p){fread(command,1,sizeof(command)-1,p);fclose(p);}
  FILE* out=fopen(getenv("MARKER_LOG"),"a");if(out){fprintf(out,"%s\n",command);fclose(out);}
}
''')
    subprocess.run([compiler,'-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
    target=shutil.which('true');assert target
    command=g.observed_command(library,tmp_path,'control',[target])
    env=dict(os.environ,MARKER_LOG=str(tmp_path/'loaded.log'));env.pop('LD_PRELOAD',None)
    code,timed=g.run_owned(command,env,tmp_path/'application.log',5)
    assert code==0 and not timed
    assert (tmp_path/'loaded.log').read_text().splitlines()==[target]
