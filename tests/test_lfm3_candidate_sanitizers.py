"""Host model tests; mocked execution provides no GPU coverage credit."""
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check_lfm3_candidate_sanitizers as gate


@pytest.mark.parametrize('text,total',[
    ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n',0),
    ('========= RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)\n',1),
    ('========= RACECHECK SUMMARY: 0 hazards displayed (1 errors, 0 warnings)\n',1),
    ('========= ERROR SUMMARY: 0 errors\n',None),
    ('',None),
    ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n'*2,None),
    ('========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n========= ERROR SUMMARY: 1 errors\n',None),
])
def test_racecheck_requires_its_own_footer_and_zero_warnings(text,total):
    assert gate.checker_total(text,'racecheck')==total


@pytest.mark.parametrize('fault',['','no-footer','duplicate-footer','internal','wrong-ids','missing-output','timeout'])
def test_integrated_screen_requires_checker_and_original_application(tmp_path,monkeypatch,fault):
    request=dict(input_ids=[1],max_new_tokens=2,check_reset=True)
    expected=dict(backend='native-cuda-experimental',input_ids=[1],generated_ids=[2,3],teacher_argmax=[2],reset_exact_logits_ok=True)
    paths={name:tmp_path/name for name in ('binary','sanitizer','checkpoint','request','expected-result','library-dir','output')}
    for name in ('binary','sanitizer'):paths[name].write_text('fake host model')
    for name in ('checkpoint','library-dir'):paths[name].mkdir()
    paths['request'].write_text(json.dumps(request));paths['expected-result'].write_text(json.dumps(expected))
    for name in ('LD_PRELOAD','LD_AUDIT','CUDA_INJECTION64_PATH'):monkeypatch.delenv(name,raising=False)
    monkeypatch.setattr(sys,'argv',['gate',*[x for name,path in paths.items() for x in ('--'+name,str(path))],
                                    '--tools','initcheck','synccheck','memcheck','--timeout','1'])
    monkeypatch.setattr(gate,'libraries',lambda text,root:{'fake':dict(path=str(paths['binary']))})
    monkeypatch.setattr(gate.shutil,'which',lambda name:'/usr/bin/true')
    calls=[]
    def run(command,env,log,timeout):
        calls.append(command)
        if '--tool' not in command:log.write_text('fake dependencies');return 0,False
        assert not any(x.startswith('--kernel') or x=='--suppressions' for x in command)
        log.write_text('fake app log')
        text='========= ERROR SUMMARY: 0 errors\n'
        if fault=='no-footer':text=''
        if fault=='duplicate-footer':text*=2
        if fault=='internal':text='Internal Sanitizer Error\n'+text
        Path(command[command.index('--log-file')+1]).write_text(text)
        if fault!='missing-output':
            actual=dict(expected)
            if fault=='wrong-ids':actual['generated_ids']=[4,5]
            Path(command[-1]).write_text(json.dumps(actual))
        return (None,True) if fault=='timeout' else (0,False)
    monkeypatch.setattr(gate,'run_owned',run)
    if fault=='timeout':
        with pytest.raises(ValueError,match='timed out'):gate.main.__wrapped__()
    else:assert gate.main.__wrapped__()==(86 if fault else 0)
    report=json.loads((paths['output']/'summary.json').read_text())
    assert report['full_acceptance'] is False and report['hashes_stable']
    assert report['kernel_exclusions']==[]
    assert len(report['records'])==(1 if fault=='timeout' else 3)
    assert all(r['group_cleanup_complete'] for r in report['records'])
    if fault:assert not all(r.get('clean') for r in report['records'])
    else:assert report['status']=='scoped-passed'
