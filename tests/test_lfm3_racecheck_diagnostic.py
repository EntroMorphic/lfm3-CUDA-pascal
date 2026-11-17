import importlib.util
from pathlib import Path

import pytest

P=Path(__file__).resolve().parents[1]/"tools/check_lfm3_racecheck_diagnostic.py"
S=importlib.util.spec_from_file_location("race_diagnostic",P);gate=importlib.util.module_from_spec(S);S.loader.exec_module(gate)


@pytest.mark.parametrize("damage",[None,"timeout","nonzero","warnings","missing-output","wrong-prompt","empty-prediction","bool-prediction","wrong-prediction"])
def test_timeout_or_incomplete_execution_is_not_a_racecheck_pass(damage):
    request=dict(input_ids=[1,2]);result=dict(backend="native-cuda-experimental",input_ids=[1,2],generated_ids=[],teacher_argmax=[7])
    code=86 if damage=="nonzero" else 0;log="0 errors, 1 warnings" if damage=="warnings" else "0 errors, 0 warnings"
    if damage=="missing-output":result=None
    elif damage=="wrong-prompt":result["input_ids"]=[3]
    elif damage=="empty-prediction":result["teacher_argmax"]=[]
    elif damage=="bool-prediction":result["teacher_argmax"]=[True]
    elif damage=="wrong-prediction":result["teacher_argmax"]=[8]
    assert gate.clean_result(code,damage=="timeout",log,result,request,dict(teacher_argmax=[7]))==(damage is None)


def test_diagnostic_reports_unlimited_launches_and_group_cleanup(tmp_path,monkeypatch):
    import json,sys
    tool=tmp_path/"sanitizer"
    tool.write_text(f"#!{sys.executable}\nimport sys,time\nif '--version' in sys.argv:print('fake version')\nelse:time.sleep(30)\n")
    tool.chmod(0o700)
    binary=tmp_path/"binary";binary.write_text("fake")
    request=tmp_path/"request.json";expected=tmp_path/"expected.json"
    ids=[1]*32768
    request.write_text(json.dumps(dict(capacity=32768,input_ids=ids,max_new_tokens=0)))
    expected.write_text(json.dumps(dict(input_ids=ids,generated_ids=[],teacher_argmax=[7])))
    output=tmp_path/"evidence"
    monkeypatch.setattr(sys,"argv",[str(P),'--sanitizer',str(tool),'--binary',str(binary),'--checkpoint',str(tmp_path),
        '--request',str(request),'--expected-result',str(expected),'--output',str(output),'--timeout','0.1'])
    assert gate.main()==86
    summary=json.loads((output/'summary.json').read_text())
    command=summary['command']
    assert command[command.index('--print-limit')+1]=='0'
    assert summary['status']=='timeout' and summary['group_cleanup_complete'] is True
    samples=[json.loads(line) for line in (output/'resources.jsonl').read_text().splitlines()]
    assert samples and all('system_memory' in row for row in samples)
