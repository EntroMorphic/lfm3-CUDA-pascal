"""CPU-only oracle and measurement contract tests, not GPU evidence."""
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import check_lfm3_projection_bedrock as g


def value(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]


@pytest.mark.parametrize('pattern',['legacy','asymmetric'])
@pytest.mark.parametrize('rows',[1,257])
def test_periodic_oracle_against_independent_literal_dot_products(pattern,rows):
    width=4;a,b=g.make_inputs(pattern,rows,width)
    actual=g.decode(g.cpu_oracle(a,b,rows,width));aa=g.decode(a);bb=g.decode(b)
    expected=[]
    for row in range(rows):
        for channel in range(3*width):
            total=sum(value(aa[channel*width+k])*value(bb[row*width+k]) for k in range(width))
            expected.append(g.bf16(total))
    assert list(actual)==expected


def test_asymmetric_inputs_and_expected_rows_are_distinguishable():
    a,b=g.make_inputs('asymmetric',257,4);expected=g.cpu_oracle(a,b,257,4)
    assert len({b[r*8:(r+1)*8] for r in range(257)})==257
    assert len({expected[r*24:(r+1)*24] for r in range(257)})==257


def test_legacy_oracle_restarts_channel_index_at_each_column():
    a,b=g.make_inputs('legacy',257,4);actual=g.decode(g.cpu_oracle(a,b,257,4))
    assert actual[0]==actual[12]==g.bf16(-6*4/16)
    assert actual[12]!=g.bf16((12%13-6)*4/16)


@pytest.mark.parametrize('damage',['truncated','late-input-corruption'])
def test_oracle_rejects_unmeasured_periodicity(damage):
    a,b=g.make_inputs('asymmetric',1,4)
    if damage=='truncated':a=a[:-2]
    else:
        a,b=g.make_inputs('asymmetric',1)
        a=a[:-2]+b'\0\0'
    with pytest.raises(ValueError):g.cpu_oracle(a,b,1,4 if damage=='truncated' else g.WIDTH)


def test_full_output_comparison_detects_last_element_and_signed_zero():
    expected=g.raw([0]*1000)
    report=g.compare(expected[:-2]+g.raw([0x8000]),expected)
    assert report['checked_elements']==1000 and report['mismatch_count']==1
    assert report['first_mismatches'][0]['index']==999
    for bad in (b'',expected[:-2],expected+b'\0\0'):
        with pytest.raises(ValueError):g.compare(bad,expected)


def test_cpu_bf16_ties_to_even():
    assert g.bf16(1+1/256)==0x3f80
    assert g.bf16(1+3/256)==0x3f82


def config():
    return dict(configuration=True,producer='gemm',rows=1,width=2048,compute='default',algo='tensor',math='default',sync='none',
                compute_enum=68,algo_enum=99,actual_math_enum=0,default_stream=True,workspace_bytes=8519680,
                cublas=130100,driver=13000,runtime=13000,sm_major=11,sm_minor=0)


@pytest.mark.parametrize('damage',[None,'count','bool','math','stream','missing','duplicate'])
def test_actual_settings_and_complete_coverage_required(damage):
    c=config();end=dict(complete=True,outputs=6144,consumer_elements=6144)
    if damage=='count':end['consumer_elements']=128
    elif damage=='bool':c['actual_math_enum']=False
    elif damage=='math':c['actual_math_enum']=2
    elif damage=='stream':c['default_stream']=False
    text=json.dumps(c)+'\n'+json.dumps(end)
    if damage=='missing':text=json.dumps(c)
    if damage=='duplicate':text+='\n'+json.dumps(end)
    settings=dict(compute='default',algo='tensor',math='default',sync='none')
    if damage:
        with pytest.raises(ValueError):g.metadata(text,'gemm',1,settings)
    else:assert g.metadata(text,'gemm',1,settings)['complete']==end


def test_canonical_error_total_not_printed_footer():
    text='========= Uninitialized __global__ memory read\n========= at consume_all(\n========= ERROR SUMMARY: 6144 errors\n========= ERROR SUMMARY: 6140 errors were not printed.\n'
    report=g.sanitizer_report(text)
    assert report['canonical_errors']==6144 and report['consumer_read_report']
    assert g.sanitizer_report(text+'========= ERROR SUMMARY: 0 errors\n')['canonical_errors'] is None


def test_address_mapping_requires_entire_read_inside_exact_allocation():
    allocations=[dict(allocation='workspace',base=4096,bytes=16)]
    text='========= Uninitialized __global__ memory read of size 4 bytes\n========= Address 0x1000\n'
    assert g.reported_addresses(text,allocations)[0]['matches']==[dict(allocation='workspace',offset_bytes=0)]
    assert g.reported_addresses(text.replace('0x1000','0x100e'),allocations)[0]['matches']==[]
    assert g.reported_addresses(text,[])[0]['matches']==[]


def test_probe_is_separate_and_does_not_initialize_gemm_output():
    source=g.SOURCE.read_text()
    import re
    assert re.findall(r'cudaMemset\(([^,]+),',source)==['workspace']
    assert 'if(workspace_seed!="none")' in source
    assert 'consume_all<<<(n+255)/256,256>>>(c,d,n)' in source
    assert 'float alpha=1,beta=0' in source
    assert 'if(sync=="producer")' in source
    assert 'CUBLAS_COMPUTE_32F_PEDANTIC' in source


@pytest.mark.parametrize('phase',['baseline','isolate'])
@pytest.mark.parametrize('instrumentation',['initcheck','none'])
@pytest.mark.parametrize('damage',[None,'last-output','missing-output','wrong-mode','tool-exit','missing-summary','timeout','probe-mutation','launch-dump','scratch-valid','scratch-missing','scratch-identity','scratch-preimage-truncated'])
def test_measurement_runner_records_whole_output_and_fails_closed(tmp_path,monkeypatch,damage,phase,instrumentation):
    original_inputs,original_oracle=g.make_inputs,g.cpu_oracle
    monkeypatch.setattr(g,'WIDTH',4)
    monkeypatch.setattr(g,'make_inputs',lambda pattern,rows:original_inputs(pattern,rows,4))
    monkeypatch.setattr(g,'cpu_oracle',lambda a,b,rows:original_oracle(a,b,rows,4))
    dependency=tmp_path/'library.so';dependency.write_bytes(b'fake library')
    monkeypatch.setattr(g,'libraries',lambda text,root:{'fake':dict(path=str(dependency),sha256=g.digest(dependency))})
    probe,sanitizer=tmp_path/'probe',tmp_path/'sanitizer'
    probe.write_bytes(b'fake executable');sanitizer.write_bytes(b'fake tool')
    out=tmp_path/'evidence';calls=[]
    def run(command,env,log,timeout):
        if Path(command[0]).name=='ldd':
            assert Path(command[0]).is_absolute()
            log.write_text('fake dependency');return 0,False
        start=command.index(str(probe))+1
        producer,rows,width,compute,algo,math,a,b,prefix,sync=command[start:start+10]
        extras=command[start+10:]
        if producer!='gemm':assert not extras  # Scratch intervention must not contaminate controls.
        rows,width=int(rows),int(width);n=rows*3*width
        expected=original_oracle(Path(a).read_bytes(),Path(b).read_bytes(),rows,width)
        data=g.raw([g.bf16(width)]*n) if producer=='kernel' else expected
        if producer=='gemm' and damage=='last-output':data=data[:-2]+g.raw([1])
        if not (producer=='gemm' and damage=='missing-output'):
            for label in ('producer','consumer'):Path(prefix+'.'+label+'.bf16').write_bytes(data)
        c=config();c.update(producer=producer,rows=rows,width=width,compute=compute,algo=algo,math=math,
                           compute_enum=69 if compute=='pedantic' else 68,algo_enum=99 if algo=='tensor' else -1,
                           actual_math_enum=2 if math=='pedantic' else 0)
        if extras:
            seed,dump=extras;c.update(workspace_seed=seed,workspace_dump_bytes=int(dump))
            if damage!='scratch-missing':Path(prefix+'.workspace.bin').write_bytes(bytes([0x5a])*int(dump))
            Path(prefix+'.workspace-before.bin').write_bytes(bytes(int(dump)-(1 if damage=='scratch-preimage-truncated' else 0)))
            if damage=='scratch-identity':c['workspace_seed']='a5'
        if producer=='gemm' and damage=='wrong-mode':c['actual_math_enum']=2
        log.write_text(json.dumps(c)+'\n'+json.dumps(dict(complete=True,outputs=n,consumer_elements=n))+'\n')
        errors=n if producer=='missing' else 0
        if instrumentation=='initcheck':
            tool_log=Path(command[command.index('--log-file')+1])
            dump='--dump-kernel-launches' in command
            if dump:
                assert command[command.index('--print-limit')+1]=='0' and producer=='gemm'
                errors+=1
            tool_log.write_text(('========= Uninitialized __global__ memory read at consume_all(\n' if producer=='missing' else '')+('========= Launch #1\n=========   Kernel: measured_kernel\n' if dump else '')+f'========= ERROR SUMMARY: {errors} errors\n')
            if producer=='gemm' and damage=='missing-summary':tool_log.write_text('========= COMPUTE-SANITIZER\n')
        else:assert '--log-file' not in command and producer!='missing'
        if producer=='gemm' and damage=='probe-mutation':probe.write_bytes(b'changed executable')
        calls.append(producer)
        if producer=='gemm' and damage=='tool-exit':return 9,False
        if producer=='gemm' and damage=='timeout':return None,True
        return (86 if errors else 0),False
    monkeypatch.setattr(g,'run_owned',run)
    argv=['measure','--probe',str(probe),'--sanitizer',str(sanitizer),'--library-dir',str(tmp_path),'--output',str(out),'--repeats','1','--phase',phase,'--instrumentation',instrumentation]
    if damage=='launch-dump' and instrumentation=='initcheck':argv.append('--capture-launches')
    if damage and damage.startswith('scratch-'):argv+=['--workspace-seed','5a','--workspace-dump-bytes','16']
    monkeypatch.setattr(sys,'argv',argv)
    failures=('missing-output','wrong-mode','tool-exit','timeout','scratch-missing','scratch-identity','scratch-preimage-truncated')+(('missing-summary',) if instrumentation=='initcheck' else ())
    if damage in failures:
        with pytest.raises((ValueError,OSError)):g.main()
    else:assert g.main()==(86 if damage=='probe-mutation' else 0)
    report=json.loads((out/'summary.json').read_text())
    assert report['full_acceptance'] is False
    assert report['hashes_stable'] is (damage!='probe-mutation')
    if damage in failures:assert report['status']=='failed'
    elif damage=='probe-mutation':assert report['status']=='failed-identity-change'
    else:
        count=4 if phase=='baseline' else 8
        assert report['status']=='measurement-completed' and report['planned_cases']==count+(2 if instrumentation=='initcheck' else 1)
        assert report['observed_gemm_cases_with_numerical_mismatch']==(count if damage=='last-output' else 0)
        if instrumentation=='none':assert report['observed_gemm_cases_with_sanitizer_errors'] is None
        if damage=='launch-dump' and instrumentation=='initcheck':
            assert report['observed_gemm_cases_with_sanitizer_errors'] is None
            assert all(r['sanitizer']['launch_dump_diagnostic'] for r in report['records'] if r['producer']=='gemm')
        if damage=='scratch-valid':
            assert all(r['workspace_preimage']['changed_bytes']==16 for r in report['records'] if r['producer']=='gemm')


def test_reported_address_map_does_not_assign_unknown_allocations():
    # Repeated addresses from separate processes are never merged into one map.
    text='========= Uninitialized __global__ memory read of size 4 bytes\n========= Address 0x1000\n'
    assert g.reported_addresses(text,[dict(allocation='a',base=8192,bytes=4096)])[0]['matches']==[]


def test_cli_help_explicitly_labels_launch_dump_as_diagnostic():
    import subprocess
    r=subprocess.run([sys.executable,str(ROOT/'tools/check_lfm3_projection_bedrock.py'),'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--capture-launches' in r.stdout and 'pollute tool totals' in ' '.join(r.stdout.split())
