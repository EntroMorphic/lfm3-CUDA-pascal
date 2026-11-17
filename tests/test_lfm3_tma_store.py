"""CPU-only checks of the minimal TMA diagnostic, not device correctness."""
from pathlib import Path
import struct
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_lfm3_tma_store import compare,cases_for


@pytest.mark.parametrize('tool',['initcheck','memcheck','synccheck'])
def test_missing_initialization_control_is_only_used_for_initcheck(tool):
    cases=cases_for(tool,3)
    assert ('missing',0) in cases if tool=='initcheck' else ('missing',0) not in cases
    assert [(m,i) for m,i in cases if m!='missing']==[(m,i) for m in ('ordinary','tma4d') for i in range(3)]


def test_unknown_diagnostic_tool_is_rejected():
    with pytest.raises(ValueError):cases_for('unknown',1)


def test_all_fp32_outputs_are_compared_bitwise():
    expected=bytes(49152)
    r=compare(expected[:-4]+struct.pack('<I',0x80000000),expected)
    assert r['checked']==12288 and r['mismatches']==1 and r['first_mismatches']==[12287]


@pytest.mark.parametrize('size',[0,4,49148,49156])
def test_incomplete_or_small_comparison_is_rejected(size):
    with pytest.raises(ValueError):compare(bytes(size),bytes(size))


def test_probe_has_documented_fence_and_full_write_wait_not_initialization():
    import re
    source=(ROOT/'native/probes/lfm3_tma_store_probe.cu').read_text()
    assert 'cudaMemset' not in source and not re.search(r'\bcublas\w*\s*\(',source,re.I)
    assert '#include <cublas' not in source.lower()
    assert 'fence.proxy.async.shared::cta;' in source
    assert 'cp.async.bulk.tensor.4d.global.shared::cta.bulk_group' in source
    assert 'cp.async.bulk.commit_group;' in source and 'cp.async.bulk.wait_group 0;' in source
    assert source.index('fence.proxy.async.shared::cta;')<source.index('__syncthreads();')<source.index('cp.async.bulk.tensor.4d')
    assert 'consume_store<<<96,128>>>' in source
