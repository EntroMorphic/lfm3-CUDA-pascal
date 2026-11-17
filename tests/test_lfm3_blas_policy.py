"""Host-only candidate isolation checks, not numerical or sanitizer acceptance."""
import hashlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_lfm3_blas_policy import prepare,POLICIES,once


@pytest.mark.parametrize('policy',POLICIES)
def test_policy_changes_only_two_isolated_sources(tmp_path,policy):
    out=tmp_path/'candidate';m=prepare(ROOT,out,policy)
    assert m['full_acceptance'] is False and m['goldens_may_change'] is False
    assert m['required_eager_checks']==6890
    changed={n for n,r in m['files'].items() if r['changed']}
    assert changed=={'native/src/lfm3_main.cu','native/probes/lfm3_projection_bedrock_probe.cu'}
    for n,r in m['files'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==r['original_sha256']
        assert hashlib.sha256((out/n).read_bytes()).hexdigest()==r['candidate_sha256']
    source=(out/'native/src/lfm3_main.cu').read_text()
    assert source.count('CUBLAS_COMPUTE_32F')==3
    assert source.count('CUBLAS_GEMM_DEFAULT_TENSOR_OP')==3
    assert 'cudaMemset(workspace' not in source
    with pytest.raises(FileExistsError):prepare(ROOT,out,policy)


@pytest.mark.parametrize('text',['missing','anchor anchor'])
def test_changed_anchors_fail_before_generation(text):
    with pytest.raises(ValueError):once(text,'anchor','replacement')
