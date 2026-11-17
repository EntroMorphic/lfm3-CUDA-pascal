"""Host isolation checks. WMMA/runtime correctness requires GPU evidence."""
import hashlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_lfm3_wmma_runtime import prepare


def test_candidate_preserves_every_source_except_linear_dispatch(tmp_path):
    output=tmp_path/'mirror';m=prepare(ROOT,output)
    assert m['full_acceptance'] is False and m['goldens_may_change'] is False and m['required_eager_checks']==6890
    assert [n for n,r in m['files'].items() if r['changed']]==['native/src/lfm3_main.cu']
    for n,r in m['files'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==r['original_sha256']
        assert hashlib.sha256((output/n).read_bytes()).hexdigest()==r['candidate_sha256']
    header=(output/'native/tests/lfm3_wmma_runtime_generated.cuh').read_text()
    probe=(ROOT/'native/probes/lfm3_wmma_policy_probe.cu').read_text()
    kernels=probe[probe.index('__global__ void wmma_partials('):probe.index('__global__ void wmma_consume(')]
    assert kernels in header and 'cudaMemset' not in header
    assert 'CUBLAS_COMPUTE_32F_PEDANTIC' not in header
    assert 'parts>16||k%parts||(k/parts)%16' in header
    source=(output/'native/src/lfm3_main.cu').read_text()
    assert source.count('cublasGemmBatchedEx')==2
    assert 'cublasSetWorkspace(blas, workspace.p, 4096*1024*2+16*1024*8)' in source
    assert 'candidate_linear.run(blas,w(name),x,out,rows,input,output);' in source
    assert 'CUBLAS_DEFAULT_MATH' in source
    assert 'CUDA::cublasLt' in (output/'CMakeLists.txt').read_text()
    with pytest.raises(FileExistsError):prepare(ROOT,output)


def test_candidate_rejects_source_anchor_changes(tmp_path):
    # once() is the same checked substitution used at each dispatch anchor.
    from prepare_lfm3_blas_policy import once
    with pytest.raises(ValueError):once('wrong dispatch','expected dispatch','candidate dispatch')
