"""Host isolation checks for the v5/v6 thin-hybrid candidate. Correctness requires GPU evidence."""
import hashlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_lfm3_thin_hybrid_runtime import prepare,region,HEADER


def test_hybrid_preserves_every_source_except_linear_dispatch(tmp_path):
    output=tmp_path/'mirror';m=prepare(ROOT,output)
    assert m['full_acceptance'] is False and m['goldens_may_change'] is False and m['required_eager_checks']==6890
    assert m['generated_header']==HEADER and (output/m['generated_header']).is_file()
    assert [n for n,r in m['files'].items() if r['changed']]==['native/src/lfm3_main.cu']
    for n,r in m['files'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==r['original_sha256']
        assert hashlib.sha256((output/n).read_bytes()).hexdigest()==r['candidate_sha256']
    header=(output/HEADER).read_text()
    v4=(ROOT/'native/probes/lfm3_wmma_policy_probe.cu').read_text()
    thin=(ROOT/'native/probes/lfm3_thin_v5_probe.cu').read_text()
    assert v4[v4.index('__global__ void wmma_partials('):v4.index('__global__ void wmma_consume(')] in header
    assert thin[thin.index('namespace v5 {'):thin.index('#ifndef CANDIDATE')] in header
    # Probe-only harness code (sentinels, timing, cuBLAS comparison) never enters the runtime.
    for probe_only in ('cudaMemset','cudaEventRecord','random_bf16','V4_HEADER','CANDIDATE','v7diag','DIAGNOSTIC'):
        assert probe_only not in header
    assert 'CUBLAS_COMPUTE_32F_PEDANTIC' not in header and 'cp.async.' not in header and 'cp.reduce' not in header
    assert 'parts>16||k%parts||(k/parts)%16' in header
    assert 'reinterpret_cast<uintptr_t>(w)%16||k%8' in header
    for kernel in ('v6::mma_thin_partials<8>','v5::mma_thin_partials<4>','v5::mma_thin_partials<16>'):
        assert kernel in header
    source=(output/'native/src/lfm3_main.cu').read_text()
    assert '#include "../tests/lfm3_thin_hybrid_runtime_generated.cuh"' in source
    assert source.count('cublasGemmBatchedEx')==2
    assert 'candidate_linear.run(blas,w(name),x,out,rows,input,output);' in source
    assert 'CUDA::cublasLt' in (output/'CMakeLists.txt').read_text()
    with pytest.raises(FileExistsError):prepare(ROOT,output)


def test_v4_inputs_are_unchanged_by_hybrid(tmp_path):
    # The v4 evidence binds this probe source; the hybrid must read it, never alter it.
    v4=ROOT/'native/probes/lfm3_wmma_policy_probe.cu'
    assert hashlib.sha256(v4.read_bytes()).hexdigest().startswith('1c6ec2f0f9c781d1d3bb')
    prepare(ROOT,tmp_path/'mirror')
    assert hashlib.sha256(v4.read_bytes()).hexdigest().startswith('1c6ec2f0f9c781d1d3bb')


def test_region_rejects_anchor_changes():
    with pytest.raises(ValueError):region('no anchors here','namespace v5 {','#ifndef CANDIDATE')
    with pytest.raises(ValueError):region('namespace v5 { namespace v5 { #ifndef CANDIDATE','namespace v5 {','#ifndef CANDIDATE')


def test_large_thin_selects_v7_without_shared_memory_kernels(tmp_path):
    m=prepare(ROOT,tmp_path/'v7',large_thin='v7')
    header=(tmp_path/'v7'/HEADER).read_text()
    assert m['large_thin']=='v7' and '#define LFM3_THIN_POLICY 7' in header
    assert 'v7::mma_thin_partials<4>' in header and 'v7::mma_thin_partials<8>' in header
    d=prepare(ROOT,tmp_path/'v6')
    assert d['large_thin']=='v6' and '#define LFM3_THIN_POLICY 6' in (tmp_path/'v6'/HEADER).read_text()
    e=prepare(ROOT,tmp_path/'v8',large_thin='v8')
    h8=(tmp_path/'v8'/HEADER).read_text()
    assert e['large_thin']=='v8' and '#define LFM3_THIN_POLICY 8' in h8 and 'v8::mma_thin_partials<8>' in h8
    assert '__shfl_xor_sync' in h8 and 'v7diag' not in h8
    with pytest.raises(ValueError):prepare(ROOT,tmp_path/'bad',large_thin='v9')
