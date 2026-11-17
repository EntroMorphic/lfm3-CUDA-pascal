"""CPU-only real-fixture extractor checks; no GPU acceptance claim."""
from pathlib import Path
import json
import struct
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_lfm3_lt_candidates import tensor


def fixture(path,entries,data=b'\x01\x02\x03\x04'):
    raw=json.dumps(entries).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+data)


def test_extracts_exact_weight_bytes_and_shape(tmp_path):
    p=tmp_path/'weights';fixture(p,{'model.layers.0.conv.in_proj.weight':dict(dtype='BF16',shape=[1,2],data_offsets=[0,4])})
    b,meta=tensor(p,'layers.0.conv.in_proj.weight')
    assert b==b'\x01\x02\x03\x04' and meta['shape']==[1,2]
    assert len(meta['tensor_sha256'])==len(meta['header_sha256'])==64


@pytest.mark.parametrize('damage',['ambiguous','missing','dtype','offset','truncated','header'])
def test_bad_weights_are_not_silently_used(tmp_path,damage):
    p=tmp_path/'weights';entry=dict(dtype='BF16',shape=[1,2],data_offsets=[0,4]);entries={'layers.0.conv.in_proj.weight':entry}
    if damage=='ambiguous':entries['other.layers.0.conv.in_proj.weight']=entry
    elif damage=='missing':entries={'other':entry}
    elif damage=='dtype':entry['dtype']='F16'
    elif damage=='offset':entry['data_offsets']=[-1,3]
    elif damage=='truncated':entry['data_offsets']=[0,8]
    fixture(p,entries)
    if damage=='header':p.write_bytes(struct.pack('<Q',100000001))
    with pytest.raises(ValueError):tensor(p,'layers.0.conv.in_proj.weight')


def test_standalone_algorithm_probe_has_no_production_mutations():
    source=(ROOT/'native/probes/lfm3_lt_policy_probe.cu').read_text()
    assert 'cudaMemset' not in source
    assert 'CUBLAS_COMPUTE_32F_PEDANTIC' not in source
    assert 'cublasLtMatmulAlgoCheck' in source and 'cublasLtMatmulAlgoGetIds' in source
    assert 'consumer_mismatches)throw' in source
