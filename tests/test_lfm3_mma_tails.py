"""Host-only independent fixture oracle checks; no GPU acceptance claim."""
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from check_lfm3_mma_tails import fixture,SHAPES
from check_lfm3_projection_bedrock import bf16


@pytest.mark.parametrize('shape',SHAPES)
def test_tail_oracle_uses_every_measured_input_word(shape):
    m,n,k,parts=shape;a,b,expected=fixture(m,n,k)
    def decode(data):return [struct.unpack('<f',struct.pack('<I',v<<16))[0] for (v,) in struct.iter_unpack('<H',data)]
    left,right=decode(a),decode(b)
    literal=[bf16(sum(left[row*k+i]*right[col*k+i] for i in range(k))) for col in range(n) for row in range(m)]
    assert literal==[v for (v,) in struct.iter_unpack('<H',expected)]
    assert len(expected)==m*n*2 and k%parts==0 and (k//parts)%16==0
