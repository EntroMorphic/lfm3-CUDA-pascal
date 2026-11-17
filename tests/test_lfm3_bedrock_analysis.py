"""Offline measurement replay counterexamples, no GPU use."""
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_lfm3_projection_bedrock import workspace_facts,log_bytes


def test_physical_change_counts_are_byte_and_word_counts_not_estimates():
    before=bytes(16);after=struct.pack('<4f',0,1,-2,3)
    facts=workspace_facts(before,after,after)
    assert facts['preimage_all_zero'] and facts['changed_fp32_words']==3
    assert facts['changed_bytes']==sum(x!=0 for x in after)
    assert facts['reference_word_mismatches']==0


@pytest.mark.parametrize('bad',[b'',b'\0',bytes(12),bytes(20)])
def test_incomplete_workspace_is_not_evidence(bad):
    with pytest.raises(ValueError):workspace_facts(None,bad,bytes(16))


def test_workspace_reference_is_bitwise_including_signed_zero():
    facts=workspace_facts(None,struct.pack('<I',0x80000000),bytes(4))
    assert facts['reference_word_mismatches']==1
    with pytest.raises(ValueError):workspace_facts(bytes(3),bytes(4),bytes(4))


def test_lossless_log_storage_is_read_as_original_bytes(tmp_path):
    import gzip
    data=b'========= example diagnostic\n'
    (tmp_path/'sanitizer.log.gz').write_bytes(gzip.compress(data,mtime=0))
    assert log_bytes(tmp_path,'sanitizer.log')==data
    (tmp_path/'sanitizer.log.gz').write_bytes(b'not gzip')
    with pytest.raises(OSError):log_bytes(tmp_path,'sanitizer.log')
