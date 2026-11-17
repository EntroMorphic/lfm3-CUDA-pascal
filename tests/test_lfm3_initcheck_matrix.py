import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[1]/"tools/check_lfm3_initcheck_matrix.py"
SPEC=importlib.util.spec_from_file_location("initcheck_matrix",PATH)
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("mode,code,log,app,expected",[
    ("kernel",0,"ERROR SUMMARY: 0 errors","PASS: 128 sampled outputs","clean"),
    ("gemm",0,"ERROR SUMMARY: 0 errors","PASS: 128 sampled outputs","clean"),
    ("missing",86,"Uninitialized __global__ memory read at read_output(","INTENTIONAL UNINITIALIZED CONTROL: 128 sampled outputs","negative-control-detected"),
    ("gemm",86,"Uninitialized __global__ memory read at read_output(","PASS: 128 sampled outputs","tracking-reproduced"),
    ("gemm",86,"Uninitialized __global__ memory read at read_output(","incorrect projection output","unexpected-failure"),
    ("gemm",1,"Uninitialized __global__ memory read at read_output(","PASS: 128 sampled outputs","unexpected-failure"),
    ("gemm",86,"Uninitialized __global__ memory read in a_library_kernel(","PASS: 128 sampled outputs","unexpected-failure"),
    ("missing",0,"ERROR SUMMARY: 0 errors","INTENTIONAL UNINITIALIZED CONTROL: 128 sampled outputs","unexpected-failure"),
    ("kernel",86,"Uninitialized __global__ memory read at read_output(","PASS: 128 sampled outputs","unexpected-failure"),
])
def test_matrix_does_not_mislabel_controls_or_numerical_failure(mode,code,log,app,expected):
    assert gate.classify(mode,code,log,app)==expected
    assert gate.classify(mode,code,log,app,timed_out=True)=="timeout"
