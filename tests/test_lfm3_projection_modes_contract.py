"""Host-only source contracts: no CUDA compilation, execution or integration."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "native/probes/lfm3_projection_modes_probe.cu").read_text()
GOLDEN = (ROOT / "native/probes/lfm3_projection_probe.cu").read_text()


def test_cli_is_a_strict_whitelist_before_cuda():
    assert 'if(argc!=5)' in SOURCE
    assert 'if(mode!="default" && mode!="pedantic" && mode!="default-algo")' in SOURCE
    assert 'if(!gemm && !missing && producer!="kernel")' in SOURCE
    assert 'find_first_not_of("0123456789")' in SOURCE
    assert 'rows<1 || rows>257 || (width!=128 && width!=2048)' in SOURCE
    assert SOURCE.index('throw std::runtime_error("bad mode")') < SOURCE.index('cudaMalloc(')


def test_supported_mode_mapping_has_no_fallback():
    assert 'mode=="pedantic"?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F' in SOURCE
    assert 'mode=="default"?CUBLAS_GEMM_DEFAULT_TENSOR_OP:CUBLAS_GEMM_DEFAULT' in SOURCE
    assert 'if(mode=="pedantic") check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));' in SOURCE
    assert 'output,compute,algo));' in SOURCE
    assert SOURCE.count('cublasGemmEx(') == 1
    assert 'cuBLAS status=' in SOURCE
    assert 'unsupported/error is failure' in SOURCE
    assert 'return 1;' in SOURCE


def test_kernels_math_and_controls_match_golden():
    for name in ('fill', 'read_output', 'varied_weights'):
        pattern = rf'__global__ void {name}\(.*?\n\}}'
        assert re.search(pattern, SOURCE, re.S).group() == re.search(pattern, GOLDEN, re.S).group()
    for line in (
        'check(cudaMalloc(&c,n*2)); check(cudaMalloc(&d,n*2)); check(cudaMalloc(&workspace,8519680));',
        'varied_weights<<<(size_t(output)*width+255)/256,256>>>(a,size_t(output)*width,width);',
        'fill<<<(size_t(rows)*width+255)/256,256>>>(b,size_t(rows)*width,0.5f);',
        '} else if(!missing) fill<<<(n+255)/256,256>>>(c,n,float(width));',
        'const size_t sample=std::min(n,size_t(128));',
        'read_output<<<1,256>>>(c,d,sample);',
        'if(!missing) for(size_t i=0;i<sample;++i) {',
        'const float expected=gemm?float(width)*float(int(i%13)-6)/16.0f:float(width);',
        'if(__bfloat162float(host[i])!=expected) throw std::runtime_error("incorrect projection output");',
    ):
        assert line in SOURCE and line in GOLDEN
    assert 'float alpha=1,beta=0;' in SOURCE
    assert 'b,CUDA_R_16BF,width,&beta,c,CUDA_R_16BF,output,compute,algo)' in SOURCE
    assert re.findall(r'(\w+)<<<[^\n]*?>>>\(c,', SOURCE) == ['fill', 'read_output']
    # The only producer writing c outside GEMM is the guarded positive fill control.
    for forbidden in ('cudaMemset', 'LD_PRELOAD', 'setenv(', 'getenv(', 'system(', 'dlopen(', 'dlsym('):
        assert forbidden not in SOURCE


def test_report_identity_and_numerical_only_scope():
    for token in ('__CUDACC_VER_MAJOR__', '__CUDACC_VER_MINOR__', '__CUDACC_VER_BUILD__',
                  'CUDART_VERSION', 'CUBLAS_VER_MAJOR', 'CUBLAS_VER_MINOR',
                  'CUBLAS_VER_PATCH', 'CUBLAS_VER_BUILD', 'cublasGetVersion(handle,&version)',
                  'cublasGetMathMode(handle,&math)', 'numerical-only (not sanitizer acceptance)'):
        assert token in SOURCE
    assert SOURCE.index('"LIBRARY cublas="') < SOURCE.index('cublasSetMathMode(')
    assert SOURCE.index('"CONFIG producer="') < SOURCE.index('cudaMalloc(')
    assert 'missing?"INTENTIONAL UNINITIALIZED CONTROL: ":"PASS: "' in SOURCE
    assert len(SOURCE.splitlines()) < 130
