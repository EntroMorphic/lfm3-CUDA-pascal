"""Generate an isolated v5/v6 thin-hybrid runtime candidate; never change production or goldens.

Identical to prepare_lfm3_wmma_runtime.py (v4) except: the v5 and v6 thin producers from
native/probes/lfm3_thin_v5_probe.cu are added verbatim, and the thin dispatch comes from
native/probes/lfm3_thin_hybrid_runtime_candidate.cuh. v4's inputs are read, never modified.
"""
import argparse
import json
from pathlib import Path
from prepare_lfm3_pedantic_diagnostic import checked,digest,CMAKE,BUILD_POLICY
from prepare_lfm3_blas_policy import once

ROOT=Path(__file__).resolve().parents[1]
HEADER='native/tests/lfm3_thin_hybrid_runtime_generated.cuh'


def region(text,start,end):
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('kernel region anchor count changed')
    return text[text.index(start):text.index(end)]


def prepare(root,output,large_thin='v6'):
    if large_thin not in ('v6','v7','v8'):raise ValueError('large_thin must be v6, v7 or v8')
    root=Path(root).absolute();output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    names=['native/src/lfm3_main.cu','native/src/lfm3_worker.cu','native/src/lfm3_decode_batch.cuh',
           'native/src/safetensors.cpp','native/probes/lfm3_decode_batch_probe.cu','LICENSE','native/third_party/PYTORCH_LICENSE']
    for p in sorted(checked(root,'native/include').rglob('*')):
        checked(root,p.relative_to(root))
        if p.is_file() and p.suffix in ('.hpp','.cuh'):names.append(p.relative_to(root).as_posix())
    generated={};records={}
    for name in names:
        original=checked(root,name).read_bytes();modified=original
        if name=='native/src/lfm3_main.cu':
            text=once(original.decode(),'#include <cublas_v2.h>','#include <cublas_v2.h>\n#include "../tests/lfm3_thin_hybrid_runtime_generated.cuh"')
            text=once(text,'  void linear(const std::string& name, const bf16* x, bf16* out, int rows, int input, int output) {',
                      '  lfm3_wmma_candidate::Linear candidate_linear;\n'
                      '  void linear(const std::string& name, const bf16* x, bf16* out, int rows, int input, int output) {')
            before='''    float alpha=1, beta=0;
    // Row-major X W^T = column-major W X^T, with W stored row-major.
    blas_check(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,output,rows,input,&alpha,
      w(name),CUDA_R_16BF,input,x,CUDA_R_16BF,input,&beta,out,CUDA_R_16BF,output,
      CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));'''
            text=once(text,before,'    candidate_linear.run(blas,w(name),x,out,rows,input,output);')
            modified=text.encode()
        generated[name]=modified;records[name]=dict(original_sha256=digest(original),candidate_sha256=digest(modified),changed=original!=modified)
    v4_probe=checked(root,'native/probes/lfm3_wmma_policy_probe.cu').read_text()
    v4_kernels=region(v4_probe,'__global__ void wmma_partials(','__global__ void wmma_consume(')
    thin_probe=checked(root,'native/probes/lfm3_thin_v5_probe.cu').read_text()
    thin_kernels=region(thin_probe,'namespace v5 {','#ifndef CANDIDATE')
    template=checked(root,'native/probes/lfm3_thin_hybrid_runtime_candidate.cuh').read_text()
    header=once(template,'// The generator places the exactly tested WMMA kernels before this class.',
                'using bf16=__nv_bfloat16;\n'+v4_kernels+thin_kernels)
    header=once(header,'#include <cublasLt.h>','#include <mma.h>\n#include <cublasLt.h>')
    header=once(header,'#pragma once\n','#pragma once\n#define LFM3_THIN_POLICY '+large_thin[1:]+'\n')
    generated[HEADER]=header.encode()
    cmake=CMAKE.replace('lfm3_pedantic_diagnostic','lfm3_thin_hybrid_runtime_candidate').replace('CUDA::cudart CUDA::cublas)','CUDA::cudart CUDA::cublas CUDA::cublasLt)')
    cmake+='diagnostic_target(lfm3-decode-batch-probe native/probes/lfm3_decode_batch_probe.cu)\n'
    manifest=dict(full_acceptance=False,diagnostic_only=True,goldens_may_change=False,required_eager_checks=6890,
                  scope=f'native WMMA for first-heuristic ID66 only, v5/{large_thin} thin hybrid; other GEMMs unchanged; not accepted',large_thin=large_thin,
                  build_policy=BUILD_POLICY,files=records,generated_header=HEADER,generated_header_sha256=digest(header.encode()),
                  kernel_source_sha256=digest(v4_probe.encode()),thin_source_sha256=digest(thin_probe.encode()),
                  v4_kernel_region_sha256=digest(v4_kernels.encode()),thin_kernel_region_sha256=digest(thin_kernels.encode()),
                  header_template_sha256=digest(template.encode()),cmake_sha256=digest(cmake.encode()))
    output.mkdir(parents=True,exist_ok=False)
    for name,data in generated.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (output/'CMakeLists.txt').write_text(cmake);(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--large-thin',choices=('v6','v7','v8'),default='v6');a=p.parse_args();prepare(ROOT,a.output,a.large_thin)
