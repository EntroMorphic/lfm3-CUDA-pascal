"""Prepare an isolated cuBLAS policy candidate; never modifies production or goldens."""
import argparse
import json
from pathlib import Path

from prepare_lfm3_pedantic_diagnostic import checked,digest,CMAKE,BUILD_POLICY

ROOT=Path(__file__).resolve().parents[1]
POLICIES={'workspace-zero':(0,False),'workspace-16k':(16384,False),
          'no-reduced-reduction':(8519680,True)}


def once(text,before,after):
    if text.count(before)!=1:raise ValueError('source anchor count changed: '+before)
    return text.replace(before,after)


def prepare(root,output,policy):
    root=Path(root).absolute();output=Path(output).absolute()
    if policy not in POLICIES:raise ValueError('unknown policy')
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    budget,disallow=POLICIES[policy]
    files=['native/src/lfm3_main.cu','native/src/lfm3_worker.cu','native/src/lfm3_decode_batch.cuh',
           'native/src/safetensors.cpp','native/probes/lfm3_decode_batch_probe.cu',
           'native/probes/lfm3_projection_bedrock_probe.cu','LICENSE','native/third_party/PYTORCH_LICENSE']
    for p in sorted(checked(root,'native/include').rglob('*')):
        checked(root,p.relative_to(root))
        if p.is_file() and p.suffix in ('.hpp','.cuh'):files.append(p.relative_to(root).as_posix())
    generated={};records={}
    for name in files:
        original=checked(root,name).read_bytes();modified=original
        if name=='native/src/lfm3_main.cu':
            text=once(original.decode(),'cublasSetWorkspace(blas, workspace.p, 4096*1024*2+16*1024*8)',
                      f'cublasSetWorkspace(blas, workspace.p, {budget})')
            if disallow:text=once(text,'cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH)',
                                 'cublasSetMathMode(blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)')
            modified=text.encode()
        elif name=='native/probes/lfm3_projection_bedrock_probe.cu':
            text=once(original.decode(),'cublasSetWorkspace(handle,workspace,8519680)',
                      f'cublasSetWorkspace(handle,workspace,{budget})')
            if disallow:
                text=once(text,'if(math=="pedantic")check(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));',
                          'if(math!="default")throw std::runtime_error("candidate requires default compute/math arguments");\n'
                          '  check(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));')
            # Allocation remains unchanged; explicitly distinguish budget from allocation metadata.
            text=once(text,'stage("before_cuda_setup");',
                      f'std::cout<<"{{\\\"candidate_policy\\\":\\\"{policy}\\\",\\\"workspace_budget_bytes\\\":{budget},\\\"full_acceptance\\\":false}}\\n";\n  stage("before_cuda_setup");')
            modified=text.encode()
        generated[name]=modified
        records[name]=dict(original_sha256=digest(original),candidate_sha256=digest(modified),changed=modified!=original)
    cmake=CMAKE.replace('lfm3_pedantic_diagnostic','lfm3_blas_policy_candidate')+'''diagnostic_target(lfm3-decode-batch-probe native/probes/lfm3_decode_batch_probe.cu)
add_executable(lfm3-policy-probe native/probes/lfm3_projection_bedrock_probe.cu)
target_compile_features(lfm3-policy-probe PRIVATE cuda_std_17)
target_link_libraries(lfm3-policy-probe PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(lfm3-policy-probe PROPERTIES CUDA_ARCHITECTURES 110)
'''
    manifest=dict(policy=policy,workspace_budget_bytes=budget,disallow_reduced_reduction=disallow,
                  diagnostic_only=True,full_acceptance=False,goldens_may_change=False,
                  required_eager_checks=6890,build_policy=BUILD_POLICY,files=records,cmake_sha256=digest(cmake.encode()))
    output.mkdir(parents=True,exist_ok=False)
    for name,data in generated.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (output/'CMakeLists.txt').write_text(cmake)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',choices=POLICIES,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(ROOT,a.output,a.policy)


if __name__=='__main__':main()
