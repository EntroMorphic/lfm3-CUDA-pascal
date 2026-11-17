"""Host-only SDK model tests. The model does not establish CUDA callback timing."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def model(tmp_path_factory):
    compiler=shutil.which('g++');assert compiler
    d=tmp_path_factory.mktemp('boundary-observer-model')
    (d/'sanitizer.h').write_text(r'''
#pragma once
#include <cstdint>
#include <cstddef>
#define SANITIZERAPI
using CUcontext=void*;using CUstream=void*;using Sanitizer_SubscriberHandle=void*;
using Sanitizer_CallbackDomain=int;using Sanitizer_CallbackId=int;using SanitizerResult=int;
constexpr int SANITIZER_SUCCESS=0,SANITIZER_CB_DOMAIN_RESOURCE=1,SANITIZER_CB_DOMAIN_LAUNCH=2;
constexpr int SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC=1,SANITIZER_CBID_RESOURCE_MODULE_LOADED=2,SANITIZER_CBID_LAUNCH_BEGIN=1;
struct Sanitizer_ResourceMemoryData {uint64_t address,size;CUcontext context;};
struct Sanitizer_ResourceModuleData {CUcontext context;void* module;size_t cubinSize;const char* pCubin;};
struct Sanitizer_LaunchData {CUcontext context;CUstream stream;void* hStream;void* module;const char* functionName;uint64_t gridId;unsigned gridDim_x=1,gridDim_y=1,gridDim_z=1,blockDim_x=256,blockDim_y=1,blockDim_z=1,clusterDim_x=1,clusterDim_y=1,clusterDim_z=1;};
using Fn=void(*)(void*,int,int,const void*);
int sanitizerSubscribe(void**,Fn,void*);
int sanitizerEnableDomain(int,void*,int);
int sanitizerMemcpyDeviceToHost(void*,void*,size_t,void*);
''')
    (d/'model.cpp').write_text(r'''
#include "sanitizer.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
Fn callback=nullptr;void* user=nullptr;
std::string fault(){const char* v=std::getenv("MODEL_FAULT");return v?v:"";}
int sanitizerSubscribe(void** h,Fn f,void* u){callback=f;user=u;*h=(void*)1;return fault()=="subscribe"?1:0;}
int sanitizerEnableDomain(int,void*,int){return 0;}
int sanitizerMemcpyDeviceToHost(void* d,void* s,size_t n,void*){if(fault()=="copy")return 1;std::memcpy(d,s,n);return 0;}
int main(){
  std::vector<unsigned char> workspace(8519680,0xa5),out(49152,0x5a),sink(49152,0x3c);
  auto alloc=[&](auto& data){Sanitizer_ResourceMemoryData d{(uint64_t)data.data(),data.size(),(void*)42};callback(user,1,1,&d);};
  alloc(workspace);alloc(out);alloc(sink);if(fault()=="duplicate")alloc(workspace);
  const char image[]="\177ELF boundary_control_fill";
  Sanitizer_ResourceModuleData module{(void*)42,(void*)43,sizeof(image),image};callback(user,1,2,&module);
  bool gemm=std::string(std::getenv("LFM_BOUNDARY_MODE"))=="gemm";
  auto launch=[&](const char* name,uint64_t id){Sanitizer_LaunchData d{};d.context=(void*)42;d.stream=fault()=="stream"?(void*)1:nullptr;d.hStream=(void*)44;d.module=(void*)43;d.functionName=name;d.gridId=id;callback(user,2,1,&d);};
  if(fault()!="missing"){launch(gemm?"nvjet_fixture":"boundary_control_fill",1);std::fill(workspace.begin(),workspace.begin()+49152,0x11);}
  launch(gemm?"splitKreduce_kernel":"boundary_control_reduce",2);std::fill(out.begin(),out.end(),0x22);
  launch(gemm?"consume_all":"boundary_control_consume",3);return 0;
}
''')
    binary=d/'model'
    subprocess.run([compiler,'-std=c++17','-pthread','-I',str(d),str(ROOT/'native/probes/lfm3_boundary_observer.cpp'),str(d/'model.cpp'),'-o',str(binary)],check=True,capture_output=True,text=True)
    return binary


@pytest.mark.parametrize('mode',['control','gemm'])
@pytest.mark.parametrize('fault',['','missing','copy','duplicate','stream','subscribe'])
def test_observer_state_and_read_only_snapshots(model,tmp_path,mode,fault):
    env=dict(os.environ,LFM_BOUNDARY_OUTPUT=str(tmp_path),LFM_BOUNDARY_MODE=mode,MODEL_FAULT=fault)
    env.pop('LD_PRELOAD',None)
    r=subprocess.run([str(model)],env=env,capture_output=True,text=True,timeout=5)
    assert r.returncode==0
    events=[json.loads(x) for x in (tmp_path/'observer.jsonl').read_text().splitlines()]
    end=events[-1];assert end['observer_complete'] and end['full_acceptance'] is False
    if fault:
        assert end['failed'] or not end['initialized']
    else:
        assert not end['failed'] and end['initialized'] and end['captures']==3 and end['launches']==3
        assert (tmp_path/'before_producer.bin').read_bytes()==bytes([0xa5])*49152
        assert (tmp_path/'before_reducer.bin').read_bytes()==bytes([0x11])*49152
        assert (tmp_path/'before_consumer.bin').read_bytes()==bytes([0x11])*49152
        if mode=='control':
            assert (tmp_path/'before_reducer_output.bin').read_bytes()==bytes([0x5a])*49152
            assert (tmp_path/'before_consumer_output.bin').read_bytes()==bytes([0x22])*49152
            assert (tmp_path/'before_consumer_sink.bin').read_bytes()==bytes([0x3c])*49152
        modules=[e for e in events if 'module_image' in e]
        assert modules and (tmp_path/modules[0]['module_image']).read_bytes().startswith(b'\x7fELF')


def test_no_forbidden_cuda_calls_or_device_mutations_in_observer():
    import re
    source=(ROOT/'native/probes/lfm3_boundary_observer.cpp').read_text()
    assert not re.search(r'\b(?:cuda\w+|cu[A-Z]\w*|sanitizerMemset|sanitizerMemcpyHostToDeviceAsync|sanitizerPatch\w*)\s*\(',source)
    assert 'sanitizerMemcpyDeviceToHost(' in source
