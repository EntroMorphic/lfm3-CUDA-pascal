"""CPU model of the Thor engine's first-layer tensors, checked against its pinned trace hashes.

Step 1 of validation/tensor-core-model-v1: the embedding rows and layer 0's operator_norm of each forward fixture,
computed in the norm kernel's own order, and saved when they hash to the Thor pins (the exact inputs of the first
prefill GEMM).   python3 tcm.py INPUT_IDS.json WORK_DIR"""
import hashlib, json, struct, sys
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
MODEL=str(ROOT.parent/'models/LFM3-Omni-2B/model.safetensors')
PINS=str(ROOT/'validation/g4-gates-v2/pins/forward.json')   # the Thor engine's pinned trace hashes
P='model.language_model.'
class Weights:
    def __init__(self):
        self.f=open(MODEL,'rb'); n=struct.unpack('<Q',self.f.read(8))[0]; self.h=json.loads(self.f.read(n)); self.base=8+n
        self.m=np.memmap(MODEL,dtype=np.uint16,mode='r',offset=self.base)
    def u16(self,name):
        t=self.h[P+name]; a,b=t['data_offsets']; return self.m[a//2:b//2].reshape(t['shape'])
def f32(u16): return (u16.astype(np.uint32)<<16).view(np.float32)
def bf16(x):  # float32 -> bf16 bits, round to nearest even (finite, in range)
    u=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
def sha(u16): return hashlib.sha256(np.ascontiguousarray(u16,dtype='<u2').tobytes()).hexdigest()
def pins(case): return json.load(open(PINS))[case]['files']
def norm_threads(rows,width):
    def power(n):
        x=1
        while x<=n//2 and x<512: x*=2
        return x
    dim0,dim1=power(width//4),power(rows); bx=min(dim0,32); by=min(dim1,512//bx); return min(dim0,512//by)
def norm_sums(xf,threads):
    """The norm kernel's sum of squares per row, in its order (FP32, no FMA)."""
    rows,width=xf.shape; sq=(xf*xf).astype(np.float32)   # __fmul_rn(v,v)
    acc=np.zeros((rows,threads,4),np.float32)
    # thread t, lane j: d = t*4 + j + i*threads*4
    per=sq.reshape(rows,width//(threads*4),threads,4)
    for i in range(per.shape[1]): acc=(acc+per[:,i]).astype(np.float32)
    s=(((acc[:,:,0]+acc[:,:,1]).astype(np.float32)+acc[:,:,2]).astype(np.float32)+acc[:,:,3]).astype(np.float32)  # [rows,threads]
    step=threads//2
    while step>=32:
        s[:,:step]=(s[:,:step]+s[:,step:2*step]).astype(np.float32); step//=2
    v=s[:,:32].copy()
    for off in (16,8,4,2,1):
        sh=v.copy(); sh[:,:32-off]=v[:,off:]   # shfl_down: lanes past the end keep their own value
        v=(v+sh).astype(np.float32)
    return v[:,0]
def rsqrt_candidates(x):
    x=x.astype(np.float32); d=1.0/np.sqrt(x.astype(np.float64))
    return {'correct':d.astype(np.float32), 'two-step':(np.float32(1)/np.sqrt(x)).astype(np.float32)}
def norm(xu,wu,eps,scale):
    xf=f32(xu); y=bf16((xf*scale[:,None]).astype(np.float32))          # __float2bfloat16_rn(__fmul_rn(x,scale))
    return bf16((f32(y)*f32(wu)[None,:]).astype(np.float32))            # __hmul: exact product, one rounding
if __name__=='__main__':
    W=Weights(); ids=json.load(open(sys.argv[1])); eps=1e-5
    for case in ('single','arithmetic','baseline','window257'):
        pin=pins(case); x=W.u16('embed_tokens.weight')[ids[case]]
        print(case,'embed', 'MATCH' if sha(x)==pin['0.embed.bf16'] else 'differ', end=' | ')
        rows,width=x.shape; t=norm_threads(rows,width); sums=norm_sums(f32(x),t)
        arg=((sums*np.float32(1.0/width)).astype(np.float32)+np.float32(eps)).astype(np.float32)
        for name,scale in rsqrt_candidates(arg).items():
            y=norm(x,W.u16('layers.0.operator_norm.weight'),eps,scale)
            print(f'operator_norm[{name}, {t} threads]', 'MATCH' if sha(y)==pin['0.layers.0.operator_norm.bf16'] else 'differ', end=' | ')
            if sha(y)==pin['0.layers.0.operator_norm.bf16']: np.save(f'{sys.argv[2]}/{case}.operator_norm.npy',y)
        print()
