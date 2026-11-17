import sys, time
import numpy as np
from multiprocessing import Pool
from tcm import Weights, f32, bf16, sha, pins
D=sys.argv[1]
def to_rz(x):
    r=x.astype(np.float32); o=np.abs(r.astype(np.float64))>np.abs(x); r[o]=np.nextafter(r[o],np.float32(0)); return r
def mma(acc,prod,g=2):
    T=np.concatenate([acc.astype(np.float64)[...,None],prod],axis=-1)
    m=np.abs(T).max(axis=-1); _,e=np.frexp(m); q=np.ldexp(1.0,e-1-23-g); q[m==0]=1.0
    return to_rz(np.trunc(T/q[...,None]).sum(axis=-1)*q)
def rows(job):
    case,r0,r1=job
    W=Weights(); wf=f32(W.u16('layers.0.conv.in_proj.weight')).astype(np.float64)
    xf=f32(np.load(f'{D}/{case}.operator_norm.npy')[r0:r1]).astype(np.float64)
    acc=np.zeros((xf.shape[0],wf.shape[0]),np.float32)
    for k in range(0,xf.shape[1],16): acc=mma(acc,xf[:,None,k:k+16]*wf[None,:,k:k+16])
    return r0,bf16(acc)
if __name__=='__main__':
    for case in sys.argv[2:]:
        R=np.load(f'{D}/{case}.operator_norm.npy').shape[0]; t=time.time()
        with Pool(8) as pool: parts=sorted(pool.map(rows,[(case,r,min(r+16,R)) for r in range(0,R,16)]),key=lambda p:p[0])
        out=np.concatenate([p[1] for p in parts]); ok=sha(out)==pins(case)['0.layers.0.conv.in_proj.bf16']
        print(f'{case}: {R} rows x 6144, {out.size} values: {"MATCH" if ok else "differ"} with Thor pin ({time.time()-t:.0f} s)',flush=True)
        if ok: np.save(f'{D}/{case}.in_proj.npy',out)
