"""Candidate arithmetic for the Thor prefill GEMM (mma.sync m16n8k16, BF16 in, FP32 accumulate), on the CPU.
Each output is a chain over K in blocks of 16 (one mma each): acc <- f(acc, 16 products). Candidates for f:
blocks of w products summed exactly, added to acc, rounded (rn: nearest even, rz: toward zero), in k order."""
import sys, json, time
import numpy as np
from tcm import Weights, f32, bf16, sha, pins
def two_sum_err(a,b,s):  # the rounding error of s = a + b in float64 (exactly), to count inexact steps
    bb=s-a; return (a-(s-bb))+(b-bb)
def to_f32(x,mode):
    r=x.astype(np.float32)
    if mode=='rz':
        over=np.abs(r.astype(np.float64))>np.abs(x)
        r[over]=np.nextafter(r[over],np.float32(0))
    return r
def thin_single(x,W):
    """thin_kernel<1>: lane l takes k = l*8 + i*256 .. +8, fma in order; then the xor butterfly."""
    xf=f32(x).astype(np.float64)[0]; wf=f32(W).astype(np.float64)   # [cols, K]
    K=xf.size; acc=np.zeros((W.shape[0],32),np.float32)
    for kb0 in range(0,K,256):
        for e in range(8):
            ks=kb0+np.arange(32)*8+e
            acc=(acc.astype(np.float64)+xf[ks][None,:]*wf[:,ks]).astype(np.float32)
    v=acc
    for off in (16,8,4,2,1):
        v=(v+v[:,np.arange(32)^off]).astype(np.float32)
    return bf16(v[:,0])[None,:]
def wide(x,W,candidates,rows_chunk=64):
    xf=f32(x).astype(np.float64); wf=f32(W).astype(np.float64); R,K=xf.shape; C=wf.shape[0]
    out={c:np.zeros((R,C),np.float32) for c in candidates}; inexact=0
    for r0 in range(0,R,rows_chunk):
        xs=xf[r0:r0+rows_chunk]; acc={c:np.zeros((xs.shape[0],C),np.float32) for c in candidates}
        for k in range(0,K,16):
            Pk=xs[:,None,k:k+16]*wf[None,:,k:k+16]          # exact: 16-bit significands
            sums={}
            for w in sorted({c[0] for c in candidates}):
                g=Pk.reshape(Pk.shape[0],C,16//w,w)
                s=g[...,0].copy()
                for i in range(1,w):
                    t=s+g[...,i]; inexact+=int(np.count_nonzero(two_sum_err(s,g[...,i],t))); s=t
                sums[w]=s
            for c in candidates:
                w,mode=c; a=acc[c]
                for j in range(16//w):
                    t=a.astype(np.float64)+sums[w][...,j]
                    a=to_f32(t,mode)
                acc[c]=a
        for c in candidates: out[c][r0:r0+rows_chunk]=acc[c]
    return {c:bf16(v) for c,v in out.items()}, inexact
if __name__=='__main__':
    W=Weights(); Wp=W.u16('layers.0.conv.in_proj.weight'); d=sys.argv[1]
    x=np.load(f'{d}/single.operator_norm.npy'); got=thin_single(x,Wp)
    print('control, single row (thin kernel, plain fma):', 'MATCH' if sha(got)==pins('single')['0.layers.0.conv.in_proj.bf16'] else 'differ', flush=True)
    cands=[(w,m) for w in (1,2,4,8,16) for m in ('rn','rz')]
    for case in sys.argv[2:]:
        x=np.load(f'{d}/{case}.operator_norm.npy'); t=time.time(); res,inexact=wide(x,Wp,cands); pin=pins(case)['0.layers.0.conv.in_proj.bf16']
        ref=res[(1,'rn')]
        print(f'{case}: {x.shape[0]} rows, {time.time()-t:.0f} s, inexact float64 block sums {inexact}')
        for c in cands:
            print(f'   block {c[0]:2d} {c[1]}: {"MATCH" if sha(res[c])==pin else "differ"}   ({int(np.count_nonzero(res[c]!=ref))} of {ref.size} BF16 outputs differ from sequential-fma rn)')
        np.savez(f'{d}/{case}.in_proj.candidates.npz',**{f'w{c[0]}_{c[1]}':v for c,v in res.items()})
