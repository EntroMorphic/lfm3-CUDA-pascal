"""All of layer 0 (a convolution layer) of a forward fixture on the CPU, each traced tensor against its Thor pin.
GEMMs use the published tensor-core model (T-FDPA, L 16, F 25, RZ); everything else is the engine's kernels
transcribed in IEEE FP32. Where a kernel calls a library function or a BF16 intrinsic whose result on Blackwell is
not specified (expf, __hadd), the candidates are tried and the one that matches is reported.
    python3 layer0.py INPUT_IDS.json CASE..."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from tcm import Weights, f32, bf16, sha, pins, norm_threads, norm_sums
F32=np.float32
def to_rz(x):
    r=x.astype(F32); o=np.abs(r.astype(np.float64))>np.abs(x); r[o]=np.nextafter(r[o],F32(0)); return r
def exp_of(v):
    _,e=np.frexp(v); e=e.astype(np.int64)-1; e[v==0]=-100000; return e
def _gemm_rows(job):
    name,xu=job; W=Weights(); wf=f32(W.u16(name)).astype(np.float64); xf=f32(xu).astype(np.float64)
    acc=np.zeros((xf.shape[0],wf.shape[0]),F32)
    for k in range(0,xf.shape[1],16):
        xs=xf[:,k:k+16]; ws=wf[:,k:k+16]; P=xs[:,None,:]*ws[None,:,:]
        ek=np.where(P==0,-100000,exp_of(xs)[:,None,:]+exp_of(ws)[None,:,:])
        emax=np.maximum(exp_of(acc.astype(np.float64)),ek.max(axis=-1)); zero=emax<-50000
        q=np.ldexp(1.0,np.where(zero,0,emax-25).astype(np.int32))
        T=np.concatenate([acc.astype(np.float64)[...,None],P],axis=-1)
        acc=to_rz(np.trunc(T/q[...,None]).sum(axis=-1)*q)
    return bf16(acc)
def gemm(pool,name,xu,chunk=4):
    return np.concatenate(pool.map(_gemm_rows,[(name,xu[r:r+chunk]) for r in range(0,xu.shape[0],chunk)]))
def hmul(au,bu): return bf16((f32(au)*f32(bu)).astype(F32))       # exact product in FP32, one rounding
def hadd(au,bu,how):
    if how=='fp32-then-bf16': return bf16((f32(au)+f32(bu)).astype(F32))
    s=f32(au).astype(np.float64)+f32(bu).astype(np.float64)           # exact; one rounding to BF16, nearest even
    r=bf16(s.astype(F32)); # correct double rounding: recompute by comparing neighbours
    lo=f32(r).astype(np.float64); u=r.astype(np.int32)
    up=f32((u+1).astype(np.uint16)).astype(np.float64); dn=f32((u-1).astype(np.uint16)).astype(np.float64)
    best=r.copy()
    for cand,cu in ((up,u+1),(dn,u-1)):
        closer=np.abs(cand-s)<np.abs(f32(best).astype(np.float64)-s)
        tie=(np.abs(cand-s)==np.abs(f32(best).astype(np.float64)-s))&((cu&1)==0)&((best.astype(np.int32)&1)==1)
        best=np.where(closer|tie,cu.astype(np.uint16),best)
    return best
def norm(xu,wu,eps,residual=None,how=None):
    xf=f32(xu); rows,width=xf.shape; sums=norm_sums(xf,norm_threads(rows,width))
    arg=((sums*F32(1.0/width)).astype(F32)+F32(eps)).astype(F32)
    scale=(1.0/np.sqrt(arg.astype(np.float64))).astype(F32)
    y=hmul(bf16((xf*scale[:,None]).astype(F32)),np.broadcast_to(wu,xu.shape))
    return y if residual is None else hadd(y,residual,how)
def conv7(proj,wu):
    T,D=proj.shape[0],proj.shape[1]//3; inp=hmul(proj[:,:D],proj[:,2*D:]); w=f32(wu).reshape(D,7)
    s=np.zeros((T,D),F32)
    for k in range(7):
        prev=np.arange(T)+k-6; x=np.where((prev>=0)[:,None],f32(inp[np.maximum(prev,0)]),F32(0))
        s=(s+(x*w[None,:,k]).astype(F32)).astype(F32)
    return hmul(proj[:,D:2*D],bf16(s))
def swiglu(au,bu,how):
    x=f32(au)
    e=np.exp(-x.astype(np.float64)).astype(F32) if how=='exp-correct' else np.exp(-x).astype(F32)
    sig=(F32(1)/(F32(1)+e).astype(F32)).astype(F32)
    return hmul(bf16((x*sig).astype(F32)),bu)
def check(case,pin,key,candidates):
    """candidates: {label: array}; returns the first that hashes to the pin."""
    for label,v in candidates.items():
        if sha(v)==pin[key]:
            print(f'  {key:34s} MATCH' + (f'  [{label}]' if len(candidates)>1 else ''),flush=True); return v,label
    print(f'  {key:34s} differ ({", ".join(candidates)})',flush=True); return None,None
if __name__=='__main__':
    W=Weights(); ids=json.load(open(sys.argv[1])); eps=1e-5; L='layers.0.'
    with Pool(10) as pool:
        for case in sys.argv[2:]:
            pin=pins(case); t=time.time(); print(f'{case}: {len(ids[case])} rows')
            h,_=check(case,pin,'0.embed.bf16',{'':np.array(W.u16('embed_tokens.weight')[ids[case]])})
            n,_=check(case,pin,'0.'+L+'operator_norm.bf16',{'':norm(h,W.u16(L+'operator_norm.weight'),eps)})
            p,_=check(case,pin,'0.'+L+'conv.in_proj.bf16',{'':gemm(pool,L+'conv.in_proj.weight',n)})
            if p is None: continue
            c,_=check(case,pin,'0.'+L+'conv.product.bf16',{'':conv7(p,W.u16(L+'conv.conv.weight'))})
            if c is None: continue
            o,_=check(case,pin,'0.'+L+'operator.bf16',{'':gemm(pool,L+'conv.out_proj.weight',c)})
            if o is None: continue
            r,how=check(case,pin,'0.'+L+'residual.bf16',{k:norm(o,W.u16(L+'post_operator_norm.weight'),eps,h,k) for k in ('fp32-then-bf16','exact-then-bf16')})
            if r is None: continue
            n2,_=check(case,pin,'0.'+L+'ffn_norm.bf16',{'':norm(r,W.u16(L+'ffn_norm.weight'),eps)})
            if n2 is None: continue
            a,_=check(case,pin,'0.'+L+'feed_forward.w1.bf16',{'':gemm(pool,L+'feed_forward.w1.weight',n2)})
            b,_=check(case,pin,'0.'+L+'feed_forward.w3.bf16',{'':gemm(pool,L+'feed_forward.w3.weight',n2)})
            if a is None or b is None: continue
            g,_=check(case,pin,'0.'+L+'feed_forward.product.bf16',{k:swiglu(a,b,k) for k in ('exp-correct','exp-f32-libm')})
            if g is None: continue
            f,_=check(case,pin,'0.'+L+'feed_forward.bf16',{'':gemm(pool,L+'feed_forward.w2.weight',g,chunk=2)})
            if f is None: continue
            out,_=check(case,pin,'0.'+L+'output.bf16',{k:norm(f,W.u16(L+'post_ffn_norm.weight'),eps,r,k) for k in (how,)})
            print(f'  ({time.time()-t:.0f} s)',flush=True)
