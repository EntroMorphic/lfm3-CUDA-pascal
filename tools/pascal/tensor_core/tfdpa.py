"""The published form of the model (MMA-Sim, arXiv:2511.10909, T-FDPA with L = 16, F = 25, final RZ, as reported for
Blackwell) against the Thor pins, next to this repository's own form. They differ only in where the window sits when
a product is the largest term: T-FDPA takes the sum of the operands' exponents, ours the product's leading bit.
    python3 tfdpa.py WORK_DIR CASE..."""
import sys, time
import numpy as np
from multiprocessing import Pool
from tcm import Weights, f32, bf16, sha, pins
D=sys.argv[1]
def to_rz(x):
    r=x.astype(np.float32); o=np.abs(r.astype(np.float64))>np.abs(x); r[o]=np.nextafter(r[o],np.float32(0)); return r
def exp_of(v):  # floor(log2 |v|) for nonzero finite v; a very small number for zero so it never sets the window
    _,e=np.frexp(v); e=e.astype(np.int64)-1; e[v==0]=-100000; return e
def step(acc,prod,ex,ew,form,F=25):
    T=np.concatenate([acc.astype(np.float64)[...,None],prod],axis=-1)
    if form=='tfdpa':
        ek=ex[:,None,:]+ew[None,:,:]; ek=np.where(prod==0,-100000,ek)
        emax=np.maximum(exp_of(acc.astype(np.float64)),ek.max(axis=-1))
    else:
        emax=exp_of(np.abs(T).max(axis=-1))
    zero=emax<-50000; q=np.ldexp(1.0,np.where(zero,0,emax-F).astype(np.int32))
    return to_rz(np.trunc(T/q[...,None]).sum(axis=-1)*q), int(np.count_nonzero(~zero))
def rows(job):
    case,r0,r1=job
    W=Weights(); wf=f32(W.u16('layers.0.conv.in_proj.weight')).astype(np.float64)
    xf=f32(np.load(f'{D}/{case}.operator_norm.npy')[r0:r1]).astype(np.float64)
    out={}; windows_differ=0; steps=0
    acc={f:np.zeros((xf.shape[0],wf.shape[0]),np.float32) for f in ('tfdpa','leading-bit')}
    for k in range(0,xf.shape[1],16):
        xs=xf[:,k:k+16]; ws=wf[:,k:k+16]; P=xs[:,None,:]*ws[None,:,:]; ex=exp_of(xs); ew=exp_of(ws)
        # how often the two forms place the window differently, on the T-FDPA trajectory
        a=acc['tfdpa'].astype(np.float64); ek=np.where(P==0,-100000,ex[:,None,:]+ew[None,:,:])
        e1=np.maximum(exp_of(a),ek.max(axis=-1)); e2=exp_of(np.abs(np.concatenate([a[...,None],P],axis=-1)).max(axis=-1))
        windows_differ+=int(np.count_nonzero(e1!=e2)); steps+=e1.size
        for f in acc: acc[f],_=step(acc[f],P,ex,ew,f)
    return r0,{f:bf16(v) for f,v in acc.items()},windows_differ,steps
if __name__=='__main__':
    for case in sys.argv[2:]:
        R=np.load(f'{D}/{case}.operator_norm.npy').shape[0]; t=time.time()
        with Pool(8) as pool: parts=sorted(pool.map(rows,[(case,r,min(r+16,R)) for r in range(0,R,16)]),key=lambda p:p[0])
        pin=pins(case)['0.layers.0.conv.in_proj.bf16']; wd=sum(p[2] for p in parts); st=sum(p[3] for p in parts)
        outs={f:np.concatenate([p[1][f] for p in parts]) for f in ('tfdpa','leading-bit')}
        print(f'{case}: T-FDPA {"MATCH" if sha(outs["tfdpa"])==pin else "differ"}, leading-bit {"MATCH" if sha(outs["leading-bit"])==pin else "differ"}; '
              f'outputs differing between the two forms {int(np.count_nonzero(outs["tfdpa"]!=outs["leading-bit"]))} of {outs["tfdpa"].size}; '
              f'steps where the windows differ {wd} of {st} ({100*wd/st:.3f}%)  ({time.time()-t:.0f} s)',flush=True)
