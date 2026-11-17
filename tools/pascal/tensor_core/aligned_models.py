"""Second family: fixed-point alignment. Within a block the accumulator and w products are aligned to the largest
term's exponent, each truncated to 24 + g bits below it (toward zero, or floor as two's complement does), summed
exactly, and the sum brought back to FP32 (rz, rn or floor). Blocks in k order."""
import sys, itertools, time
import numpy as np
from multiprocessing import Pool
from tcm import Weights, f32, bf16, sha, pins
D=sys.argv[1]; CASE=sys.argv[2]
def to_f32(x,mode):
    r=x.astype(np.float32)
    if mode=='rz':
        o=np.abs(r.astype(np.float64))>np.abs(x); r[o]=np.nextafter(r[o],np.float32(0))
    elif mode=='floor':
        o=r.astype(np.float64)>x; r[o]=np.nextafter(r[o],np.float32(-np.inf))
    return r
def block(acc,prod,g,trunc,final):
    T=np.concatenate([acc.astype(np.float64)[...,None],prod],axis=-1)
    m=np.abs(T).max(axis=-1); _,e=np.frexp(m)                      # m = f * 2^e, f in [0.5, 1): top bit at e-1
    q=np.ldexp(1.0,e-1-23-g); q[m==0]=1.0
    scaled=T/q[...,None]
    scaled=np.trunc(scaled) if trunc=='rz' else np.floor(scaled)
    return to_f32(scaled.sum(axis=-1)*q,final)
def run(cand):
    w,g,trunc,final=cand
    W=Weights(); wf=f32(W.u16('layers.0.conv.in_proj.weight')).astype(np.float64)
    xf=f32(np.load(f'{D}/{CASE}.operator_norm.npy')).astype(np.float64); R,K=xf.shape; C=wf.shape[0]
    acc=np.zeros((R,C),np.float32)
    for k in range(0,K,16):
        Pk=xf[:,None,k:k+16]*wf[None,:,k:k+16]
        for j in range(0,16,w): acc=block(acc,Pk[...,j:j+w],g,trunc,final)
    out=bf16(acc); return cand, sha(out)==pins(CASE)['0.layers.0.conv.in_proj.bf16'], out
if __name__=='__main__':
    cands=list(itertools.product((16,8,4),(0,1,2,3,4,5,6,8),('rz','floor'),('rz','rn','floor')))
    ref=np.load(f'{D}/{CASE}.in_proj.candidates.npz')['w16_rn']; t=time.time(); hits=[]
    with Pool(8) as pool:
        for cand,ok,out in pool.imap_unordered(run,cands):
            d=int(np.count_nonzero(out!=ref))
            if ok: hits.append(cand); np.save(f'{D}/{CASE}.in_proj.match.npy',out)
            print(f'w={cand[0]:2d} g={cand[1]} term={cand[2]:5s} final={cand[3]:5s}: {"MATCH" if ok else "differ"}  ({d} outputs differ from exact-16 rn)',flush=True)
    print(f'{len(cands)} candidates, {time.time()-t:.0f} s; matches: {hits}')
