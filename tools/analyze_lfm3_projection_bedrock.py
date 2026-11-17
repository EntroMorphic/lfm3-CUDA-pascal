"""Replay saved bedrock measurements without GPU access; no acceptance inference."""
import argparse
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import re
import struct

from check_lfm3_projection_bedrock import make_inputs,periodic_units


def need(condition,message):
    if not condition:raise ValueError(message)


def log_bytes(directory,name):
    p=directory/name
    return p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())


def workspace_facts(before,after,expected):
    need(bool(after) and len(after)==len(expected) and len(after)%4==0,'workspace snapshot size mismatch')
    count=len(after)//4
    result=dict(bytes=len(after),fp32_words=count,reference_word_mismatches=sum(a!=b for a,b in zip(struct.unpack('<'+'I'*count,after),struct.unpack('<'+'I'*count,expected))),sha256=hashlib.sha256(after).hexdigest())
    if before is not None:
        need(len(before)==len(after),'workspace preimage size mismatch')
        result.update(preimage_all_zero=not any(before),changed_bytes=sum(a!=b for a,b in zip(before,after)),changed_fp32_words=sum(a!=b for a,b in zip(struct.unpack('<'+'I'*count,before),struct.unpack('<'+'I'*count,after))))
    return result


def analyze(root):
    def phase(name):
        d=root/name;s=json.loads((d/'summary.json').read_text());m=json.loads((d/'raw-manifest.json').read_text())['files']
        need(s['status']=='measurement-completed' and s['hashes_stable'] and not s['full_acceptance'],'phase incomplete or identity drift')
        gemms=[r for r in s['records'] if r['producer']=='gemm']
        need(bool(gemms),'no GEMM observations')
        for r in gemms:
            need(r['status']=='measured' and r['group_cleanup_complete'] and not r['timed_out'],'case incomplete')
            for kind in ('producer','consumer'):
                c=r[kind+'_comparison'];key=f"case-{r['index']:03d}/output.{kind}.bf16"
                need(c['checked_elements']==r['rows']*6144 and c['mismatch_count']==0,'numerical coverage/mismatch')
                need(c['sha256']==m[key]['sha256'] and m[key]['bytes']==r['rows']*6144*2,'raw output binding mismatch')
            for name in (f"case-{r['index']:03d}/sanitizer.log",f"case-{r['index']:03d}/application.log"):
                if name in m:need(hashlib.sha256(log_bytes(d,name)).hexdigest()==m[name]['sha256'],'raw log changed')
        return d,s,gemms
    _,numerical,ng=phase('projection-bedrock-numerical-v3')
    need(len(ng)==12 and numerical['instrumentation']=='none','wrong numerical fixture scope')
    need({(r['pattern'],r['rows'],r['repeat']) for r in ng}==set(itertools.product(('legacy','asymmetric'),(1,257),range(3))),'numerical repetitions missing')
    _,isolation,ig=phase('projection-bedrock-isolate-row1-v4')
    need(len(ig)==24 and all(r['rows']==1 and r['settings']['sync']=='none' for r in ig),'factorial scope mismatch')
    grid={}
    for c,a,m in itertools.product(('default','pedantic'),('tensor','default'),('default','pedantic')):
        observations=[r for r in ig if tuple(r['settings'][k] for k in ('compute','algo','math'))==(c,a,m)]
        need(len(observations)==3 and {r['repeat'] for r in observations}=={0,1,2},'factorial repetitions missing')
        totals=[r['sanitizer']['canonical_errors'] for r in observations]
        need(totals==([12288]*3 if c=='default' else [0]*3),'reported factorial result changed')
        grid['/'.join((c,a,m))]=totals
    _,_,large=phase('projection-bedrock-compute-only-row257-v7')
    need(len(large)==3 and all(r['rows']==257 and r['sanitizer']['canonical_errors']==0 for r in large),'large pedantic scope mismatch')
    need(all(r['settings']==dict(compute='pedantic',algo='tensor',math='default',sync='none') for r in large),'more than compute type changed')
    d,_,trace=phase('projection-bedrock-launches-row1-v6')
    default=next(r for r in trace if r['settings']['compute']=='default')
    addresses=default['reported_read_addresses']
    need(len(addresses)==12288 and all(x['read_bytes']==4 and len(x['matches'])==1 and x['matches'][0]['allocation']=='workspace' for x in addresses),'read allocation mapping mismatch')
    need(sorted(x['matches'][0]['offset_bytes'] for x in addresses)==list(range(0,49152,4)),'workspace read range mismatch')
    kernels=default['sanitizer']['printed_kernel_names'];need(len(kernels)==3 and kernels[0].startswith('nvjet_') and 'splitKreduce' in kernels[1] and 'consume_all' in kernels[2],'kernel sequence mismatch')
    a,b=make_inputs('asymmetric',1);left=periodic_units(a,6144,2048);right=periodic_units(b,1,2048)[0]
    dot=[sum(x*y for x,y in zip(v,right))*512/1024 for v in left]
    reference=b''.join(struct.pack('<f',v/2) for v in dot)*2
    scratch=[];snapshots=[]
    for seed in ('none','a5','5a'):
        d,s,gemms=phase('projection-bedrock-workspace-'+seed+'-v9')
        need(len(gemms)==3,'scratch repetitions missing')
        for r in gemms:
            data=(d/f"case-{r['index']:03d}/output.workspace.bin").read_bytes();facts=workspace_facts(None,data,reference)
            need(facts['reference_word_mismatches']==0,'scratch partial-dot hypothesis failed')
            need(r['sanitizer']['canonical_errors']==(13824 if seed=='none' else 0),'scratch diagnostic totals changed')
            scratch.append(dict(seed=seed,repeat=r['repeat'],**facts));snapshots.append(data)
    need(len(set(snapshots))==1,'scratch depends on seed/repeat')
    trace_counts={}
    for seed,name in [('none','projection-bedrock-workspace-trace-none-v11'),('a5','projection-bedrock-workspace-trace-a5-v10'),('5a','projection-bedrock-workspace-trace-5a-v10')]:
        d,s,gemms=phase(name);need(len(gemms)==1,'trace scope changed');r=gemms[0]
        need(r['sanitizer']['printed_kernel_names']==kernels,'seeding changed observed kernel path')
        text=log_bytes(d,f"case-{r['index']:03d}/sanitizer.log").decode()
        need(re.findall(r'(?m)^=========\s+Stream: (.+)$',text)==['0x0']*3,'trace stream changed')
        trace_counts[seed]=dict(device_read_reports=len(r['reported_read_addresses']),host_api_reports=text.count('Host API memory access error'),printed_launches=r['sanitizer']['printed_launches'])
        need(trace_counts[seed]==dict(device_read_reports=12288 if seed=='none' else 0,host_api_reports=1536 if seed=='none' else 0,printed_launches=3),'seeded trace report counts changed')
    d,s,gemms=phase('projection-bedrock-workspace-preimage-v12');r=gemms[0];case=d/f"case-{r['index']:03d}"
    facts=workspace_facts((case/'output.workspace-before.bin').read_bytes(),(case/'output.workspace.bin').read_bytes(),reference)
    need(facts['preimage_all_zero'] and facts['changed_bytes']==23222 and facts['changed_fp32_words']==12152 and not facts['reference_word_mismatches'],'preimage measurement changed')
    log=log_bytes(case,'sanitizer.log').decode();host_events=[m.start() for m in re.finditer('Host API memory access error',log)]
    before=sum(pos<log.index('========= Launch #1') for pos in host_events);after=sum(pos>log.index('========= Launch #3') for pos in host_events)
    need(before==after==1536,'pre/post API report counts changed')
    return dict(status='saved observations revalidated',full_acceptance=False,numerical_gemm_observations=12,factorial_error_totals=grid,reported_workspace_reads=dict(count=12288,read_bytes=4,first_offset=0,last_offset=49148),kernel_sequence=kernels,workspace_seed_observations=scratch,seed_trace_counts=trace_counts,workspace_preimage=dict(**facts,host_api_reports_before_first_launch=before,host_api_reports_after_last_launch=after),unknowns=['exact instruction/write-tracking mechanism','whether internal reads include an ordering defect','completed default257-row whole-consumer initcheck','separate synccheck and full32K racecheck root causes'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-root',type=Path,default=Path(__file__).resolve().parents[1]/'validation');p.add_argument('--output',type=Path)
    args=p.parse_args();result=analyze(args.validation_root)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(text)
    print(text)


if __name__=='__main__':main()
