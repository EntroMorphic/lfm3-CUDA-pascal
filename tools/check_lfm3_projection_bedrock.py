"""Whole-output GEMM measurement, not production or sanitizer acceptance.

Inputs are host-generated. The CPU oracle factors the actual periodic input
bytes, not CUDA results, and uses integer dot products plus BF16 ties-to-even.
"""
import argparse
from array import array
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time

from check_lfm3_cublas_modes import digest, environment, libraries, run_owned
from lfm3_owned_process import owned_main

WIDTH = 2048
SOURCE = Path(__file__).resolve().parents[1] / 'native/probes/lfm3_projection_bedrock_probe.cu'


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def raw(values):
    result = array('H', values)
    if sys.byteorder != 'little':
        result.byteswap()
    return result.tobytes()


def decode(data):
    if len(data) % 2:
        raise ValueError('odd BF16 byte count')
    result = array('H'); result.frombytes(data)
    if sys.byteorder != 'little':
        result.byteswap()
    return result


def make_inputs(pattern, rows, width=WIDTH):
    if pattern not in ('legacy', 'asymmetric') or rows not in (1, 257) or width not in (4, WIDTH):
        raise ValueError('unsupported fixture')
    a, b = bytearray(), bytearray()
    for channel in range(3 * width):
        x, y = channel % 13 - 6, (channel // 13) % 7 - 3
        units = [4*x]*4 if pattern == 'legacy' else [x*u+y*v for u,v in zip((1,2,3,4),(1,0,-1,0))]
        a.extend(raw(bf16(u/32) for u in units) * (width//4))
    for row in range(rows):
        x, y = row % 17 + 1, row // 17 + 1
        units = [16]*4 if pattern == 'legacy' else [x*p+y*q for p,q in zip((-1,0,1,2),(1,-1,1,-1))]
        b.extend(raw(bf16(u/32) for u in units) * (width//4))
    return bytes(a), bytes(b)


def periodic_units(data, vectors, width):
    if len(data) != vectors*width*2:
        raise ValueError('matrix size mismatch')
    result = []
    for row in range(vectors):
        block = data[row*width*2:(row+1)*width*2]
        if block != block[:8]*(width//4):
            raise ValueError('oracle requires measured four-element periodic inputs')
        units = []
        for bits in decode(block[:8]):
            value = struct.unpack('<f', struct.pack('<I', bits << 16))[0] * 32
            if not value.is_integer() or not -64 <= value <= 64:
                raise ValueError('oracle inputs outside exact integer fixture domain')
            units.append(int(value))
        result.append(units)
    return result


def cpu_oracle(a, b, rows, width=WIDTH):
    left = periodic_units(a, 3*width, width)
    right = periodic_units(b, rows, width)
    values = array('H')
    for vector in right:
        for weights in left:
            # Every input is units/32. Conservative absolute-product bound
            # certifies that FP32's integer significand suffices for this fixture.
            if sum(abs(x*y) for x,y in zip(weights,vector))*(width//4) >= 2**24:
                raise ValueError('exact FP32 accumulation bound exceeded')
            numerator = sum(x*y for x,y in zip(weights,vector))*(width//4)
            values.append(bf16(numerator/1024))
    if sys.byteorder != 'little': values.byteswap()
    return values.tobytes()


def compare(actual, expected):
    if len(actual) != len(expected) or not expected or len(expected) % 2:
        raise ValueError('full output absent, truncated or oversized')
    mismatches = []
    count = 0
    for index, (a,b) in enumerate(zip(decode(actual), decode(expected))):
        if a != b:
            count += 1
            if len(mismatches) < 16:
                mismatches.append(dict(index=index, actual_bf16_bits=a, expected_bf16_bits=b))
    return dict(checked_elements=len(expected)//2, mismatch_count=count,
                first_mismatches=mismatches, sha256=hashlib.sha256(actual).hexdigest())


def metadata(text, producer, rows, settings):
    entries = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
    configs = [x for x in entries if x.get('configuration') is True]
    complete = [x for x in entries if x.get('complete') is True]
    if len(configs) != 1 or len(complete) != 1:
        raise ValueError('missing or duplicate application identity/completion')
    c = configs[0]
    expected = dict(producer=producer, rows=rows, width=WIDTH, **settings)
    if any(c.get(k) != v for k,v in expected.items()):
        raise ValueError('wrong application configuration')
    if any(type(c.get(k)) is not int for k in ('rows','width','actual_math_enum','compute_enum','algo_enum','workspace_bytes','cublas','driver','runtime','sm_major','sm_minor')):
        raise ValueError('API identity fields must be integers')
    if (c.get('actual_math_enum') != (2 if settings['math']=='pedantic' else 0) or
        c.get('compute_enum') != (69 if settings['compute']=='pedantic' else 68) or
        c.get('algo_enum') != (99 if settings['algo']=='tensor' else -1) or
        c.get('default_stream') is not True or c.get('workspace_bytes') != 8519680):
        raise ValueError('unexpected actual API settings')
    if any(type(complete[0].get(k)) is not int or complete[0][k] != rows*3*WIDTH for k in ('outputs','consumer_elements')):
        raise ValueError('incomplete output coverage')
    allocations=[x for x in entries if 'allocation' in x]
    if allocations:
        if {x.get('allocation') for x in allocations}!={'a','b','c','d','workspace'} or len(allocations)!=5:
            raise ValueError('incomplete allocation map')
        for x in allocations:
            if any(type(x.get(k)) is not int or x[k]<=0 for k in ('base','bytes')):raise ValueError('invalid allocation range')
    return dict(configuration=c, complete=complete[0], stages=[x for x in entries if 'stage' in x], allocations=allocations)


def reported_addresses(text,allocations):
    result=[]
    for block in text.split('========= Uninitialized __global__ memory read')[1:]:
        size=re.search(r'of size (\d+) bytes',block);address=re.search(r'Address (0x[0-9a-fA-F]+)',block)
        if size and address:
            n,ptr=int(size[1]),int(address[1],16)
            matches=[dict(allocation=a['allocation'],offset_bytes=ptr-a['base']) for a in allocations if a['base']<=ptr and ptr+n<=a['base']+a['bytes']]
            result.append(dict(address=ptr,read_bytes=n,matches=matches))
    return result


def sanitizer_report(text):
    totals = re.findall(r'(?m)^=+\s+ERROR SUMMARY: (\d+) errors\s*$', text)
    return dict(canonical_errors=int(totals[0]) if len(totals)==1 else None,
                consumer_read_report='consume_all(' in text and 'Uninitialized __global__ memory read' in text,
                internal_splitK_report='splitKreduce_kernel' in text,
                internal_sanitizer_error='Internal Sanitizer Error' in text)


@owned_main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('probe','sanitizer','library-dir','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--phase', choices=('baseline','isolate'), default='baseline')
    p.add_argument('--instrumentation',choices=('initcheck','none'),default='initcheck')
    p.add_argument('--rows',choices=(1,257),type=int,nargs='+')
    p.add_argument('--compute',choices=('default','pedantic'),nargs='+')
    p.add_argument('--algo',choices=('tensor','default'),nargs='+')
    p.add_argument('--math',choices=('default','pedantic'),nargs='+')
    p.add_argument('--capture-launches',action='store_true',help='diagnostic logging; printed launch records pollute tool totals')
    p.add_argument('--workspace-seed',choices=('none','a5','5a'),default='none',help='scratch diagnostic only, never production acceptance')
    p.add_argument('--workspace-dump-bytes',type=int,default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=float, default=90)
    args = p.parse_args()
    if not 1<=args.repeats<=5 or not 0<args.timeout<=180:
        p.error('bounded repeats/timeout required')
    if args.capture_launches and args.instrumentation!='initcheck':p.error('launch capture requires instrumentation')
    if args.phase=='baseline' and any((args.compute,args.algo,args.math)):p.error('baseline settings are fixed; use isolate')
    if not 0<=args.workspace_dump_bytes<=8519680 or args.workspace_dump_bytes%4:p.error('invalid workspace dump range')
    if any(values and len(set(values))!=len(values) for values in (args.compute,args.algo,args.math)):p.error('duplicate setting values')
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(status='failed', full_acceptance=False, phase=args.phase, records=[],
                  instrumentation=args.instrumentation,
                  launch_capture_diagnostic=args.capture_launches,
                  workspace_seed=args.workspace_seed,workspace_dump_bytes=args.workspace_dump_bytes,
                  scope='whole-output diagnostic; no kernel-selection or root-cause inference')
    pins = {}
    def pin(path):
        path = Path(path).resolve(); pins[str(path)] = digest(path)
    def save():
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for path in (args.probe,args.sanitizer,SOURCE,Path(__file__),Path(__file__).with_name('check_lfm3_cublas_modes.py'),Path(__file__).with_name('lfm3_owned_process.py')): pin(path)
        env = environment(args.library_dir.resolve(), args.library_dir.resolve())
        ldd=shutil.which('ldd')
        if ldd is None:raise ValueError('ldd not available')
        code, timed = run_owned([ldd, str(args.probe.resolve())], env,args.output/'ldd.log',10)
        if code!=0 or timed: raise ValueError('dependency resolution failed')
        dependencies = libraries((args.output/'ldd.log').read_text(),args.library_dir.resolve())
        for entry in dependencies.values(): pin(entry['path'])
        report.update(start_sha256=pins.copy(),dependencies=dependencies,ld_library_path=env['LD_LIBRARY_PATH'])
        fixtures = {}
        patterns = ('legacy','asymmetric') if args.phase=='baseline' else ('asymmetric',)
        shapes = tuple(args.rows) if args.rows else (1,257) if args.phase=='baseline' else (257,)
        if len(set(shapes))!=len(shapes):raise ValueError('duplicate shapes are not repetitions')
        for pattern,rows in itertools.product(patterns,shapes):
            directory=args.output/f'fixture-{pattern}-{rows}';directory.mkdir()
            a,b=make_inputs(pattern,rows);expected=cpu_oracle(a,b,rows)
            files=[directory/'a.bf16',directory/'b.bf16',directory/'expected.bf16']
            for path,data in zip(files,(a,b,expected)):path.write_bytes(data);pin(path)
            fixtures[pattern,rows]=(files,expected)
        control_pattern='legacy' if args.phase=='baseline' else 'asymmetric'
        if (control_pattern,1) not in fixtures:
            directory=args.output/'fixture-control-1';directory.mkdir()
            a,b=make_inputs(control_pattern,1);expected=cpu_oracle(a,b,1)
            files=[directory/'a.bf16',directory/'b.bf16',directory/'expected.bf16']
            for path,data in zip(files,(a,b,expected)):path.write_bytes(data);pin(path)
            fixtures[control_pattern,1]=(files,expected)
        default=dict(compute='default',algo='tensor',math='default',sync='none')
        cases=[('kernel',control_pattern,1,default,0)]
        if args.instrumentation=='initcheck':cases.append(('missing',control_pattern,1,default,0))
        modes=[default] if args.phase=='baseline' else [dict(compute=c,algo=a,math=m,sync='none') for c,a,m in itertools.product(args.compute or ('default','pedantic'),args.algo or ('tensor','default'),args.math or ('default','pedantic'))]
        cases += [('gemm',pattern,rows,settings,repeat) for pattern,rows,settings,repeat in itertools.product(patterns,shapes,modes,range(args.repeats))]
        report['planned_cases']=len(cases);report['start_sha256']=pins.copy();save()
        for index,(producer,pattern,rows,settings,repeat) in enumerate(cases):
            d=args.output/f'case-{index:03d}';d.mkdir();files,expected=fixtures[pattern,rows]
            case_seed=args.workspace_seed if producer=='gemm' else 'none'
            case_dump=args.workspace_dump_bytes if producer=='gemm' else 0
            cmd=[str(args.probe.resolve()),producer,str(rows),str(WIDTH),settings['compute'],settings['algo'],settings['math'],str(files[0].resolve()),str(files[1].resolve()),str((d/'output').resolve()),settings['sync']]
            if case_seed!='none' or case_dump:cmd += [case_seed,str(case_dump)]
            if args.instrumentation=='initcheck':cmd=[str(args.sanitizer.resolve()),'--tool','initcheck','--error-exitcode','86','--print-limit','4','--log-file',str((d/'sanitizer.log').resolve()),*cmd]
            if args.capture_launches and producer=='gemm':
                cmd[cmd.index('--print-limit')+1]='0'  # Launch records share the tool's reporting limit.
                cmd[1:1]=['--dump-kernel-launches','--print-session-details']
            record=dict(index=index,producer=producer,pattern=pattern,rows=rows,settings=settings,repeat=repeat,command=cmd,status='incomplete',workspace_seed=case_seed,workspace_dump_bytes=case_dump)
            report['records'].append(record);save();start=time.monotonic()
            code,timed=run_owned(cmd,env,d/'application.log',args.timeout)
            record.update(returncode=code,timed_out=timed,seconds=time.monotonic()-start,group_cleanup_complete=True)
            log=(d/'sanitizer.log').read_text(errors='replace') if (d/'sanitizer.log').exists() else ''
            record['sanitizer']=sr=sanitizer_report(log)
            sr['instrumented']=args.instrumentation=='initcheck'
            sr['launch_dump_diagnostic']=args.capture_launches and producer=='gemm'
            sr['printed_launches']=len(re.findall(r'(?m)^========= Launch #\d+\s*$',log))
            sr['printed_kernel_names']=re.findall(r'(?m)^=========\s+Kernel: (.+)$',log)
            if timed: raise ValueError('bounded case timeout; stop without coverage claim')
            if code not in (0,86): raise ValueError('application/tool failed; inspect raw evidence')
            record['application']=metadata((d/'application.log').read_text(),producer,rows,settings)
            c=record['application']['configuration']
            if (case_seed!='none' or case_dump) and (c.get('workspace_seed')!=case_seed or c.get('workspace_dump_bytes')!=case_dump):raise ValueError('scratch diagnostic settings missing or different')
            if case_dump:
                scratch=(d/'output.workspace.bin').read_bytes()
                if len(scratch)!=case_dump:raise ValueError('workspace dump incomplete')
                record['workspace_postcompletion']=dict(bytes=len(scratch),sha256=hashlib.sha256(scratch).hexdigest())
                preimage=d/'output.workspace-before.bin'
                if preimage.exists():
                    before=preimage.read_bytes()
                    if len(before)!=case_dump:raise ValueError('workspace preimage incomplete')
                    record['workspace_preimage']=dict(bytes=len(before),sha256=hashlib.sha256(before).hexdigest(),changed_bytes=sum(a!=b for a,b in zip(before,scratch)))
            record['reported_read_addresses']=reported_addresses(log,record['application']['allocations'])
            if args.instrumentation=='initcheck' and (sr['canonical_errors'] is None or sr['internal_sanitizer_error']):raise ValueError('incomplete sanitizer report')
            if sr['launch_dump_diagnostic'] and (not sr['printed_launches'] or len(sr['printed_kernel_names'])!=sr['printed_launches']):raise ValueError('launch capture incomplete')
            if args.instrumentation=='none' and code!=0:raise ValueError('uninstrumented application failed')
            expected_case=raw([bf16(WIDTH)]*(rows*3*WIDTH)) if producer=='kernel' else expected
            for label in ('producer','consumer'):
                data=(d/f'output.{label}.bf16').read_bytes()
                if producer=='missing':
                    if len(data)!=rows*3*WIDTH*2:raise ValueError('negative control output incomplete')
                else:record[label+'_comparison']=compare(data,expected_case)
            if producer=='missing':
                if code!=86 or not sr['consumer_read_report'] or not sr['canonical_errors']:raise ValueError('negative control not detected')
            elif producer=='kernel':
                if code!=0 or sr['canonical_errors'] or any(record[x+'_comparison']['mismatch_count'] for x in ('producer','consumer')):raise ValueError('fill control failed')
            else:
                if args.instrumentation=='initcheck' and (code==0)!=(sr['canonical_errors']==0):raise ValueError('exit status and sanitizer summary disagree')
            record['status']='measured';save()
        report['status']='measurement-completed'
        gemms=[r for r in report['records'] if r['producer']=='gemm']
        report['observed_gemm_cases']=len(gemms)
        report['observed_gemm_cases_with_numerical_mismatch']=sum(any(r[k+'_comparison']['mismatch_count'] for k in ('producer','consumer')) for r in gemms)
        report['observed_gemm_cases_with_sanitizer_errors']=sum((r['sanitizer']['canonical_errors'] or 0)>0 for r in gemms) if args.instrumentation=='initcheck' and not args.capture_launches else None
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        report['end_sha256']={path:digest(path) if Path(path).is_file() else None for path in pins}
        report['hashes_stable']=report['end_sha256']==pins
        if not report['hashes_stable']:report['status']='failed-identity-change'
        save()
    print(json.dumps(dict(status=report['status'],cases=len(report['records']),full_acceptance=False)))
    return 0 if report['status']=='measurement-completed' else 86


if __name__=='__main__':raise SystemExit(main())
