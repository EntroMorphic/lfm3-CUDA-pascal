"""CPU-only BF16 byte-structure/compression census; no GPU speedup prediction."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import sys
import zlib

from benchmark_common import write_json


def entropy(values) -> float:
    counts = Counter(values); total = sum(counts.values())
    return -sum(n/total*math.log2(n/total) for n in counts.values()) if total else 0.


def census(data: bytes) -> dict:
    if not data or len(data) % 2: raise ValueError("Expected nonempty BF16-aligned bytes")
    words = array("H"); words.frombytes(data)
    if sys.byteorder != "little": words.byteswap()
    low, high = data[0::2], data[1::2]
    raw = zlib.compress(data, 6)
    planar = zlib.compress(low + high, 6)
    if zlib.decompress(raw) != data or zlib.decompress(planar) != low + high:
        raise ValueError("Lossless compression round-trip failed")
    restored = bytearray(len(data)); restored[0::2] = low; restored[1::2] = high
    if restored != data: raise ValueError("Byte-plane layout round-trip failed")
    return dict(sample_bytes=len(data), empirical_value_entropy_bits=entropy(words),
        low_byte_entropy_bits=entropy(low), high_byte_entropy_bits=entropy(high),
        raw_zlib_bytes=len(raw), planar_zlib_bytes=len(planar),
        raw_compressed_fraction=len(raw)/len(data), planar_compressed_fraction=len(planar)/len(data))


def tensor_entries(model: Path):
    if model.is_dir():
        single=model/"model.safetensors"; index=model/"model.safetensors.index.json"
        if single.exists() and index.exists(): raise ValueError("Ambiguous checkpoint; select a file or index")
        model=single if single.exists() else index
    if model.name.endswith(".index.json"):
        mapping=json.loads(model.read_text())["weight_map"]
        names=set(mapping.values()); files=[]
        for name in names:
            path=(model.parent/name).resolve()
            if Path(name).name!=name or not path.is_relative_to(model.parent.resolve()):
                raise ValueError("Shard must be a sibling file within checkpoint directory")
            files.append(path)
    else: files=[model]
    entries=[]
    for path in sorted(files):
        with path.open("rb") as stream:
            prefix=stream.read(8)
            if len(prefix)!=8: raise ValueError("Missing Safetensors header")
            size=int.from_bytes(prefix,"little")
            if size<2 or size>100_000_000 or size>path.stat().st_size-8: raise ValueError("Invalid Safetensors header size")
            header=json.loads(stream.read(size))
        payload=8+size; file_size=path.stat().st_size
        for name,meta in header.items():
            if name=="__metadata__" or meta.get("dtype")!="BF16" or len(meta.get("shape",[]))<2: continue
            start,end=meta["data_offsets"]
            elements=math.prod(meta["shape"])
            if not 0<=start<end<=file_size-payload or end-start!=2*elements:
                raise ValueError("Invalid BF16 tensor span")
            entries.append(dict(name=name,path=path,offset=payload+start,bytes=end-start,shape=meta["shape"]))
    return sorted(entries,key=lambda r:r["name"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=6)
    p.add_argument("--sample-bytes",type=int,default=2*1024*1024)
    args=p.parse_args(argv)
    if args.samples<1 or args.sample_bytes<2 or args.sample_bytes%2: p.error("Samples must be positive and sample bytes even")
    if args.output.exists(): p.error("Output already exists")
    entries=tensor_entries(args.model)
    if not entries: raise ValueError("No BF16 matrix tensors")
    count=min(args.samples,len(entries))
    selected=[entries[round(i*(len(entries)-1)/max(1,count-1))] for i in range(count)]
    reports=[]
    import hashlib
    for entry in selected:
        size=min(args.sample_bytes,entry["bytes"])
        relative=((entry["bytes"]-size)//4)*2  # Centered, BF16-aligned sample.
        with entry["path"].open("rb") as stream:
            stream.seek(entry["offset"]+relative); data=stream.read(size)
        if len(data)!=size: raise ValueError("Truncated matrix sample")
        reports.append(dict(tensor=entry["name"],shard=str(entry["path"]),shape=entry["shape"],
            tensor_bytes=entry["bytes"],sample_file_offset=entry["offset"]+relative,
            sample_sha256=hashlib.sha256(data).hexdigest(),**census(data)))
    result=dict(scope="Sampled BF16 marginal entropy and CPU lossless compression. Ratios are sample/codec-specific; GPU decode cost, whole-checkpoint compression and inference throughput remain unmeasured.",
        model=str(args.model),bf16_matrix_tensors=len(entries),samples=reports,
        codec=dict(name="zlib",level=6,build_version=zlib.ZLIB_VERSION,runtime_version=zlib.ZLIB_RUNTIME_VERSION),
        warning="A marginal entropy estimate assumes a symbol model; it is not a universal compression bound. GPU speed depends on decoding, cache behavior, layout and added memory traffic.")
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
