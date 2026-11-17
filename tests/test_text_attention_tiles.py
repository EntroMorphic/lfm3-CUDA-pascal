"""Host proofs of tiled key coverage; not GPU numerical acceptance."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def planner(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "g++ is required for attention range coverage"
    folder = tmp_path_factory.mktemp("attention_tiles")
    source = folder / "tiles.cpp"
    source.write_text(r'''
#include <lfm/text_attention_tiles.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
using namespace lfm::text;

void coverage(int total,int first,int rows,int window,bool enumerate) {
  auto r=attention_key_range(32768,0,total,first,rows,window);
  assert(r.begin>=0 && r.begin<r.end && r.end<=total);
  assert(r.span()<=total && rows<=scratch_rows(32768));
  for(int q=first;q<first+rows;++q) {
    const int lower=window?std::max(0,q-window+1):0;
    // Interval proof: the complete unmasked set [lower,q] is retained.
    assert(r.begin<=lower && q<r.end);
    // Every excluded key is below lower or above q, hence masked.
    assert(r.begin==0 || r.begin-1<lower);
    assert(r.end==total || r.end>q);
    if(enumerate) for(int k=0;k<total;++k) {
      const bool visible=k>=lower && k<=q;
      const bool retained=k>=r.begin && k<r.end;
      assert(!visible || retained);
      // Applying the original mask after trimming yields exactly the same set.
      assert(visible==(retained && k>=lower && k<=q));
    }
  }
  if(total>4096) {
    assert(r.end==first+rows);
    assert(r.begin==(window?std::max(0,first-window+1):0));
  } else {
    assert(r.begin==0 && r.end==total);
  }
}

int main(int argc,char** argv) {
  assert(argc==2);
  std::string mode=argv[1];
  if(mode=="coverage") {
    // All query starts, lengths, queries and keys at small lengths, including
    // the scratch-tile boundary (128 -> 129), not only aligned tiles.
    for(int total=1;total<=145;++total)
      for(int first=0;first<total;++first)
        for(int rows=1;rows<=std::min(128,total-first);++rows)
          for(int window:{0,1,2,7,128,256}) coverage(total,first,rows,window,true);
    // Exhaust every optimized start and feasible tile length at 4097,
    // including unaligned starts around the sliding-window boundary.
    for(int first=0;first<4097;++first)
      for(int rows=1;rows<=std::min(128,4097-first);++rows)
        for(int window:{0,256}) coverage(4097,first,rows,window,false);
    // All tile lengths 1..128 and every start in the last 512 positions,
    // near the real context limit; interval checks cover every excluded key.
    for(int total:{4097,32767,32768})
      for(int first=total-512;first<total;++first)
        for(int rows=1;rows<=std::min(128,total-first);++rows)
          for(int window:{0,256}) coverage(total,first,rows,window,false);
    for(int first:{0,1,127,128,255,256,257,32640})
      for(int rows:{1,127,128})
        for(int window:{0,256}) coverage(32768,first,rows,window,true);
  } else if(mode=="optimized-windows") {
    for(int total:{4097,4098,32764,32767,32768})
      for(int window:{0,1,2,7,128,256,65536})
        for(int first:{0,1,127,128,255,256,257,2047,2048,total-128,total-1})
          for(int wanted:{1,127,128}) {
            const int rows=std::min(wanted,total-first);
            coverage(total,first,rows,window,true);
          }
  } else if(mode=="legacy") {
    // Every <=4096 capacity and every eager prefill length retains its exact
    // old GEMM/softmax extent, including the one-row sliding special case.
    for(int capacity=1;capacity<=4096;++capacity)
      for(int total=1;total<=capacity;++total)
        for(int window:{0,256}) {
          auto r=attention_key_range(capacity,0,total,0,total,window);
          assert(r.begin==0 && r.end==total);
        }
    // Exhaust every decode position at both eager and long capacities.
    for(int capacity:{1,128,4096,4097,32768})
      for(int cursor=0;cursor<capacity;++cursor)
        for(int window:{0,1,256,32768}) {
          auto r=attention_key_range(capacity,cursor,1,0,1,window);
          assert(r.begin==(window?std::max(0,cursor+1-window):0));
          assert(r.end==cursor+1);
        }
    for(int total=1;total<=128;++total)
      for(int window:{0,256}) {
        auto r=attention_key_range(32768,0,total,0,total,window);
        assert(r.begin==0 && r.end==total);
      }
    // A long-capacity runner still retains the original full key range for
    // EVERY short prompt, even when its queries use the 128-row schedule.
    for(int capacity:{4097,4112,32768})
      for(int total=1;total<=4096;++total)
        for(int first=0;first<total;first+=128)
          for(int window:{0,256}) {
            const int rows=std::min(128,total-first);
            auto r=attention_key_range(capacity,0,total,first,rows,window);
            assert(r.begin==0 && r.end==total);
          }
    for(int total:{4097,4098})
      for(int first=0;first<total;first+=128) {
        auto r=attention_key_range(4112,0,total,first,std::min(128,total-first),256);
        assert(r.end==std::min(total,first+128));
        assert(r.begin==std::max(0,first-255));
      }
  } else if(mode=="counts") {
    const int total=32768,heads=16;
    for(int window:{0,256}) {
      int64_t positions=0;
      for(int first=0;first<total;first+=128) {
        auto r=attention_key_range(total,0,total,first,128,window);
        positions+=int64_t(heads)*128*r.span();
      }
      std::cout<<window<<" "<<int64_t(heads)*total*total<<" "<<positions<<"\n";
    }
  } else if(mode=="invalid") {
    auto bad=[](int cap,int cursor,int total,int first,int rows,int window) {
      bool rejected=false;
      try {attention_key_range(cap,cursor,total,first,rows,window);}
      catch(const std::runtime_error&) {rejected=true;}
      assert(rejected);
    };
    bad(0,0,1,0,1,0); bad(262145,0,1,0,1,0);
    bad(32768,-1,1,0,1,0); bad(32768,32768,1,0,1,0);
    bad(32768,32767,2,0,1,0); bad(32768,0,0,0,1,0);
    bad(32768,0,129,-1,1,0); bad(32768,0,129,129,1,0);
    bad(32768,0,129,0,0,0); bad(32768,0,129,0,129,0);
    bad(32768,0,129,128,2,0); bad(32768,0,129,0,1,-1);
    bad(32768,1,129,0,128,0);
  } else {assert(false);}
}
''')
    executable = folder / "tiles"
    include = Path(__file__).resolve().parents[1] / "native/include"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(include), str(source), "-o", str(executable)], check=True,
    )
    return executable


@pytest.mark.parametrize("mode", ["coverage", "optimized-windows", "legacy", "invalid"])
def test_attention_ranges(planner, mode):
    subprocess.run([str(planner), mode], check=True, timeout=120)


def test_32k_score_position_reduction(planner):
    result = subprocess.run([str(planner), "counts"], check=True, text=True, capture_output=True)
    counts = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    assert counts == [
        (0, 17_179_869_184, 8_623_489_024),
        # Spans: 128, 256, then 254 tiles of 383 (=128+256-1).
        (256, 17_179_869_184, 200_019_968),
    ]
