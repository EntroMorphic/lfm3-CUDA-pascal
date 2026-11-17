"""Independent Python arithmetic oracle for the documented filter policy.

Not a Torch RNG comparison or a claim of ties identical to every backend.
"""
import math
from pathlib import Path
import random
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def distribution(tmp_path_factory):
    compiler=shutil.which("g++")
    if not compiler:pytest.skip("g++ required")
    root=Path(__file__).resolve().parents[1]
    directory=tmp_path_factory.mktemp("distribution")
    source=directory/"distribution.cpp"
    source.write_text(r'''
#include <lfm/text_sampling.hpp>
#include <iostream>
#include <iomanip>
int main(){
  int n,k;double temperature,p,m,penalty;
  std::cin>>n>>temperature>>k>>p>>m>>penalty;
  std::vector<float> x(n);for(auto&v:x)std::cin>>v;
  lfm::TextSamplingOptions o;o.do_sample=true;o.temperature=temperature;o.top_k=k;
  o.top_p=p;o.min_p=m;o.repetition_penalty=penalty;
  lfm::TextSampler sampler(o);
  auto d=sampler.distribution(x.data(),x.size(),{1,1,3},{2,2});
  std::cout<<std::setprecision(18);
  for(auto v:d)std::cout<<v.token_id<<" "<<v.probability<<"\n";
}
''')
    exe=directory/"distribution"
    subprocess.run([compiler,"-std=c++17","-O2","-I",str(root/"native/include"),str(source),"-o",str(exe)],check=True)
    return exe


def reference(values,temperature,k,p,min_p,penalty):
    adjusted=[(i,(v*penalty if v<0 else v/penalty) if i in {1,3} else v)
              for i,v in enumerate(values) if i!=2]
    adjusted.sort(key=lambda x:(-x[1],x[0]))
    if k:adjusted=adjusted[:k]
    weights=[(i,math.exp((v-adjusted[0][1])/temperature)) for i,v in adjusted]
    weights=[(i,w) for i,w in weights if w>0 and w>=min_p]
    threshold=p*sum(w for _,w in weights)
    kept=[];cumulative=0.0
    for i,w in weights:
        kept.append((i,w));cumulative+=w
        if p<1 and cumulative>=threshold:break
    return [(i,w/cumulative) for i,w in kept]


def test_independent_filtered_probabilities(distribution):
    rng=random.Random(1977)
    for index in range(80):
        # Exact binary-fraction logits keep float input conversion unambiguous.
        values=[rng.randrange(-64,65)/8 for _ in range(17)]
        temperature=rng.choice([0.25,0.7,1.0,2.0])
        k=rng.choice([0,1,5,17]);p=rng.choice([1.0,0.963,0.7,0.2])
        min_p=rng.choice([0.0,0.01,0.2]);penalty=rng.choice([0.5,1.0,1.1,2.0])
        payload=" ".join(map(str,[len(values),temperature,k,p,min_p,penalty,*values]))
        result=subprocess.run([str(distribution)],input=payload,text=True,capture_output=True,check=True)
        actual=[(int(line.split()[0]),float(line.split()[1])) for line in result.stdout.splitlines()]
        expected=reference(values,temperature,k,p,min_p,penalty)
        assert [i for i,_ in actual]==[i for i,_ in expected],index
        assert [w for _,w in actual]==pytest.approx([w for _,w in expected],abs=1e-12,rel=1e-12),index
