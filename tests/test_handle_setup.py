from pathlib import Path
import shutil
import subprocess

import pytest


def test_constructor_setup_releases_every_failure_stage(tmp_path):
    compiler=shutil.which("g++")
    if not compiler:pytest.skip("g++ required")
    root=Path(__file__).resolve().parents[1]
    source=tmp_path/"handle.cpp"
    source.write_text(r'''
#include <lfm/handle_setup.hpp>
#include <cassert>
#include <stdexcept>
int live=0,destroyed=0;
struct Runner {
  int handle=0;
  Runner(int fail) {
    handle=++live;
    lfm::finish_handle_setup(handle,[](int) noexcept {--live;++destroyed;},[&] {
      for(int stage=1;stage<=3;++stage) if(stage==fail) throw std::runtime_error("setup failure");
    });
  }
  ~Runner(){if(handle){--live;++destroyed;}}
};
int main(){
  for(int stage=1;stage<=3;++stage){
    int before=destroyed;bool failed=false;
    try{Runner value(stage);}catch(const std::runtime_error&){failed=true;}
    assert(failed && live==0 && destroyed==before+1);
  }
  {Runner value(0);assert(live==1);}
  assert(live==0 && destroyed==4);
  int h=42;
  try{lfm::finish_handle_setup(h,[](int) noexcept {},[]{throw 7;});}catch(int value){assert(value==7);}
  assert(h==0);
}
''')
    binary=tmp_path/"handle"
    subprocess.run([compiler,"-std=c++17","-Wall","-Wextra","-Werror","-fsanitize=undefined",
                    "-I",str(root/"native/include"),str(source),"-o",str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
