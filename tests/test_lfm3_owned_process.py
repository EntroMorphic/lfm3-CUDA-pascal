import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_owned_process as owned


@pytest.mark.parametrize("leader", ["sleep", "exit", "error"])
def test_owned_group_cleans_helper_that_outlives_leader(tmp_path, leader):
    marker = tmp_path / "child.pid"
    child = 'import time;time.sleep(30)'
    script = ("import subprocess,sys,time\n"
              f"p=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
              f"open({str(marker)!r},'w').write(str(p.pid))\n" +
              ("time.sleep(30)" if leader == "sleep" else "sys.exit(1)" if leader == "error" else "sys.exit(0)"))
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        code, timed_out = owned.wait_owned(process, timeout=0.3)
        assert timed_out == (leader == "sleep")
        assert code == (None if leader == "sleep" else 1 if leader == "error" else 0)
        assert marker.exists()
        assert owned.live_group(process.pid) == []
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait(timeout=2)


def test_owned_runner_refuses_shared_process_group():
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        with pytest.raises(ValueError, match="new process session"):
            owned.wait_owned(process, timeout=0.1)
    finally:
        process.kill()
        process.wait(timeout=2)


def test_cleanup_failure_is_not_returned_as_success(monkeypatch):
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    monkeypatch.setattr(owned, "live_group", lambda group: [999999])
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        owned.wait_owned(process, timeout=1, cleanup_timeout=0.02)
    assert process.poll() is not None


def test_startup_signal_keeps_ownership_until_cleanup(monkeypatch):
    import signal
    original=owned.subprocess.Popen
    children=[]
    def interrupt(*args,**kwargs):
        process=original(*args,**kwargs)
        children.append(process)
        os.kill(os.getpid(),signal.SIGINT)
        return process
    monkeypatch.setattr(owned.subprocess,"Popen",interrupt)
    owner=owned.OwnedJob()
    try:
        with pytest.raises(KeyboardInterrupt):
            owner.start([sys.executable,"-c","import time;time.sleep(30)"])
    finally:
        owner.close()
    assert children and children[0].poll() is not None and owner.closed


def test_resource_snapshot_includes_reparented_same_group_helper(tmp_path):
    marker=tmp_path/"helper.pid"
    code=("import subprocess,sys\n"
          "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
          f"open({str(marker)!r},'w').write(str(child.pid))\n")
    owner=owned.OwnedJob()
    try:
        process=owner.start([sys.executable,"-c",code])
        assert process.wait(timeout=3)==0
        child=int(marker.read_text())
        sample=owned.resource_snapshot(process.pid)
        assert child in [row['pid'] for row in sample['processes']]
        assert sample['system_memory']['MemAvailable_bytes']>0
    finally:
        owner.close()
    assert owned.live_group(process.pid)==[]
