

import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 300


def verified_execution(result):
    if (
        result.get("verified") is not True
        or result.get("exit_code") != 0
        or result.get("timed_out")
    ):
        return False
    evaluation = result.get("evaluation", {})
    if evaluation.get("exit_code") != 0:
        return False
    try:
        metrics = json.loads(evaluation["stdout"])
        return (
            all(
                math.isfinite(metrics[k]) and metrics[k] >= 0
                for k in ["train_rmse", "test_rmse", "zero_test_rmse"]
            )
            and metrics["test_rmse"] == result["test_rmse"]
        )
    except (KeyError, TypeError, ValueError):
        return False


class CodingTool:
    def __init__(self, base=None, timeout=TIMEOUT):
        if not 0 < timeout <= TIMEOUT:
            raise ValueError("Timeout must be at most 300 seconds")
        self.timeout = timeout
        self.directory = (
            Path(base or ROOT / "results/agent_runs") / uuid.uuid4().hex
        ).resolve()
        self.directory.mkdir(parents=True)

    def execute(self, source):
        started = time.monotonic()
        directory = self.directory / uuid.uuid4().hex
        directory.mkdir()
        path = directory / "solution.py"
        path.write_text(source)
        
        
        bootstrap = """import sys, os, runpy, resource
from pathlib import Path
work=Path(sys.argv[1]).resolve(); root=Path(sys.argv[2]).resolve()
allowed=[work,root/'task5',root/'data',Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path('/usr/lib'),Path('/usr/share')]
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        mode=args[1]; flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        if writing and not p.is_relative_to(work): raise PermissionError('Write outside run directory')
        if not writing and not any(p.is_relative_to(a) for a in allowed): raise PermissionError('Read outside allowed directories')
    if event.startswith(('socket.','subprocess.')) or event in ['os.system','os.exec','os.fork','os.posix_spawn']:
        raise PermissionError('Network and child processes disabled')
    if event in ['os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate','os.rename','os.link','os.symlink']:
        paths=args[:2] if event in ['os.rename','os.link','os.symlink'] else args[:1]
        if any(not Path(os.fsdecode(p)).resolve().is_relative_to(work) for p in paths):
            raise PermissionError('Mutation outside run directory')
resource.setrlimit(resource.RLIMIT_CPU,(300,300))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024*1024,64*1024*1024))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
sys.path.insert(0,str(root)); sys.dont_write_bytecode=True
sys.addaudithook(audit)
runpy.run_path(str(work/'solution.py'),run_name='__main__')
"""
        command = [sys.executable, "-I", "-c", bootstrap, str(directory), str(ROOT)]
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(directory),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(directory / "mpl"),
        }
        out = directory / "stdout.txt"
        err = directory / "stderr.txt"
        with out.open("w") as stdout, err.open("w") as stderr:
            process = subprocess.Popen(
                command,
                cwd=directory,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timed_out = False
            try:
                code = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        result = {
            "directory": str(directory),
            "exit_code": code,
            "stdout": out.read_text(errors="replace")[-100000:],
            "stderr": err.read_text(errors="replace")[-100000:],
            "timed_out": timed_out,
            "verified": False,
        }
        
        
        candidate = directory / "candidate.pt"
        if code == 0 and candidate.exists():
            try:
                remaining = max(0.001, self.timeout - (time.monotonic() - started))
                check = subprocess.run(
                    [sys.executable, "-m", "task5.evaluate", str(candidate)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                )
                result["evaluation"] = {
                    "exit_code": check.returncode,
                    "stdout": check.stdout,
                    "stderr": check.stderr,
                }
                if check.returncode == 0:
                    metrics = json.loads(check.stdout)
                    result.update(
                        verified=True, test_rmse=metrics["test_rmse"], metrics=metrics
                    )
            except subprocess.TimeoutExpired:
                result.update(
                    timed_out=True,
                    evaluation={
                        "error": "Independent evaluation exceeded remaining execution budget"
                    },
                )
        (directory / "execution.json").write_text(json.dumps(result, indent=2))
        return result
