"""Mounted read-only; no application imports or credentials in the child."""
import os
import resource
import runpy
import sys
import shlex
import shutil

kind, seconds, payload = sys.argv[1], int(sys.argv[2]), sys.argv[3]
resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
memory=int(os.environ['VENUS_SANDBOX_MEMORY_BYTES'])
# CUDA reserves large virtual address ranges. Only a host-verified PBS cgroup
# profile can use physical-memory accounting instead of the CPU address cap.
if os.environ.get('VENUS_SANDBOX_MEMORY_POLICY') != 'pbs':
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
resource.setrlimit(resource.RLIMIT_FSIZE,(100*1024**2,100*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_NPROC,(16,16))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.umask(0o077)
os.chdir('/workspace')
# Bound CPU parallelism before denying changes to affinity in untrusted code.
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
# Load only the host-mounted policy; /workspace cannot shadow this import.
policy=runpy.run_path('/sandbox_seccomp.py')
policy['install']()
if kind=='python':
    sys.path.insert(0,'/workspace');sys.argv=[payload]
    runpy.run_path(payload,run_name='__main__')
elif kind=='shell':
    # A plain external command can replace the runner; /bin/sh would fork even
    # for `python analysis.py`. Shell syntax still stays under the syscall policy.
    lexer=shlex.shlex(payload,posix=True,punctuation_chars=True);lexer.whitespace_split=True
    tokens=list(lexer)
    if tokens and not any(t in {';','|','||','&','&&','<','>','>>','(',')'} for t in tokens) and not any(c in payload for c in '$`*?\n'):
        args=shlex.split(payload)
        if shutil.which(args[0]):os.execvp(args[0],args)
    os.execv('/bin/sh',['sh','-c',payload])
else:raise SystemExit('Unsupported execution')
