"""Entry point mounted read-only into the network-isolated research runner."""
import os
import resource
import runpy
import sys

script,seconds=sys.argv[1],int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds))
memory = (2 if os.getenv("VENUS_SCIENTIST_ONLINE") == "1" else 8) * 1024**3
resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
resource.setrlimit(resource.RLIMIT_FSIZE,(100*1024**2,100*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
os.chdir('/workspace')
sys.path.insert(0,'/workspace')
sys.argv=[script]
runpy.run_path(script,run_name='__main__')
