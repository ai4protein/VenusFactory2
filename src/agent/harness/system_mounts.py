"""Minimal read-only runtime dependencies, never the host's whole /etc tree."""
from pathlib import Path


def system_mounts():
    args=[]
    for name in ['/usr','/bin','/lib','/lib64']:
        if Path(name).exists():args += ['--ro-bind',name,name]
    args += ['--dir','/etc']
    for name in ['/etc/ssl/certs','/etc/ld.so.cache','/etc/resolv.conf','/etc/localtime']:
        if Path(name).exists():args += ['--ro-bind',str(Path(name).resolve()),name]
    return args
