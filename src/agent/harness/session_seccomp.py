"""Guest-side syscall filter; installed before any session-authored code.

Single-process analysis with threads: prevents fork-based memory multiplication
on hosts without delegated cgroups. Separate scientific jobs use queued MCP.
"""
import ctypes
import errno


def install():
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    class Compare(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_int),('a',ctypes.c_uint64),('b',ctypes.c_uint64)]
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Compare)]
    lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
    context=lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:raise RuntimeError('Cannot initialize syscall filter')
    def deny(name,error=errno.EPERM,comparison=None):
        syscall=lib.seccomp_syscall_resolve_name(name.encode())
        if syscall<0:return
        comparisons=(Compare*1)(comparison) if comparison is not None else None
        if lib.seccomp_rule_add_array(context,0x50000|error,syscall,1 if comparisons is not None else 0,comparisons)!=0:
            raise RuntimeError('Cannot install syscall restriction: '+name)
    try:
        for name in ['fork','vfork','unshare','setns','mount','umount2','pivot_root','chroot',
                     'ptrace','process_vm_readv','process_vm_writev','pidfd_getfd','kcmp',
                     'bpf','perf_event_open','userfaultfd','keyctl','add_key','request_key',
                     'io_uring_setup','io_uring_enter','io_uring_register','memfd_create',
                     'open_by_handle_at','name_to_handle_at','swapon','swapoff','reboot',
                     'kexec_load','kexec_file_load','init_module','finit_module','delete_module',
                     'shmget','shmat','shmdt','shmctl','msgget','msgsnd','msgrcv','msgctl',
                     'semget','semop','semtimedop','semctl','sched_setaffinity']:
            deny(name)
        # glibc falls back to clone for pthreads when clone3 returns ENOSYS.
        deny('clone3',errno.ENOSYS)
        deny('clone',comparison=Compare(0,7,0x10000,0)) # CLONE_THREAD absent
        # No nested namespaces, even in a thread-like clone request.
        for flag in [0x00020000,0x02000000,0x04000000,0x08000000,0x10000000,0x20000000,0x40000000]:
            deny('clone',comparison=Compare(0,7,flag,flag))
        if lib.seccomp_load(context)!=0:raise RuntimeError('Cannot activate syscall filter')
    finally:
        lib.seccomp_release(context)
