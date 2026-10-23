"""Limit torch's CPU threads to the CPUs the container may use.

torch starts os.cpu_count() threads. In a container with a CPU quota (a RunPod pod showed 128 cores and allowed 13.6)
they throttle each other and every CPU operation runs 10 to 40 times slower: the probe's regressions took 490 s
instead of 12 s. OMP_NUM_THREADS, when set, decides (the flat notebook sets it for everything it starts).
"""

import os
from pathlib import Path

CPU_MAX = "/sys/fs/cgroup/cpu.max"  # cgroup v2: "<quota> <period>" in microseconds, or "max <period>"


def allowed_cpus(cpu_max: str = CPU_MAX) -> int:
    """CPUs of the cgroup quota; the CPU count without one."""
    try:
        quota, period = Path(cpu_max).read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def limit_threads(limit: int = 8, cpu_max: str = CPU_MAX) -> int:
    """Set torch's CPU threads to min(limit, allowed CPUs) unless OMP_NUM_THREADS is set. Returns the number."""
    import torch

    if os.environ.get("OMP_NUM_THREADS"):
        return torch.get_num_threads()
    threads = max(1, min(limit, allowed_cpus(cpu_max)))
    torch.set_num_threads(threads)
    return threads
