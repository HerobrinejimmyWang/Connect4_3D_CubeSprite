import multiprocessing
import os
from pathlib import Path


def _read_int(path):
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit():
    """Return a Linux cgroup CPU quota, or None when no quota is active."""
    try:
        quota_text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
        quota_value, period_value = quota_text.split()[:2]
        if quota_value != "max":
            quota = int(quota_value)
            period = int(period_value)
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass

    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and period is not None and quota > 0 and period > 0:
        return max(1, quota // period)
    return None


def available_cpu_count():
    """Return the CPU count available to this process, including cgroup quotas."""
    candidates = [max(1, int(multiprocessing.cpu_count()))]
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(max(1, len(os.sched_getaffinity(0))))
        except OSError:
            pass
    cgroup_limit = _cgroup_cpu_limit()
    if cgroup_limit is not None:
        candidates.append(cgroup_limit)
    return max(1, min(candidates))
