"""Waits for the machine to be idle before a timed run, so another program's
load is not measured as the engine's."""
import os, subprocess, sys, time


def vram_used_mb():
    """GPU memory in use across the whole card, in MiB."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return float(out.split()[0])


def _cpu_busy(seconds):
    def snapshot():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4]
    total0, idle0 = snapshot()
    time.sleep(seconds)
    total1, idle1 = snapshot()
    return 1 - (idle1 - idle0) / (total1 - total0)


def wait_quiet(max_wait_s=3600):
    """Wait until the CPU and the GPU are both doing nothing right now. Sampling
    them directly, unlike the load average, does not count the previous run, so
    a sweep of many short runs is not spent waiting for an average to decay."""
    start = time.time()
    while True:
        busy = _cpu_busy(5)
        gpu = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True).stdout.split()[0])
        if busy < 0.05 and gpu < 5:
            return
        if time.time() - start > max_wait_s:
            sys.exit(f"machine not quiet after {max_wait_s} s (cpu {busy:.0%}, gpu {gpu:.0f}%); not measuring")
        time.sleep(10)


def wait_idle(max_wait_s=900):
    """Wait until the 1-minute load average is under 1 and the 5-minute one
    under 1.5. The 5-minute average catches load that comes in bursts, such as a
    busy browser tab, which the 1-minute one can miss between them."""
    start = time.time()
    while True:
        one, five, _ = os.getloadavg()
        if one < 1 and five < 1.5:
            return
        if time.time() - start > max_wait_s:
            sys.exit(f"machine not idle after {max_wait_s} s (load {one:.1f} / {five:.1f}); not measuring")
        time.sleep(10)
