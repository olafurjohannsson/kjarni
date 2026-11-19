"""Waits for the machine to be idle before a timed run, so another program's
load is not measured as the engine's."""
import glob, os, subprocess, sys, time

# A CPU that is still hot from the previous run clocks lower, so back-to-back runs
# would measure the heat as much as the engine. Each run starts at or below this
# package temperature (°C); BENCH_MAX_C overrides it.
MAX_C = float(os.environ.get("BENCH_MAX_C", "52"))


def cpu_temp_c():
    """The CPU package temperature from Linux's coretemp sensor, or None without one."""
    for name in glob.glob("/sys/class/hwmon/hwmon*/name"):
        if open(name).read().strip() != "coretemp":
            continue
        d = os.path.dirname(name)
        for label in glob.glob(os.path.join(d, "temp*_label")):
            if open(label).read().startswith("Package"):
                return int(open(label.replace("_label", "_input")).read()) / 1000
    return None


def _wait_cool(max_wait_s=1800):
    start = time.time()
    while (t := cpu_temp_c()) is not None and t > MAX_C:
        if time.time() - start > max_wait_s:
            sys.exit(f"CPU still at {t:.0f} °C after {max_wait_s} s (limit {MAX_C:.0f}); not measuring")
        time.sleep(10)


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
            _wait_cool()
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
            _wait_cool()
            return
        if time.time() - start > max_wait_s:
            sys.exit(f"machine not idle after {max_wait_s} s (load {one:.1f} / {five:.1f}); not measuring")
        time.sleep(10)
