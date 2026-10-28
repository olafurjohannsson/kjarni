"""Waits for the machine to be idle before a timed run, so another program's
load is not measured as the engine's."""
import os, sys, time


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
