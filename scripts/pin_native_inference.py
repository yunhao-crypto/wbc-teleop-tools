#!/usr/bin/env python3
"""Pin the native locomotion inference thread to dedicated CPUs at runtime.

The MimicLite policy runs ONNX Runtime with one intra-op thread, so its
inference executes only on the persistent AsyncInferenceState worker of
locomotion_exec. That worker is unnamed, so it is identified by sampling the
SCHED_OTHER locomotion threads with perf and taking the thread whose samples
fall in libonnxruntime. Real-time (SCHED_FIFO/RR) threads are never sampled or
changed. The mask lasts until locomotion restarts. Failure never stops the
session. Use --log to save decisions as JSON lines.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ORT = 'libonnxruntime'


def parse_cpus(text):
    cpus = set()
    for part in text.split(','):
        low, _, high = part.partition('-')
        low, high = int(low), int(high or low)
        if low < 0 or high < low:
            raise ValueError(f'invalid CPU list: {text}')
        cpus.update(range(low, high+1))
    if not cpus:
        raise ValueError(f'invalid CPU list: {text}')
    return cpus


def ort_samples(script_text):
    """perf script ``-F tid,cpu,ip,dso`` lines -> {tid: Counter(cpu)} for ORT samples, and all samples."""
    ort, every = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for line in script_text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            tid, cpu = int(fields[0]), int(fields[1].strip('[]'))
        except ValueError:
            continue
        every[tid][cpu] += 1
        if ORT in fields[-1]:
            ort[tid][cpu] += 1
    return ort, every


def choose(ort, minimum=20, share=.8):
    totals = {tid: sum(c.values()) for tid, c in ort.items()}
    total = sum(totals.values())
    if total < minimum:
        return None
    tid = max(totals, key=totals.get)
    return tid if totals[tid] >= share*total else None


def locomotion_pid():
    out = subprocess.run(['systemctl', 'show', '-p', 'MainPID', '--value', 'locomotion'],
                         capture_output=True, text=True, timeout=5).stdout.strip()
    pid = int(out or 0)
    if pid <= 0 or Path(os.readlink(f'/proc/{pid}/exe')).name != 'locomotion_exec':
        raise RuntimeError(f'locomotion_exec is not running (MainPID={out!r})')
    return pid


def other_threads(pid):
    """SCHED_OTHER threads only; real-time threads are left untouched."""
    tids = []
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            stat = (task/'stat').read_text()
        except OSError:
            continue
        if int(stat[stat.rfind(')')+2:].split()[38]) == 0:
            tids.append(int(task.name))
    return sorted(tids)


def sample(tids, seconds, frequency):
    with tempfile.TemporaryDirectory(prefix='inference-pin-') as tmp:
        data = Path(tmp)/'perf.data'
        # Per-thread (-t) recording omits the CPU field unless --sample-cpu is given.
        subprocess.run(['perf', 'record', '-q', '-N', '-B', '--sample-cpu', '-e', 'cpu-clock', '-F', str(frequency),
                        '-t', ','.join(map(str, tids)), '-o', str(data), '--', 'sleep', str(seconds)],
                       capture_output=True, timeout=seconds+20, check=True)
        script = subprocess.run(['perf', 'script', '-i', str(data), '-F', 'tid,cpu,ip,dso'],
                                capture_output=True, text=True, timeout=60, check=True).stdout
    return ort_samples(script)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpus', default='4', help='CPU list for the inference thread, e.g. 4 or 0-5')
    p.add_argument('--log', type=Path)
    p.add_argument('--launcher-pid', type=int, help='Stop searching when the launching WTT session exits')
    p.add_argument('--activation-file', type=Path,
                   help='Wait for this session to confirm native activation before sampling')
    p.add_argument('--timeout', type=float, default=180.)
    p.add_argument('--window', type=float, default=1.)
    p.add_argument('--frequency', type=int, default=499)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open('a')

    def write(event, **fields):
        if log is not None:
            log.write(json.dumps(dict(event=event, realtime_ns=time.time_ns(), **fields))+'\n')
            log.flush()

    def say(text):
        log_note = f' (log {args.log})' if args.log is not None else ''
        print('MIMICLITE_INFERENCE_PIN: '+text+log_note, file=sys.stderr, flush=True)

    try:
        cpus = parse_cpus(args.cpus)
        deadline = time.monotonic()+args.timeout
        launcher = Path(f'/proc/{args.launcher_pid}') if args.launcher_pid else None
        if args.activation_file is not None:
            while time.monotonic() < deadline and (launcher is None or launcher.exists()):
                if args.activation_file.is_file():
                    break
                time.sleep(.2)
            else:
                write('give_up', reason='native activation was not confirmed')
                return 0
        pid = locomotion_pid()
        write('start', pid=pid, cpus=sorted(cpus), helper_affinity=sorted(os.sched_getaffinity(0)))

        def try_sample():
            tids = other_threads(pid)
            try:
                return tids, sample(tids, args.window, args.frequency)
            except subprocess.CalledProcessError as error:
                # perf -t fails when a listed thread exits before recording starts; resample.
                write('sample_failed', threads=len(tids), returncode=error.returncode,
                      stderr=(error.stderr or b'').decode(errors='replace')[-500:])
                return tids, None

        while (launcher is None or launcher.exists()) and (
                args.activation_file is None or args.activation_file.is_file()):
            tids, result = try_sample()
            target = None
            if result is not None:
                ort, every = result
                target = choose(ort)
                write('sample', threads=len(tids), ort={str(t): sum(c.values()) for t, c in ort.items()}, target=target)
            if target is not None and target in other_threads(pid):
                break
            if time.monotonic() >= deadline:
                write('give_up', reason='no single thread running ONNX Runtime')
                say(f'no inference activity found within {args.timeout:.0f} s; not pinned')
                return 0
            time.sleep(1.)
        if ((launcher is not None and not launcher.exists()) or
                (args.activation_file is not None and not args.activation_file.is_file())):
            write('give_up', reason='native session ended')
            return 0
        before = sorted(os.sched_getaffinity(target))
        os.sched_setaffinity(target, cpus)
        after = sorted(os.sched_getaffinity(target))
        if set(after) != cpus:
            raise RuntimeError(f'requested CPUs {sorted(cpus)} but thread {target} has {after}')
        write('pinned', tid=target, before=before, after=after)
        if log is not None:
            for _ in range(3):
                _, result = try_sample()
                if result is not None:
                    ort, every = result
                    intruders = {str(t): sum(n for cpu, n in c.items() if cpu in cpus)
                                 for t, c in every.items() if t != target}
                    write('verify', tid=target, ort_cpus={str(k): v for k, v in ort.get(target, {}).items()},
                          other_locomotion_samples_on_target_cpus={t: n for t, n in intruders.items() if n})
                    break
        say(f'thread {target} {before} -> {after}')
        return 0
    except KeyboardInterrupt:
        write('interrupted')
        return 0
    except Exception as error:
        write('error', error=repr(error))
        say(f'not pinned: {error!r}')
        return 1
    finally:
        if log is not None:
            log.close()


if __name__ == '__main__':
    raise SystemExit(main())
