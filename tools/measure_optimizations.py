#!/usr/bin/env python3
"""Run solver commands sequentially and retain unfiltered process measurements.

Single command (literal argv, no shell):
  python tools/measure_optimizations.py --out-dir measurements/run1 -- build/amls_solve.exe --case path:32 --nev 4 --threads 1 --quiet

Batch: --commands-file commands.json containing argument arrays or objects:
  [{"name": "baseline", "argv": ["build/amls_solve.exe", "--case", "path:32"],
    "env": {"AMLS_BATCH_OFF": "1", "AMLS_N7_SERIAL": "1", "AMLS_NO_ADAPT": null}}]
An env string sets a variable; null removes it. Each run starts from a fresh
copy of the parent environment with MinGW prepended to PATH. Nonzero exits do
not stop the batch. Raw stdout/stderr are binary files. On Windows, samples.csv
records process CPU seconds and working set every 0.5 seconds by default.
"""

import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def load_commands(args):
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if args.commands_file:
        if command:
            raise ValueError("use either --commands-file or a command after --")
        entries = json.loads(args.commands_file.read_text(encoding="utf-8-sig"))
    else:
        entries = [command] if command else []
    if not isinstance(entries, list) or not entries:
        raise ValueError("provide a nonempty JSON command list or a command after --")
    commands = []
    for index, entry in enumerate(entries, 1):
        if isinstance(entry, dict):
            name = entry.get("name", f"run-{index}")
            argv = entry.get("argv")
            overrides = entry.get("env", {})
        else:
            name, argv, overrides = f"run-{index}", entry, {}
        if not isinstance(name, str) or not name:
            raise ValueError(f"command {index}: name must be a nonempty string")
        if (not isinstance(argv, list) or not argv or
                any(not isinstance(arg, str) or "\0" in arg for arg in argv) or not argv[0]):
            raise ValueError(f"command {index}: argv must be a nonempty list of strings")
        if (not isinstance(overrides, dict) or any(
                not isinstance(key, str) or not key or "=" in key or "\0" in key or
                (value is not None and (not isinstance(value, str) or "\0" in value))
                for key, value in overrides.items())):
            raise ValueError(f"command {index}: env must map variable names to strings or null")
        commands.append((name, argv, overrides))
    return commands


def executable_info(command, cwd, env):
    """Resolve once and launch this same absolute path, so the hash identifies it."""
    local = cwd / command
    if local.is_file():
        path = local.resolve()
    elif os.name == "nt" and local.with_suffix(local.suffix + ".exe").is_file():
        path = local.with_suffix(local.suffix + ".exe").resolve()
    else:
        found = shutil.which(command, path=env.get("PATH", ""))
        if not found:
            raise FileNotFoundError(f"executable not found: {command}")
        path = Path(found).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return str(path), digest


class WindowsSampler:
    """Keep an independent process handle for a final CPU sample after exit."""

    def __init__(self, pid):
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        self.MemoryCounters = MemoryCounters
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def sample(self):
        row = {}
        errors = []
        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if self.kernel.GetProcessTimes(self.handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                       ctypes.byref(kernel), ctypes.byref(user)):
            kernel_seconds = ((kernel.dwHighDateTime << 32) | kernel.dwLowDateTime) * 1e-7
            user_seconds = ((user.dwHighDateTime << 32) | user.dwLowDateTime) * 1e-7
            row.update(kernel_seconds=kernel_seconds, user_seconds=user_seconds,
                       cpu_seconds=kernel_seconds + user_seconds)
        else:
            errors.append(f"GetProcessTimes:{ctypes.get_last_error()}")
        memory = self.MemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        if self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(memory), memory.cb):
            row.update(working_set_bytes=memory.WorkingSetSize,
                       peak_working_set_bytes=memory.PeakWorkingSetSize)
        else:
            errors.append(f"GetProcessMemoryInfo:{ctypes.get_last_error()}")
        row["error"] = ";".join(errors)
        return row

    def close(self):
        self.kernel.CloseHandle(self.handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands-file", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("measurements") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
                        help="new directory for logs and measurements (must not exist)")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--mingw-bin", type=Path,
                        default=Path(os.environ.get("MINGW_BIN", "D:/msys2/mingw64/bin")),
                        help="prepend to child PATH (default: MINGW_BIN or D:/msys2/mingw64/bin)")
    parser.add_argument("--timeout", type=float, help="per-command timeout in seconds")
    parser.add_argument("--sample-interval", type=float, default=0.5,
                        help="CPU/memory sampling interval in seconds (default: 0.5)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        commands = load_commands(args)
        if args.timeout is not None and not 0 < args.timeout < float("inf"):
            raise ValueError("--timeout must be a finite positive number")
        if not 0 < args.sample_interval < float("inf"):
            raise ValueError("--sample-interval must be a finite positive number")
        cwd = args.cwd.resolve(strict=True)
        if not cwd.is_dir():
            raise ValueError("--cwd must be a directory")
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    base_env = os.environ.copy()
    mingw_bin = str(args.mingw_bin.resolve())
    base_env["PATH"] = mingw_bin + os.pathsep + base_env.get("PATH", "")
    runs = []
    for index, (name, argv, overrides) in enumerate(commands, 1):
        env = base_env.copy()
        for key, value in overrides.items():
            # Windows environment variable names are case-insensitive.
            if os.name == "nt":
                for existing in list(env):
                    if existing.upper() == key.upper():
                        del env[existing]
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        run_dir = out_dir / f"{index:03d}"
        run_dir.mkdir()
        print(f"[{index}/{len(commands)}] {name}: {json.dumps(argv, ensure_ascii=False)}", flush=True)
        executable = digest = error = sample_error = returncode = None
        status = "completed"
        sampler = None
        sample_count = 0
        final_sample = None
        # Hashing happens before the process wall timer, once per run.
        try:
            executable, digest = executable_info(argv[0], cwd, env)
        except OSError as exc:
            status, error = "launch_error", str(exc)
        with (run_dir / "stdout.log").open("wb") as stdout, \
                (run_dir / "stderr.log").open("wb") as stderr, \
                (run_dir / "samples.csv").open("w", newline="", encoding="utf-8") as samples:
            writer = csv.DictWriter(samples, fieldnames=[
                "wall_seconds", "cpu_seconds", "kernel_seconds", "user_seconds",
                "working_set_bytes", "peak_working_set_bytes", "error"])
            writer.writeheader()
            started_at = datetime.now(timezone.utc).isoformat()
            start = time.perf_counter()

            def sample():
                nonlocal sample_count, final_sample
                if sampler is not None:
                    final_sample = sampler.sample()
                    writer.writerow({"wall_seconds": time.perf_counter() - start, **final_sample})
                    samples.flush()
                    sample_count += 1

            try:
                if status != "launch_error":
                    process = subprocess.Popen([executable, *argv[1:]], cwd=cwd, env=env,
                                               stdout=stdout, stderr=stderr, shell=False)
                    if os.name == "nt":
                        try:
                            sampler = WindowsSampler(process.pid)
                        except OSError as exc:
                            sample_error = str(exc)
                    else:
                        sample_error = "CPU/working-set sampling requires Windows"
                    try:
                        sample()
                        while True:
                            remaining = None if args.timeout is None else args.timeout - (time.perf_counter() - start)
                            if remaining is not None and remaining <= 0:
                                if process.poll() is not None:
                                    returncode = process.returncode
                                    break
                                raise subprocess.TimeoutExpired(argv, args.timeout)
                            try:
                                returncode = process.wait(timeout=args.sample_interval if remaining is None
                                                          else min(args.sample_interval, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                sample()
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait()
                        status, error = "timeout", f"exceeded {args.timeout} seconds"
                    except KeyboardInterrupt:
                        process.kill()
                        returncode = process.wait()
                        status, error = "interrupted", "interrupted by user"
                wall = time.perf_counter() - start
                sample()
            except OSError as exc:
                status, error = "launch_error", str(exc)
                wall = time.perf_counter() - start
            finally:
                if sampler is not None:
                    sampler.close()
        measurement = {
            "index": index, "name": name, "argv": argv, "cwd": str(cwd),
            "env": overrides, "mingw_bin": mingw_bin, "effective_path": env.get("PATH"),
            "executable": executable, "executable_sha256": digest,
            "started_at_utc": started_at, "status": status,
            "exitcode": returncode, "wall_seconds": wall,
            "timeout_seconds": args.timeout, "error": error,
            "sample_interval_seconds": args.sample_interval, "sample_count": sample_count,
            "sampling_error": sample_error, "final_sample": final_sample,
            "samples": str(run_dir / "samples.csv"),
            "stdout": str(run_dir / "stdout.log"), "stderr": str(run_dir / "stderr.log"),
        }
        (run_dir / "exitcode.txt").write_text(
            "unavailable\n" if returncode is None else f"{returncode}\n", encoding="utf-8")
        (run_dir / "wall_seconds.txt").write_text(f"{wall:.9f}\n", encoding="utf-8")
        (run_dir / "run.json").write_text(
            json.dumps(measurement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        runs.append(measurement)
        (out_dir / "summary.json").write_text(
            json.dumps({"runs": runs}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  status={status} exitcode={returncode} wall_seconds={wall:.6f}", flush=True)
        if status == "interrupted":
            return 130
    print(f"saved: {out_dir}", flush=True)
    return 0 if all(run["status"] == "completed" and run["exitcode"] == 0 for run in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
