"""
Compress a single video file using NVIDIA NVEnc (h264_nvenc).
Target bitrate = min(original_bitrate, requested_bitrate).
Output format: MP4.  Press P during encoding to pause / resume.

Usage:
    python compress_video.py <input_file> [--output <output_file>] [--bitrate <bitrate>]
"""

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import threading
import time

# ── GUI pause/resume hooks ────────────────────────────────────────
# Set _gui_pause to request a pause; set _gui_resume to request a resume.
# Both are consumed (cleared) immediately after being detected so they
# act as edge-triggered signals rather than level-triggered state.
_gui_pause  = threading.Event()
_gui_resume = threading.Event()

# ── Platform-specific keyboard / process helpers ─────────────────

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    import msvcrt

    _PROCESS_SUSPEND_RESUME = 0x0800

    def _suspend_process(pid: int):
        """Suspend all threads in a process (Windows only)."""
        handle = ctypes.windll.kernel32.OpenProcess(
            _PROCESS_SUSPEND_RESUME, False, pid
        )
        if handle:
            ctypes.windll.ntdll.NtSuspendProcess(handle)
            ctypes.windll.kernel32.CloseHandle(handle)

    def _resume_process(pid: int):
        """Resume all threads in a process (Windows only)."""
        handle = ctypes.windll.kernel32.OpenProcess(
            _PROCESS_SUSPEND_RESUME, False, pid
        )
        if handle:
            ctypes.windll.ntdll.NtResumeProcess(handle)
            ctypes.windll.kernel32.CloseHandle(handle)

    def _pause_key_pressed() -> bool:
        """Return True if the user pressed P/Space OR the GUI requested a pause."""
        if _gui_pause.is_set():
            _gui_pause.clear()
            return True
        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"p", b"P", b" ")
        return False

    def _wait_for_resume_key():
        """Block until the user presses P/Space OR the GUI requests a resume."""
        import time as _time
        _gui_resume.clear()  # discard any stale resume signal
        while True:
            if _gui_resume.is_set():
                _gui_resume.clear()
                return
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"p", b"P", b" "):
                    return
            _time.sleep(0.1)

else:
    # Non-Windows fallback — pause not supported
    def _suspend_process(pid: int):
        os.kill(pid, 19)  # SIGSTOP

    def _resume_process(pid: int):
        os.kill(pid, 18)  # SIGCONT

    def _pause_key_pressed() -> bool:
        return False

    def _wait_for_resume_key():
        pass


def check_pause_between_videos():
    """
    Call between videos to let the user pause the batch.
    Non-blocking check; if P was pressed or GUI pause requested, blocks until resumed.
    """
    if not _pause_key_pressed():
        return
    print("\n[PAUSED] Batch paused. Press P or click Resume to continue …")
    _wait_for_resume_key()
    print("[RESUMED] Continuing …\n")


# ── Helpers ──────────────────────────────────────────────────────

def _parse_bitrate_to_kbps(bitrate_str: str) -> int:
    """Convert a bitrate string like '700k' or '1M' to an integer in kbps."""
    bitrate_str = bitrate_str.strip().lower()
    if bitrate_str.endswith("m"):
        return int(float(bitrate_str[:-1]) * 1000)
    if bitrate_str.endswith("k"):
        return int(float(bitrate_str[:-1]))
    return int(float(bitrate_str) / 1000)


def probe_video(input_path: str) -> dict:
    """
    Single ffprobe call to get duration and video bitrate.

    Returns:
        {"duration": float | None, "video_bitrate_kbps": int | None}
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v:0",
        input_path,
    ]
    result_dict: dict = {"duration": None, "video_bitrate_kbps": None}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return result_dict
        info = json.loads(proc.stdout)

        fmt = info.get("format", {})
        if "duration" in fmt:
            result_dict["duration"] = float(fmt["duration"])

        streams = info.get("streams", [])
        if streams and "bit_rate" in streams[0]:
            result_dict["video_bitrate_kbps"] = int(streams[0]["bit_rate"]) // 1000
        elif "bit_rate" in fmt:
            result_dict["video_bitrate_kbps"] = max(
                int(fmt["bit_rate"]) // 1000 - 128, 1
            )
    except Exception:
        pass
    return result_dict


def _format_size(size_bytes: float) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


# Container / muxing overhead multiplier (≈2 %)
_CONTAINER_OVERHEAD = 1.02
# Audio bitrate used for encoding (kbps)
_AUDIO_BITRATE_KBPS = 128


def estimate_compressed_size(
    duration_s: float,
    video_bitrate_kbps: int,
    audio_bitrate_kbps: int = _AUDIO_BITRATE_KBPS,
) -> int:
    """
    Return the *approximate* output file size in bytes.

    Formula:
        bytes ≈ (video_kbps + audio_kbps) × 1000 / 8 × duration_s × overhead
    """
    if duration_s is None or duration_s <= 0:
        return 0
    bits = (video_bitrate_kbps + audio_bitrate_kbps) * 1000 * duration_s
    return int(bits / 8 * _CONTAINER_OVERHEAD)


def print_progress_bar(
    current: float, total: float, paused: bool = False, bar_length: int = 40,
    start_time: float = None,
):
    """Print a progress bar to stdout (overwrites current line)."""
    if total <= 0:
        return
    fraction = min(current / total, 1.0)
    filled = int(bar_length * fraction)
    bar = "█" * filled + "░" * (bar_length - filled)
    pct = fraction * 100
    cur_m, cur_s = divmod(int(current), 60)
    tot_m, tot_s = divmod(int(total), 60)

    if start_time and current > 0 and fraction < 1.0:
        wall_elapsed = time.time() - start_time
        speed = current / wall_elapsed          # video-seconds per wall-second
        eta_s = int((total - current) / speed)
        eta_m, eta_s = divmod(eta_s, 60)
        eta_str = f"  ETA {eta_m:02d}:{eta_s:02d}"
    else:
        eta_str = ""

    status = " ⏸ PAUSED " if paused else ""
    sys.stdout.write(
        f"\r  Progress: |{bar}| {pct:5.1f}%  "
        f"[{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}]"
        f"{eta_str}{status}   "
    )
    sys.stdout.flush()


def _parse_time_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.micro to seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


# ── Main compression function ───────────────────────────────────

def compress_video(input_path: str, output_path: str, bitrate: str = "700k"):
    """
    Compress a video using FFmpeg with h264_nvenc encoder.
    Shows a real-time progress bar.  Press P to pause/resume.

    Args:
        input_path:  Absolute path to the source video.
        output_path: Absolute path for the compressed output (.mp4).
        bitrate:     Target video bitrate (default "700k").

    Returns:
        True  — compression succeeded (output is smaller than original).
        False — compression failed (FFmpeg error).
        None  — skipped (bitrate already low, or output was larger than
                original and was discarded).
    """
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return False

    original_size = os.path.getsize(input_path)

    # ── Single probe for duration + bitrate ──────────────────────
    probe = probe_video(input_path)
    duration = probe["duration"]
    original_kbps = probe["video_bitrate_kbps"]
    target_kbps = _parse_bitrate_to_kbps(bitrate)

    # ── Determine effective bitrate: min(original, target) ───────
    if original_kbps is not None:
        effective_kbps = min(original_kbps, target_kbps)
        print(f"[INFO] Original bitrate: {original_kbps} kbps | "
              f"Target cap: {target_kbps} kbps | "
              f"Using: {effective_kbps} kbps")
        if original_kbps <= target_kbps:
            print(f"[SKIP] Original bitrate ({original_kbps}k) is already at or "
                  f"below the target ({target_kbps}k). Skipping compression.")
            return None
    else:
        effective_kbps = target_kbps
        print(f"[WARN] Could not detect original bitrate. "
              f"Using target: {target_kbps} kbps")

    effective_bitrate = f"{effective_kbps}k"

    # ── Estimated output size ────────────────────────────────────
    est_size = estimate_compressed_size(duration, effective_kbps)
    if est_size > 0:
        est_savings = original_size - est_size
        print(f"[EST]  Estimated output: {_format_size(est_size)}  |  "
              f"Savings: {_format_size(est_savings)}  "
              f"({est_savings / original_size * 100:.1f}%)"
              if est_savings > 0 else
              f"[EST]  Estimated output: {_format_size(est_size)}  |  "
              f"⚠ May be larger than original ({_format_size(original_size)})")

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "hevc_nvenc",
        "-b:v", effective_bitrate,
        "-maxrate", effective_bitrate,
        "-bufsize", f"{effective_kbps * 2}k",
        "-preset", "p4",
        "-rc", "vbr",
        "-c:a", "copy",
        "-movflags", "+faststart",
    ]

    if duration and duration > 0:
        cmd += ["-progress", "pipe:1", "-nostats"]
    cmd.append(output_path)

    print(f"[COMPRESS] Encoding: {os.path.basename(input_path)}")
    if _IS_WINDOWS:
        print(f"           Press P to pause / resume")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stderr_lines = []

        def _read_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)

        t = threading.Thread(target=_read_stderr, daemon=True)
        t.start()

        paused = False
        last_progress_s = 0.0
        encode_start = time.time()

        if duration and duration > 0:
            for line in proc.stdout:
                # ── Pause / resume on keypress ───────────────────
                if _pause_key_pressed():
                    if not paused:
                        _suspend_process(proc.pid)
                        paused = True
                        print_progress_bar(
                            last_progress_s, duration, paused=True,
                            start_time=encode_start,
                        )
                        print(
                            "\n[PAUSED] FFmpeg suspended. "
                            "Press P to resume …"
                        )
                        # Block here until the user presses P again
                        _wait_for_resume_key()
                        _resume_process(proc.pid)
                        paused = False
                        print("[RESUMED] Encoding continues …")

                line = line.strip()
                if line.startswith("out_time="):
                    time_str = line.split("=", 1)[1]
                    if time_str and time_str != "N/A":
                        last_progress_s = _parse_time_to_seconds(time_str)
                        print_progress_bar(last_progress_s, duration,
                                           start_time=encode_start)
                elif line.startswith("progress=end"):
                    print_progress_bar(duration, duration)
                    print()

        proc.wait()
        t.join(timeout=5)

        if proc.returncode != 0:
            print(f"\n[ERROR] FFmpeg failed for {input_path}")
            err_text = "".join(stderr_lines)
            print(err_text[-2000:] if len(err_text) > 2000 else err_text)
            # Clean up partial output
            if os.path.isfile(output_path):
                os.remove(output_path)
            return False

        # ── Size guard: discard if output >= original ────────────
        compressed_size = (
            os.path.getsize(output_path) if os.path.isfile(output_path) else 0
        )
        if compressed_size >= original_size:
            print(
                f"[DISCARD] Compressed file ({_format_size(compressed_size)}) "
                f"is not smaller than the original "
                f"({_format_size(original_size)}). Discarding output."
            )
            os.remove(output_path)
            return None

        print(f"[OK] Compressed successfully -> {output_path}")
        return True

    except FileNotFoundError:
        print(
            "[ERROR] ffmpeg/ffprobe not found. "
            "Make sure FFmpeg is installed and on your PATH."
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compress a video with H.264 NVEnc to MP4."
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the output file. Defaults to <name>_compressed.mp4.",
    )
    parser.add_argument(
        "--bitrate", "-b", default="700k",
        help="Target video bitrate (default: 700k).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_compressed.mp4"

    result = compress_video(input_path, output_path, args.bitrate)
    if result is True:
        sys.exit(0)
    elif result is None:
        sys.exit(2)  # skipped / discarded
    else:
        sys.exit(1)  # failed


if __name__ == "__main__":
    main()
