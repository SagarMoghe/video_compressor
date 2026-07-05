"""
Compress a single video file using AV1 (default: NVIDIA NVENC av1_nvenc).
Target bitrate = min(original_bitrate, requested_bitrate).
Output format: MP4.  Press P during encoding to pause / resume.

Usage:
    python compress_video.py <input_file> [--output <output_file>] [--bitrate <bitrate>]
                           [--encoder <ffmpeg_encoder>]
"""

import argparse
import collections
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time

# ── GUI pause/resume/stop hooks ───────────────────────────────────
# Set _gui_pause to request a pause; set _gui_resume to request a resume.
# Both are consumed (cleared) immediately after being detected so they
# act as edge-triggered signals rather than level-triggered state.
# _gui_stop is set when the application is closing — compression should abort.
_gui_pause  = threading.Event()
_gui_resume = threading.Event()
_gui_stop   = threading.Event()

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
        if _gui_stop.is_set():
            return False
        if _gui_pause.is_set():
            _gui_pause.clear()
            return True
        try:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                return key in (b"p", b"P", b" ")
        except Exception:
            pass
        return False

    def _wait_for_resume_key():
        """Block until the user presses P/Space OR the GUI requests a resume, or stop is signalled."""
        while not _gui_stop.is_set():
            if _gui_resume.is_set():
                _gui_resume.clear()
                return
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b"p", b"P", b" "):
                        return
            except Exception:
                pass
            time.sleep(0.05)

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
        kwargs = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
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


def format_size(size_bytes: float) -> str:
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
_DEFAULT_ENCODER = "av1_nvenc"


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


_SRT_TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})(.*)$")


def _srt_time_to_ms(value: str) -> int:
    """Convert an SRT timestamp (HH:MM:SS,mmm) to milliseconds."""
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(",")
    return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(ms)


def _ms_to_srt_time(value_ms: int) -> str:
    """Convert milliseconds to SRT timestamp format."""
    value_ms = max(0, int(value_ms))
    total_s, ms = divmod(value_ms, 1000)
    hh, rem = divmod(total_s, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _shift_srt_file(input_path: str, offset_s: float) -> str:
    """Create a temporary SRT with all cues shifted by offset_s seconds."""
    offset_ms = int(round(offset_s * 1000))
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    blocks = re.split(r"\r?\n\r?\n", content.strip()) if content.strip() else []
    shifted_blocks = []
    for block in blocks:
        lines = block.splitlines()
        timing_idx = -1
        for i, line in enumerate(lines):
            if _SRT_TIME_RE.match(line.strip()):
                timing_idx = i
                break
        if timing_idx < 0:
            shifted_blocks.append(block)
            continue

        match = _SRT_TIME_RE.match(lines[timing_idx].strip())
        start_ms = _srt_time_to_ms(match.group(1)) + offset_ms
        end_ms = _srt_time_to_ms(match.group(2)) + offset_ms

        if end_ms <= 0:
            continue  # cue moved fully before zero; drop it
        start_ms = max(0, start_ms)
        end_ms = max(0, end_ms)
        if end_ms <= start_ms:
            end_ms = start_ms + 1

        lines[timing_idx] = f"{_ms_to_srt_time(start_ms)} --> {_ms_to_srt_time(end_ms)}{match.group(3)}"
        shifted_blocks.append("\n".join(lines))

    fd, temp_path = tempfile.mkstemp(prefix="shifted_subs_", suffix=".srt")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
        if shifted_blocks:
            f.write("\n\n".join(shifted_blocks) + "\n")
        else:
            f.write("")
    return temp_path


def _escape_subtitles_path(path: str) -> str:
    """Escape a filesystem path for use inside ffmpeg's subtitles filter."""
    norm = os.path.abspath(path).replace("\\", "/")
    norm = norm.replace(":", r"\:")
    norm = norm.replace("'", r"\'")
    norm = norm.replace(",", r"\,")
    norm = norm.replace("[", r"\[").replace("]", r"\]")
    return norm


def _toggle_ffmpeg_pause(proc: subprocess.Popen) -> bool:
    """Toggle ffmpeg's built-in pause state by writing 'p' to stdin."""
    try:
        if proc.stdin:
            proc.stdin.write("p\n")
            proc.stdin.flush()
            return True
    except Exception:
        pass
    return False


def _remux_to_mp4_copy(input_path: str, output_path: str) -> bool:
    """Fast path: remux input to MP4 with stream copy (no re-encode)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-probesize", "50M",
        "-analyzeduration", "100M",
        "-fflags", "+discardcorrupt",
        "-i", input_path,
        "-map", "0",
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        kwargs = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        if proc.returncode == 0 and os.path.isfile(output_path):
            return True
        if os.path.isfile(output_path):
            os.remove(output_path)
        return False
    except Exception:
        if os.path.isfile(output_path):
            os.remove(output_path)
        return False


# ── Main compression function ───────────────────────────────────

def compress_video(
    input_path: str,
    output_path: str,
    bitrate: str = "700k",
    encoder: str = _DEFAULT_ENCODER,
    subtitle_path: str | None = None,
    subtitle_offset_s: float = 0.0,
):
    """
    Compress a video using FFmpeg.
    Shows a real-time progress bar.  Press P to pause/resume.

    Args:
        input_path:  Absolute path to the source video.
        output_path: Absolute path for the compressed output (.mp4).
        bitrate:     Target video bitrate (default "700k").
        encoder:     FFmpeg video encoder (default "av1_nvenc").
        subtitle_path: Optional path to an .srt subtitle file to burn in.
        subtitle_offset_s: Subtitle offset in seconds (positive delays subtitles).

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
    has_subtitles = bool(subtitle_path and str(subtitle_path).strip())

    # ── Single probe for duration + bitrate ──────────────────────
    probe = probe_video(input_path)
    duration = probe["duration"]
    original_kbps = probe["video_bitrate_kbps"]
    target_kbps = _parse_bitrate_to_kbps(bitrate)

    # ── Determine effective bitrate: min(original, target) ───────
    is_already_mp4 = os.path.splitext(input_path)[1].lower() == ".mp4"

    if original_kbps is not None:
        effective_kbps = min(original_kbps, target_kbps)
        print(f"[INFO] Original bitrate: {original_kbps} kbps | "
              f"Target cap: {target_kbps} kbps | "
              f"Using: {effective_kbps} kbps")
        if original_kbps <= target_kbps and is_already_mp4 and not has_subtitles:
            print(f"[SKIP] Original bitrate ({original_kbps}k) is already at or "
                  f"below the target ({target_kbps}k) and file is already MP4. "
                  f"Skipping compression.")
            return None
        elif original_kbps <= target_kbps and not has_subtitles:
            print(f"[INFO] Original bitrate ({original_kbps}k) is at or below target "
                  f"({target_kbps}k), file is not MP4. Trying fast MP4 remux first.")
            if _remux_to_mp4_copy(input_path, output_path):
                print(f"[OK] Remuxed successfully -> {output_path}")
                return True
            print("[WARN] Fast remux failed (codec/container incompatibility). "
                  "Falling back to full re-encode.")
        elif original_kbps <= target_kbps and has_subtitles:
            print("[INFO] Subtitle burn-in requested; forcing re-encode (no skip/remux).")
    else:
        effective_kbps = target_kbps
        print(f"[WARN] Could not detect original bitrate. "
              f"Using target: {target_kbps} kbps")

    effective_bitrate = f"{effective_kbps}k"

    # ── Estimated output size ────────────────────────────────────
    est_size = estimate_compressed_size(duration, effective_kbps)
    if est_size > 0:
        est_savings = original_size - est_size
        print(f"[EST]  Estimated output: {format_size(est_size)}  |  "
              f"Savings: {format_size(est_savings)}  "
              f"({est_savings / original_size * 100:.1f}%)"
              if est_savings > 0 else
              f"[EST]  Estimated output: {format_size(est_size)}  |  "
              f"⚠ May be larger than original ({format_size(original_size)})")

    shifted_subtitle_path = None
    subtitle_filter = None
    if subtitle_path:
        subtitle_path = os.path.abspath(subtitle_path)
        if not os.path.isfile(subtitle_path):
            print(f"[ERROR] Subtitle file not found: {subtitle_path}")
            return False
        source_subtitle = subtitle_path
        if abs(subtitle_offset_s) > 1e-9:
            shifted_subtitle_path = _shift_srt_file(subtitle_path, subtitle_offset_s)
            source_subtitle = shifted_subtitle_path
            print(f"[INFO] Applied subtitle offset: {subtitle_offset_s:+.3f}s")
        subtitle_filter = f"subtitles='{_escape_subtitles_path(source_subtitle)}'"
        print(f"[INFO] Burning subtitles from: {subtitle_path}")

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y", "-probesize", "50M", "-analyzeduration", "100M", "-fflags", "+discardcorrupt",
        "-i", input_path,
    ]
    if subtitle_filter:
        cmd += ["-vf", subtitle_filter]
    cmd += [
        "-c:v", encoder,
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
        popen_kwargs = {}
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )

        stderr_lines: collections.deque = collections.deque(maxlen=200)

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
                # ── Stop check (app closing) ──────────────────────
                if _gui_stop.is_set():
                    proc.terminate()
                    proc.wait(timeout=5)
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                    return False

                # ── Pause / resume on keypress ───────────────────
                if _pause_key_pressed():
                    if not paused:
                        if not _toggle_ffmpeg_pause(proc):
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
                        if not _toggle_ffmpeg_pause(proc):
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
        else:
            # Keep pause handling available even when progress mode is disabled.
            while proc.poll() is None:
                if _gui_stop.is_set():
                    proc.terminate()
                    proc.wait(timeout=5)
                    if os.path.isfile(output_path):
                        os.remove(output_path)
                    return False
                if _pause_key_pressed() and not paused:
                    if not _toggle_ffmpeg_pause(proc):
                        _suspend_process(proc.pid)
                    paused = True
                    print("\n[PAUSED] FFmpeg suspended. Press P to resume …")
                    _wait_for_resume_key()
                    if not _toggle_ffmpeg_pause(proc):
                        _resume_process(proc.pid)
                    paused = False
                    print("[RESUMED] Encoding continues …")
                time.sleep(0.1)

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

        # ── Size guard: discard if output >= original (only for mp4 sources) ─
        compressed_size = (
            os.path.getsize(output_path) if os.path.isfile(output_path) else 0
        )
        if compressed_size >= original_size:
            if is_already_mp4 and not has_subtitles:
                print(
                    f"[DISCARD] Compressed file ({format_size(compressed_size)}) "
                    f"is not smaller than the original "
                    f"({format_size(original_size)}). Discarding output."
                )
                os.remove(output_path)
                return None
            else:
                if has_subtitles:
                    print(
                        f"[KEEP] Compressed file ({format_size(compressed_size)}) "
                        f"is not smaller than the original "
                        f"({format_size(original_size)}), but keeping output "
                        f"because subtitles were burned in."
                    )
                else:
                    print(
                        f"[KEEP] Compressed file ({format_size(compressed_size)}) "
                        f"is not smaller than the original "
                        f"({format_size(original_size)}), but keeping MP4 "
                        f"(original was not MP4)."
                    )

        print(f"[OK] Compressed successfully -> {output_path}")
        return True

    except FileNotFoundError:
        print(
            "[ERROR] ffmpeg/ffprobe not found. "
            "Make sure FFmpeg is installed and on your PATH."
        )
        return False
    finally:
        if shifted_subtitle_path and os.path.isfile(shifted_subtitle_path):
            try:
                os.remove(shifted_subtitle_path)
            except OSError:
                pass


def create_subtitle_preview(
    input_path: str,
    output_path: str,
    subtitle_path: str,
    subtitle_offset_s: float = 0.0,
    preview_seconds: float = 30.0,
    preview_start_s: float = 0.0,
) -> bool:
    """Render a short preview clip with burned subtitles to help tune subtitle offset."""
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return False
    if not os.path.isfile(subtitle_path):
        print(f"[ERROR] Subtitle file not found: {subtitle_path}")
        return False

    shifted_subtitle_path = None
    try:
        source_subtitle = os.path.abspath(subtitle_path)
        start_s = max(0.0, float(preview_start_s))
        duration_s = max(1.0, float(preview_seconds))

        # Preview clips reset video time to 0 at the selected start, so shift
        # subtitles by (offset - preview_start) to preserve absolute timing.
        preview_sub_shift_s = float(subtitle_offset_s) - start_s
        if abs(preview_sub_shift_s) > 1e-9:
            shifted_subtitle_path = _shift_srt_file(source_subtitle, preview_sub_shift_s)
            source_subtitle = shifted_subtitle_path

        subtitle_filter = f"subtitles='{_escape_subtitles_path(source_subtitle)}'"
        print(
            f"[PREVIEW] start={start_s:.3f}s  duration={duration_s:.3f}s  "
            f"subtitle_offset={float(subtitle_offset_s):+.3f}s"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-probesize", "50M",
            "-analyzeduration", "100M",
            "-fflags", "+discardcorrupt",
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration_s:.3f}",
            "-i", input_path,
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]

        kwargs = {}
        if _IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        if proc.returncode != 0:
            print("[ERROR] Failed to generate subtitle preview clip.")
            err = proc.stderr or ""
            print(err[-2000:] if len(err) > 2000 else err)
            if os.path.isfile(output_path):
                os.remove(output_path)
            return False
        return os.path.isfile(output_path)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install FFmpeg and add it to PATH.")
        return False
    except Exception as exc:
        print(f"[ERROR] Subtitle preview failed: {exc}")
        if os.path.isfile(output_path):
            os.remove(output_path)
        return False
    finally:
        if shifted_subtitle_path and os.path.isfile(shifted_subtitle_path):
            try:
                os.remove(shifted_subtitle_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Compress a video to AV1 (default av1_nvenc) and output MP4."
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
    parser.add_argument(
        "--encoder", "-e", default=_DEFAULT_ENCODER,
        help=(
            "FFmpeg video encoder to use "
            f"(default: {_DEFAULT_ENCODER}, e.g. libsvtav1, libaom-av1)."
        ),
    )
    parser.add_argument(
        "--subtitle", "-s", default=None,
        help="Optional .srt subtitle file to burn into the video.",
    )
    parser.add_argument(
        "--subtitle-offset", type=float, default=0.0,
        help="Subtitle offset in seconds (positive delays subtitles).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_compressed.mp4"

    result = compress_video(
        input_path,
        output_path,
        args.bitrate,
        args.encoder,
        args.subtitle,
        args.subtitle_offset,
    )
    if result is True:
        sys.exit(0)
    elif result is None:
        sys.exit(2)  # skipped / discarded
    else:
        sys.exit(1)  # failed


if __name__ == "__main__":
    main()
