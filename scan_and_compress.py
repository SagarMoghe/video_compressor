"""
Recursively scan a directory for video files over 1 GB and compress them
using H.264 NVEnc @ 700 kbps -> MP4.

Usage:
    python scan_and_compress.py <input_path> [--min-size <GB>] [--bitrate <bitrate>]
                                              [--workers <N>] [--dry-run]

Examples:
    python scan_and_compress.py "D:\\Videos"
    python scan_and_compress.py "D:\\Videos" --min-size 0.5 --dry-run
    python scan_and_compress.py "D:\\Videos" --workers 3
"""

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# Import compression function directly — avoids spawning a new Python
# interpreter per video (saves ~0.5-1 s startup overhead each time).
from compress_video import (
    compress_video,
    check_pause_between_videos,
    estimate_compressed_size,
    probe_video,
    _parse_bitrate_to_kbps,
)

# Video file extensions to consider (frozenset for faster lookups)
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".ts", ".vob", ".3gp", ".mts",
    ".m2ts", ".divx", ".ogv", ".f4v", ".asf",
})

# Size constants
ONE_GB = 1 << 30  # 1,073,741,824 bytes

# Folder name where originals are moved after successful compression
ORIGINALS_FOLDER = "_originals_to_delete"


# ── Helpers ──────────────────────────────────────────────────────

def format_size(size_bytes: float) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def _scandir_recursive(root: str, min_bytes: int):
    """
    Generator that yields (path, size) for every qualifying video file
    under *root*, using os.scandir for speed (avoids extra stat calls).
    """
    try:
        entries = os.scandir(root)
    except PermissionError:
        return

    with entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name == ORIGINALS_FOLDER:
                    continue  # skip already-moved originals
                yield from _scandir_recursive(entry.path, min_bytes)
            elif entry.is_file(follow_symlinks=False):
                name_lower = entry.name.lower()
                if name_lower.endswith("_compressed.mp4"):
                    continue
                ext = os.path.splitext(name_lower)[1]
                if ext not in VIDEO_EXTENSIONS:
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    continue
                if size >= min_bytes:
                    yield entry.path, size


def find_large_videos(root_path: str, min_bytes: int) -> List[dict]:
    """Return a list of dicts for every qualifying video, sorted largest-first."""
    results = [
        {"path": p, "size": s}
        for p, s in _scandir_recursive(root_path, min_bytes)
    ]
    results.sort(key=lambda v: v["size"], reverse=True)
    return results


def move_to_originals(video_path: str, root_path: str) -> Optional[str]:
    """
    Move the original video into an _originals_to_delete folder,
    preserving the relative directory structure.
    Returns the new path, or None on failure.
    """
    rel = os.path.relpath(video_path, root_path)
    dest = os.path.join(root_path, ORIGINALS_FOLDER, rel)
    dest_dir = os.path.dirname(dest)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(video_path, dest)
        return dest
    except OSError as e:
        print(f"[WARN] Could not move original: {e}")
        return None


# ── Single-video workflow (called per file) ──────────────────────

def _process_one(
    index: int,
    total: int,
    video: dict,
    bitrate: str,
    root: str,
) -> dict:
    """
    Compress one video and move the original on success.
    Returns a result dict for the summary / log.
    """
    path = video["path"]
    original_size = video["size"]
    base, _ = os.path.splitext(path)
    output_path = f"{base}_compressed.mp4"

    print(f"\n{'─'*60}")
    print(f"[{index}/{total}] Compressing: {path}")
    print(f"         Original size: {format_size(original_size)}")
    start = time.time()

    result = compress_video(path, output_path, bitrate)
    elapsed = time.time() - start

    rec = {
        "path": path,
        "original_size": original_size,
        "new_size": 0,
        "elapsed": elapsed,
        "status": "failed",
        "moved_to": "",
    }

    if result is True:
        rec["status"] = "ok"
        new_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
        rec["new_size"] = new_size
        savings = original_size - new_size
        print(f"[OK] Finished in {elapsed:.1f}s  |  "
              f"New size: {format_size(new_size)}  |  "
              f"Saved: {format_size(savings)}")
        moved = move_to_originals(path, root)
        if moved:
            rec["moved_to"] = moved
            print(f"[MOVED] Original moved to: {moved}")
    elif result is None:
        rec["status"] = "skipped"
        print(f"[SKIP] Skipped / discarded after {elapsed:.1f}s")
    else:
        print(f"[FAIL] Compression failed after {elapsed:.1f}s")

    return rec


# ── CSV log ──────────────────────────────────────────────────────

def write_log(log_path: str, records: List[dict]):
    """Write a simple CSV log of all processed videos."""
    fieldnames = ["path", "status", "original_size", "new_size", "elapsed", "moved_to"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[LOG] Results written to: {log_path}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan for large videos and compress them with NVEnc H.264."
    )
    parser.add_argument("input_path", help="Root directory to scan recursively.")
    parser.add_argument(
        "--min-size", type=float, default=1.0,
        help="Minimum file size in GB to consider (default: 1.0)."
    )
    parser.add_argument(
        "--bitrate", default="700k",
        help="Target video bitrate (default: 700k)."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel compression workers (default: 1). "
             "NVEnc supports multiple simultaneous sessions."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only list matching files; do not compress."
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Skip interactive file selection and compress all qualifying files."
    )
    args = parser.parse_args()

    root = os.path.abspath(args.input_path)
    if not os.path.isdir(root):
        print(f"[ERROR] Not a valid directory: {root}")
        sys.exit(1)

    min_bytes = int(args.min_size * ONE_GB)
    originals_dir = os.path.join(root, ORIGINALS_FOLDER)
    workers = max(1, args.workers)

    print(f"{'='*60}")
    print(f"  Video Compression Scanner")
    print(f"  Root path      : {root}")
    print(f"  Min size       : {format_size(min_bytes)}")
    print(f"  Bitrate        : {args.bitrate}")
    print(f"  Workers        : {workers}")
    print(f"  Dry run        : {args.dry_run}")
    print(f"  Originals dir  : {originals_dir}")
    print(f"  Pause          : Press P during encoding to pause/resume")
    print(f"{'='*60}\n")

    print("[SCAN] Searching for videos …")
    videos = find_large_videos(root, min_bytes)

    if not videos:
        print("[DONE] No videos found matching the criteria.")
        sys.exit(0)

    total_size = sum(v["size"] for v in videos)
    print(f"[SCAN] Found {len(videos)} video(s) totalling {format_size(total_size)}:\n")

    # Probe each video for duration/bitrate so we can show estimates
    target_kbps = _parse_bitrate_to_kbps(args.bitrate)
    total_est = 0
    for i, v in enumerate(videos, 1):
        probe = probe_video(v["path"])
        v["_probe"] = probe  # cache for later
        orig_kbps = probe.get("video_bitrate_kbps")
        duration = probe.get("duration")

        eff_kbps = min(orig_kbps, target_kbps) if orig_kbps else target_kbps
        est = estimate_compressed_size(duration, eff_kbps) if duration else 0
        v["_est_size"] = est
        est_savings = v["size"] - est if est > 0 else 0
        total_est += est

        print(f"  {i:>3}. {v['path']}")
        print(f"       Size: {format_size(v['size'])}", end="")
        if est > 0:
            pct = est_savings / v["size"] * 100 if v["size"] > 0 else 0
            print(f"  →  Est. output: {format_size(est)}  "
                  f"(save ~{format_size(est_savings)}, {pct:.0f}%)")
        else:
            print("  →  Est. output: N/A")

    if total_est > 0:
        total_savings = total_size - total_est
        print(f"\n  Total estimated output : {format_size(total_est)}")
        print(f"  Total estimated savings: {format_size(total_savings)} "
              f"({total_savings / total_size * 100:.0f}%)")
    print()

    if args.dry_run:
        print("[DRY-RUN] Exiting without compressing.")
        sys.exit(0)

    # ── Interactive file selection ───────────────────────────────
    if not args.auto:
        print("─" * 60)
        print("  Select files to compress:")
        print("    y = yes  |  n = no  |  a = all remaining  |  s = skip remaining  |  q = quit")
        print("─" * 60)
        selected = []
        accept_all = False
        for i, v in enumerate(videos, 1):
            if accept_all:
                selected.append(v)
                continue

            est = v.get("_est_size", 0)
            est_str = format_size(est) if est > 0 else "N/A"
            savings = v["size"] - est if est > 0 else 0
            pct = savings / v["size"] * 100 if v["size"] > 0 and est > 0 else 0

            prompt = (
                f"  [{i}/{len(videos)}] {os.path.basename(v['path'])}\n"
                f"          {format_size(v['size'])} → ~{est_str}"
            )
            if est > 0:
                prompt += f"  (save ~{format_size(savings)}, {pct:.0f}%)"
            prompt += "\n          Include? [y/n/a/s/q]: "

            while True:
                choice = input(prompt).strip().lower()
                if choice in ("y", "n", "a", "s", "q", ""):
                    break
                print("          Invalid choice. Enter y, n, a, s, or q.")

            if choice == "q":
                print("\n[QUIT] Exiting.")
                sys.exit(0)
            elif choice == "s":
                print(f"  Skipping remaining {len(videos) - i + 1} file(s).")
                break
            elif choice == "a":
                accept_all = True
                selected.append(v)
                remaining = len(videos) - i
                if remaining > 0:
                    print(f"  Including this and all remaining {remaining} file(s).")
            elif choice in ("y", ""):
                selected.append(v)
            # 'n' → just skip this file

        videos = selected
        print()

        if not videos:
            print("[DONE] No files selected for compression.")
            sys.exit(0)

        sel_size = sum(v["size"] for v in videos)
        sel_est = sum(v.get("_est_size", 0) for v in videos)
        print(f"[SELECTED] {len(videos)} file(s)  |  "
              f"Total: {format_size(sel_size)}  |  "
              f"Est. output: {format_size(sel_est)}\n")

    # ── Compress ─────────────────────────────────────────────────
    all_records = []
    total_count = len(videos)
    overall_start = time.time()

    if workers == 1:
        # Sequential — simpler output, no interleaving
        for i, v in enumerate(videos, 1):
            check_pause_between_videos()
            rec = _process_one(i, total_count, v, args.bitrate, root)
            all_records.append(rec)
    else:
        # Parallel — spin up a thread pool
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, v in enumerate(videos, 1):
                fut = pool.submit(_process_one, i, total_count, v, args.bitrate, root)
                futures[fut] = v
            for fut in as_completed(futures):
                all_records.append(fut.result())

    overall_elapsed = time.time() - overall_start

    # ── Summary ──────────────────────────────────────────────────
    succeeded = sum(1 for r in all_records if r["status"] == "ok")
    skipped   = sum(1 for r in all_records if r["status"] == "skipped")
    failed    = sum(1 for r in all_records if r["status"] == "failed")
    total_saved = sum(r["original_size"] - r["new_size"]
                      for r in all_records if r["status"] == "ok")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  Total      : {total_count}")
    print(f"  Success    : {succeeded}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print(f"  Total saved: {format_size(total_saved)}")
    print(f"  Elapsed    : {overall_elapsed:.1f}s")
    if succeeded > 0:
        print(f"  Originals  : {originals_dir}")
        print(f"               Review and delete manually when ready.")
    print(f"{'='*60}")

    # Write a log CSV next to the script
    log_path = os.path.join(root, "compression_log.csv")
    write_log(log_path, all_records)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()


