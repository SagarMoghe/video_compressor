"""
GUI front-end for the video compression pipeline.

Workflow:
  1. Browse for a root folder (or pick individual files).
  2. Click "Scan" → probes each video and shows original vs. estimated size.
  3. Tick the checkboxes for the files you want to compress.
  4. Click "Compress Selected" → runs the job in a background thread.

Requires: compress_video.py and scan_and_compress.py in the same directory.
"""

import json
import os
import queue
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from compress_video import (
    _gui_pause,
    _gui_resume,
    _parse_bitrate_to_kbps,
    estimate_compressed_size,
    format_size,
    probe_video,
)
from scan_and_compress import (
    _process_one,
    find_large_videos,
    write_log,
)


# ── Config persistence ────────────────────────────────────────────────────────

_CONFIG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_compress_config.json")
_MAX_RECENT   = 10


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(config: dict):
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


def _add_recent_folder(config: dict, folder: str) -> dict:
    recent = config.get("recent_folders", [])
    folder = os.path.normpath(folder)
    if folder in recent:
        recent.remove(folder)
    recent.insert(0, folder)
    config["recent_folders"] = recent[:_MAX_RECENT]
    return config


# ── FFmpeg auto-detection ──────────────────────────────────────────────────────

def _find_ffmpeg_dir() -> str:
    """
    Return the directory containing ffmpeg.exe, or '' if already on PATH.
    Search order:
      1. Already on PATH (shutil.which)
      2. imageio-ffmpeg bundled binary
      3. Common Windows install locations
    """
    if shutil.which("ffmpeg"):
        return ""  # already accessible

    # imageio-ffmpeg bundles its own binaries
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return os.path.dirname(exe)
    except Exception:
        pass

    candidates = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.join(os.path.expanduser("~"), "ffmpeg", "bin"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "ffmpeg", "bin"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "ffmpeg", "bin"),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "ffmpeg.exe")):
            return path

    return ""  # not found — user must provide


def _inject_ffmpeg_path(ffmpeg_dir: str):
    """Prepend ffmpeg_dir to PATH so subprocess calls can find ffmpeg/ffprobe."""
    if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MIN_SIZE_GB = 0.0
DEFAULT_BITRATE     = "700k"
COL_CHECK   = "Select"
COL_NAME    = "File Name"
COL_ORIG    = "Original Size"
COL_BITRATE = "Current Bitrate"
COL_EST     = "Est. Output"
COL_SAVE    = "Savings"
COL_PCT     = "Save %"
COLUMNS     = (COL_CHECK, COL_NAME, COL_ORIG, COL_BITRATE, COL_EST, COL_SAVE, COL_PCT)

# Pre-computed column indices for fast access in hot loops
_IDX_CHECK   = COLUMNS.index(COL_CHECK)
_IDX_NAME    = COLUMNS.index(COL_NAME)
_IDX_ORIG    = COLUMNS.index(COL_ORIG)
_IDX_BITRATE = COLUMNS.index(COL_BITRATE)
_IDX_EST     = COLUMNS.index(COL_EST)
_IDX_SAVE    = COLUMNS.index(COL_SAVE)
_IDX_PCT     = COLUMNS.index(COL_PCT)

# Unit multipliers for size-string parsing (used by _sort_key)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "PB": 1024**5}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _probe_and_estimate(video: dict, target_kbps: int) -> dict:
    """Add _probe / _est_size keys to the video dict and return it."""
    probe = probe_video(video["path"])
    video["_probe"] = probe
    orig_kbps = probe.get("video_bitrate_kbps")
    duration  = probe.get("duration")
    eff_kbps  = min(orig_kbps, target_kbps) if orig_kbps else target_kbps
    video["_est_size"] = estimate_compressed_size(duration, eff_kbps) if duration else 0
    return video


def _compute_row_strings(orig: int, est: int) -> tuple[str, str, str]:
    """Return (est_str, save_str, pct_str) for a single video row."""
    save = orig - est if est > 0 else 0
    est_str  = format_size(est)  if est  > 0 else "N/A"
    save_str = format_size(save) if save > 0 else "N/A"
    pct_str  = f"{save / orig * 100:.0f}%" if orig > 0 and est > 0 else "N/A"
    return est_str, save_str, pct_str


# ── Stdout redirector ─────────────────────────────────────────────────────────

class _StdoutRedirector:
    """
    Captures sys.stdout writes and forwards them to a queue so the GUI
    log pane can display all terminal output (including ffmpeg progress).
    """
    def __init__(self, log_queue: queue.Queue, original):
        self._q        = log_queue
        self._original = original

    def write(self, text: str):
        if text:
            self._q.put(text)

    def flush(self):
        pass

    def __getattr__(self, name):
        return getattr(self._original, name)


# ── Main Application ───────────────────────────────────────────────────────────

class CompressApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Compressor")
        self.resizable(True, True)
        self.minsize(900, 500)

        # State
        self._videos: list[dict] = []          # full list from last scan
        self._check_vars: list[tk.BooleanVar] = []
        self._log_queue: queue.Queue = queue.Queue()
        self._job_running = False
        self._paused = False
        self._bitrate_refresh_id = None
        self._sort_reverse: dict[str, bool] = {}  # track sort direction per column

        self._config = _load_config()
        self._build_ui()
        self._restore_config()
        self._poll_log()

        # Redirect stdout so all print() / progress-bar writes appear in the log
        sys.stdout = _StdoutRedirector(self._log_queue, sys.stdout)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        # ── Top bar: folder + settings ─────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Folder:").pack(side="left")
        self._folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._folder_var, width=55).pack(side="left", padx=4)
        ttk.Button(top, text="Browse…", command=self._browse_folder).pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(top, text="Min size (GB):").pack(side="left")
        self._minsize_var = tk.StringVar(value=str(DEFAULT_MIN_SIZE_GB))
        ttk.Entry(top, textvariable=self._minsize_var, width=6).pack(side="left", padx=4)

        ttk.Label(top, text="Bitrate:").pack(side="left")
        self._bitrate_var = tk.StringVar(value=DEFAULT_BITRATE)
        ttk.Entry(top, textvariable=self._bitrate_var, width=7).pack(side="left", padx=4)
        self._bitrate_var.trace_add("write", self._on_bitrate_changed)

        ttk.Button(top, text="Scan", command=self._start_scan).pack(side="left", padx=8)

        # ── FFmpeg path row ────────────────────────────────────────────────────
        ff_row = ttk.Frame(self)
        ff_row.pack(fill="x", padx=6, pady=(0, 2))

        ttk.Label(ff_row, text="FFmpeg bin:").pack(side="left")
        self._ffmpeg_var = tk.StringVar(value=_find_ffmpeg_dir())
        self._ffmpeg_entry = ttk.Entry(ff_row, textvariable=self._ffmpeg_var, width=60)
        self._ffmpeg_entry.pack(side="left", padx=4)
        ttk.Button(ff_row, text="Browse…", command=self._browse_ffmpeg).pack(side="left")

        if not self._ffmpeg_var.get() and not shutil.which("ffmpeg"):
            self._ffmpeg_entry.configure(style="Error.TEntry")
            ttk.Label(
                ff_row, text="⚠ ffmpeg not found — browse to its bin folder",
                foreground="red",
            ).pack(side="left", padx=6)

        # ── Paned window (table + log) ────────────────────────────────────────
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True, padx=6, pady=2)

        # ── Table ──────────────────────────────────────────────────────────────
        tbl_frame = ttk.Frame(paned)

        vsb = ttk.Scrollbar(tbl_frame, orient="vertical")
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal")
        self._tree = ttk.Treeview(
            tbl_frame,
            columns=COLUMNS,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)

        col_widths = {
            COL_CHECK:   55,
            COL_NAME:    320,
            COL_ORIG:    110,
            COL_BITRATE: 120,
            COL_EST:     110,
            COL_SAVE:    100,
            COL_PCT:     70,
        }
        for col in COLUMNS:
            self._tree.heading(col, text=col, command=lambda c=col: self._sort_by(c))
            self._tree.column(col, width=col_widths[col], anchor="center" if col != COL_NAME else "w")

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<Button-1>", self._on_row_click)

        paned.add(tbl_frame, weight=3)

        # ── Bottom bar: select-all + compress button + status ─────────────────
        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=6, pady=4)

        ttk.Button(bot, text="Select All",   command=self._select_all).pack(side="left", padx=2)
        ttk.Button(bot, text="Deselect All", command=self._deselect_all).pack(side="left", padx=2)

        self._compress_btn = ttk.Button(
            bot, text="Compress Selected", command=self._start_compress
        )
        self._compress_btn.pack(side="right", padx=6)

        self._pause_btn = ttk.Button(
            bot, text="⏸ Pause", command=self._toggle_pause
        )
        self._pause_btn.pack(side="right", padx=2)
        self._pause_btn.state(["disabled"])

        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(bot, textvariable=self._status_var, anchor="w").pack(
            side="left", fill="x", expand=True, padx=8
        )

        # ── Summary bar ────────────────────────────────────────────────────────
        summary = ttk.Frame(self)
        summary.pack(fill="x", padx=6, pady=(0, 2))

        self._summary_var = tk.StringVar(value="")
        ttk.Label(summary, textvariable=self._summary_var, anchor="w",
                  font=("TkDefaultFont", 9, "bold")).pack(fill="x")

        # ── Log pane ───────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(paned, text="Log")

        self._log_text = tk.Text(log_frame, height=7, state="disabled", wrap="none")
        log_sb = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)

        paned.add(log_frame, weight=1)

    # ── Browse ─────────────────────────────────────────────────────────────────

    def _restore_config(self):
        """Populate UI fields from the saved config."""
        c = self._config
        if c.get("recent_folders"):
            self._folder_var.set(c["recent_folders"][0])
        if c.get("ffmpeg_dir"):
            self._ffmpeg_var.set(c["ffmpeg_dir"])
            _inject_ffmpeg_path(c["ffmpeg_dir"])
        if c.get("bitrate"):
            self._bitrate_var.set(c["bitrate"])
        if c.get("min_size_gb") is not None:
            self._minsize_var.set(str(c["min_size_gb"]))

    def _persist_config(self):
        """Save current UI settings to the config file."""
        folder = self._folder_var.get().strip()
        if folder:
            self._config = _add_recent_folder(self._config, folder)
        ffmpeg_dir = self._ffmpeg_var.get().strip()
        if ffmpeg_dir:
            self._config["ffmpeg_dir"] = ffmpeg_dir
        self._config["bitrate"] = self._bitrate_var.get().strip()
        try:
            self._config["min_size_gb"] = float(self._minsize_var.get())
        except ValueError:
            pass
        _save_config(self._config)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select folder to scan")
        if path:
            self._folder_var.set(path)
            self._persist_config()

    def _browse_ffmpeg(self):
        path = filedialog.askdirectory(title="Select FFmpeg bin folder (containing ffmpeg.exe)")
        if path:
            self._ffmpeg_var.set(path)
            _inject_ffmpeg_path(path)
            self._persist_config()

    def _ensure_ffmpeg(self) -> bool:
        """Inject ffmpeg dir into PATH and return True if ffmpeg is now accessible."""
        _inject_ffmpeg_path(self._ffmpeg_var.get().strip())
        if shutil.which("ffmpeg"):
            return True
        messagebox.showerror(
            "FFmpeg not found",
            "Cannot locate ffmpeg.exe.\n\n"
            "Please use the 'FFmpeg bin: Browse…' button to point to the folder "
            "containing ffmpeg.exe and ffprobe.exe.",
        )
        return False

    # ── Scan ───────────────────────────────────────────────────────────────────

    def _start_scan(self):
        if not self._ensure_ffmpeg():
            return
        folder = self._folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder first.")
            return
        try:
            min_gb = float(self._minsize_var.get())
        except ValueError:
            messagebox.showerror("Error", "Min size must be a number.")
            return

        self._set_status("Scanning…")
        self._compress_btn.state(["disabled"])
        self._clear_table()
        self._persist_config()
        threading.Thread(target=self._scan_worker, args=(folder, min_gb), daemon=True).start()

    def _scan_worker(self, folder: str, min_gb: float):
        try:
            min_bytes   = int(min_gb * (1 << 30))
            target_kbps = _parse_bitrate_to_kbps(self._bitrate_var.get())
            videos = find_large_videos(folder, min_bytes)

            size_filter = f" ≥ {min_gb} GB" if min_gb > 0 else ""
            self._log(f"Found {len(videos)} video(s){size_filter} in {folder}")

            # Probe videos in parallel (ffprobe is I/O-bound)
            probed = [None] * len(videos)
            max_workers = min(4, len(videos)) or 1

            def _probe_one(idx_video):
                idx, v = idx_video
                result = _probe_and_estimate(v, target_kbps)
                probe  = result.get("_probe", {})
                self._log(f"  Probed {idx + 1}/{len(videos)}: "
                          f"{os.path.basename(v['path'])}  "
                          f"duration={probe.get('duration')}s  "
                          f"bitrate={probe.get('video_bitrate_kbps')}kbps  "
                          f"est={format_size(result.get('_est_size', 0))}")
                return idx, result

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for idx, result in pool.map(_probe_one, enumerate(videos)):
                    probed[idx] = result

            self.after(0, self._populate_table, probed)
        except Exception as exc:
            self.after(0, self._set_status, f"Scan error: {exc}")
            self._log(f"[ERROR] {exc}")

    # ── Table population ───────────────────────────────────────────────────────

    def _populate_table(self, videos: list[dict]):
        self._check_vars = []
        self._clear_table()
        self._videos = videos  # must be set AFTER _clear_table resets it

        for idx, v in enumerate(videos):
            orig      = v["size"]
            est       = v.get("_est_size", 0)
            orig_kbps = v.get("_probe", {}).get("video_bitrate_kbps")
            bitrate_str = f"{orig_kbps} kbps" if orig_kbps else "N/A"
            est_str, save_str, pct_str = _compute_row_strings(orig, est)

            var = tk.BooleanVar(value=True)
            self._check_vars.append(var)

            self._tree.insert(
                "", "end",
                iid=str(idx),
                values=(
                    "☑",
                    os.path.basename(v["path"]),
                    format_size(orig),
                    bitrate_str,
                    est_str,
                    save_str,
                    pct_str,
                ),
                tags=("checked",),
            )

        self._tree.tag_configure("checked",   background="#e8f5e9")
        self._tree.tag_configure("unchecked", background="#ffffff")

        self._compress_btn.state(["!disabled"])
        selected = self._update_summary()
        self._set_status(
            f"Found {len(videos)} file(s). {selected} selected. "
            "Uncheck any you don't want, then click Compress Selected."
        )

    def _on_bitrate_changed(self, *_):
        """Debounce bitrate edits — refresh estimates 400 ms after last keystroke."""
        if self._bitrate_refresh_id:
            self.after_cancel(self._bitrate_refresh_id)
        self._bitrate_refresh_id = self.after(400, self._refresh_estimates)

    def _refresh_estimates(self):
        """Recompute estimated sizes for all scanned videos using the current bitrate."""
        if not self._videos:
            return
        try:
            target_kbps = _parse_bitrate_to_kbps(self._bitrate_var.get())
        except (ValueError, ZeroDivisionError):
            return  # invalid bitrate string — wait for user to finish typing

        children = self._tree.get_children()
        for row_id in children:
            v = self._videos[int(row_id)]
            probe     = v.get("_probe", {})
            orig_kbps = probe.get("video_bitrate_kbps")
            duration  = probe.get("duration")

            eff_kbps = min(orig_kbps, target_kbps) if orig_kbps else target_kbps
            est = estimate_compressed_size(duration, eff_kbps) if duration else 0
            v["_est_size"] = est

            est_str, save_str, pct_str = _compute_row_strings(v["size"], est)

            vals = list(self._tree.item(row_id, "values"))
            vals[_IDX_EST]  = est_str
            vals[_IDX_SAVE] = save_str
            vals[_IDX_PCT]  = pct_str
            self._tree.item(row_id, values=vals)

        self._update_summary()

    def _clear_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._check_vars = []
        self._videos = []

    # ── Aggregate summary ────────────────────────────────────────────────────

    def _update_summary(self):
        """Recompute and display aggregate totals for selected files.
        Also updates the status bar selection count. Returns the selected count."""
        total = len(self._check_vars)
        if not self._videos:
            self._summary_var.set("")
            return 0

        total_orig = 0
        total_est = 0
        count = 0
        for v, var in zip(self._videos, self._check_vars):
            if var.get():
                count += 1
                total_orig += v["size"]
                est = v.get("_est_size", 0)
                total_est += est if est > 0 else v["size"]

        self._set_status(f"{count} / {total} file(s) selected.")

        if count == 0:
            self._summary_var.set("No files selected.")
            return 0

        total_save = total_orig - total_est
        pct = (total_save / total_orig * 100) if total_orig > 0 else 0
        self._summary_var.set(
            f"Selected: {count} file(s)  │  "
            f"Original: {format_size(total_orig)}  │  "
            f"Estimated: {format_size(total_est)}  │  "
            f"Savings: {format_size(total_save)} ({pct:.1f}%)"
        )
        return count

    # ── Row click → toggle checkbox ────────────────────────────────────────────

    def _on_row_click(self, event):
        row_id = self._tree.identify_row(event.y)
        if not row_id:
            return
        idx = int(row_id)
        if idx >= len(self._check_vars):
            return

        var = self._check_vars[idx]
        var.set(not var.get())
        checked = var.get()

        vals = list(self._tree.item(row_id, "values"))
        vals[0] = "☑" if checked else "☐"
        self._tree.item(row_id, values=vals, tags=("checked" if checked else "unchecked",))

        self._update_summary()

    # ── Select / Deselect All ──────────────────────────────────────────────────

    def _select_all(self):   self._set_all(True)
    def _deselect_all(self): self._set_all(False)

    def _set_all(self, state: bool):
        children = self._tree.get_children()
        for row_id in children:
            idx = int(row_id)
            var = self._check_vars[idx]
            var.set(state)
            vals = list(self._tree.item(row_id, "values"))
            vals[0] = "☑" if state else "☐"
            self._tree.item(row_id, values=vals, tags=("checked" if state else "unchecked",))
        self._update_summary()

    # ── Sort ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sort_key(col: str, value: str):
        """Return a sort key that handles sizes, bitrates, and percentages numerically."""
        if col in (COL_ORIG, COL_EST, COL_SAVE):
            parts = value.split()
            try:
                return float(parts[0]) * _SIZE_UNITS.get(parts[1], 1)
            except (IndexError, ValueError):
                return -1  # N/A sorts first
        if col == COL_BITRATE:
            # "1234 kbps" → 1234
            try:
                return float(value.split()[0])
            except (IndexError, ValueError):
                return -1
        if col == COL_PCT:
            # "42%" → 42
            try:
                return float(value.rstrip("%"))
            except ValueError:
                return -1
        return value.lower()

    def _sort_by(self, col):
        """Sort tree rows by *col*, toggling ascending / descending on repeat click."""
        reverse = self._sort_reverse.get(col, False)
        items = [
            (self._sort_key(col, self._tree.set(k, col)), k)
            for k in self._tree.get_children("")
        ]
        items.sort(reverse=reverse)
        for rank, (_, k) in enumerate(items):
            self._tree.move(k, "", rank)
        # Toggle direction for next click
        self._sort_reverse[col] = not reverse

    # ── Pause / Resume ─────────────────────────────────────────────────────────

    def _toggle_pause(self):
        if self._paused:
            self._paused = False
            self._pause_btn.configure(text="⏸ Pause")
            _gui_resume.set()
        else:
            self._paused = True
            self._pause_btn.configure(text="▶ Resume")
            _gui_pause.set()

    # ── Compress ───────────────────────────────────────────────────────────────

    def _start_compress(self):
        if not self._ensure_ffmpeg():
            return
        if self._job_running:
            messagebox.showinfo("Running", "A compression job is already running.")
            return

        selected = [
            v for v, var in zip(self._videos, self._check_vars) if var.get()
        ]
        if not selected:
            messagebox.showwarning("Nothing selected", "Please select at least one file.")
            return

        bitrate = self._bitrate_var.get().strip() or DEFAULT_BITRATE
        root    = self._folder_var.get().strip()

        if not messagebox.askyesno(
            "Confirm",
            f"Compress {len(selected)} file(s) at {bitrate}?\n\n"
            "Originals will be moved to _originals_to_delete/",
        ):
            return

        self._compress_btn.state(["disabled"])
        self._pause_btn.state(["!disabled"])
        self._paused = False
        _gui_pause.clear()
        _gui_resume.clear()
        self._job_running = True
        threading.Thread(
            target=self._compress_worker,
            args=(selected, bitrate, root),
            daemon=True,
        ).start()

    def _compress_worker(self, videos: list[dict], bitrate: str, root: str):
        total   = len(videos)
        records = []
        for i, v in enumerate(videos, 1):
            self._log(f"\n[{i}/{total}] {os.path.basename(v['path'])}")
            self.after(0, self._set_status, f"Compressing {i}/{total}: {os.path.basename(v['path'])}")
            rec = _process_one(i, total, v, bitrate, root)
            records.append(rec)
            status_icon = {"ok": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(rec["status"], "?")
            self._log(f"  [{status_icon}] {os.path.basename(v['path'])}")

        # Write log CSV
        log_path = os.path.join(root, "compression_log.csv")
        write_log(log_path, records)

        ok   = sum(1 for r in records if r["status"] == "ok")
        fail = sum(1 for r in records if r["status"] == "failed")
        skip = sum(1 for r in records if r["status"] == "skipped")
        self._log(f"\nDone. OK={ok}  Skipped={skip}  Failed={fail}  Log={log_path}")
        self.after(0, self._set_status,
                   f"Done. {ok} succeeded, {skip} skipped, {fail} failed. Log: {log_path}")
        self.after(0, self._compress_btn.state, ["!disabled"])
        self.after(0, self._pause_btn.state, ["disabled"])
        self.after(0, self._pause_btn.configure, {"text": "⏸ Pause"})
        self._job_running = False
        self._paused = False

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_queue.put(msg + "\n")

    # Name of the tkinter mark that anchors the current progress line
    _PROGRESS_MARK = "progress_line_start"

    def _poll_log(self):
        try:
            while True:
                text = self._log_queue.get_nowait()
                self._log_text.configure(state="normal")

                if "\r" in text:
                    # Progress bar: \r means "overwrite the current line".
                    # Use a named mark so we only ever replace the progress
                    # line itself — never the log lines above it.
                    content = text.split("\r")[-1]  # keep last segment after \r

                    if self._PROGRESS_MARK in self._log_text.mark_names():
                        # Already have a progress line — replace it in-place.
                        self._log_text.delete(self._PROGRESS_MARK, "end-1c")
                    else:
                        # First progress write — ensure we start on a fresh line.
                        last_char = self._log_text.get("end-2c", "end-1c")
                        if last_char != "\n":
                            self._log_text.insert("end", "\n")
                        # Plant the mark with left-gravity so it never drifts
                        # forward as we insert text after it.
                        self._log_text.mark_set(self._PROGRESS_MARK, "end-1c")
                        self._log_text.mark_gravity(self._PROGRESS_MARK, "left")

                    if content:
                        self._log_text.insert("end", content)

                else:
                    # Regular text — clear the progress mark first so the
                    # next \r starts a fresh line rather than overwriting this.
                    if self._PROGRESS_MARK in self._log_text.mark_names():
                        last_char = self._log_text.get("end-2c", "end-1c")
                        if last_char != "\n" and not text.startswith("\n"):
                            self._log_text.insert("end", "\n")
                        self._log_text.mark_unset(self._PROGRESS_MARK)
                    self._log_text.insert("end", text)

                self._log_text.see("end")
                self._log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status_var.set(msg)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CompressApp()
    app.mainloop()
