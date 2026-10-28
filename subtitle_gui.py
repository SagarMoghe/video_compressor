"""Background subtitle-generation window for the compression front end."""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from generate_subtitles import (
    DEVICES, MODELS, SubtitleCancelled, generate_english_srt, load_model,
    subtitle_output_path,
)


class SubtitleWindow(tk.Toplevel):
    def __init__(
        self, parent: tk.Tk, videos: list[str],
        on_generated: Callable[[str, str], None],
        log: Callable[[str], None],
        is_compressing: Callable[[], bool],
    ):
        super().__init__(parent)
        self.title("Japanese audio to English subtitles")
        self.geometry("760x470")
        self.running = False
        self._videos = list(dict.fromkeys(videos))
        self._on_generated = on_generated
        self._log = log
        self._is_compressing = is_compressing
        self._cancel = threading.Event()
        self._events = queue.Queue()
        self._poll_id = None
        self.protocol("WM_DELETE_WINDOW", self._close)

        ttk.Label(
            self, text="Translate Japanese speech to English with faster-whisper.",
        ).pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(
            self, text="Saves <video>.en.srt beside each video. Existing files are never overwritten.",
        ).pack(anchor="w", padx=10)
        self._files = tk.Listbox(self, height=8)
        self._files.pack(fill="both", expand=True, padx=10, pady=8)
        self._refresh_files()
        self._browse_btn = ttk.Button(self, text="Choose Videos...", command=self._browse)
        self._browse_btn.pack(anchor="w", padx=10)

        settings = ttk.Frame(self)
        settings.pack(fill="x", padx=10, pady=8)
        ttk.Label(settings, text="Model:").pack(side="left")
        self._model = tk.StringVar(value="small")
        self._model_combo = ttk.Combobox(
            settings, textvariable=self._model, values=MODELS,
            state="readonly", width=12,
        )
        self._model_combo.pack(side="left", padx=5)
        ttk.Label(settings, text="Device:").pack(side="left", padx=(10, 0))
        self._device = tk.StringVar(value="cuda")
        self._device_combo = ttk.Combobox(
            settings, textvariable=self._device, values=DEVICES,
            state="readonly", width=8,
        )
        self._device_combo.pack(side="left", padx=5)
        self._batched = tk.BooleanVar(value=True)
        self._batched_check = ttk.Checkbutton(
            settings, text="Batched CUDA (8 chunks)", variable=self._batched,
        )
        self._batched_check.pack(side="left", padx=10)
        ttk.Label(
            self,
            text="CPU: INT8. CUDA: FP16. Uncheck batched CUDA for sequential quality fallback.",
        ).pack(anchor="w", padx=10)
        self._status = tk.StringVar(value="Choose videos or use the checked videos from the main window.")
        ttk.Label(self, textvariable=self._status, wraplength=730).pack(
            anchor="w", padx=10, pady=8,
        )
        self._progress_label = tk.StringVar(value="Current video: not started")
        ttk.Label(self, textvariable=self._progress_label).pack(anchor="w", padx=10)
        self._progress_bar = ttk.Progressbar(self, maximum=100, mode="determinate")
        self._progress_bar.pack(fill="x", padx=10, pady=(4, 10))
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=10, pady=(0, 10))
        self._generate_btn = ttk.Button(buttons, text="Generate English SRT", command=self._start)
        self._generate_btn.pack(side="left")
        self._cancel_btn = ttk.Button(
            buttons, text="Cancel", command=self._request_cancel, state="disabled",
        )
        self._cancel_btn.pack(side="left", padx=8)

    def _refresh_files(self):
        self._files.delete(0, "end")
        for path in self._videos:
            self._files.insert("end", path)

    def _browse(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Select videos with Japanese audio",
            filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts"),
                       ("All files", "*.*")],
        )
        if paths:
            self._videos = list(dict.fromkeys(paths))
            self._refresh_files()

    def _start(self):
        if self.running:
            return
        if self._is_compressing():
            messagebox.showinfo(
                "Running", "Wait for compression to finish before generating subtitles.",
                parent=self,
            )
            return
        if not self._videos:
            messagebox.showwarning("No videos", "Choose at least one video.", parent=self)
            return
        for path in self._videos:
            if not os.path.isfile(path):
                messagebox.showerror("Missing video", path, parent=self)
                return
            if os.path.exists(subtitle_output_path(path)):
                messagebox.showerror(
                    "Subtitle already exists",
                    f"Move or rename this file before generating again:\n{subtitle_output_path(path)}",
                    parent=self,
                )
                return
        outputs = [os.path.normcase(os.path.abspath(subtitle_output_path(p))) for p in self._videos]
        if len(set(outputs)) != len(outputs):
            messagebox.showerror(
                "Conflicting output paths",
                "Some selected videos have the same base name and would write the same SRT. "
                "Process them separately, renaming the output between runs.",
                parent=self,
            )
            return
        self._cancel.clear()
        self._set_running(True)
        self._status.set("Loading model (first use downloads weights)...")
        self._update_progress(None, "Loading model...")
        threading.Thread(
            target=self._worker,
            args=(tuple(self._videos), self._model.get(), self._device.get(), self._batched.get()),
            daemon=True,
        ).start()
        self._poll_id = self.after(100, self._poll)

    def _set_running(self, running: bool):
        self.running = running
        state = ["disabled" if running else "!disabled"]
        for widget in (
            self._generate_btn, self._browse_btn, self._model_combo,
            self._device_combo, self._batched_check,
        ):
            widget.state(state)
        self._cancel_btn.state(["!disabled" if running else "disabled"])

    def _worker(self, videos: tuple[str, ...], model_size: str, device: str, batched: bool = True):
        succeeded = failed = 0
        try:
            model = load_model(model_size, device)
            for index, path in enumerate(videos, 1):
                if self._cancel.is_set():
                    raise SubtitleCancelled()
                self._events.put(("translation", (
                    f"Preparing video {index}/{len(videos)}: {path}", None,
                )))
                try:
                    output = generate_english_srt(
                        path, model, self._cancel,
                        lambda text, fraction: self._events.put(("translation", (text, fraction))),
                        batch_size=8 if device == "cuda" and batched else 1,
                    )
                except SubtitleCancelled:
                    raise
                except Exception as exc:
                    failed += 1
                    self._events.put(("error", f"{path}: {exc}"))
                else:
                    succeeded += 1
                    self._events.put(("generated", (path, output)))
        except SubtitleCancelled:
            pass  # The done event reports cancellation and preserves completed files.
        except Exception as exc:
            failed += 1
            self._events.put(("error", str(exc)))
        finally:
            self._events.put(("done", (succeeded, failed, self._cancel.is_set())))

    def _poll(self):
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == "translation":
                    text, fraction = value
                    if not self._cancel.is_set():
                        self._status.set(text)
                        self._update_progress(fraction)
                elif kind == "generated":
                    video, output = value
                    self._update_progress(1.0)
                    self._log(f"[SUBTITLES] Created: {output}")
                    self._on_generated(video, output)
                elif kind == "error":
                    self._stop_progress()
                    self._progress_label.set("Current video: failed")
                    self._log(f"[SUBTITLES ERROR] {value}")
                    messagebox.showerror("Subtitle generation failed", value, parent=self)
                elif kind == "done":
                    succeeded, failed, cancelled = value
                    self._set_running(False)
                    self._stop_progress()
                    if cancelled:
                        self._progress_label.set("Current video: cancelled")
                    elif failed:
                        self._progress_label.set("Finished with errors; see log")
                    self._status.set(
                        f"{'Cancelled' if cancelled else 'Done'}. "
                        f"{succeeded} created, {failed} failed."
                    )
                    self._poll_id = None
                    return
        except queue.Empty:
            pass
        self._poll_id = self.after(100, self._poll)

    def _stop_progress(self):
        value = self._progress_bar["value"]
        self._progress_bar.stop()
        self._progress_bar.configure(value=value)

    def _update_progress(self, fraction: float | None, activity: str = "Preparing audio..."):
        if fraction is None:
            if str(self._progress_bar["mode"]) != "indeterminate":
                self._progress_bar.configure(mode="indeterminate", value=0)
                self._progress_bar.start(15)
            self._progress_label.set(activity)
        else:
            self._progress_bar.stop()
            percent = max(0.0, min(fraction, 1.0)) * 100
            self._progress_bar.configure(mode="determinate", value=percent)
            self._progress_label.set(f"Current video: {percent:.0f}%")

    def _request_cancel(self):
        self._cancel.set()
        self._stop_progress()
        self._progress_label.set("Current video: cancelling...")
        self._cancel_btn.state(["disabled"])
        self._status.set("Cancelling after the current model load or inference step finishes...")

    def _close(self):
        if self.running:
            self._request_cancel()
            return
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.destroy()
