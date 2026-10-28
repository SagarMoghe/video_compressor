"""Translate Japanese video audio to English SubRip subtitles with faster-whisper."""

from __future__ import annotations

import math
import os
import threading
from importlib.metadata import PackageNotFoundError, distribution
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
DEVICES = ("cpu", "cuda")
_DLL_DIRECTORIES: dict[str, object] = {}


def _configure_cuda_libraries():
    """Keep installed NVIDIA DLL directories registered for lazy runtime loads."""
    if os.name != "nt":
        return
    for package in ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12"):
        try:
            installed = distribution(package)
        except PackageNotFoundError:
            continue  # System-installed CUDA libraries can still be used.
        directories = {
            os.path.dirname(os.path.abspath(installed.locate_file(file)))
            for file in installed.files or ()
            if str(file).lower().endswith(".dll")
        }
        for directory in sorted(directories):
            if directory not in _DLL_DIRECTORIES:
                _DLL_DIRECTORIES[directory] = os.add_dll_directory(directory)
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


class SubtitleCancelled(Exception):
    """Raised when subtitle generation is cancelled."""


def _check_cancel(cancel: threading.Event):
    if cancel.is_set():
        raise SubtitleCancelled("Subtitle generation cancelled.")


def subtitle_output_path(video_path: str) -> str:
    return os.path.splitext(video_path)[0] + ".en.srt"


def load_model(model_size: str, device: str) -> WhisperModel:
    if model_size not in MODELS or device not in DEVICES:
        raise ValueError("Choose a supported multilingual model and CPU or CUDA device.")
    if device == "cuda":
        _configure_cuda_libraries()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import faster-whisper. Install the optional dependency in "
            "the Python environment running this app:\n"
            "python -m pip install -r requirements-subtitles.txt\n"
            f"Import error: {exc}"
        ) from exc
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", f"faster-whisper-{model_size}",
    )
    options = {"local_files_only": True} if os.path.isdir(local_path) else {}
    return WhisperModel(
        local_path if options else model_size, device=device,
        compute_type="int8" if device == "cpu" else "float16",
        **options,
    )


def _srt_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid subtitle timestamp: {seconds}")
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def generate_english_srt(
    video_path: str,
    model: WhisperModel,
    cancel: threading.Event,
    progress: Callable[[str, float | None], None],
    batch_size: int = 1,
) -> str:
    """Write a new .en.srt beside the video, removing partial output on failure.

    PyAV decodes the video's first audio stream directly; no FFmpeg executable
    or intermediate WAV is needed. The supplied model can be reused for a batch.
    """
    _check_cancel(cancel)
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    output_path = subtitle_output_path(video_path)
    # Exclusive creation protects existing subtitles, including concurrent jobs.
    with open(output_path, "x", encoding="utf-8", newline="\n") as output:
        try:
            if batch_size > 1:
                from faster_whisper import BatchedInferencePipeline

                transcriber = BatchedInferencePipeline(model=model)
                batch_options = {
                    "batch_size": batch_size,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 6,
                }
            else:
                transcriber = model
                batch_options = {
                    "condition_on_previous_text": False,
                    "word_timestamps": True,
                    "hallucination_silence_threshold": 2.0,
                }
            segments, info = transcriber.transcribe(
                video_path,
                language="ja",
                task="translate",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                without_timestamps=False,
                **batch_options,
            )
            duration = info.duration
            known_duration = math.isfinite(duration) and duration > 0
            fraction = 0.0
            progress("Audio prepared; translating...", 0.0 if known_duration else None)
            count = 0
            for segment in segments:
                _check_cancel(cancel)
                text = " ".join(segment.text.split())
                if not text:
                    continue
                start = _srt_timestamp(segment.start)
                end = _srt_timestamp(segment.end)
                if round(segment.end * 1000) <= round(segment.start * 1000):
                    raise ValueError("Subtitle end must be later than its start.")
                count += 1
                output.write(f"{count}\n{start} --> {end}\n{text}\n\n")
                if known_duration:
                    fraction = max(fraction, min(segment.end / duration, 0.99))
                progress(
                    f"{os.path.basename(video_path)}: translated through {end}",
                    fraction if known_duration else None,
                )
            _check_cancel(cancel)
            if not count:
                raise ValueError("No speech was found; no subtitle file was created.")
            output.flush()
        except BaseException:
            output.close()
            os.remove(output_path)
            raise
    return output_path
