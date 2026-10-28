# Video compressor and English subtitles

Run `python gui_compress.py` or double-click `launch_video_compressor.pyw`.
Compression and subtitle previews require FFmpeg and ffprobe; select their bin
directory in the GUI if they are not on PATH.

## Compression

Choose a folder, scan, select videos, and click **Compress Selected**.
The GUI supports size estimates, encoder/bitrate settings, pause/resume/stop,
per-video SRT assignment, subtitle offsets, and preview clips.

Compression caps the video bitrate at the lower of the original and requested
rates. Low-bitrate MP4s without subtitle burn-in are skipped; other low-bitrate
videos first try stream-copy remuxing to MP4. Encoding retains the fallback
encoder chain and progressive 80% bitrate retries. Failed outputs are removed
and output media is validated before success is reported. Larger outputs are
discarded for plain MP4 compression but retained for conversion or burn-in.

Successful compression moves originals into `_originals_to_delete` under the
selected folder and writes `compression_log.csv`. Review outputs before manually
deleting originals. Failures and skips leave originals in place.

The existing command-line entry points remain available:
`python compress_video.py --help` and `python scan_and_compress.py --help`.

## Japanese audio to English SRT

Install dependencies into the same Python environment that runs the GUI:

```powershell
python -m pip install -r requirements-subtitles.txt
# Windows CUDA users:
python -m pip install -r requirements-cuda.txt
python gui_compress.py
```

PyAV is constrained below version 19 for faster-whisper compatibility.
Windows CUDA packages provide CUDA 12 cuBLAS and cuDNN 9; the app registers their
DLL directories automatically. A compatible NVIDIA driver is still required.
Restart the GUI after installing dependencies.

Click **Generate English SRT...** to use checked videos or choose videos directly.
Select a multilingual model and device: CUDA uses FP16, CPU uses INT8.
`small` and CUDA are the defaults. To reuse local weights offline, select
`large-v3` with its complete model in `models\faster-whisper-large-v3`.
Other models use `models\faster-whisper-<model-name>` if present, otherwise the
Hugging Face cache/download. Local model loading disables downloads; keep the
tokenizer, configuration and vocabulary files alongside `model.bin`.
`models` is excluded from Git.

The first audio stream must contain Japanese. Processing is local, with
`language="ja"` and `task="translate"`. PyAV decodes audio without an FFmpeg
executable. Each video produces a UTF-8 `<video-name>.en.srt` beside it.
Existing SRTs are never overwritten; failed/cancelled partial files are removed.
Completed files are kept and assigned to matching GUI rows at zero offset.

**Batched CUDA (8 chunks)** processes audio chunks in parallel using more VRAM.
Uncheck it for sequential inference, which is also used on CPU.
Both modes generate timestamps and use 500 ms VAD silence splitting.
Batched decoding uses a 1.1 repetition penalty and blocks six-token repeats.
Sequential decoding disables previous-text conditioning and enables word
timestamps with a 2-second hallucination-silence threshold. These settings are
not a guarantee against hallucinations, and may suppress quiet/repeated dialogue.
Review the translation and timing before burning subtitles into a video;
duplicate cues are not automatically deleted.

Progress shows activity during loading/preparation and percentage completion as
segments arrive. Each video reaches 100% only after its SRT is saved. Cancellation
waits for the current model load or inference step; it is not immediate.
Compression and translation cannot run simultaneously.
