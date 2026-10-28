import os
from pathlib import Path
import sys
import tempfile
import threading
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from generate_subtitles import (
    SubtitleCancelled, _configure_cuda_libraries, _srt_timestamp, generate_english_srt, load_model,
    subtitle_output_path,
)
from subtitle_gui import SubtitleWindow


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / "episode.mkv"
        self.video.touch()
        self.output = Path(subtitle_output_path(str(self.video)))
        self.cancel = threading.Event()
        self.progress = Mock()

    def generate(self, segments):
        model = Mock()
        model.transcribe.return_value = (iter(segments), SimpleNamespace(duration=4000.0))
        output = generate_english_srt(str(self.video), model, self.cancel, self.progress)
        return model, Path(output).read_text(encoding="utf-8")

    def test_japanese_is_translated_to_english_srt(self):
        model, text = self.generate([
            SimpleNamespace(start=0.125, end=2.9996, text=" Hello\n world! "),
            SimpleNamespace(start=3.0, end=4.0, text="   "),
            SimpleNamespace(start=3601.5, end=3602.75, text="Goodbye."),
        ])
        model.transcribe.assert_called_once_with(
            str(self.video), language="ja", task="translate", beam_size=5, vad_filter=True,
            without_timestamps=False,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False, word_timestamps=True,
            hallucination_silence_threshold=2.0,
        )
        self.assertEqual(
            text,
            "1\n00:00:00,125 --> 00:00:03,000\nHello world!\n\n"
            "2\n01:00:01,500 --> 01:00:02,750\nGoodbye.\n\n",
        )
        self.assertEqual(self.progress.call_count, 3)
        self.assertEqual(self.progress.call_args_list[0].args[1], 0.0)
        self.assertAlmostEqual(self.progress.call_args_list[1].args[1], 2.9996 / 4000)
        self.assertAlmostEqual(self.progress.call_args_list[2].args[1], 3602.75 / 4000)

    def test_progress_does_not_reach_completion_before_file_is_saved(self):
        self.generate([SimpleNamespace(start=0.0, end=4000.0, text="Hello")])
        self.assertEqual(self.progress.call_args.args[1], 0.99)

    def test_batched_translation_keeps_english_task_and_original_timestamps(self):
        model = Mock()
        pipeline = Mock()
        pipeline.transcribe.return_value = (
            iter([SimpleNamespace(start=60.5, end=62.0, text="Hello.")]),
            SimpleNamespace(duration=120.0),
        )
        with patch("faster_whisper.BatchedInferencePipeline", return_value=pipeline) as create:
            generate_english_srt(
                str(self.video), model, self.cancel, self.progress, batch_size=8,
            )
        create.assert_called_once_with(model=model)
        pipeline.transcribe.assert_called_once_with(
            str(self.video), language="ja", task="translate", beam_size=5,
            vad_filter=True, batch_size=8, without_timestamps=False,
            repetition_penalty=1.1, no_repeat_ngram_size=6,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        model.transcribe.assert_not_called()
        self.assertIn("00:01:00,500 --> 00:01:02,000", self.output.read_text(encoding="utf-8"))
        self.assertAlmostEqual(self.progress.call_args.args[1], 62 / 120)

    def test_batch_failure_is_reported_and_partial_srt_removed(self):
        pipeline = Mock()
        pipeline.transcribe.side_effect = RuntimeError("CUDA out of memory")
        with patch("faster_whisper.BatchedInferencePipeline", return_value=pipeline):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                generate_english_srt(
                    str(self.video), Mock(), self.cancel, self.progress, batch_size=8,
                )
        self.assertFalse(self.output.exists())

    def test_unknown_duration_reports_activity_instead_of_percentage(self):
        for duration in (0.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                model = Mock()
                model.transcribe.return_value = (
                    iter([SimpleNamespace(start=0.0, end=1.0, text="Hello")]),
                    SimpleNamespace(duration=duration),
                )
                progress = Mock()
                generate_english_srt(str(self.video), model, self.cancel, progress)
                self.assertTrue(all(call.args[1] is None for call in progress.call_args_list))
                self.output.unlink()

    def test_preserves_existing_subtitle(self):
        self.output.write_text("Existing translation", encoding="utf-8")
        model = Mock()
        with self.assertRaises(FileExistsError):
            generate_english_srt(str(self.video), model, self.cancel, self.progress)
        model.transcribe.assert_not_called()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "Existing translation")

    def test_no_speech_does_not_leave_empty_file(self):
        with self.assertRaisesRegex(ValueError, "No speech"):
            self.generate([])
        self.assertFalse(self.output.exists())

    def test_generator_failure_removes_partial_output(self):
        def segments():
            yield SimpleNamespace(start=0.0, end=1.0, text="Hello")
            raise RuntimeError("Decoder failure")
        with self.assertRaisesRegex(RuntimeError, "Decoder failure"):
            self.generate(segments())
        self.assertFalse(self.output.exists())

    def test_cancel_during_inference_removes_partial_output(self):
        def segments():
            yield SimpleNamespace(start=0.0, end=1.0, text="Hello")
            self.cancel.set()
            yield SimpleNamespace(start=1.0, end=2.0, text="Goodbye")
        with self.assertRaises(SubtitleCancelled):
            self.generate(segments())
        self.assertFalse(self.output.exists())

    def test_cancel_before_start_creates_nothing(self):
        self.cancel.set()
        with self.assertRaises(SubtitleCancelled):
            self.generate([])
        self.assertFalse(self.output.exists())

    def test_invalid_timing_removes_output(self):
        for start, end in [(float("nan"), 1), (-1, 1), (2, 1), (1, float("inf"))]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    self.generate([SimpleNamespace(start=start, end=end, text="Hello")])
                self.assertFalse(self.output.exists())

    def test_timestamp_carries_at_hour_boundary(self):
        self.assertEqual(_srt_timestamp(3599.9996), "01:00:00,000")

    def test_missing_video(self):
        self.video.unlink()
        with self.assertRaises(FileNotFoundError):
            self.generate([])
        self.assertFalse(self.output.exists())

    def test_model_device_and_precision(self):
        whisper = SimpleNamespace(WhisperModel=Mock())
        with patch.dict(sys.modules, {"faster_whisper": whisper}), patch(
            "generate_subtitles.os.path.isdir", return_value=False,
        ):
            load_model("small", "cpu")
            whisper.WhisperModel.assert_called_with("small", device="cpu", compute_type="int8")
            load_model("large-v3", "cuda")
            whisper.WhisperModel.assert_called_with(
                "large-v3", device="cuda", compute_type="float16",
            )

    @unittest.skipUnless(os.name == "nt", "Windows DLL registration")
    def test_cuda_dll_directories_are_registered_once_and_kept_alive(self):
        installed = Mock()
        installed.files = [Path("vendor") / "bin" / "library.dll"]
        installed.locate_file.side_effect = lambda file: Path(self.temp.name) / file
        directory = os.path.join(self.temp.name, "vendor", "bin")
        with patch("generate_subtitles.distribution", return_value=installed), patch(
            "generate_subtitles.os.add_dll_directory",
        ) as register, patch.dict("generate_subtitles._DLL_DIRECTORIES", {}, clear=True), patch.dict(
            os.environ, {"PATH": "original"},
        ):
            _configure_cuda_libraries()
            _configure_cuda_libraries()
            register.assert_called_once_with(directory)
            self.assertEqual(os.environ["PATH"], directory + os.pathsep + "original")

    def test_local_model_is_loaded_without_downloads(self):
        whisper = SimpleNamespace(WhisperModel=Mock())
        import generate_subtitles
        expected = os.path.join(
            os.path.dirname(os.path.abspath(generate_subtitles.__file__)),
            "models", "faster-whisper-large-v3",
        )
        with patch.dict(sys.modules, {"faster_whisper": whisper}), patch(
            "generate_subtitles.os.path.isdir", return_value=True,
        ):
            load_model("large-v3", "cuda")
        whisper.WhisperModel.assert_called_once_with(
            expected, device="cuda", compute_type="float16", local_files_only=True,
        )

    def test_english_only_and_turbo_models_are_not_offered(self):
        for name in ("small.en", "turbo", "distil-large-v3"):
            with self.assertRaises(ValueError):
                load_model(name, "cpu")

    def test_missing_optional_dependency_is_actionable(self):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-subtitles.txt"):
                load_model("small", "cpu")


class SubtitleWindowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.generated = Mock()
        self.log = Mock()
        self.compressing = Mock(return_value=False)
        self.window = SubtitleWindow(
            self.root, [], self.generated, self.log, self.compressing,
        )
        self.window.withdraw()

    def test_cuda_is_default_device(self):
        self.assertEqual(self.window._device.get(), "cuda")

    def test_running_state_locks_settings_and_restores_readonly_combos(self):
        self.window._set_running(True)
        for widget in (
            self.window._generate_btn, self.window._browse_btn,
            self.window._model_combo, self.window._device_combo, self.window._batched_check,
        ):
            self.assertTrue(widget.instate(["disabled"]))
        self.assertTrue(self.window._cancel_btn.instate(["!disabled"]))
        self.window._set_running(False)
        self.assertTrue(self.window._model_combo.instate(["readonly", "!disabled"]))
        self.assertTrue(self.window._device_combo.instate(["readonly", "!disabled"]))
        self.assertTrue(self.window._cancel_btn.instate(["disabled"]))

    def test_translation_progress_switches_from_activity_to_percentage(self):
        self.window._update_progress(None)
        self.assertEqual(str(self.window._progress_bar["mode"]), "indeterminate")
        self.window._events.put(("translation", ("Translated part of video", 0.42)))
        self.window._events.put(("done", (0, 0, False)))
        self.window._poll()
        self.assertEqual(str(self.window._progress_bar["mode"]), "determinate")
        self.assertEqual(float(self.window._progress_bar["value"]), 42.0)
        self.assertEqual(self.window._progress_label.get(), "Current video: 42%")

    def test_cancelled_progress_does_not_show_success(self):
        self.window._update_progress(0.35)
        self.window._request_cancel()
        self.window._events.put(("translation", ("Late progress", 0.9)))
        self.window._events.put(("done", (0, 0, True)))
        self.window._poll()
        self.assertEqual(float(self.window._progress_bar["value"]), 35.0)
        self.assertEqual(self.window._progress_label.get(), "Current video: cancelled")

    def test_background_result_is_assigned_on_ui_thread(self):
        self.window.running = True
        self.window._events.put(("generated", ("episode.mkv", "episode.en.srt")))
        self.window._events.put(("done", (1, 0, False)))
        self.window._poll()
        self.generated.assert_called_once_with("episode.mkv", "episode.en.srt")
        self.assertFalse(self.window.running)
        self.assertEqual(self.window._status.get(), "Done. 1 created, 0 failed.")
        self.assertEqual(float(self.window._progress_bar["value"]), 100.0)
        self.assertEqual(str(self.window._model_combo["state"]), "readonly")

    def test_compression_started_after_opening_blocks_generation(self):
        self.compressing.return_value = True
        with patch("subtitle_gui.messagebox.showinfo") as info:
            self.window._start()
        info.assert_called_once()
        self.assertFalse(self.window.running)

    def test_close_cancels_but_keeps_window_alive_until_worker_finishes(self):
        self.window.running = True
        self.window._close()
        self.assertTrue(self.window._cancel.is_set())
        self.assertTrue(self.window.winfo_exists())

    def test_worker_reuses_model_and_continues_after_file_error(self):
        model = Mock()
        with patch("subtitle_gui.load_model", return_value=model) as load:
            with patch(
                "subtitle_gui.generate_english_srt",
                side_effect=[RuntimeError("Bad audio"), "second.en.srt"],
            ) as generate:
                self.window._worker(("first.mkv", "second.mkv"), "small", "cpu")
        load.assert_called_once_with("small", "cpu")
        self.assertEqual(generate.call_count, 2)
        self.assertIs(generate.call_args_list[0].args[1], model)
        self.assertIs(generate.call_args_list[1].args[1], model)
        self.assertEqual(generate.call_args.kwargs["batch_size"], 1)
        events = []
        while not self.window._events.empty():
            events.append(self.window._events.get_nowait())
        self.assertIn(("error", "first.mkv: Bad audio"), events)
        self.assertIn(("generated", ("second.mkv", "second.en.srt")), events)
        self.assertEqual(events[-1], ("done", (1, 1, False)))

    def test_cuda_worker_requests_batch_size_eight(self):
        with patch("subtitle_gui.load_model"), patch(
            "subtitle_gui.generate_english_srt", return_value="episode.en.srt",
        ) as generate:
            self.window._worker(("episode.mkv",), "large-v3", "cuda")
        self.assertEqual(generate.call_args.kwargs["batch_size"], 8)

    def test_cuda_worker_can_use_sequential_quality_fallback(self):
        with patch("subtitle_gui.load_model"), patch(
            "subtitle_gui.generate_english_srt", return_value="episode.en.srt",
        ) as generate:
            self.window._worker(("episode.mkv",), "large-v3", "cuda", False)
        self.assertEqual(generate.call_args.kwargs["batch_size"], 1)

    def test_repeated_dialogue_is_not_removed_by_srt_writer(self):
        model = Mock()
        model.transcribe.return_value = (
            iter([
                SimpleNamespace(start=1.0, end=2.0, text="Hello."),
                SimpleNamespace(start=3.0, end=4.0, text="Hello."),
            ]),
            SimpleNamespace(duration=5.0),
        )
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "episode.mkv"
            video.touch()
            output = generate_english_srt(str(video), model, threading.Event(), Mock())
            text = Path(output).read_text(encoding="utf-8")
        self.assertEqual(text.count("Hello."), 2)

    def test_full_frontend_attaches_generated_subtitles(self):
        from gui_compress import CompressApp

        app = CompressApp()
        app.withdraw()
        try:
            video = os.path.abspath("episode.mkv")
            output = subtitle_output_path(video)
            app._populate_table([{"path": video, "size": 1024}])
            app._open_subtitle_generator()
            app._subtitle_window.withdraw()
            app._on_subtitle_generated(video, output)
            self.assertEqual(app._videos[0]["_subtitle_path"], output)
            self.assertEqual(app._videos[0]["_subtitle_offset"], 0.0)
            self.assertEqual(app._tree.set("0", "Subtitle"), "episode.en.srt")
            app._subtitle_window.running = True
            with patch("gui_compress.messagebox.showinfo") as info:
                with patch.object(app, "_ensure_ffmpeg") as ffmpeg:
                    app._start_compress()
            info.assert_called_once()
            ffmpeg.assert_not_called()
        finally:
            sys.stdout = app._original_stdout
            app.destroy()

    def test_compression_settings_restore_and_save_without_changing_config_file(self):
        from gui_compress import CompressApp

        app = CompressApp()
        app.withdraw()
        try:
            config = {
                "bitrate": "500k", "encoder": "av1_nvenc", "subtitle_file": "",
                "subtitle_offset_s": 0.0, "min_size_gb": 0.5,
                "preview_start_s": 12.5, "preview_duration_s": 15.0,
            }
            app._config = config.copy()
            app._restore_config()
            self.assertEqual(app._bitrate_var.get(), "500k")
            self.assertEqual(app._subtitle_offset_var.get(), "0.0")
            self.assertEqual(app._preview_start_var.get(), "12.5")
            app._preview_duration_var.set("invalid")
            app._bitrate_var.set("600k")
            with patch("gui_compress._save_config") as save:
                app._persist_config()
            saved = save.call_args.args[0]
            self.assertEqual(saved["bitrate"], "600k")
            self.assertEqual(saved["min_size_gb"], 0.5)
            self.assertEqual(saved["preview_duration_s"], 15.0)
        finally:
            sys.stdout = app._original_stdout
            app.destroy()


if __name__ == "__main__":
    unittest.main()
