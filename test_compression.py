import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import compress_video as compression
from scan_and_compress import _process_one, find_large_videos


class CompressionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / "episode.mp4"
        self.video.write_bytes(b"original")
        self.output = self.root / "episode_compressed.mp4"
        self.subtitle = self.root / "episode.en.srt"
        self.subtitle.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello.\n\n", encoding="utf-8")
        compression._gui_stop.clear()
        self.addCleanup(compression._gui_stop.clear)
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def retry(self, **overrides):
        options = dict(
            input_path=str(self.video), output_path=str(self.output),
            encoder="av1_nvenc", effective_kbps=300, duration=5,
            subtitle_filter=None, original_size=8, is_already_mp4=True,
        )
        options.update(overrides)
        return compression._encode_with_retries(**options)

    def test_estimates_reuse_cached_probe_and_cap_bitrate(self):
        video = {"_probe": {"duration": 60, "video_bitrate_kbps": 400}}
        self.assertEqual(
            compression.update_size_estimate(video, 700),
            compression.estimate_compressed_size(60, 400),
        )
        self.assertEqual(video["_est_size"], compression.estimate_compressed_size(60, 400))
        self.assertEqual(compression.update_size_estimate({}, 700), 0)

    def test_shifted_subtitles_are_cleaned_up_even_when_encoding_raises(self):
        shifted = self.root / "shifted.srt"
        shifted.touch()
        with patch("compress_video._shift_srt_file", return_value=str(shifted)):
            with self.assertRaisesRegex(RuntimeError, "Encoding failed"):
                with compression._subtitle_filter(str(self.subtitle), 0.5) as subtitle_filter:
                    self.assertIn("shifted.srt", subtitle_filter)
                    raise RuntimeError("Encoding failed")
        self.assertFalse(shifted.exists())
        self.assertTrue(self.subtitle.exists())

    def test_unshifted_filter_does_not_create_temporary_subtitles(self):
        with patch("compress_video._shift_srt_file") as shift:
            with compression._subtitle_filter(str(self.subtitle), 0) as subtitle_filter:
                self.assertIn("episode.en.srt", subtitle_filter)
            with compression._subtitle_filter(None, 0) as subtitle_filter:
                self.assertIsNone(subtitle_filter)
        shift.assert_not_called()
        self.assertTrue(self.subtitle.exists())

    def test_bitrate_retries_and_encoder_fallback_order_are_preserved(self):
        with patch("compress_video._check_gpu_available", return_value=True), patch(
            "compress_video._get_fallback_encoders", return_value=["libx264"],
        ), patch("compress_video._attempt_encode", return_value=(False, False)) as attempt:
            self.assertFalse(self.retry())
        self.assertEqual(
            [(call.args[2], call.args[4]) for call in attempt.call_args_list],
            [("av1_nvenc", 300), ("av1_nvenc", 240), ("av1_nvenc", 192),
             ("libx264", 300), ("libx264", 240), ("libx264", 192)],
        )

    def test_fatal_encoder_error_skips_remaining_bitrates(self):
        self.output.write_bytes(b"small")
        with patch("compress_video._check_gpu_available", return_value=True), patch(
            "compress_video._get_fallback_encoders", return_value=["libx264"],
        ), patch(
            "compress_video._attempt_encode", side_effect=[(False, True), (True, False)],
        ) as attempt, patch("compress_video._is_valid_output_media", return_value=True):
            self.assertTrue(self.retry())
        self.assertEqual([call.args[4] for call in attempt.call_args_list], [300, 300])

    def test_larger_output_policy(self):
        for is_mp4, subtitle_filter, expected in (
            (True, None, None), (False, None, True), (True, "subtitles='file.srt'", True),
        ):
            with self.subTest(is_mp4=is_mp4, subtitle_filter=subtitle_filter):
                self.output.write_bytes(b"larger than original")
                with patch("compress_video._check_gpu_available", return_value=True), patch(
                    "compress_video._attempt_encode", return_value=(True, False),
                ), patch("compress_video._is_valid_output_media", return_value=True):
                    self.assertIs(self.retry(is_already_mp4=is_mp4, subtitle_filter=subtitle_filter), expected)
                self.assertEqual(self.output.exists(), expected is True)

    def test_stop_does_not_start_another_attempt(self):
        compression._gui_stop.set()
        with patch("compress_video._check_gpu_available", return_value=True), patch(
            "compress_video._attempt_encode",
        ) as attempt:
            self.assertFalse(self.retry())
        attempt.assert_not_called()

    def test_low_bitrate_mp4_skips_but_subtitles_force_encoding(self):
        with patch(
            "compress_video.probe_video", return_value={"duration": 5, "video_bitrate_kbps": 200},
        ), patch("compress_video._encode_with_retries", return_value=True) as encode:
            self.assertIsNone(compression.compress_video(str(self.video), str(self.output)))
            encode.assert_not_called()
            self.assertTrue(compression.compress_video(
                str(self.video), str(self.output), subtitle_path=str(self.subtitle),
            ))
            self.assertIn("episode.en.srt", encode.call_args.args[5])

    def test_scan_skips_originals_and_generated_outputs(self):
        originals = self.root / "_originals_to_delete"
        originals.mkdir()
        (originals / "old.mkv").touch()
        self.output.touch()
        self.assertEqual(
            [v["path"] for v in find_large_videos(str(self.root), 0)], [str(self.video)],
        )

    def test_originals_move_only_after_success(self):
        for outcome in (None, False, True):
            with self.subTest(outcome=outcome), patch(
                "scan_and_compress.compress_video", return_value=outcome,
            ):
                if outcome is True:
                    self.output.write_bytes(b"small")
                record = _process_one(
                    1, 1, {"path": str(self.video), "size": 8}, "700k", "av1_nvenc", str(self.root),
                )
                self.assertEqual(self.video.exists(), outcome is not True)
                self.assertEqual(bool(record["moved_to"]), outcome is True)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_remux_and_subtitle_preview(self):
        source = self.root / "source.mkv"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10",
             "-t", "3", "-c:v", "libx264", str(source)],
            check=True, capture_output=True,
        )
        self.assertTrue(compression._remux_to_mp4_copy(str(source), str(self.output)))
        preview = self.root / "preview.mp4"
        self.assertTrue(compression.create_subtitle_preview(
            str(source), str(preview), str(self.subtitle),
            subtitle_offset_s=0.25, preview_start_s=0.5, preview_seconds=1,
        ))
        self.assertTrue(compression._is_valid_output_media(str(preview)))
        self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
