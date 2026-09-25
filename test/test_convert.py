import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ebooklib import epub

from epubvox import convert as qtb


def make_chapter(number=1, title="第1章 丹田被毁", body=("“师兄，苦了你了。”秦若水柔声道。", "他没有回答")):
    return qtb.Chapter(number, title, (title, *body))


class ParseChapterRangesTest(unittest.TestCase):
    def test_default_is_whole_book(self):
        self.assertEqual(qtb.parse_chapter_ranges(None, 4), [1, 2, 3, 4])

    def test_ranges_lists_and_open_end(self):
        self.assertEqual(qtb.parse_chapter_ranges("1-3,5,8-", 10), [1, 2, 3, 5, 8, 9, 10])
        self.assertEqual(qtb.parse_chapter_ranges("2-2", 5), [2])

    def test_invalid_specs_raise(self):
        for spec in ("abc", "0-3", "3-2", "1-99", "1-2-3"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                qtb.parse_chapter_ranges(spec, 10)


class FilenameTest(unittest.TestCase):
    def test_number_padded_with_title(self):
        self.assertEqual(qtb.chapter_filename(make_chapter()), "0001-第1章 丹田被毁.wav")

    def test_illegal_characters_and_edges(self):
        self.assertEqual(qtb.safe_filename('a/b:c?d. '), "a_b_c_d")
        self.assertEqual(qtb.safe_filename("..."), "untitled")
        self.assertLessEqual(len(qtb.safe_filename("章" * 200)), 80)


class SplitChunksTest(unittest.TestCase):
    def test_title_stands_alone_and_gets_a_full_stop(self):
        self.assertEqual(qtb.split_chunks(make_chapter())[0], "第1章 丹田被毁。")

    def test_chunks_respect_limit_and_keep_all_text(self):
        body = tuple(f"第{i}句话。他说：“好的。”" for i in range(40))
        chunks = qtb.split_chunks(make_chapter(body=body), max_chars=50)
        self.assertTrue(all(len(c) <= 50 for c in chunks[1:]))
        self.assertEqual("".join(chunks[1:]), "".join(body))

    def test_closing_quote_stays_with_its_sentence(self):
        body = tuple("“师兄，苦了你了。”秦若水柔声道。" for _ in range(10))
        chunks = qtb.split_chunks(make_chapter(body=body), max_chars=20)
        self.assertFalse(any(c.startswith("”") for c in chunks))

    def test_unpunctuated_paragraph_is_terminated(self):
        self.assertTrue(qtb.split_chunks(make_chapter(body=("他没有回答",)))[-1].endswith("。"))


class FingerprintTest(unittest.TestCase):
    def test_changes_with_voice_settings_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.wav"
            ref.write_bytes(b"voice-a")
            base = qtb.settings_fingerprint("m", ref, "text", "chinese")
            self.assertEqual(base, qtb.settings_fingerprint("m", ref, "text", "chinese"))
            self.assertNotEqual(base, qtb.settings_fingerprint("m", ref, None, "chinese"))
            self.assertNotEqual(base, qtb.settings_fingerprint("m", ref, "text", "english"))
            self.assertNotEqual(base, qtb.settings_fingerprint("other", ref, "text", "chinese"))
            ref.write_bytes(b"voice-b")
            self.assertNotEqual(base, qtb.settings_fingerprint("m", ref, "text", "chinese"))


class ConvertBookResumeTest(unittest.TestCase):
    """Ctrl+C mid-chapter, then re-running the same command must continue instead of starting over."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out"
        self.cache = self.out / ".cache" / "fp"
        body = tuple(f"这是第{i}段很长很长的文字，用来凑够多个片段。" * 4 for i in range(12))
        self.chapters = [make_chapter(1, "第1章 甲", body), make_chapter(2, "第2章 乙", body)]
        self.chunks_in_chapter = len(qtb.split_chunks(self.chapters[0]))
        self.calls = 0

    def synth(self, text):
        self.calls += 1
        return np.full(2400, 0.1, dtype=np.float32)

    def run_book(self, synth, numbers=(1, 2)):
        self.logs = []
        return qtb.convert_book(self.chapters, list(numbers), self.out, self.cache, synth, log=self.logs.append)

    def test_interrupt_then_resume_skips_finished_work(self):
        self.assertGreater(self.chunks_in_chapter, 4)
        finished_before_interrupt = 3

        def interrupting(text):
            if self.calls == finished_before_interrupt:  # Ctrl+C arrives while chunk 4 is being synthesized
                raise KeyboardInterrupt
            return self.synth(text)

        with self.assertRaises(KeyboardInterrupt):
            self.run_book(interrupting)
        self.assertEqual(list(self.out.glob("*.wav")), [])  # unfinished chapter leaves no wav
        self.assertEqual(len(list(self.cache.glob("*.npy"))), finished_before_interrupt)  # finished chunks are kept

        self.calls = 0
        self.assertEqual(self.run_book(self.synth), 2)
        self.assertEqual(self.calls, 2 * self.chunks_in_chapter - finished_before_interrupt)  # only the remainder
        self.assertIn(f"    resuming: {finished_before_interrupt} chunks already finished", self.logs)
        self.assertEqual(sorted(p.name for p in self.out.glob("*.wav")), ["0001-第1章 甲.wav", "0002-第2章 乙.wav"])
        self.assertEqual(list(self.cache.glob("*.npy")), [])  # cache freed once a chapter is done

        self.calls = 0
        self.assertEqual(self.run_book(self.synth), 0)  # everything finished: nothing to do
        self.assertEqual(self.calls, 0)

    def test_progress_reports_totals_chunks_and_eta(self):
        states = []
        qtb.convert_book(self.chapters, [1, 2], self.out, self.cache, self.synth,
                         log=lambda _: None, on_progress=states.append)
        first = states[0]
        self.assertEqual((first["state"], first["selected"], first["todo"], first["converted"]), ("running", 2, 2, 0))
        announced = [s["chapter"] for s in states if s["chapter"]]
        self.assertEqual(announced[0], {"number": 1, "title": "第1章 甲", "chunk": 0, "chunks": self.chunks_in_chapter})
        self.assertEqual(announced[-1]["chunk"], announced[-1]["chunks"])
        last = states[-1]
        self.assertEqual((last["converted"], last["chapter"], last["eta_seconds"]), (2, None, 0))

    def test_progress_counts_chapters_that_were_already_done(self):
        self.run_book(self.synth, numbers=(1,))
        states = []
        qtb.convert_book(self.chapters, [1, 2], self.out, self.cache, self.synth,
                         log=lambda _: None, on_progress=states.append)
        self.assertEqual((states[0]["already_done"], states[0]["todo"], states[0]["selected"]), (1, 1, 2))

    def test_partial_selection_only_converts_requested_chapters(self):
        self.run_book(self.synth, numbers=(2,))
        self.assertEqual([p.name for p in self.out.glob("*.wav")], ["0002-第2章 乙.wav"])


class ExtractChaptersTest(unittest.TestCase):
    def test_skips_front_matter_and_numbers_chapters_in_reading_order(self):
        book = epub.EpubBook()
        book.set_identifier("t")
        book.set_title("测试书")
        book.set_language("zh")
        pages = {
            "info.xhtml": "<p>书籍信息</p><p>作者：某人</p>",
            "chapter1.xhtml": "<h1>第1章 开始</h1><p>第一段。</p>",
            "chapter2.xhtml": "<h1>第2章 继续</h1><p>第二段。</p>",
            "chapter3.xhtml": "<h1>第3章 结束</h1><p>第三段。</p>",
        }
        items = []
        for name, html in pages.items():
            item = epub.EpubHtml(title=name, file_name=name, lang="zh")
            item.content = html
            book.add_item(item)
            items.append(item)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = items
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.epub"
            epub.write_epub(str(path), book)
            chapters = qtb.extract_chapters(path)
        self.assertEqual([(c.number, c.title) for c in chapters], [(1, "第1章 开始"), (2, "第2章 继续"), (3, "第3章 结束")])

    def test_book_title_and_author_come_from_the_epub_metadata(self):
        book = epub.EpubBook()
        book.set_identifier("t")
        book.set_title("贫道看事，只杀不渡！")
        book.add_author("某作者")
        page = epub.EpubHtml(title="c", file_name="c.xhtml", lang="zh")
        page.content = "<h1>第1章 开始</h1><p>正文。</p>"
        book.add_item(page)
        book.add_item(epub.EpubNcx())
        book.spine = [page]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "some file.epub"
            epub.write_epub(str(path), book)
            self.assertEqual(qtb.read_book_info(path), ("贫道看事，只杀不渡！", "某作者"))
            book.metadata = {}
            book.set_identifier("t")
            epub.write_epub(str(path), book)
            self.assertEqual(qtb.read_book_info(path), ("some file", ""))


class ReplacementsTest(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.txt"
            path.write_text(text, encoding="utf-8")
            return qtb.load_replacements(path)

    def test_rules_rewrite_text_before_synthesis(self):
        rules = self.load("# fix names\n\n重楼==虫楼\n请收藏本站.*?。==\n")
        spoken = []
        synth = qtb.with_replacements(lambda text: spoken.append(text) or np.ones(3, dtype=np.float32), rules)
        synth("他见到重楼。请收藏本站www。")
        self.assertEqual(spoken, ["他见到虫楼。"])

    def test_text_removed_entirely_is_silence_not_a_model_call(self):
        synth = qtb.with_replacements(lambda text: self.fail("should not synthesize"), self.load("广告。==\n"))
        self.assertEqual(len(synth("广告。")), 0)

    def test_no_rules_leaves_synth_untouched(self):
        synth = lambda text: np.zeros(1)
        self.assertIs(qtb.with_replacements(synth, []), synth)

    def test_bad_lines_are_reported_with_their_line_number(self):
        for text in ("no separator here\n", "==empty search\n", "([==x\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r":1:"):
                self.load(text)


class CtrlCTest(unittest.TestCase):
    def test_ctrl_c_is_restored_even_when_the_launcher_ignored_it(self):
        """`( cmd & )` leaves SIGINT ignored for cmd and everything it starts, which made Stop a silent no-op."""
        code = ("import signal; from epubvox import convert as q\n"
                "ignored = signal.getsignal(signal.SIGINT) == signal.SIG_IGN\n"
                "q.enable_ctrl_c()\n"
                "print(ignored, signal.getsignal(signal.SIGINT) is signal.default_int_handler)")
        result = subprocess.run(["sh", "-c", 'trap "" INT; exec "$0" -c "$1"', sys.executable, code],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.stdout.strip(), "True True", result.stderr)


class ProgressFileTest(unittest.TestCase):
    def test_state_is_published_as_json_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "progress.json"
            progress = qtb.ProgressFile(path)
            progress.update({"state": "running", "selected": 3})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["selected"], 3)
            progress.finish("error", "boom")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual((data["state"], data["message"], data["selected"]), ("error", "boom", 3))
            self.assertIn("updated", data)
            self.assertEqual([p.name for p in path.parent.iterdir()], ["progress.json"])  # no .part left behind

    def test_extra_facts_are_repeated_in_every_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            progress = qtb.ProgressFile(path, {"book": "My Book"})
            progress.update({"state": "running"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["book"], "My Book")
            progress.finish("finished")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["book"], "My Book")

    def test_without_a_path_it_does_nothing(self):
        progress = qtb.ProgressFile(None)
        progress.update({"state": "running"})
        progress.finish("finished")
        self.assertEqual(progress.state["state"], "finished")


class OutputFormatTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        wav = qtb.make_output_format("wav")
        self.assertEqual((wav.extension, wav.encoder, wav.bitrate), (".wav", None, None))
        self.assertEqual(qtb.make_output_format("mp3").bitrate, "64k")
        self.assertEqual(qtb.make_output_format("m4a").bitrate, "32k")
        self.assertEqual(qtb.make_output_format("mp3", "32k").bitrate, "32k")

    def test_speed_is_kept_and_routes_fast_wav_through_ffmpeg(self):
        self.assertEqual(qtb.make_output_format("m4a", speed=1.5).speed, 1.5)
        self.assertIsNone(qtb.make_output_format("wav").encoder)
        self.assertEqual(qtb.make_output_format("wav", speed=1.5).encoder, "pcm_s16le")

    def test_invalid_combinations_raise(self):
        for name, bitrate in (("wav", "64k"), ("mp3", "abc"), ("mp3", "48"), ("m4a", "1000k"), ("flac", None)):
            with self.subTest(name=name, bitrate=bitrate), self.assertRaises(ValueError):
                qtb.make_output_format(name, bitrate)
        for speed in (0.4, 4.1):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                qtb.make_output_format("m4a", speed=speed)


class PreviewTest(unittest.TestCase):
    def test_preview_keeps_the_title_and_the_first_paragraphs_only(self):
        chapter = make_chapter(7, "第7章 甲", ("一" * 150 + "。", "二" * 150 + "。", "三" * 150 + "。"))
        preview = qtb.preview_chapter(chapter, max_chars=200)
        self.assertEqual((preview.number, preview.title), (7, "第7章 甲"))
        self.assertEqual(preview.paragraphs, chapter.paragraphs[:3])

    def test_write_preview_saves_one_file_and_leaves_no_cache(self):
        spoken = []

        def synth(text):
            spoken.append(text)
            return np.ones(240, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "previews" / "sample.wav"
            qtb.write_preview(make_chapter(), synth, target, qtb.WAV)
            self.assertEqual(sorted(p.name for p in Path(tmp).rglob("*") if p.is_file()), ["sample.wav"])
        self.assertEqual(spoken[0], "第1章 丹田被毁。")


class ModelsTest(unittest.TestCase):
    def test_qwen_8bit_is_the_default_and_listed_first(self):
        self.assertEqual(qtb.MODEL_ID, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
        self.assertEqual(qtb.KNOWN_MODELS[0][0], qtb.MODEL_ID)
        self.assertTrue(all("Qwen3-TTS" in model for model, _ in qtb.KNOWN_MODELS))


class DefaultsTest(unittest.TestCase):
    def test_only_the_book_is_needed(self):
        args = qtb.build_parser().parse_args(["book.epub"])
        self.assertEqual((args.ref_audio, args.audio_format, args.bitrate, args.speed, args.chapters),
                         (qtb.DEFAULT_REF_AUDIO, "m4a", None, 1.5, None))

    def test_output_folder_is_the_short_book_title(self):
        for stem, folder in (("贫道看事 ，只杀不渡！", "贫道看事"), ("丹田被毁：百炼成仙 - 王屋山人", "丹田被毁"),
                             ("阻我功名", "阻我功名"), ("Dune: Part One", "Dune"), ("The Great Gatsby", "The Great Gatsby"),
                             ("：only punctuation first", "：only punctuation first")):
            with self.subTest(stem=stem):
                self.assertEqual(qtb.default_output(Path(f"/x/{stem}.epub")), Path(folder))

    def test_extension_drives_filename_and_finished_detection(self):
        self.assertEqual(qtb.chapter_filename(make_chapter(), ".m4a"), "0001-第1章 丹田被毁.m4a")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("0001-a.wav", "0002-b.mp3", "0003-c.mp3.part"):
                (out / name).touch()
            self.assertEqual(qtb.finished_numbers(out, ".wav"), {1})
            self.assertEqual(qtb.finished_numbers(out, ".mp3"), {2})


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class EncodingTest(unittest.TestCase):
    TONE = (0.3 * np.sin(2 * np.pi * 440 * np.arange(qtb.SAMPLE_RATE) / qtb.SAMPLE_RATE)).astype(np.float32)

    def encode(self, name, bitrate=None):
        fmt = qtb.make_output_format(name, bitrate)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"0001-x{fmt.extension}"
            qtb.check_encoder(fmt)
            qtb.save_audio(path, self.TONE, fmt)
            return path.read_bytes()[:12], sorted(p.name for p in Path(tmp).iterdir())

    def test_mp3_is_a_real_mp3_and_leaves_no_partial_file(self):
        head, names = self.encode("mp3", "32k")
        self.assertTrue(head.startswith(b"ID3") or head[0] == 0xFF)
        self.assertEqual(names, ["0001-x.mp3"])

    def test_m4a_is_a_real_mp4_audio_file(self):
        head, names = self.encode("m4a")
        self.assertEqual(head[4:8], b"ftyp")
        self.assertEqual(names, ["0001-x.m4a"])

    def test_speed_shortens_the_audio(self):
        fmt = qtb.make_output_format("wav", speed=1.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fast.wav"
            qtb.save_audio(path, self.TONE, fmt)
            import soundfile as sf
            self.assertAlmostEqual(sf.info(path).duration, 1 / 1.5, delta=0.05)

    def test_unusable_encoder_fails_fast(self):
        broken = qtb.OutputFormat("x", ".x", "no_such_encoder", "64k", ("-f", "mp3"))
        with self.assertRaises(RuntimeError):
            qtb.check_encoder(broken)

    def test_convert_book_writes_and_then_skips_mp3_chapters(self):
        chapters = [make_chapter(1, "第1章 甲"), make_chapter(2, "第2章 乙")]
        fmt = qtb.make_output_format("mp3", "32k")

        def synth(text):
            return self.TONE[:2400]

        with tempfile.TemporaryDirectory() as tmp:
            out, cache = Path(tmp) / "out", Path(tmp) / "cache"
            self.assertEqual(qtb.convert_book(chapters, [1, 2], out, cache, synth, fmt=fmt, log=lambda _: None), 2)
            self.assertEqual(sorted(p.name for p in out.glob("*.mp3")), ["0001-第1章 甲.mp3", "0002-第2章 乙.mp3"])
            self.assertEqual(qtb.convert_book(chapters, [1, 2], out, cache, synth, fmt=fmt, log=lambda _: None), 0)

    def test_chapters_are_tagged_with_album_artist_title_and_track(self):
        chapters = [make_chapter(1, "第1章 甲"), make_chapter(2, "第2章 乙")]
        for name in ("m4a", "mp3"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "out"
                qtb.convert_book(chapters, [2], out, Path(tmp) / "cache", lambda text: self.TONE[:2400],
                                 fmt=qtb.make_output_format(name), log=lambda _: None,
                                 album="贫道看事", artist="某作者")
                probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                                        str(out / f"0002-第2章 乙.{name}")], capture_output=True, text=True)
                tags = {k.lower(): v for k, v in json.loads(probe.stdout)["format"]["tags"].items()}
                self.assertEqual((tags["album"], tags["artist"], tags["title"], tags["track"]),
                                 ("贫道看事", "某作者", "第2章 乙", "2/2"))


if __name__ == "__main__":
    unittest.main()
