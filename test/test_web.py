import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote, urlencode

from ebooklib import epub

from epubvox import web

# Stands in for the converter (epubvox/convert.py): same --name=value options, writes the same progress JSON, honours Ctrl+C.
STUB_CONVERTER = '''
import json, signal, sys, time
options = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
progress = options["progress-file"]

def publish(state):
    with open(progress, "w") as handle:
        json.dump({"book": "book", "selected": 1, "already_done": 0, "converted": 0, "chapter": None, **state}, handle)

if options["model"] == "stub-slow":
    def stop(*_):
        publish({"state": "interrupted"})
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    publish({"state": "running"})
    time.sleep(30)
else:
    publish({"state": "running"})
    time.sleep(0.2)
    if "preview" in options:
        with open(options["preview"], "wb") as handle:
            handle.write(b"fake audio")
    publish({"state": "finished", "converted": 1})
'''


def make_epub(path: Path, chapter_count: int = 3) -> None:
    book = epub.EpubBook()
    book.set_identifier("t")
    book.set_title("测试书")
    book.set_language("zh")
    items = []
    for number in range(1, chapter_count + 1):
        item = epub.EpubHtml(title=f"c{number}", file_name=f"chapter{number}.xhtml", lang="zh")
        item.content = f"<h1>第{number}章 甲</h1><p>第{number}段。</p>"
        book.add_item(item)
        items.append(item)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = items
    epub.write_epub(str(path), book)


class SettingsStoreTest(unittest.TestCase):
    def test_a_known_book_returns_as_run_and_a_new_book_inherits_voice_but_not_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = web.SettingsStore(Path(tmp) / "settings.json")
            first = {**web.DEFAULT_FORM, "epub": "/b/a.epub", "ref_audio": "/v.wav", "ref_text": "hello",
                     "audio_format": "mp3", "bitrate": "48k", "chapters": "80-", "output": "/o/a"}
            store.remember(first)
            self.assertEqual(store.form_for("/b/a.epub"), first)
            fresh = store.form_for("/b/new.epub")
            self.assertEqual((fresh["ref_audio"], fresh["ref_text"], fresh["audio_format"], fresh["bitrate"]),
                             ("/v.wav", "hello", "mp3", "48k"))
            self.assertEqual((fresh["epub"], fresh["chapters"], fresh["output"]), ("/b/new.epub", "", ""))

    def test_persists_across_restarts_and_lists_recent_books_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            store = web.SettingsStore(path)
            for epub_path in ("/a.epub", "/b.epub", "/a.epub"):
                store.remember({**web.DEFAULT_FORM, "epub": epub_path})
            reloaded = web.SettingsStore(path).snapshot()
            self.assertEqual(reloaded["recent"], ["/a.epub", "/b.epub"])
            self.assertEqual(reloaded["last"]["epub"], "/a.epub")

    def test_recent_books_are_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = web.SettingsStore(Path(tmp) / "settings.json")
            for number in range(web.MAX_RECENT_BOOKS + 5):
                store.remember({**web.DEFAULT_FORM, "epub": f"/book{number}.epub"})
            self.assertEqual(len(store.snapshot()["recent"]), web.MAX_RECENT_BOOKS)

    def test_missing_or_corrupt_file_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            self.assertEqual(web.SettingsStore(path).snapshot()["last"], web.DEFAULT_FORM)
            for junk in ("{not json", "[1, 2]"):
                path.write_text(junk, encoding="utf-8")
                self.assertEqual(web.SettingsStore(path).snapshot()["last"], web.DEFAULT_FORM)


class CleanFormAndCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.epub = Path(self.tmp.name) / "book.epub"
        make_epub(self.epub)
        self.voice = Path(self.tmp.name) / "voice.wav"
        self.voice.write_bytes(b"RIFF")
        self.raw = {**web.DEFAULT_FORM, "epub": str(self.epub), "ref_audio": str(self.voice)}

    def clean(self, **changes):
        return web.clean_form({**self.raw, **changes}, lambda path: 3)

    def test_valid_form_gets_absolute_paths_and_a_default_output_folder(self):
        form = self.clean()
        self.assertEqual(form["epub"], str(self.epub.resolve()))
        self.assertEqual(form["output"], str((Path.cwd() / "book").resolve()))
        self.assertEqual((form["audio_format"], form["bitrate"]), ("m4a", "32k"))

    def test_wav_has_no_bitrate_and_a_blank_bitrate_uses_the_format_default(self):
        self.assertEqual(self.clean(audio_format="wav", bitrate="64k")["bitrate"], "")
        self.assertEqual(self.clean(audio_format="mp3", bitrate="")["bitrate"], "64k")

    def test_problems_are_reported_in_plain_words(self):
        cases = [({"epub": "/nope.epub"}, ".epub"), ({"ref_audio": ""}, "reference voice"),
                 ({"lang": "no way!"}, "Language"), ({"bitrate": "fast"}, "bitrate"),
                 ({"chapters": "1-99"}, "outside 1-3"), ({"chapters": "x"}, "bad chapter range")]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.clean(**changes)

    def test_speed_and_rules_file_are_checked_and_passed_on(self):
        rules = Path(self.tmp.name) / "fixes.txt"
        rules.write_text("重楼==虫楼\n", encoding="utf-8")
        form = self.clean(speed="1.25", replace=str(rules))
        command = web.build_command(form, Path("/p"), ("uv", "run"))
        self.assertIn("--speed=1.25", command)
        self.assertIn(f"--replace={rules.resolve()}", command)
        self.assertEqual(self.clean(speed="")["speed"], "1.5")
        broken = Path(self.tmp.name) / "broken.txt"
        broken.write_text("no separator\n", encoding="utf-8")
        for changes, message in (({"speed": "fast"}, "Speed"), ({"speed": "9"}, "speed"),
                                 ({"replace": "/nope.txt"}, "rules file"), ({"replace": str(broken)}, ":1:")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.clean(**changes)

    def test_preview_file_is_named_after_book_and_model(self):
        form = {"epub": "/b/贫道看事，只杀不渡！.epub", "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                "audio_format": "m4a"}
        path = web.preview_path(Path("/d"), form)
        self.assertEqual(path, Path("/d/previews/贫道看事__mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit.m4a"))
        command = web.build_command(self.clean(), Path("/p"), ("uv", "run"), preview=path)
        self.assertIn(f"--preview={path}", command)

    def test_command_uses_name_equals_value_options(self):
        form = self.clean(chapters="2-", ref_text="-starts with a dash")
        command = web.build_command(form, Path("/p/progress.json"), ("uv", "run"), Path("/s/book.py"))
        self.assertEqual(command[:4], ["uv", "run", "/s/book.py", form["epub"]])
        for expected in ("--chapters=2-", "--ref-text=-starts with a dash", "--bitrate=32k", "--format=m4a",
                         "--progress-file=/p/progress.json"):
            self.assertIn(expected, command)

    def test_command_leaves_out_options_that_do_not_apply(self):
        command = web.build_command(self.clean(audio_format="wav"), Path("/p"), ("uv", "run"))
        self.assertEqual([part for part in command
                          if part.startswith(("--bitrate", "--ref-text", "--chapters", "--replace"))], [])

    def test_only_localhost_is_accepted_as_host(self):
        self.assertTrue(all(web.is_local_host(h) for h in ("127.0.0.1:8765", "localhost:1", "localhost")))
        self.assertFalse(any(web.is_local_host(h) for h in ("evil.example", "127.0.0.1.evil.example:80", "", None)))


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.stub = root / "stub_converter.py"
        self.stub.write_text(STUB_CONVERTER, encoding="utf-8")
        self.epub = root / "book.epub"
        make_epub(self.epub)
        self.voice = root / "voice.wav"
        self.voice.write_bytes(b"RIFF")
        self.output = root / "out"
        self.app = web.App(root / "data", runner=(sys.executable,), script=self.stub)
        self.server = web.make_server(self.app, 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.app.job.stop()
        self.app.job.wait(10)
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        data = json.loads(response.read() or b"null")
        connection.close()
        return response.status, data

    def form(self, **changes):
        return {**web.DEFAULT_FORM, "epub": str(self.epub), "ref_audio": str(self.voice), "ref_text": "hi",
                "output": str(self.output), "model": "stub-fast", **changes}

    def wait_for(self, predicate, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.call("GET", "/api/state")[1]
            if predicate(state):
                return state
            time.sleep(0.05)
        self.fail("timed out waiting for the conversion")

    def test_starts_idle_with_default_settings(self):
        status, state = self.call("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertFalse(state["job"]["running"])
        self.assertEqual((state["progress"], state["settings"]["recent"]), (None, []))
        self.assertEqual(state["settings"]["last"]["audio_format"], "m4a")

    def test_page_is_served(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", "/")
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"<title>epubvox</title>", response.read())

    def test_favicon_request_is_answered_without_an_error(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", "/favicon.ico")
        self.assertEqual(connection.getresponse().status, 204)

    def test_start_runs_to_completion_and_remembers_the_settings(self):
        status, result = self.call("POST", "/api/start", self.form())
        self.assertEqual((status, result["ok"]), (200, True))
        state = self.wait_for(lambda s: not s["job"]["running"] and (s["progress"] or {}).get("state") == "finished")
        self.assertEqual(state["job"]["exit_code"], 0)
        epub_key = str(self.epub.resolve())
        self.assertEqual(state["settings"]["recent"], [epub_key])
        remembered = self.call("GET", "/api/form?epub=" + quote(epub_key))[1]
        self.assertEqual((remembered["ref_text"], remembered["output"]), ("hi", str(self.output.resolve())))

    def test_stop_interrupts_a_running_conversion(self):
        self.assertEqual(self.call("POST", "/api/start", self.form(model="stub-slow"))[0], 200)
        self.wait_for(lambda s: s["job"]["running"] and (s["progress"] or {}).get("state") == "running")
        self.assertEqual(self.call("POST", "/api/stop", {})[0], 200)
        state = self.wait_for(lambda s: not s["job"]["running"])
        self.assertEqual((state["progress"]["state"], state["job"]["exit_code"]), ("interrupted", 130))

    def test_a_second_start_is_refused_while_one_is_running(self):
        self.call("POST", "/api/start", self.form(model="stub-slow"))
        self.wait_for(lambda s: s["job"]["running"])
        status, data = self.call("POST", "/api/start", self.form(model="stub-slow"))
        self.assertEqual(status, 400)
        self.assertIn("already running", data["error"])

    def test_bad_input_gets_a_400_with_a_message(self):
        status, data = self.call("POST", "/api/start", self.form(epub="/nope.epub"))
        self.assertEqual(status, 400)
        self.assertIn(".epub", data["error"])
        status, data = self.call("POST", "/api/start", self.form(chapters="1-99"))
        self.assertEqual((status, "outside 1-3" in data["error"]), (400, True))

    def test_foreign_hosts_and_non_json_posts_are_rejected(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.putrequest("GET", "/api/state", skip_host=True)
        connection.putheader("Host", "evil.example")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 403)
        status, _ = self.call("POST", "/api/stop", None, {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_book_info_counts_what_is_already_on_disk(self):
        self.output.mkdir()
        (self.output / "0001-第1章 甲.m4a").touch()
        query = urlencode({"epub": str(self.epub), "chapters": "", "output": str(self.output), "audio_format": "m4a"})
        info = self.call("GET", "/api/book?" + query)[1]
        self.assertEqual((info["total"], info["selected"], info["done"], info["first_unfinished"]), (3, 3, 1, 2))
        query = urlencode({"epub": str(self.epub), "chapters": "2-3", "output": str(self.output), "audio_format": "m4a"})
        info = self.call("GET", "/api/book?" + query)[1]
        self.assertEqual((info["selected"], info["done"], info["first_unfinished"]), (2, 0, 2))

    def test_file_picker_lists_folders_and_only_the_wanted_files(self):
        folder = str(self.epub.parent)
        books = self.call("GET", "/api/browse?" + urlencode({"path": folder, "kind": "epub"}))[1]
        names = {entry["name"] for entry in books["entries"]}
        self.assertIn("book.epub", names)
        self.assertNotIn("voice.wav", names)
        audio = self.call("GET", "/api/browse?" + urlencode({"path": folder, "kind": "audio"}))[1]
        names = {entry["name"] for entry in audio["entries"]}
        self.assertIn("voice.wav", names)
        self.assertNotIn("book.epub", names)

    def fetch(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, response.getheader("Content-Type"), body

    def test_preview_is_kept_per_book_and_model_and_can_be_played(self):
        self.assertEqual(self.call("POST", "/api/preview", self.form(chapters="2-"))[0], 200)
        self.wait_for(lambda state: not state["job"]["running"])
        status, previews = self.call("GET", "/api/previews?" + urlencode({"epub": str(self.epub)}))
        self.assertEqual(status, 200)
        self.assertEqual([p["model"] for p in previews], ["stub-fast"])
        status, content_type, body = self.fetch("/api/preview-audio?" + urlencode({"name": previews[0]["name"]}))
        self.assertEqual((status, content_type, body), (200, "audio/mp4", b"fake audio"))
        other = Path(self.tmp.name) / "other.epub"
        make_epub(other)
        self.assertEqual(self.call("GET", "/api/previews?" + urlencode({"epub": str(other)}))[1], [])

    def test_preview_audio_only_serves_files_from_the_previews_folder(self):
        for name in ("../settings.json", "nope.m4a", ""):
            with self.subTest(name=name):
                self.assertEqual(self.fetch("/api/preview-audio?" + urlencode({"name": name}))[0], 404)

    def test_state_offers_the_known_models(self):
        models = self.call("GET", "/api/state")[1]["models"]
        self.assertEqual(models[0]["id"], web.converter.MODEL_ID)

    def test_rules_picker_lists_text_files(self):
        folder = Path(self.tmp.name)
        for name in ("fixes.txt", "voice.wav"):
            (folder / name).touch()
        names = [entry["name"] for entry in self.app.browse(str(folder), "text")["entries"]]
        self.assertIn("fixes.txt", names)
        self.assertNotIn("voice.wav", names)

    def test_corrupt_progress_file_reads_as_no_progress(self):
        self.app.job.progress_path.write_text("{half written", encoding="utf-8")
        self.assertIsNone(self.app.job.progress())


if __name__ == "__main__":
    unittest.main()
