"""Local web page for epubvox: pick a book and a voice, set the options, preview, watch progress, stop and continue.

    epubvox web            # opens http://127.0.0.1:8765

Settings are remembered in ~/.local/state/epubvox/settings.json: per book (re-select a book and everything you ran
with comes back, so continuing is "change the chapter range, press Start") and as "last used" (a new book starts with
your voice, format and bitrate). Conversion runs in a separate process, so Stop is the same safe Ctrl+C as on the
command line, and Start again resumes where it stopped. The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from epubvox import convert as converter

CONVERTER = Path(converter.__file__).resolve()
PAGE = Path(__file__).resolve().with_name("web.html")
DEFAULT_DATA_DIR = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "epubvox"
DEFAULT_PORT = 8765
LOCAL_HOSTS = ("127.0.0.1", "localhost")
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
MAX_BODY_BYTES = 64 * 1024
MAX_RECENT_BOOKS = 20
MAX_LISTED_FILES = 500
LOG_TAIL_BYTES = 8192
LOG_TAIL_LINES = 30

FORM_FIELDS = ("epub", "ref_audio", "ref_text", "lang", "audio_format", "bitrate", "speed", "replace", "chapters",
               "output", "model")
BOOK_SPECIFIC_FIELDS = ("chapters", "output")  # everything else follows you to the next book
DEFAULT_FORM = {"epub": "", "ref_audio": str(converter.DEFAULT_REF_AUDIO), "ref_text": "", "lang": "chinese",
                "audio_format": "m4a", "bitrate": "32k", "speed": str(converter.DEFAULT_SPEED), "replace": "",
                "chapters": "", "output": "", "model": converter.MODEL_ID}
PICKER_SUFFIXES = {"epub": {".epub"}, "audio": AUDIO_SUFFIXES, "text": {".txt"}}
AUDIO_TYPES = {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav"}


def is_local_host(header: str | None) -> bool:
    """Only answer requests addressed to localhost, which blocks DNS-rebinding attacks from other websites."""
    return (header or "").rsplit(":", 1)[0] in LOCAL_HOSTS


def default_output(epub: Path) -> Path:
    return Path.cwd() / converter.default_output(epub)


# --- settings -------------------------------------------------------------------------------------------------

class SettingsStore:
    """The last form used, plus one form per book, kept in a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            last, books = data.get("last", {}), data.get("books", {})
        except (OSError, ValueError, AttributeError):
            last, books = {}, {}
        return {"last": {**DEFAULT_FORM, **last}, "books": dict(books)}

    def snapshot(self) -> dict:
        with self._lock:
            return {"last": dict(self._data["last"]), "recent": list(reversed(self._data["books"]))}

    def form_for(self, epub: str) -> dict:
        """A book you ran before comes back exactly as you ran it; a new book inherits voice and format only."""
        with self._lock:
            saved = self._data["books"].get(epub)
            if saved is not None:
                return {**DEFAULT_FORM, **saved, "epub": epub}
            inherited = {k: v for k, v in self._data["last"].items() if k not in BOOK_SPECIFIC_FIELDS}
            return {**DEFAULT_FORM, **inherited, "epub": epub}

    def remember(self, form: dict) -> None:
        with self._lock:
            books = {k: v for k, v in self._data["books"].items() if k != form["epub"]}
            books[form["epub"]] = dict(form)
            self._data = {"last": dict(form), "books": dict(list(books.items())[-MAX_RECENT_BOOKS:])}
            payload = json.dumps(self._data, ensure_ascii=False, indent=2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            converter.replace_atomically(lambda partial: partial.write_text(payload, encoding="utf-8"), self.path)


# --- validating the form and building the converter command -------------------------------------------------------

def clean_form(raw: dict, count_chapters: Callable[[Path], int]) -> dict:
    """Validate what the browser sent. Returns the cleaned form, or raises ValueError with a message for the user."""
    form = {name: str(raw.get(name) or "").strip() for name in FORM_FIELDS}
    for name in ("lang", "model", "audio_format", "speed"):
        form[name] = form[name] or DEFAULT_FORM[name]
    epub = Path(form["epub"]).expanduser()
    if epub.suffix.lower() != ".epub" or not epub.is_file():
        raise ValueError("Pick an existing .epub file.")
    ref_audio = Path(form["ref_audio"]).expanduser()
    if not form["ref_audio"] or not ref_audio.is_file():
        raise ValueError("Pick an existing reference voice audio file.")
    if not re.fullmatch(r"[A-Za-z_-]{2,20}", form["lang"]):
        raise ValueError("Language must be a code such as chinese, english or auto.")
    try:
        speed = float(form["speed"])
    except ValueError:
        raise ValueError("Speed must be a number such as 1, 1.25 or 1.5.") from None
    bitrate = None if form["audio_format"] == "wav" else form["bitrate"] or None
    fmt = converter.make_output_format(form["audio_format"], bitrate, speed)
    replace = Path(form["replace"]).expanduser() if form["replace"] else None
    if replace is not None:
        if not replace.is_file():
            raise ValueError("Pick an existing pronunciation rules file, or leave it empty.")
        converter.load_replacements(replace)  # reports a bad line before the run starts
    if form["chapters"]:
        converter.parse_chapter_ranges(form["chapters"], count_chapters(epub))
    output = Path(form["output"]).expanduser() if form["output"] else default_output(epub)
    return {**form, "epub": str(epub.resolve()), "ref_audio": str(ref_audio.resolve()),
            "bitrate": fmt.bitrate or "", "speed": f"{speed:g}", "replace": str(replace.resolve()) if replace else "",
            "output": str(output.resolve())}


def preview_path(data_dir: Path, form: dict) -> Path:
    """One sample per book and model, so several models can be compared side by side."""
    book = converter.default_output(Path(form["epub"])).name
    model = re.sub(r"[^\w.-]+", "_", form["model"].replace("/", "--"))
    return data_dir / "previews" / f"{book}__{model}.{form['audio_format']}"


def build_command(form: dict, progress_path: Path, runner: tuple[str, ...], script: Path = CONVERTER,
                  preview: Path | None = None) -> list[str]:
    """Options use the --name=value form so a value that starts with "-" can never be mistaken for an option."""
    command = [*runner, str(script), form["epub"], f"--ref-audio={form['ref_audio']}", f"--lang={form['lang']}",
               f"--format={form['audio_format']}", f"--output={form['output']}", f"--model={form['model']}",
               f"--speed={form['speed']}", f"--progress-file={progress_path}"]
    if form["replace"]:
        command.append(f"--replace={form['replace']}")
    if form["audio_format"] != "wav" and form["bitrate"]:
        command.append(f"--bitrate={form['bitrate']}")
    if form["ref_text"]:
        command.append(f"--ref-text={form['ref_text']}")
    if form["chapters"]:
        command.append(f"--chapters={form['chapters']}")
    if preview is not None:
        command.append(f"--preview={preview}")
    return command


# --- books and the running job ------------------------------------------------------------------------------------

class BookCache:
    """Parsing a big EPUB takes seconds, so keep the last few parsed books (keyed by path and modification time)."""

    def __init__(self, size: int = 4) -> None:
        self.size = size
        self._items: dict[tuple[str, int], list] = {}
        self._lock = threading.Lock()

    def chapters(self, epub: Path) -> list:
        key = (str(epub), epub.stat().st_mtime_ns)
        with self._lock:
            if key in self._items:
                return self._items[key]
        parsed = converter.extract_chapters(epub)  # slow, so outside the lock
        with self._lock:
            self._items[key] = parsed
            while len(self._items) > self.size:
                self._items.pop(next(iter(self._items)))
        return parsed


class Job:
    """At most one converter process. Stop sends it Ctrl+C (SIGINT), exactly as a terminal would."""

    def __init__(self, data_dir: Path) -> None:
        self.log_path = data_dir / "run.log"
        self.progress_path = data_dir / "progress.json"
        self._process: subprocess.Popen | None = None

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def exit_code(self) -> int | None:
        return None if self._process is None else self._process.poll()

    def start(self, command: list[str]) -> None:
        if self.running():
            raise ValueError("A conversion is already running.")
        self.progress_path.unlink(missing_ok=True)
        try:
            with open(self.log_path, "wb") as log:
                self._process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
        except OSError as error:
            raise ValueError(f"Could not start the converter: {error}") from error

    def stop(self) -> None:
        if self.running():
            try:
                os.killpg(self._process.pid, signal.SIGINT)
            except ProcessLookupError:  # it finished a moment ago
                pass

    def wait(self, timeout: float) -> None:
        if self._process is not None:
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired:
                pass

    def log_tail(self) -> list[str]:
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - LOG_TAIL_BYTES))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-LOG_TAIL_LINES:]

    def progress(self) -> dict | None:
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None


class App:
    """Everything the page can ask for. The HTTP layer below only translates requests to these methods."""

    def __init__(self, data_dir: Path, runner: tuple[str, ...] | None = None, script: Path = CONVERTER) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir = data_dir / "previews"
        self.store = SettingsStore(data_dir / "settings.json")
        self.job = Job(data_dir)
        self.books = BookCache()
        self.runner = runner or (sys.executable,)  # the same Python, so the converter has the same packages
        self.script = script
        self._start_lock = threading.Lock()

    def _count_chapters(self, epub: Path) -> int:
        try:
            return len(self.books.chapters(epub))
        except Exception as error:
            raise ValueError(f"Could not read this EPUB: {error}") from error

    def state(self) -> dict:
        return {"settings": self.store.snapshot(),
                "models": [{"id": model, "label": label} for model, label in converter.KNOWN_MODELS],
                "job": {"running": self.job.running(), "exit_code": self.job.exit_code()},
                "progress": self.job.progress(), "log": self.job.log_tail()}

    def save(self, raw: dict) -> dict:
        form = clean_form(raw, self._count_chapters)
        self.store.remember(form)
        return {"ok": True, "form": form}

    def start(self, raw: dict) -> dict:
        form = clean_form(raw, self._count_chapters)
        with self._start_lock:
            self.job.start(build_command(form, self.job.progress_path, self.runner, self.script))
            self.store.remember(form)
        return {"ok": True, "form": form}

    def preview(self, raw: dict) -> dict:
        """Speak the start of the first selected chapter with these settings, as a job the page can watch and stop."""
        form = clean_form(raw, self._count_chapters)
        target = preview_path(self.previews_dir.parent, form)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._start_lock:
            self.job.start(build_command(form, self.job.progress_path, self.runner, self.script, preview=target))
        return {"ok": True, "name": target.name}

    def previews(self, epub_path: str) -> list[dict]:
        """Samples made for this book, newest first; the model is read back from the file name."""
        if not epub_path or not self.previews_dir.is_dir():
            return []
        prefix = converter.default_output(Path(epub_path)).name + "__"
        files = [p for p in self.previews_dir.iterdir() if p.name.startswith(prefix) and p.suffix in AUDIO_TYPES]
        return [{"name": p.name, "model": p.stem[len(prefix):].replace("--", "/"), "mtime": p.stat().st_mtime}
                for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]

    def preview_file(self, name: str) -> Path | None:
        """Only a file directly inside the previews folder, never a path the browser made up."""
        candidate = self.previews_dir / Path(name).name
        if not name or Path(name).name != name or candidate.suffix not in AUDIO_TYPES or not candidate.is_file():
            return None
        return candidate

    def stop(self, _payload: dict | None = None) -> dict:
        self.job.stop()
        return {"ok": True}

    def book_info(self, epub_path: str, chapters_spec: str, output: str, audio_format: str) -> dict:
        """What the page shows under the form: chapter count, selection, and how much is already on disk."""
        epub = Path(epub_path).expanduser()
        if epub.suffix.lower() != ".epub" or not epub.is_file():
            return {"error": "Pick an existing .epub file."}
        try:
            total = self._count_chapters(epub)
            numbers = converter.parse_chapter_ranges(chapters_spec or None, total)
            extension = converter.make_output_format(audio_format).extension
        except ValueError as error:
            return {"error": str(error)}
        out_dir = Path(output).expanduser() if output else default_output(epub)
        done = converter.finished_numbers(out_dir, extension) if out_dir.is_dir() else set()
        first_unfinished = next((number for number in numbers if number not in done), None)
        return {"error": "", "title": epub.stem, "total": total, "selected": len(numbers),
                "done": len(done & set(numbers)), "first_unfinished": first_unfinished, "output": str(out_dir)}

    def browse(self, path: str, kind: str) -> dict:
        """List a folder on this computer for the file picker (the server only listens on localhost)."""
        folder = Path(path).expanduser() if path else Path.home()
        if folder.is_file():
            folder = folder.parent
        while not folder.is_dir() and folder != folder.parent:
            folder = folder.parent
        wanted = PICKER_SUFFIXES.get(kind, PICKER_SUFFIXES["epub"])
        try:
            children = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as error:
            raise ValueError(f"Cannot open {folder}: {error}") from error
        entries = [{"name": child.name, "dir": child.is_dir()} for child in children
                   if not child.name.startswith(".") and (child.is_dir() or child.suffix.lower() in wanted)]
        return {"path": str(folder), "parent": None if folder == folder.parent else str(folder.parent),
                "entries": entries[:MAX_LISTED_FILES]}


# --- HTTP ---------------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    app: App  # bound by make_server

    def log_message(self, format: str, *args) -> None:  # keep the terminal quiet
        pass

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _local_only(self) -> bool:
        if is_local_host(self.headers.get("Host")):
            return True
        self._json({"error": "forbidden host"}, 403)
        return False

    def do_GET(self) -> None:
        if not self._local_only():
            return
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self._reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/favicon.ico":  # browsers ask for it; answering keeps their console clean
                self._reply(204, b"", "image/x-icon")
            elif url.path == "/api/state":
                self._json(self.app.state())
            elif url.path == "/api/form":
                self._json(self.app.store.form_for(query.get("epub", "")))
            elif url.path == "/api/book":
                self._json(self.app.book_info(query.get("epub", ""), query.get("chapters", ""),
                                              query.get("output", ""), query.get("audio_format", "wav")))
            elif url.path == "/api/browse":
                self._json(self.app.browse(query.get("path", ""), query.get("kind", "epub")))
            elif url.path == "/api/previews":
                self._json(self.app.previews(query.get("epub", "")))
            elif url.path == "/api/preview-audio":
                path = self.app.preview_file(query.get("name", ""))
                if path is None:
                    self._json({"error": "not found"}, 404)
                else:
                    self._reply(200, path.read_bytes(), AUDIO_TYPES[path.suffix])
            else:
                self._json({"error": "not found"}, 404)
        except ValueError as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:  # a local tool: tell the page what went wrong instead of dropping the connection
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)

    def do_POST(self) -> None:
        if not self._local_only():
            return
        # Browsers only send application/json cross-site after a CORS preflight, which this server never grants.
        if self.headers.get_content_type() != "application/json":
            return self._json({"error": "JSON body required"}, 415)
        actions: dict[str, Callable[[dict], dict]] = {
            "/api/start": self.app.start, "/api/save": self.app.save, "/api/stop": self.app.stop,
            "/api/preview": self.app.preview}
        action = actions.get(urlparse(self.path).path)
        if action is None:
            return self._json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                return self._json({"error": "request too large"}, 413)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON object expected")
            self._json(action(payload))
        except ValueError as error:
            self._json({"error": str(error)}, 400)
        except Exception as error:
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    converter.enable_ctrl_c()  # also keeps the converter we launch stoppable: children inherit an ignored SIGINT
    parser = argparse.ArgumentParser(prog="epubvox web", description="Open the epubvox web page (local only).")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"default: {DEFAULT_PORT}")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="settings, progress and run log (default: ~/.local/state/epubvox)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)
    app = App(args.data_dir)
    try:
        server = make_server(app, args.port)
    except OSError as error:
        parser.error(f"cannot listen on port {args.port}: {error}")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"epubvox is running at {url}")
    print("Ctrl+C stops the server and safely stops any running conversion.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if app.job.running():
            print("Stopping the running conversion; start it again later to continue ...")
            app.job.stop()
            app.job.wait(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
