"""Convert an EPUB into one audio file per chapter with Qwen3-TTS on Apple Silicon (MLX), cloning your voice.

    epubvox BOOK.epub                          whole book: m4a 32k, 1.5x, your voice
    epubvox -c 80- BOOK.epub                   chapter 80 to the end
    epubvox --speed 1 -f mp3 -b 48k -r other.wav BOOK.epub
    epubvox --replace fixes.txt BOOK.epub      pronunciation fixes
    epubvox -l BOOK.epub                       title, author and chapters

Only the book is required. The voice is $EPUBVOX_VOICE or ~/.config/epubvox/voice.wav, and the output folder is
./<title up to the first ，：！ or " - ">, e.g. ./贫道看事.
--chapters accepts 5, 1-100, 500- (to the end) and comma lists such as 1-10,50,60-70.
Chapter numbers are positions in the book (the 0001 prefix). A book that splits or skips chapters can drift from the
number in its titles, so check `--list`.
--format picks m4a (AAC: small, best for iPhone), mp3 or wav; --bitrate sets the size (m4a 32k, mp3 64k).
--speed is applied when encoding (pitch unchanged), so changing it never re-runs the model. Needs ffmpeg.
Files are tagged with album and artist (from the EPUB, or --album/--artist), the chapter title and the track number.
--replace FILE takes `search==replace` regex lines (# comments) applied to the text before it is spoken, e.g.
`重楼==虫楼` or `请收藏本站.*?。==` to drop an ad line; file names keep the book's own text.
Files are named 0001-<chapter title>.<ext>. Ctrl+C is safe: re-run the same command and finished chapters are skipped
while the interrupted chapter continues from its last finished chunk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ebooklib
import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import epub

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
KNOWN_MODELS = (  # (id, label) offered by the web page
    (MODEL_ID, "Qwen3-TTS 1.7B 8-bit (default)"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit", "Qwen3-TTS 1.7B 4-bit (smaller download)"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "Qwen3-TTS 1.7B bf16 (full precision, slowest)"),
)
PREVIEW_CHARS = 200
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "epubvox"
DEFAULT_REF_AUDIO = Path(os.environ.get("EPUBVOX_VOICE") or CONFIG_DIR / "voice.wav").expanduser()
DEFAULT_SPEED = 1.5
SPEED_RANGE = (0.5, 4.0)
BOOK_TITLE_END = re.compile(r"\s*(?:[，,：:！!？?（(【\[]|\s-\s)")  # "贫道看事 ，只杀不渡！" -> "贫道看事"
SAMPLE_RATE = 24000
CHUNK_CHARS = 180
GAP_SECONDS = 0.4
MAX_TOKENS = 4096
MAX_ATTEMPTS = 3
MIN_FALLBACK_CHARS = 500
MIN_HEADED_CHAPTERS = 3

TEXT_TAGS = ["h1", "h2", "h3", "h4", "p", "li"]
CHAPTER_HEADING = re.compile(r"^(第\s*[0-9零〇一二三四五六七八九十百千万两]+\s*[章回节節]|chapter\s+\d+)", re.IGNORECASE)
SENTENCE = re.compile(r"[^。！？!?…；;.]*[。！？!?…；;.]+[”’」』）)\"']*|[^。！？!?…；;.]+$")
ENDINGS = tuple("。！？!?…；;.”’」』）)\"'")

Synth = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class Chapter:
    number: int
    title: str
    paragraphs: tuple[str, ...]  # paragraphs[0] is the title


# --- EPUB -> chapters -------------------------------------------------------------------------------------------

def _document_paragraphs(item) -> tuple[str, ...]:
    soup = BeautifulSoup(item.get_body_content(), "lxml")
    tags = [tag for tag in soup.find_all(TEXT_TAGS) if not tag.find(TEXT_TAGS)]
    texts = (re.sub(r"\s+", " ", tag.get_text()).strip() for tag in tags)
    return tuple(text for text in texts if text)


def _spine_documents(book: epub.EpubBook) -> list[tuple[str, tuple[str, ...]]]:
    documents = []
    for entry in book.spine:
        item = book.get_item_with_id(entry[0] if isinstance(entry, (tuple, list)) else entry)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            paragraphs = _document_paragraphs(item)
            if paragraphs:
                documents.append((item.get_name(), paragraphs))
    return documents


def extract_chapters(epub_path: Path) -> list[Chapter]:
    """Chapters are the spine documents that open with a chapter heading (第N章 / Chapter N).

    Without recognizable headings, every substantial non-navigation document counts as a chapter.
    """
    documents = _spine_documents(epub.read_epub(str(epub_path)))
    chosen = [doc for doc in documents if CHAPTER_HEADING.match(doc[1][0])]
    if len(chosen) < MIN_HEADED_CHAPTERS:
        chosen = [doc for doc in documents
                  if sum(map(len, doc[1])) >= MIN_FALLBACK_CHARS
                  and not Path(doc[0]).name.lower().startswith(("nav", "toc"))]
    return [Chapter(number, paragraphs[0], paragraphs) for number, (_, paragraphs) in enumerate(chosen, start=1)]


def read_book_info(epub_path: Path) -> tuple[str, str]:
    """(title, author) from the EPUB's Dublin Core metadata; the file name and "" when missing."""
    book = epub.read_epub(str(epub_path))

    def first(name: str) -> str:
        values = book.get_metadata("DC", name)
        return values[0][0].strip() if values and values[0][0] else ""

    return first("title") or epub_path.stem, first("creator")


# --- pronunciation fixes ----------------------------------------------------------------------------------------

Rules = list[tuple[re.Pattern, str]]


def load_replacements(path: Path) -> Rules:
    """One `search==replace` rule per line (search is a regex; an empty replace deletes); # starts a comment."""
    rules = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        search, separator, replacement = line.partition("==")
        if not separator or not search:
            raise ValueError(f"{path}:{line_number}: expected search==replace")
        try:
            rules.append((re.compile(search), replacement))
        except re.error as error:
            raise ValueError(f"{path}:{line_number}: bad pattern: {error}") from None
    return rules


def with_replacements(synth: Synth, rules: Rules) -> Synth:
    """Rewrite each chunk just before it is spoken; file names and titles on disk stay as written in the book."""
    if not rules:
        return synth

    def replaced(text: str) -> np.ndarray:
        for pattern, replacement in rules:
            text = pattern.sub(replacement, text)
        return synth(text) if text.strip() else np.zeros(0, dtype=np.float32)

    return replaced


# --- selection and naming ---------------------------------------------------------------------------------------

def parse_chapter_ranges(spec: str | None, total: int) -> list[int]:
    """'1-3,7,10-' -> [1, 2, 3, 7, 10, ...total]; None -> every chapter."""
    if not spec:
        return list(range(1, total + 1))
    numbers: set[int] = set()
    for part in (piece.strip() for piece in spec.split(",")):
        match = re.fullmatch(r"(\d+)(-(\d*))?", part)
        if not match:
            raise ValueError(f"bad chapter range {part!r}; use e.g. 5, 1-100, 500- or 1-10,50")
        start = int(match.group(1))
        end = start if not match.group(2) else int(match.group(3) or total)
        if start < 1 or end < start or end > total:
            raise ValueError(f"chapter range {part!r} is outside 1-{total}")
        numbers.update(range(start, end + 1))
    return sorted(numbers)


def safe_filename(title: str, max_len: int = 80) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", title)
    return re.sub(r"\s+", " ", cleaned).strip(" .")[:max_len].rstrip(" .") or "untitled"


def chapter_filename(chapter: Chapter, extension: str = ".wav") -> str:
    return f"{chapter.number:04d}-{safe_filename(chapter.title)}{extension}"


def finished_numbers(out_dir: Path, extension: str = ".wav") -> set[int]:
    """A chapter is finished when a file with the target extension exists (.part files never match)."""
    return {int(path.name[:4]) for path in out_dir.glob(f"[0-9][0-9][0-9][0-9]-*{extension}")}


# --- output formats ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputFormat:
    name: str
    extension: str
    encoder: str | None = None  # ffmpeg encoder; None means a plain WAV written with soundfile
    bitrate: str | None = None
    muxer_args: tuple[str, ...] = ()
    speed: float = 1.0  # applied at encode time (Qwen3-TTS ignores its own speed argument), so the cache stays 1x


WAV = OutputFormat("wav", ".wav")
FORMAT_NAMES = ("wav", "mp3", "m4a")
DEFAULT_BITRATES = {"mp3": "64k", "m4a": "32k"}
FFMPEG_ENCODERS = {
    "mp3": ("libmp3lame", ("-f", "mp3")),
    "m4a": ("aac", ("-f", "ipod", "-movflags", "+faststart")),
}
BITRATE = re.compile(r"\d{1,3}k")


def make_output_format(name: str, bitrate: str | None = None, speed: float = 1.0) -> OutputFormat:
    if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
        raise ValueError(f"--speed must be between {SPEED_RANGE[0]} and {SPEED_RANGE[1]}")
    if name == "wav":
        if bitrate:
            raise ValueError("--bitrate only applies to mp3 and m4a")
        # soundfile cannot change the tempo, so a sped-up WAV goes through ffmpeg too
        return WAV if speed == 1.0 else OutputFormat("wav", ".wav", "pcm_s16le", None, ("-f", "wav"), speed)
    if name not in FFMPEG_ENCODERS:
        raise ValueError(f"unknown format {name!r}; choose from {', '.join(FORMAT_NAMES)}")
    bitrate = bitrate or DEFAULT_BITRATES[name]
    if not BITRATE.fullmatch(bitrate):
        raise ValueError(f"bad bitrate {bitrate!r}; use e.g. 32k, 48k or 64k")
    encoder, muxer_args = FFMPEG_ENCODERS[name]
    return OutputFormat(name, f".{name}", encoder, bitrate, muxer_args, speed)


def default_output(epub_path: Path) -> Path:
    """./<book title without its subtitle>, so a long file name still maps to a short, stable folder."""
    short = BOOK_TITLE_END.split(epub_path.stem, maxsplit=1)[0].strip()
    return Path(short or epub_path.stem)


# --- chunking ---------------------------------------------------------------------------------------------------

def sentence_mark(lang: str) -> str:
    return "。" if lang in ("chinese", "auto") else "."


def _terminate(text: str, mark: str) -> str:
    return text if text.endswith(ENDINGS) else text + mark


def split_chunks(chapter: Chapter, mark: str = "。", max_chars: int = CHUNK_CHARS) -> list[str]:
    """The title alone, then the body packed sentence by sentence into chunks of up to max_chars."""
    sentences = [s for paragraph in chapter.paragraphs[1:] for s in SENTENCE.findall(_terminate(paragraph, mark))]
    chunks, current = [_terminate(chapter.title, mark)], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = ""
        current += sentence
    return chunks + ([current] if current else [])


# --- cache and atomic writes ------------------------------------------------------------------------------------

def settings_fingerprint(model_id: str, ref_audio: Path, ref_text: str | None, lang: str, rules: str = "") -> str:
    """Cached audio is only reusable while the voice, model, language and pronunciation rules are unchanged."""
    digest = hashlib.sha256()
    for part in (model_id, ref_text or "", lang):
        digest.update(part.encode("utf-8") + b"\0")
    digest.update(ref_audio.read_bytes())
    if rules:  # only when used, so caches made before --replace existed stay valid
        digest.update(b"rules\0" + rules.encode("utf-8"))
    return digest.hexdigest()[:12]


def chunk_path(cache_dir: Path, number: int, index: int, text: str) -> Path:
    return cache_dir / f"{number:04d}_{index:03d}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}.npy"


def replace_atomically(write: Callable[[Path], None], target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    write(partial)
    os.replace(partial, target)


def save_chunk(path: Path, audio: np.ndarray) -> None:
    def write(partial: Path) -> None:
        with open(partial, "wb") as handle:
            np.save(handle, audio)

    replace_atomically(write, path)


def save_wav(path: Path, audio: np.ndarray) -> None:
    replace_atomically(lambda partial: sf.write(partial, audio, SAMPLE_RATE, format="WAV"), path)


def encode_with_ffmpeg(path: Path, audio: np.ndarray, fmt: OutputFormat, tags: dict[str, str] | None = None) -> None:
    """Pipe raw mono float32 audio to ffmpeg, so no intermediate WAV is written."""
    metadata = [arg for key, value in (tags or {}).items() if value for arg in ("-metadata", f"{key}={value}")]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
               *(["-af", f"atempo={fmt.speed}"] if fmt.speed != 1.0 else []),  # faster, same pitch
               "-c:a", fmt.encoder, *(["-b:a", fmt.bitrate] if fmt.bitrate else []), *metadata,
               *fmt.muxer_args, str(path)]
    result = subprocess.run(command, input=np.asarray(audio, dtype="<f4").tobytes(), capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not encode {fmt.name} at {fmt.bitrate}: {detail}")


def save_audio(path: Path, audio: np.ndarray, fmt: OutputFormat, tags: dict[str, str] | None = None) -> None:
    """tags (album, artist, title, track) are written by ffmpeg; a plain 1x WAV has none."""
    if fmt.encoder is None:
        save_wav(path, audio)
    else:
        replace_atomically(lambda partial: encode_with_ffmpeg(partial, audio, fmt, tags), path)


def check_encoder(fmt: OutputFormat) -> None:
    """Fail now, not hours into a run, if ffmpeg or the requested encoder and bitrate are unusable."""
    if fmt.encoder is None:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found: install it with `brew install ffmpeg`, or use --format wav")
    with tempfile.TemporaryDirectory() as tmp:
        encode_with_ffmpeg(Path(tmp) / f"probe{fmt.extension}", np.zeros(SAMPLE_RATE, dtype=np.float32), fmt)


def _load_chunk(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError):  # damaged cache entry: synthesize it again
        path.unlink(missing_ok=True)
        return None


# --- synthesis --------------------------------------------------------------------------------------------------

def _is_valid(audio: np.ndarray) -> bool:
    return audio.size > 0 and bool(np.isfinite(audio).all()) and float(np.abs(audio).max()) > 1e-3


def _generate(model, text: str, ref_audio: Path, ref_text: str | None, lang: str) -> tuple[np.ndarray, bool]:
    parts, truncated = [], False
    for result in model.generate(text=text, ref_audio=str(ref_audio), ref_text=ref_text,
                                 lang_code=lang, max_tokens=MAX_TOKENS):
        if result.sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"unexpected sample rate {result.sample_rate}, expected {SAMPLE_RATE}")
        parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        truncated = truncated or result.token_count >= MAX_TOKENS
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), truncated


def make_synth(model_id: str, ref_audio: Path, ref_text: str | None, lang: str) -> Synth:
    model = None  # loaded on first use, so re-running a finished book returns instantly

    def synth(text: str) -> np.ndarray:
        nonlocal model
        if model is None:
            from mlx_audio.tts.utils import load_model  # heavy: import only when a book is really converted

            print(f"Loading {model_id} ...", flush=True)
            model = load_model(model_id)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            audio, truncated = _generate(model, text, ref_audio, ref_text, lang)
            if _is_valid(audio) and not truncated:
                return audio
            print(f"\n    warning: unusable audio (attempt {attempt}/{MAX_ATTEMPTS}) for {text[:30]!r}", flush=True)
        raise RuntimeError(f"{model_id} returned unusable audio {MAX_ATTEMPTS} times for {text[:60]!r}")

    return synth


def render_chapter(chapter: Chapter, synth: Synth, cache_dir: Path, mark: str = "。",
                   on_chunk: Callable[[int, int], None] = lambda index, total: None) -> np.ndarray:
    chunks = split_chunks(chapter, mark)
    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    on_chunk(0, len(chunks))
    for index, text in enumerate(chunks, start=1):
        path = chunk_path(cache_dir, chapter.number, index, text)
        audio = _load_chunk(path)
        if audio is None:
            audio = synth(text)
            save_chunk(path, audio)
        pieces += [audio, gap]
        on_chunk(index, len(chunks))
    return np.concatenate(pieces)


def preview_chapter(chapter: Chapter, max_chars: int = PREVIEW_CHARS) -> Chapter:
    """The title plus whole paragraphs until about max_chars of text, so a sample ends on a sentence."""
    paragraphs, length = [], 0
    for paragraph in chapter.paragraphs:
        if length >= max_chars:
            break
        paragraphs.append(paragraph)
        length += len(paragraph)
    return Chapter(chapter.number, chapter.title, tuple(paragraphs))


def write_preview(chapter: Chapter, synth: Synth, target: Path, fmt: OutputFormat, mark: str = "。") -> None:
    """A short sample made exactly like a real chapter (voice, speed, rules, format), without touching the book's cache."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as cache:
        audio = render_chapter(preview_chapter(chapter), synth, Path(cache), mark)
    save_audio(target, audio, fmt)


# --- book-level driver and CLI ----------------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m"


def _show_progress(index: int, total: int) -> None:
    if sys.stdout.isatty():
        print(f"    chunk {index}/{total}", end="\r", flush=True)


class ProgressFile:
    """Publishes the run state as JSON, replaced atomically so a reader never sees a half-written file."""

    def __init__(self, path: Path | None, extra: dict | None = None) -> None:
        self.path = path
        self.extra = extra or {}  # fixed facts about the run (book, output folder), repeated in every update
        self.state: dict = {}

    def update(self, state: dict) -> None:
        self.state = state
        if self.path is not None:
            payload = json.dumps({**self.extra, **state, "updated": time.time()}, ensure_ascii=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            replace_atomically(lambda partial: partial.write_text(payload, encoding="utf-8"), self.path)

    def finish(self, status: str, message: str = "") -> None:
        self.update({**self.state, "state": status, "message": message})


def _chunk_reporter(state: dict, chapter: Chapter, on_progress: Callable[[dict], None]) -> Callable[[int, int], None]:
    def report(index: int, total: int) -> None:
        _show_progress(index, total)
        on_progress({**state, "chapter": {"number": chapter.number, "title": chapter.title,
                                          "chunk": index, "chunks": total}})

    return report


def convert_book(chapters: list[Chapter], numbers: list[int], out_dir: Path, cache_dir: Path,
                 synth: Synth, mark: str = "。", fmt: OutputFormat = WAV,
                 log: Callable[[str], None] = print,
                 on_progress: Callable[[dict], None] = lambda state: None,
                 album: str = "", artist: str = "") -> int:
    """Convert the selected chapters, skipping finished ones. Returns how many were converted.

    on_progress receives a fresh state dict after every chunk and every chapter (see ProgressFile).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = finished_numbers(out_dir, fmt.extension)
    todo = [number for number in numbers if number not in done]
    label = f"{fmt.name} {fmt.bitrate}" if fmt.bitrate else fmt.name
    log(f"{len(numbers)} chapters selected: {len(numbers) - len(todo)} already done, {len(todo)} to convert ({label})")
    started = time.time()
    state = {"state": "running", "started": started, "selected": len(numbers),
             "already_done": len(numbers) - len(todo), "todo": len(todo), "converted": 0,
             "chapter": None, "seconds_per_chapter": None, "eta_seconds": None}
    on_progress(state)
    for position, number in enumerate(todo, start=1):
        chapter = chapters[number - 1]
        log(f"[{number:04d}/{len(chapters)}] {chapter.title}")
        cached = len(list(cache_dir.glob(f"{number:04d}_*.npy")))
        if cached:
            log(f"    resuming: {cached} chunks already finished")
        audio = render_chapter(chapter, synth, cache_dir, mark, _chunk_reporter(state, chapter, on_progress))
        target = out_dir / chapter_filename(chapter, fmt.extension)
        save_audio(target, audio, fmt, {"album": album, "artist": artist, "title": chapter.title,
                                        "track": f"{number}/{len(chapters)}"})
        for stale in cache_dir.glob(f"{number:04d}_*.npy"):
            stale.unlink()
        average = (time.time() - started) / position
        log(f"    done: {len(audio) / SAMPLE_RATE / 60:.1f} min audio, {target.stat().st_size / 1e6:.1f} MB"
            f" | avg {average:.0f}s/chapter | ETA {format_duration(average * (len(todo) - position))}")
        state = {**state, "converted": position, "chapter": None,
                 "seconds_per_chapter": average, "eta_seconds": average * (len(todo) - position)}
        on_progress(state)
    return len(todo)


def enable_ctrl_c() -> None:
    """A shell that starts us in the background leaves SIGINT ignored, and children inherit that.

    Take Ctrl+C back, or Stop in the web UI (which sends SIGINT) would silently do nothing.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)


def build_parser() -> argparse.ArgumentParser:
    from epubvox import __version__

    parser = argparse.ArgumentParser(
        prog="epubvox",
        description="Convert an EPUB into one audio file per chapter, read in your cloned voice by Qwen3-TTS (MLX). "
                    "Resumable: after Ctrl+C, run the same command again.",
        epilog="The web page has the same options: epubvox web [--port N] [--no-open]",
    )
    parser.add_argument("epub", type=Path, metavar="BOOK.epub")
    parser.add_argument("-V", "--version", action="version", version=f"epubvox {__version__}")
    parser.add_argument("-l", "--list", action="store_true", help="print the title, author and chapters, then exit")
    parser.add_argument("-c", "--chapters", metavar="RANGE",
                        help="5, 1-100, 500- or 1-10,50 (default: the whole book); positions as shown by --list")
    parser.add_argument("-o", "--output", type=Path, metavar="DIR",
                        help="output folder (default: ./<book title up to the first ，：！ or ' - '>)")
    parser.add_argument("-r", "--ref-audio", type=Path, default=DEFAULT_REF_AUDIO, metavar="FILE",
                        help="voice clip to clone (default: $EPUBVOX_VOICE or ~/.config/epubvox/voice.wav)")
    parser.add_argument("--ref-text", metavar="TEXT",
                        help="exact words spoken in the clip; makes the cloned voice closer")
    parser.add_argument("-f", "--format", dest="audio_format", choices=FORMAT_NAMES, default="m4a",
                        help="m4a (default; AAC, best for iPhone), mp3 or wav (lossless)")
    parser.add_argument("-b", "--bitrate", metavar="RATE", help="for m4a/mp3, e.g. 24k, 32k, 48k (m4a 32k, mp3 64k)")
    parser.add_argument("-s", "--speed", type=float, default=DEFAULT_SPEED,
                        help=f"reading speed, pitch unchanged (default: {DEFAULT_SPEED}; 1 = natural pace)")
    parser.add_argument("--replace", type=Path, metavar="FILE",
                        help="pronunciation fixes: one `search==replace` regex per line, # for comments")
    parser.add_argument("--album", help="album tag (default: the book title in the EPUB)")
    parser.add_argument("--artist", help="artist tag (default: the author in the EPUB)")
    parser.add_argument("--lang", default="chinese", help="Qwen3-TTS language (default: chinese)")
    parser.add_argument("-m", "--model", default=MODEL_ID, metavar="ID",
                        help=f"Qwen3-TTS model (default: {MODEL_ID}; also ...-Base-4bit and ...-Base-bf16)")
    parser.add_argument("--preview", type=Path, metavar="FILE",
                        help=f"speak the first ~{PREVIEW_CHARS} characters of the first selected chapter into FILE, "
                             "then exit")
    parser.add_argument("--progress-file", type=Path, help=argparse.SUPPRESS)  # written for the web page
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_ctrl_c()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.epub.is_file():
        parser.error(f"EPUB not found: {args.epub}")
    chapters = extract_chapters(args.epub)
    if not chapters:
        parser.error("no chapters found in this EPUB")
    title, author = read_book_info(args.epub)
    if args.list:
        print(f"{title} / {author or 'unknown author'}")
        for chapter in chapters:
            print(f"{chapter.number:04d}  {sum(map(len, chapter.paragraphs)):>6,} chars  {chapter.title}")
        print(f"{len(chapters)} chapters")
        return 0
    if not args.ref_audio.is_file():
        parser.error(f"voice clip not found: {args.ref_audio} (put your clip there, set EPUBVOX_VOICE, "
                     "or pass -r FILE)")
    try:
        numbers = parse_chapter_ranges(args.chapters, len(chapters))
    except ValueError as error:
        parser.error(str(error))
    try:
        fmt = make_output_format(args.audio_format, args.bitrate, args.speed)
        check_encoder(fmt)
        rules_text = args.replace.read_text(encoding="utf-8") if args.replace else ""
        rules = load_replacements(args.replace) if args.replace else []
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    out_dir = args.output or default_output(args.epub)
    cache_dir = out_dir / ".cache" / settings_fingerprint(args.model, args.ref_audio, args.ref_text, args.lang,
                                                          rules_text)
    if not args.ref_text:
        print("note: no --ref-text, so cloning uses the speaker embedding only (less faithful to the voice).")
    progress = ProgressFile(args.progress_file, {"book": args.epub.stem, "output": str(out_dir), "format": fmt.name,
                                                 "preview": bool(args.preview)})
    try:
        synth = with_replacements(make_synth(args.model, args.ref_audio, args.ref_text, args.lang), rules)
        if args.preview:
            write_preview(chapters[numbers[0] - 1], synth, args.preview, fmt, sentence_mark(args.lang))
            progress.finish("finished")
            print(f"Preview saved: {args.preview}")
            return 0
        convert_book(chapters, numbers, out_dir, cache_dir, synth, mark=sentence_mark(args.lang), fmt=fmt,
                     on_progress=progress.update, album=args.album or title, artist=args.artist or author)
    except KeyboardInterrupt:
        progress.finish("interrupted")
        print("\nInterrupted. Re-run the same command to resume: finished chapters are skipped and the current "
              "chapter continues from its last finished chunk.")
        return 130
    except Exception as error:
        progress.finish("error", str(error))
        raise
    progress.finish("finished")
    print(f"Finished. Files are in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
