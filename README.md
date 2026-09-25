# epubvox

Turn an EPUB into an audiobook, one file per chapter, read in **your own voice**. It runs
[Qwen3-TTS](https://huggingface.co/Qwen) locally on an Apple Silicon Mac through [MLX](https://github.com/ml-explore/mlx)
(via [mlx-audio](https://github.com/Blaizzy/mlx-audio)): no cloud service, no GPU setup. Tuned for long Chinese web
novels, and works for other languages Qwen3-TTS speaks.

```console
$ epubvox book.epub          # the whole book, in your voice
$ epubvox web                # or the same in a local web page
```

- **Your voice:** clones any short clip of the voice you want.
- **Resumable:** stop any time (Ctrl+C or the Stop button); run it again and it continues from the last finished
  sentence group (about 180 characters).
- **iPhone-friendly output:** `m4a` (default), `mp3` or `wav` at the bitrate you choose, named
  `0001-第1章 丹田被毁.m4a` and tagged with album, artist, chapter title and track number.
- **1.5× by default:** any speed from 0.5× to 4×, pitch unchanged.
- **Preview:** hear the start of a chapter with the current voice, speed and model before converting a whole book.
- **Pronunciation fixes:** a small rules file rewrites words the voice gets wrong, or removes ad lines.

## Credits

epubvox started as a fork of **[audiblez](https://github.com/santinic/audiblez)** by **Claudio Santini**, which turns
EPUBs into audiobooks with Kokoro-82M. It is rewritten around Qwen3-TTS voice cloning on MLX and narrowed to EPUB
conversion, and it keeps audiblez's MIT license (see [LICENSE](LICENSE)). Thank you, Claudio.

## Install

You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and ffmpeg:

```bash
brew install uv ffmpeg
uv tool install git+https://github.com/tubalu/epubvox
```

Then put a clean recording of the voice you want (a few seconds to half a minute) at `~/.config/epubvox/voice.wav`,
or point `EPUBVOX_VOICE` at it, or pass `-r FILE` each time:

```bash
mkdir -p ~/.config/epubvox && cp my-voice.wav ~/.config/epubvox/voice.wav
```

The first conversion downloads the model (`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`) into the Hugging Face cache.
Update with `uv tool upgrade epubvox`; remove with `uv tool uninstall epubvox`.

To try it without installing: `uvx --from git+https://github.com/tubalu/epubvox epubvox web`.

## Command line

```bash
epubvox book.epub                        # whole book: m4a 32k, 1.5x, your voice, into ./<book title>/
epubvox -l book.epub                     # title, author and the chapters it found
epubvox -c 80- book.epub                 # chapter 80 to the end
epubvox -s 1 -f mp3 -b 48k book.epub     # natural speed, mp3 at 48k
epubvox --replace fixes.txt book.epub    # with pronunciation fixes
epubvox -h                               # every option
```

Quote paths with spaces or punctuation: `epubvox "贫道看事，只杀不渡！.epub"`.

| Option | Meaning |
|---|---|
| `-l`, `--list` | print the title, author and chapters, then exit |
| `-c`, `--chapters RANGE` | `5`, `1-100`, `500-` (to the end) or `1-10,50`; default is the whole book |
| `-o`, `--output DIR` | output folder; default `./<title up to the first ，：！ or " - ">`, e.g. `./贫道看事` |
| `-r`, `--ref-audio FILE` | the voice to clone; default `$EPUBVOX_VOICE` or `~/.config/epubvox/voice.wav` |
| `--ref-text TEXT` | the exact words spoken in that clip (optional; the voice matches better) |
| `-f`, `--format` | `m4a` (default, AAC, best for iPhone), `mp3` or `wav` (lossless) |
| `-b`, `--bitrate RATE` | for m4a/mp3, e.g. `24k`, `32k`, `48k` (defaults: m4a 32k, mp3 64k) |
| `-s`, `--speed N` | reading speed, default `1.5`; `1` is the natural pace |
| `--replace FILE` | pronunciation fixes, see below |
| `--album`, `--artist` | tags; default to the title and author in the EPUB |
| `--lang` | Qwen3-TTS language, default `chinese` |
| `-m`, `--model ID` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` (default), `…-Base-4bit` (smaller) or `…-Base-bf16` |
| `--preview FILE` | speak the first ~200 characters of the first selected chapter into FILE, then exit |
| `-V`, `--version` | print the version |

Chapter numbers are **positions in the book** (the `0001` in the file name). A book that splits or skips chapters can
drift from the number in its titles, so check `-l`. Exit status is 0 on success, 2 for a usage error and 130 after
Ctrl+C.

### Pronunciation fixes

A plain `.txt` file, one rule per line: `search==replace`. `search` is a regular expression, an empty `replace`
deletes, and lines starting with `#` are comments. Rules change only what is spoken; file names keep the book's text.

```
# names the voice reads wrong
重楼==虫楼
# drop the site's ad line
请收藏本站.*?。==
```

## Web page

```bash
epubvox web                  # opens http://127.0.0.1:8765 (-p PORT, --no-open)
```

- **Book**, **voice** and **pronunciation fixes** have Browse… pickers, and there is a list of recent books.
- Settings are remembered in `~/.local/state/epubvox/`. Pick a book you ran before and everything comes back exactly
  as you ran it, so continuing is "change the chapter range, press Start". A new book starts with your voice, format,
  bitrate, speed, rules and model, but a fresh chapter range and output folder.
- Under the form: how many chapters the book has, how many are selected and already converted, and a link to continue
  from the first unfinished one.
- **Preview** speaks the start of the first selected chapter with the current settings. Each model keeps its own sample
  per book, with a player under the form, so you can compare them by ear.
- The Progress panel shows chapters done, the current chapter part by part, elapsed time, time per chapter, time left
  and the converter's log.
- The server only listens on your own computer (127.0.0.1). Ctrl+C stops it and safely stops a running conversion.

## Stopping and continuing

Ctrl+C in the terminal, or Stop in the web page, is always safe. Run the same command (or press Start) again: finished
chapters are skipped and the interrupted chapter continues from its last finished chunk. A chapter counts as finished
when a file with the same extension is in the output folder, so changing the format or folder starts those chapters
over. Partial work lives in `<output>/.cache/` and is removed chapter by chapter.

Speed is applied when each file is encoded, and the cache holds natural-pace audio, so changing `--speed` never re-runs
the model. Changing the voice, reference text, language, model or rules file starts the interrupted chapter over.

## Sizes and speed

One 10.25-minute chapter at natural pace (at the default 1.5× a file is about ⅔ of this):

| Format | Size |
|---|---|
| wav | 29.5 MB |
| mp3 64k / 48k / 32k | 4.9 / 3.7 / 2.5 MB |
| m4a 48k / 32k / 24k | 3.8 / 2.5 / 1.9 MB |

On an M2 Ultra a chapter of that length takes about 3 minutes to synthesize; other Macs will differ.

## Development

```bash
git clone https://github.com/tubalu/epubvox && cd epubvox
uv run epubvox -h                                # runs from the source tree
uv run python -m unittest discover -s test       # tests
```

| Path | What it is |
|---|---|
| `src/epubvox/convert.py` | the converter and its command line |
| `src/epubvox/web.py`, `web.html` | the local web server (standard library only) and its page |
| `src/epubvox/cli.py` | the `epubvox` command: `web` goes to the web page, everything else to the converter |
| `test/` | unit tests |

## License

MIT, as audiblez. See [LICENSE](LICENSE).
