"""epubvox: turn an EPUB into a per-chapter audiobook in your own cloned voice (Qwen3-TTS on Apple Silicon)."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("epubvox")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"
