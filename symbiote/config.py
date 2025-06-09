"""TOML config loader with hot-reload."""

import pathlib
import typing as _t

import toml


class Config:
    """Loads and caches a TOML configuration file."""

    def __init__(self, path: str | pathlib.Path | None = None):
        if path is None:
            self.path = (
                pathlib.Path(__file__).with_suffix(".conf").with_name("symb.conf")
            )
        else:
            self.path = pathlib.Path(path)
        self._last_mtime = 0
        self._cache: dict[str, _t.Any] = {}

    def load(self) -> dict:
        """
        Load the config file.

        The file is re-read from disk if it has been modified.
        """
        if not self.path.exists():
            return {}
        mtime = self.path.stat().st_mtime
        if mtime != self._last_mtime:
            self._cache = toml.loads(self.path.read_text(encoding="utf-8"))
            self._last_mtime = mtime
        return self._cache


_config = Config()
load_config = _config.load
