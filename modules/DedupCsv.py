from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock


@dataclass
class DedupHit:
    sha256: str
    voiced_path: str


class DedupCsv:
    """Ultra-simple dedup registry.

    File format (UTF-8):
        sha256;path_to_voiced_file

    Reads are best-effort. Writes are protected by FileLock.
    If sha256 appears multiple times, the *last* entry wins.
    """

    def __init__(self, dedup_csv: str | Path):
        self.path = Path(dedup_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path) + ".lock")

    @staticmethod
    def sha256(path: str | Path, text: str, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        h.update(b"\x00TEXT\x00")
        h.update((text or "").encode("utf-8", errors="replace"))
        return h.hexdigest()

    def load_map(self) -> dict[str, str]:
        m: dict[str, str] = {}
        if not self.path.exists():
            return m
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(";", 1)
                    if len(parts) != 2:
                        continue
                    sha, voiced = parts[0].strip(), parts[1].strip()
                    if not sha or not voiced:
                        continue
                    m[sha] = voiced
        except Exception:
            # best-effort reader
            return m
        return m

    def append(self, sha256: str, voiced_path: str):
        voiced_path = str(voiced_path)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"{sha256};{voiced_path}\n")

    @staticmethod
    def link_or_copy(src: str | Path, dst: str | Path):
        src_p = Path(src)
        dst_p = Path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)

        if dst_p.exists():
            try:
                dst_p.unlink()
            except Exception:
                pass

        shutil.copy2(src_p, dst_p)
