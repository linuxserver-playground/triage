"""The read-only knowledge the model searches: the hub's `data` branch and the site's docs, each a shallow clone
refreshed hourly, so the bot never adds to the Pages rate limit other readers share. CONVENTIONS.md and the
image index go into the cached system prompt."""

import json
import logging
import re
import subprocess
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)

DOCS_PATH = "src/content/docs"
MAX_READ = 30_000
MAX_HITS = 60


class Sources:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.hub = settings.cache_dir / "hub"
        self.site = settings.cache_dir / "site"

    @property
    def roots(self) -> dict[str, Path]:
        return {"hub": self.hub, "docs": self.site / DOCS_PATH}

    def refresh(self) -> None:
        """Clone or fetch both trees. A failure keeps the copy already on disk; the next refresh tries again."""
        base = f"https://github.com/{self.settings.org}"
        self._sync(self.hub, f"{base}/{self.settings.hub_repo}.git", "data", sparse=None)
        self._sync(self.site, f"{base}/{self.settings.site_repo}.git", "main", sparse=DOCS_PATH)

    def _sync(self, path: Path, url: str, branch: str, sparse: str | None) -> None:
        def git(*args: str, cwd: Path | None = None) -> None:
            subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=600)

        try:
            if not (path / ".git").exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                extra = ["--filter=blob:none", "--sparse"] if sparse else []
                git("clone", "--quiet", "--depth", "1", "--single-branch", "--branch", branch, *extra, url, str(path))
                if sparse:
                    git("sparse-checkout", "set", sparse, cwd=path)
            else:
                # The hub's data branch is one root commit replaced on every publish, so fetch and reset, then drop
                # the old commit so the clone never grows.
                git("fetch", "--quiet", "--depth", "1", "origin", branch, cwd=path)
                git("reset", "--quiet", "--hard", "FETCH_HEAD", cwd=path)
                git("reflog", "expire", "--expire=now", "--all", cwd=path)
                git("gc", "--quiet", "--prune=now", cwd=path)
            log.info("refreshed %s", url)
        except (subprocess.SubprocessError, OSError) as exc:
            detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else exc
            log.warning("could not refresh %s: %s", url, detail)

    # -- what the tools read --------------------------------------------------------------------------------------

    def resolve(self, source: str, path: str) -> Path:
        root = self.roots.get(source)
        if root is None:
            raise ValueError(f"unknown source {source!r}: use hub or docs")
        target = (root / path.lstrip("/")).resolve()
        if target != root.resolve() and root.resolve() not in target.parents:
            raise ValueError("path leaves the source")
        if ".git" in target.relative_to(root.resolve()).parts:
            raise ValueError("path leaves the source")
        return target

    def list_dir(self, source: str, path: str = "") -> str:
        target = self.resolve(source, path)
        if not target.is_dir():
            raise ValueError(f"{path or '/'} is not a directory")
        entries = sorted(p for p in target.iterdir() if p.name != ".git")
        lines = [f"{p.name}/" if p.is_dir() else f"{p.name}\t{p.stat().st_size} bytes" for p in entries]
        return "\n".join(lines) or "(empty)"

    def read_file(self, source: str, path: str, offset: int = 0) -> str:
        target = self.resolve(source, path)
        if not target.is_file():
            raise ValueError(f"{path} is not a file")
        text = target.read_text(errors="replace")
        chunk = text[offset : offset + MAX_READ]
        if offset + MAX_READ < len(text):
            chunk += (
                f"\n[... {len(text) - offset - MAX_READ} more characters: read again with offset {offset + MAX_READ}]"
            )
        return chunk

    def grep(self, source: str, pattern: str, path: str = "") -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        target = self.resolve(source, path)
        root = self.roots[source].resolve()
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        hits: list[str] = []
        for file in files:
            if ".git" in file.relative_to(root).parts or file.suffix in (".png", ".jpg", ".webp", ".gz"):
                continue
            try:
                lines = file.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    hits.append(f"{file.relative_to(root)}:{number}: {line.strip()[:300]}")
                    if len(hits) >= MAX_HITS:
                        return "\n".join(hits) + f"\n[stopped at {MAX_HITS} matches: narrow the pattern or the path]"
        return "\n".join(hits) or "no matches"

    # -- what the system prompt carries --------------------------------------------------------------------------

    def image_index(self) -> str:
        """One line per image: name, kind, live branches and the start of its summary."""
        images = self.hub / "images"
        if not images.is_dir():
            return "(the hub's data is not available yet)"
        lines = []
        for folder in sorted(p for p in images.iterdir() if p.is_dir()):
            manifests = sorted(folder.glob("manifest.*.json"))
            branches = [m.name.removeprefix("manifest.").removesuffix(".json") for m in manifests]
            kind, summary = "app", ""
            if manifests:
                try:
                    data = json.loads(manifests[0].read_text())
                    kind = str(data.get("kind") or "app")
                    summary = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", str(data.get("summary") or ""))[:110]
                except (OSError, ValueError):
                    pass
            lines.append(f"{folder.name} | {kind} | {', '.join(branches)} | {summary}")
        return "\n".join(lines)
