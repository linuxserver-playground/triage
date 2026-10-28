"""Settings from the environment. Every setting has a default that works on n97; only the two secrets
(ANTHROPIC_API_KEY, read by the SDK itself, and the App's private key file) and TRIAGE_APP_ID must be set."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

INTERNAL_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
"""GitHub's author_association values that mean the author is on the team, not an outside contributor."""


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    app_id: str = ""
    org: str = "linuxserver-playground"
    data_dir: Path = Path("/config")
    control_url: str = "https://ci.thelamer.duckdns.org:8443"
    poll_seconds: int = 30
    refresh_seconds: int = 3600
    """How often the hub's data branch, the site's docs and CONVENTIONS are fetched again."""
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    monthly_cap_usd: float = 100.0
    authors: Literal["all", "external"] = "all"
    """`all` answers members too (the demo); `external` answers only people outside the team (production)."""
    max_followups: int = 3
    max_turns: int = 16
    dry_run: bool = False
    """Run the model but post nothing: the comment and labels go to the log only."""
    hub_repo: str = "hub"
    site_repo: str = "site"
    site_url: str = "https://linuxserver-playground.github.io/site"

    @property
    def app_key(self) -> Path:
        return self.data_dir / "app.pem"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "triage.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        authors = env.get("TRIAGE_AUTHORS", "all").strip().lower()
        if authors not in ("all", "external"):
            raise ValueError(f"TRIAGE_AUTHORS must be all or external, not {authors!r}")
        return cls(
            app_id=env.get("TRIAGE_APP_ID", "").strip(),
            org=env.get("TRIAGE_ORG", cls.org),
            data_dir=Path(env.get("TRIAGE_DATA_DIR", str(cls.data_dir))),
            control_url=env.get("TRIAGE_CONTROL_URL", cls.control_url).rstrip("/"),
            poll_seconds=int(env.get("TRIAGE_POLL_SECONDS", cls.poll_seconds)),
            refresh_seconds=int(env.get("TRIAGE_REFRESH_SECONDS", cls.refresh_seconds)),
            model=env.get("TRIAGE_MODEL", cls.model),
            effort=env.get("TRIAGE_EFFORT", cls.effort),
            monthly_cap_usd=float(env.get("TRIAGE_MONTHLY_CAP_USD", cls.monthly_cap_usd)),
            authors="all" if authors == "all" else "external",
            max_followups=int(env.get("TRIAGE_MAX_FOLLOWUPS", cls.max_followups)),
            max_turns=int(env.get("TRIAGE_MAX_TURNS", cls.max_turns)),
            dry_run=_bool(env.get("TRIAGE_DRY_RUN"), cls.dry_run),
            site_url=env.get("TRIAGE_SITE_URL", cls.site_url).rstrip("/"),
        )
