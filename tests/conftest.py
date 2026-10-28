import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from triage.agent import Agent
from triage.config import Settings
from triage.sources import DOCS_PATH, Sources
from triage.state import State

ORG = "example-org"
REPO = f"{ORG}/docker-sonarr"
BOT = "lsio-triage[bot]"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Each test gets its own data dir; the hub and docs trees are tiny fixtures laid where the clones would be."""
    settings = Settings(app_id="1", org=ORG, data_dir=tmp_path, control_url="https://control.test")
    images = settings.cache_dir / "hub" / "images" / "sonarr"
    (images / "builds").mkdir(parents=True)
    (images / "manifest.main.json").write_text(
        json.dumps({"name": "sonarr", "summary": "[Sonarr](https://sonarr.tv/) is a PVR for usenet users."})
    )
    (images / "latest.json").write_text(json.dumps({"main": {"tag": "4.0.20.3014-ls4"}}))
    (images / "builds" / "4.0.20.3014-ls4.diff.json").write_text('{"packages": {"changed": ["sqlite 3.50"]}}')
    (settings.cache_dir / "hub" / ".git").mkdir()
    (settings.cache_dir / "hub" / ".git" / "config").write_text("secret-ish")
    docs = settings.cache_dir / "site" / DOCS_PATH / "general"
    docs.mkdir(parents=True)
    (docs / "docker-compose.md").write_text("# Docker compose\n\nSet PUID and PGID to your user.\n")
    return settings


@pytest.fixture
def state(settings: Settings) -> State:
    return State(settings.db_path)


@pytest.fixture
def sources(settings: Settings) -> Sources:
    return Sources(settings)


class FakeGitHub:
    """Just enough of triage.github.GitHub, recording what the bot writes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.issues: dict[tuple[str, int], dict[str, Any]] = {}
        self.thread_comments: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self.posted: list[tuple[str, int, str]] = []
        self.labelled: list[tuple[str, int, list[str]]] = []
        self.files: dict[str, str] = {"CONVENTIONS.md": "# Conventions\n"}

    def login(self) -> str:
        return BOT

    def repo(self, name: str) -> str:
        return f"{self.settings.org}/{name.removeprefix(self.settings.org + '/')}"

    def issue(self, repo: str, number: int) -> dict[str, Any]:
        return self.issues[(repo, number)]

    def comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.thread_comments.get((repo, number), [])

    def pull(self, repo: str, number: int) -> dict[str, Any]:
        return {"head": {"sha": "a" * 40, "ref": "fix", "repo": {"full_name": "someone/docker-sonarr"}}, "base": {}}

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        return [{"filename": "README.md", "status": "modified", "additions": 1, "deletions": 1, "patch": "-a\n+b"}]

    def file(self, repo: str, path: str, ref: str | None = None) -> str:
        return self.files[path]

    def search_issues(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [{"html_url": f"https://github.com/{REPO}/issues/1", "state": "closed", "title": "Database is locked"}]

    def post_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        self.posted.append((repo, number, body))
        comment = {
            "user": {"login": BOT},
            "created_at": "2030-01-01T00:00:10Z",
            "body": body,
            "html_url": f"https://github.com/{repo}/issues/{number}#issuecomment-1",
        }
        self.thread_comments.setdefault((repo, number), []).append(comment)
        return comment

    def add_labels(self, repo: str, number: int, labels: list[str]) -> None:
        self.labelled.append((repo, number, labels))


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, args: dict[str, Any], id: str = "tool_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id)


def answer(post: bool = True, labels: list[str] | None = None, comment: str = "Thanks, please add your logs.") -> str:
    return json.dumps({"post": post, "classification": "bug", "labels": labels or ["needs-info"], "comment": comment})


def response(stop_reason: str, *content: SimpleNamespace, model: str = "claude-sonnet-5-5") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=200, cache_read_input_tokens=50_000, cache_creation_input_tokens=0
    )
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), usage=usage, model=model)


class FakeClaude:
    """Plays back scripted responses and keeps every request, as `client.beta.messages.create` would see it."""

    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def make_agent(settings: Settings, github: FakeGitHub, sources: Sources, claude: FakeClaude, **overrides: Any) -> Agent:
    agent = Agent(replace(settings, **overrides), github, sources, client=claude)  # type: ignore[arg-type]
    agent.refresh()
    return agent
