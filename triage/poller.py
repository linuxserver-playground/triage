"""The loop: every poll, read control's public lists of open issues and pull requests, decide which threads need
an answer, run the model and post. Control already watches the whole org through its webhooks, so the bot needs
no webhook, no open port and no credential of control's."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .agent import Agent, Result, render_thread
from .config import INTERNAL_ASSOCIATIONS, Settings
from .github import GitHub
from .sources import Sources
from .state import State

log = logging.getLogger(__name__)

OPT_OUT = "no-bot"
FOOTER = (
    "\n\n---\n<sub>This is an automated first response written by an AI (Claude) from the playground's own data. "
    "It may be wrong: a maintainer will read this thread and follow up. Maintainers can add the `{label}` label "
    "to stop these replies.</sub>"
)


@dataclass(frozen=True)
class Item:
    repo: str
    number: int
    kind: str  # issue or pull
    author: str
    labels: tuple[str, ...]
    created_at: str
    updated_at: str


def is_bot(login: str | None) -> bool:
    return not login or login.endswith("[bot]")


def ts(value: str | None) -> str:
    """GitHub's `Z` and control's `+00:00` side by side compare wrongly as strings; normalise to `+00:00`."""
    return (value or "").replace("Z", "+00:00")


class Poller:
    def __init__(
        self,
        settings: Settings,
        state: State,
        github: GitHub,
        agent: Agent,
        sources: Sources,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings
        self.state = state
        self.github = github
        self.agent = agent
        self.sources = sources
        self.client = client or httpx.Client(timeout=20.0, headers={"User-Agent": "lsio-triage"})
        self.started_at = state.started_at()
        self.refreshed = 0.0
        self.over_cap_logged = ""

    def items(self) -> list[Item]:
        """Control's open issues and pull requests (public JSON, no auth)."""
        found = []
        for kind, path, key in (("issue", "issues", "issues"), ("pull", "pulls", "pulls")):
            response = self.client.get(f"{self.settings.control_url}/api/{path}.json")
            response.raise_for_status()
            for row in response.json()[key]:
                data = row.get("pull", row) if kind == "pull" else row
                found.append(
                    Item(
                        repo=data["repo"],
                        number=int(data["number"]),
                        kind=kind,
                        author=data.get("author") or "",
                        labels=tuple(data.get("labels") or ()),
                        created_at=ts(data["created_at"]),
                        updated_at=ts(data["updated_at"]),
                    )
                )
        return found

    def due(self, item: Item) -> bool:
        """Cheap checks on control's row alone, before any GitHub call."""
        if item.created_at < self.started_at or is_bot(item.author) or OPT_OUT in item.labels:
            return False
        known = self.state.thread(item.repo, item.number)
        return known is None or known.seen_updated_at != item.updated_at

    def poll_once(self) -> None:
        if time.monotonic() - self.refreshed > self.settings.refresh_seconds or not self.refreshed:
            self.sources.refresh()
            self.agent.refresh()
            self.refreshed = time.monotonic()
        try:
            items = self.items()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("could not read control: %s", exc)
            return
        for item in sorted(items, key=lambda i: i.created_at):
            if not self.due(item):
                continue
            try:
                self.handle(item)
            except Exception:
                # Marked seen, so a thread that breaks the bot is tried again only after it changes.
                log.exception("failed on %s#%s", item.repo, item.number)
                self.state.mark(item.repo, item.number, item.kind, item.updated_at, "error")

    def handle(self, item: Item) -> None:
        issue = self.github.issue(item.repo, item.number)
        labels = {label["name"] for label in issue.get("labels") or []}
        if OPT_OUT in labels:
            self.state.mark(item.repo, item.number, item.kind, item.updated_at, "opted out")
            return
        if self.settings.authors == "external" and issue.get("author_association") in INTERNAL_ASSOCIATIONS:
            self.state.mark(item.repo, item.number, item.kind, item.updated_at, "member")
            return

        bot = self.github.login()
        comments = self.github.comments(item.repo, item.number)
        mine = [c for c in comments if (c.get("user") or {}).get("login") == bot]
        known = self.state.thread(item.repo, item.number)
        if not mine:
            if known is not None and known.outcome != "error":
                # Seen before and deliberately not answered (no post, a refusal, over the cap): stay quiet.
                self.state.mark(item.repo, item.number, item.kind, item.updated_at, known.outcome)
                return
            mode = "first"
        else:
            author = (issue.get("user") or {}).get("login")
            since = mine[-1]["created_at"]
            replies = [c for c in comments if c["created_at"] > since and (c.get("user") or {}).get("login") == author]
            if not replies or (known and known.replies > self.settings.max_followups):
                self.state.mark(item.repo, item.number, item.kind, item.updated_at, "nothing new")
                return
            mode = "followup"

        spent = self.state.month_spend()
        if spent >= self.settings.monthly_cap_usd:
            month = time.strftime("%Y-%m", time.gmtime())
            if self.over_cap_logged != month:
                self.over_cap_logged = month
                log.warning(
                    "monthly cap of $%.2f reached ($%.2f spent): not answering", self.settings.monthly_cap_usd, spent
                )
            self.state.mark(item.repo, item.number, item.kind, item.updated_at, "over cap")
            return

        pull = files = None
        if item.kind == "pull":
            pull = self.github.pull(item.repo, item.number)
            files = self.github.pull_files(item.repo, item.number)
        thread = render_thread(item.repo, issue, comments, bot, pull, files)
        log.info("triaging %s#%s (%s, %s)", item.repo, item.number, item.kind, mode)
        result = self.agent.triage(thread, mode)
        url = self.publish(item, result)
        self.state.record_run(item.repo, item.number, mode, result.usage, result.outcome, url)
        self.state.mark(item.repo, item.number, item.kind, item.updated_at, result.outcome, replied=url is not None)
        log.info(
            "%s#%s: %s, %s, labels %s, $%.4f, %d tool calls%s",
            item.repo,
            item.number,
            result.outcome,
            result.classification,
            result.labels,
            result.usage.usd,
            len(result.tool_calls),
            f", posted {url}" if url else "",
        )

    def publish(self, item: Item, result: Result) -> str | None:
        if result.outcome != "ok" or not result.post or not result.comment.strip():
            return None
        body = result.comment.strip() + FOOTER.format(label=OPT_OUT)
        if self.settings.dry_run:
            log.info("dry run, would post on %s#%s with labels %s:\n%s", item.repo, item.number, result.labels, body)
            return None
        comment: dict[str, Any] = self.github.post_comment(item.repo, item.number, body)
        try:
            self.github.add_labels(item.repo, item.number, result.labels)
        except httpx.HTTPError as exc:
            log.warning("could not label %s#%s: %s", item.repo, item.number, exc)
        return str(comment.get("html_url") or "")

    def run(self) -> None:
        log.info(
            "watching %s via %s every %ss (authors: %s, model %s, cap $%.0f/month%s)",
            self.settings.org,
            self.settings.control_url,
            self.settings.poll_seconds,
            self.settings.authors,
            self.settings.model,
            self.settings.monthly_cap_usd,
            ", dry run" if self.settings.dry_run else "",
        )
        while True:
            self.poll_once()
            time.sleep(self.settings.poll_seconds)
