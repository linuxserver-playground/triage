from dataclasses import replace
from typing import Any

import httpx
import pytest

from triage.config import Settings
from triage.poller import FOOTER, Item, Poller
from triage.sources import Sources
from triage.state import State, Usage

from .conftest import BOT, REPO, FakeClaude, FakeGitHub, answer, make_agent, response, text_block

OPENED = "2030-01-01T00:00:00+00:00"


def control(issues: list[dict[str, Any]], pulls: list[dict[str, Any]] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/issues.json":
            return httpx.Response(200, json={"issues": issues})
        return httpx.Response(200, json={"pulls": [{"pull": p, "run": None} for p in pulls or []]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def row(number: int = 12, author: str = "reporter", updated: str = OPENED, **extra: Any) -> dict[str, Any]:
    return {
        "repo": REPO,
        "number": number,
        "author": author,
        "labels": [],
        "created_at": OPENED,
        "updated_at": updated,
        **extra,
    }


def gh_issue(number: int = 12, association: str = "NONE", **extra: Any) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Container fails to start",
        "user": {"login": "reporter"},
        "author_association": association,
        "labels": [],
        "body": "It crashes.",
        "created_at": "2030-01-01T00:00:00Z",
        **extra,
    }


def poller(
    settings: Settings,
    state: State,
    sources: Sources,
    github: FakeGitHub,
    claude: FakeClaude,
    rows: list[dict[str, Any]],
) -> Poller:
    with state.connect() as conn:  # the bot started before these threads were opened
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('started_at', '2029-12-31T00:00:00+00:00')")
    p = Poller(settings, state, github, make_agent(settings, github, sources, claude), sources, client=control(rows))  # type: ignore[arg-type]
    p.refreshed = 1e12  # the fixture trees stand in for the clones; never run git in tests
    return p


def test_a_new_issue_gets_one_comment_with_the_footer_and_labels(
    settings: Settings, state: State, sources: Sources
) -> None:
    github = FakeGitHub(settings)
    github.issues[(REPO, 12)] = gh_issue()
    claude = FakeClaude(response("end_turn", text_block(answer())))
    p = poller(settings, state, sources, github, claude, [row()])

    p.poll_once()
    assert len(github.posted) == 1
    body = github.posted[0][2]
    assert body.startswith("Thanks, please add your logs.") and body.endswith(FOOTER.format(label="no-bot"))
    assert github.labelled == [(REPO, 12, ["needs-info"])]
    thread = state.thread(REPO, 12)
    assert thread and thread.replies == 1 and thread.outcome == "ok"
    assert state.month_spend() > 0

    p.poll_once()  # nothing changed on control: no GitHub call, no model call
    assert len(claude.requests) == 1


def test_old_bot_and_opted_out_threads_are_never_read(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)  # no issues: any GitHub read would raise KeyError and mark an error
    claude = FakeClaude()
    rows = [
        row(1, created_at="2029-01-01T00:00:00+00:00"),
        row(2, author="renovate[bot]"),
        row(3, labels=["no-bot"]),
    ]
    p = poller(settings, state, sources, github, claude, rows)
    p.poll_once()
    assert claude.requests == [] and all(state.thread(REPO, n) is None for n in (1, 2, 3))


def test_external_mode_skips_members(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)
    github.issues[(REPO, 12)] = gh_issue(association="MEMBER")
    claude = FakeClaude()
    p = poller(replace(settings, authors="external"), state, sources, github, claude, [row()])
    p.poll_once()
    thread = state.thread(REPO, 12)
    assert claude.requests == [] and thread and thread.outcome == "member"


def test_the_author_gets_follow_ups_up_to_the_cap(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)
    github.issues[(REPO, 12)] = gh_issue()
    claude = FakeClaude(*(response("end_turn", text_block(answer())) for _ in range(3)))
    rows = [row()]
    p = poller(replace(settings, max_followups=1), state, sources, github, claude, rows)
    p.poll_once()

    # Someone else comments: the thread moves, but the bot only answers its author.
    github.thread_comments[(REPO, 12)].append({"user": {"login": "passerby"}, "created_at": "2030-01-01T00:01:00Z"})
    rows[0] = row(updated="2030-01-01T00:01:00+00:00")
    p.client = control(rows)
    p.poll_once()
    assert len(claude.requests) == 1

    github.thread_comments[(REPO, 12)].append({"user": {"login": "reporter"}, "created_at": "2030-01-01T00:02:00Z"})
    rows[0] = row(updated="2030-01-01T00:02:00+00:00")
    p.client = control(rows)
    p.poll_once()
    assert len(claude.requests) == 2 and len(github.posted) == 2
    assert "The author replied" in claude.requests[1]["messages"][0]["content"]

    github.thread_comments[(REPO, 12)].append({"user": {"login": "reporter"}, "created_at": "2030-01-01T00:03:00Z"})
    rows[0] = row(updated="2030-01-01T00:03:00+00:00")
    p.client = control(rows)
    p.poll_once()
    assert len(claude.requests) == 2  # one follow-up was the cap


def test_no_post_dry_run_and_cap(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)
    for n in (1, 2, 3):
        github.issues[(REPO, n)] = gh_issue(n)
    claude = FakeClaude(
        response("end_turn", text_block(answer(post=False))), response("end_turn", text_block(answer()))
    )
    p = poller(replace(settings, dry_run=True), state, sources, github, claude, [row(1), row(2)])
    p.poll_once()
    assert github.posted == [] and len(claude.requests) == 2

    state.record_run(REPO, 99, "first", Usage("claude-sonnet-5-5", usd=100.0), "ok", None)
    p.client = control([row(3)])
    p.poll_once()
    thread = state.thread(REPO, 3)
    assert len(claude.requests) == 2 and thread and thread.outcome == "over cap"


def test_pull_requests_carry_their_diff(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)
    github.issues[(REPO, 7)] = gh_issue(7, pull_request={"url": "x"})
    claude = FakeClaude(response("end_turn", text_block(answer())))
    p = poller(settings, state, sources, github, claude, [])
    p.client = control([], [row(7, head_sha="a" * 40)])
    p.poll_once()
    content = claude.requests[0]["messages"][0]["content"]
    assert 'kind="pull request"' in content and "--- README.md" in content


def test_a_failure_is_retried_only_when_the_thread_changes(settings: Settings, state: State, sources: Sources) -> None:
    github = FakeGitHub(settings)  # issue 12 missing: handle raises
    p = poller(settings, state, sources, github, FakeClaude(), [row()])
    p.poll_once()
    thread = state.thread(REPO, 12)
    assert thread and thread.outcome == "error"
    assert not p.due(Item(REPO, 12, "issue", "reporter", (), OPENED, OPENED))
    assert p.due(Item(REPO, 12, "issue", "reporter", (), OPENED, "2030-01-02T00:00:00+00:00"))


def test_started_at_is_kept(state: State) -> None:
    assert state.started_at() == state.started_at()


@pytest.mark.parametrize("login", [BOT, "dependabot[bot]", ""])
def test_bots_are_not_answered(settings: Settings, state: State, sources: Sources, login: str) -> None:
    p = poller(settings, state, sources, FakeGitHub(settings), FakeClaude(), [])
    assert not p.due(Item(REPO, 1, "issue", login, (), OPENED, OPENED))
