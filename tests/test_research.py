from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from triage.config import Settings
from triage.pricing import price
from triage.research import Researcher
from triage.sources import Sources

from .conftest import BOT, REPO, FakeClaude, FakeGitHub, answer, make_agent, response, text_block, tool_block

HAIKU = "claude-haiku-5-5"


def server_block(name: str, args: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(type="server_tool_use", name=name, input=args, id="srv_1")


def test_research_reads_upstream_and_returns_its_findings(settings: Settings) -> None:
    github = FakeGitHub(settings)
    search = server_block("web_search", {"query": "sonarr database is locked"})
    claude = FakeClaude(
        response("pause_turn", search, model=HAIKU, searches=1),
        response(
            "tool_use",
            tool_block("github_file", {"repo": "Sonarr/Sonarr", "path": "src/db.cs", "ref": "v4"}),
            model=HAIKU,
        ),
        response(
            "tool_use",
            tool_block("github_compare", {"repo": "Sonarr/Sonarr", "base": "v4.0.20", "head": "v4.0.21"}),
            model=HAIKU,
        ),
        response(
            "end_turn", text_block("**Answer** Fixed in 4.0.21: https://github.com/Sonarr/Sonarr/pull/1"), model=HAIKU
        ),
    )
    findings = Researcher(settings, github, claude).research("Is 'database is locked' fixed upstream?")  # type: ignore[arg-type]

    assert findings.text.startswith("**Answer** Fixed in 4.0.21")
    assert [lookup.split()[0] for lookup in findings.lookups] == ["web_search", "github_file", "github_compare"]
    # The paused server-tool turn goes back unchanged, and the client tools' results follow it.
    assert claude.requests[1]["messages"][1]["content"] == [search]
    file_result = claude.requests[2]["messages"][-1]["content"][0]["content"]
    assert file_result == "1: open()\n2: retry()"
    compare_result = claude.requests[3]["messages"][-1]["content"][0]["content"]
    assert "abcdef1 Retry locked database" in compare_result and "src/db.cs (modified" in compare_result

    request = claude.requests[0]
    assert request["model"] == HAIKU and "fallbacks" not in request  # Haiku has no server-side fallback
    assert {tool["name"] for tool in request["tools"]} >= {"web_search", "web_fetch", "github_file"}
    assert findings.usage.usd == pytest.approx(
        sum(price(HAIKU, r.usage) for r in [response("x", model=HAIKU, searches=1), *[response("x", model=HAIKU)] * 3])
    )


def test_research_stops_reading_past_its_budget(settings: Settings) -> None:
    claude = FakeClaude(
        response("tool_use", tool_block("github_releases", {"repo": "Sonarr/Sonarr"}), model=HAIKU, prompt=90_000),
        response("end_turn", text_block("done"), model=HAIKU),
    )
    Researcher(settings, FakeGitHub(settings), claude).research("q")  # type: ignore[arg-type]
    assert [r["tool_choice"]["type"] for r in claude.requests] == ["auto", "none"]


def test_a_failed_research_call_is_reported_not_raised(settings: Settings) -> None:
    def broken(**kwargs: object) -> None:
        raise anthropic.BadRequestError(
            "bad", response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")), body=None
        )

    claude = FakeClaude()
    claude.beta.messages.create = broken
    findings = Researcher(settings, FakeGitHub(settings), claude).research("q")  # type: ignore[arg-type]
    assert findings.text.startswith("Research failed")


def test_issue_search_spans_github_for_research(settings: Settings) -> None:
    github = FakeGitHub(settings)
    Researcher(settings, github, FakeClaude()).run_tool("github_search_issues", {"query": "locked"})  # type: ignore[arg-type]
    assert github.searches == [("locked", False)]


def test_sonnet_delegates_research_up_to_the_cap(settings: Settings, sources: Sources) -> None:
    ask = [tool_block("research", {"question": f"q{i}"}, id=f"r{i}") for i in range(2)]
    claude = FakeClaude(
        response("tool_use", ask[0]),
        response("end_turn", text_block("Upstream fixed it in 4.0.21."), model=HAIKU, searches=2),
        response("tool_use", ask[1]),
        response(
            "end_turn",
            text_block(answer(comment="Fixed upstream.\n\n<details><summary>What I checked</summary></details>")),
        ),
    )
    agent = make_agent(settings, FakeGitHub(settings), sources, claude, max_research=1)
    thread = f'<thread repo="{REPO}">x</thread>'
    result = agent.triage(thread, "first")

    first = claude.requests[2]["messages"][-1]["content"][0]
    assert first["content"] == "Upstream fixed it in 4.0.21." and not first["is_error"]
    second = claude.requests[3]["messages"][-1]["content"][0]
    assert second["is_error"] and "no research left" in second["content"]
    assert result.usage.model == f"claude-sonnet-5-5+{HAIKU}"
    assert result.usage.usd > 3 * price("claude-sonnet-5-5", response("x").usage)  # Haiku and the searches count
    assert "research" in {tool["name"] for tool in claude.requests[0]["tools"]}
    assert "up to 1 calls per thread" in claude.requests[0]["system"][0]["text"]
    assert BOT not in result.comment


def test_price_counts_web_searches() -> None:
    assert price(HAIKU, response("x", model=HAIKU, searches=3).usage) - price(
        HAIKU, response("x", model=HAIKU).usage
    ) == pytest.approx(0.03)
