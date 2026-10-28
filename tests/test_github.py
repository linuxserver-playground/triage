import httpx
import pytest

from triage.config import Settings
from triage.github import GitHub


def client(private: bool, seen: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/repos/example-org/docker-sonarr":
            return httpx.Response(200, json={"private": private})
        if request.url.path == "/search/issues":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(200, json={"encoding": "base64", "content": "aW1hZ2UueW1s"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def github(settings: Settings, private: bool, seen: list[httpx.Request]) -> GitHub:
    gh = GitHub(settings, client(private, seen))
    gh._token = ("token", 1e12)  # skip the App JWT exchange
    return gh


def test_only_public_repositories_are_read(settings: Settings) -> None:
    seen: list[httpx.Request] = []
    assert github(settings, False, seen).file("example-org/docker-sonarr", "image.yml") == "image.yml"
    with pytest.raises(ValueError, match="not a public repository"):
        github(settings, True, seen).file("example-org/docker-sonarr", "image.yml")
    github(settings, False, seen).search_issues("locked")
    assert seen[-1].url.params["q"] == "org:example-org is:public locked"


@pytest.mark.parametrize("name", ["", "other/repo", ".github", "../x"])
def test_repo_names_stay_in_the_org(settings: Settings, name: str) -> None:
    with pytest.raises(ValueError):
        GitHub(settings).repo(name)
    assert GitHub(settings).repo("example-org/docker-sonarr") == "example-org/docker-sonarr"


def test_an_app_not_installed_on_the_org_is_named(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/app":
            return httpx.Response(200, json={"slug": "other-app", "id": 42})
        return httpx.Response(404, json={"message": "Not Found"})

    gh = GitHub(settings, httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(gh, "app_jwt", lambda: "jwt")
    with pytest.raises(RuntimeError, match=r"'other-app' \(id 42.*not installed on example-org"):
        gh.token()
