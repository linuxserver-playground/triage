"""The GitHub REST calls the bot makes, as its own App. It reads issues, pull requests and files, and writes
only comments and labels; the App holds no other write permission."""

import base64
import datetime
import logging
import time
from typing import Any

import httpx
import jwt

from .config import Settings

log = logging.getLogger(__name__)

API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


class GitHub:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=30.0, headers={"User-Agent": "lsio-triage"})
        self._token: tuple[str, float] | None = None
        self._login: str | None = None

    # -- auth -----------------------------------------------------------------------------------------------------

    def app_jwt(self) -> str:
        """Issued a minute in the past to survive clock skew; GitHub caps the lifetime at ten minutes."""
        now = int(time.time())
        key = self.settings.app_key.read_text()
        return jwt.encode({"iat": now - 60, "exp": now + 9 * 60, "iss": self.settings.app_id}, key, algorithm="RS256")

    def token(self) -> str:
        """An installation token, reused until five minutes before it expires."""
        if self._token and self._token[1] - time.time() > 300:
            return self._token[0]
        auth = {**HEADERS, "Authorization": f"Bearer {self.app_jwt()}"}
        response = self.client.get(f"{API}/orgs/{self.settings.org}/installation", headers=auth)
        response.raise_for_status()
        installation = response.json()["id"]
        response = self.client.post(f"{API}/app/installations/{installation}/access_tokens", headers=auth)
        response.raise_for_status()
        data = response.json()
        expires = datetime.datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")).timestamp()
        self._token = (str(data["token"]), expires)
        return self._token[0]

    def login(self) -> str:
        """The bot's own login (`<slug>[bot]`), so it recognises its earlier comments."""
        if self._login is None:
            response = self.client.get(f"{API}/app", headers={**HEADERS, "Authorization": f"Bearer {self.app_jwt()}"})
            response.raise_for_status()
            self._login = f"{response.json()['slug']}[bot]"
        return self._login

    # -- requests -------------------------------------------------------------------------------------------------

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = {**HEADERS, "Authorization": f"Bearer {self.token()}", **kwargs.pop("headers", {})}
        response = self.client.request(method, f"{API}{path}", headers=headers, **kwargs)
        response.raise_for_status()
        return response

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params or None).json()

    def paginate(self, path: str, limit: int = 300, **params: Any) -> list[Any]:
        items: list[Any] = []
        page = 1
        while len(items) < limit:
            batch = self.get(path, per_page=100, page=page, **params)
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return items[:limit]

    def repo(self, name: str) -> str:
        """Only repositories of our own org: the model names a repo, the code adds the owner."""
        name = name.removeprefix(f"{self.settings.org}/")
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"not a repository of {self.settings.org}: {name!r}")
        return f"{self.settings.org}/{name}"

    def issue(self, repo: str, number: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"/repos/{repo}/issues/{number}")
        return result

    def comments(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/issues/{number}/comments")

    def pull(self, repo: str, number: int) -> dict[str, Any]:
        result: dict[str, Any] = self.get(f"/repos/{repo}/pulls/{number}")
        return result

    def pull_files(self, repo: str, number: int) -> list[dict[str, Any]]:
        return self.paginate(f"/repos/{repo}/pulls/{number}/files")

    def file(self, repo: str, path: str, ref: str | None = None) -> str:
        """A file's text at a ref (a PR's head sha, a branch); a directory comes back as a listing. Public
        repositories only: whatever the model reads may end up in a public comment."""
        if self.get(f"/repos/{repo}").get("private", True):
            raise ValueError(f"{repo} is not a public repository")
        data = self.get(f"/repos/{repo}/contents/{path.lstrip('/')}", **({"ref": ref} if ref else {}))
        if isinstance(data, list):
            return "\n".join(f"{item['type']}\t{item['path']}" for item in data)
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return str(data.get("content") or "")

    def search_issues(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        data = self.get("/search/issues", q=f"org:{self.settings.org} is:public {query}", per_page=limit)
        items: list[dict[str, Any]] = data.get("items") or []
        return items

    def post_comment(self, repo: str, number: int, body: str) -> dict[str, Any]:
        result: dict[str, Any] = self.request(
            "POST", f"/repos/{repo}/issues/{number}/comments", json={"body": body}
        ).json()
        return result

    def add_labels(self, repo: str, number: int, labels: list[str]) -> None:
        if labels:
            self.request("POST", f"/repos/{repo}/issues/{number}/labels", json={"labels": labels})
