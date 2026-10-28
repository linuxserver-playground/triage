"""python -m triage [run]                   the service: poll control, answer, post
python -m triage once <repo> <number>      triage one thread and print the result; posts nothing
python -m triage spend                     this month's spend and the newest runs"""

import json
import logging
import sys

from .agent import Agent, render_thread
from .config import Settings
from .github import GitHub
from .poller import Poller
from .sources import Sources
from .state import State


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    command = argv[0] if argv else "run"
    state = State(settings.db_path)

    if command == "spend":
        print(f"${state.month_spend():.2f} this month of a ${settings.monthly_cap_usd:.0f} cap")
        with state.connect() as conn:
            for row in conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 20"):
                print(f"{row['at']} {row['repo']}#{row['number']} {row['mode']} {row['outcome']} ${row['usd']:.4f}")
        return 0

    if not settings.app_id or not settings.app_key.exists():
        print(f"TRIAGE_APP_ID and {settings.app_key} are required", file=sys.stderr)
        return 2
    github = GitHub(settings)
    sources = Sources(settings)
    agent = Agent(settings, github, sources)

    if command == "run":
        Poller(settings, state, github, agent, sources).run()
        return 0
    if command == "once" and len(argv) == 3:
        repo, number = github.repo(argv[1]), int(argv[2])
        sources.refresh()
        agent.refresh()
        issue = github.issue(repo, number)
        pull = github.pull(repo, number) if "pull_request" in issue else None
        files = github.pull_files(repo, number) if pull else None
        thread = render_thread(repo, issue, github.comments(repo, number), github.login(), pull, files)
        result = agent.triage(thread, "first")
        state.record_run(repo, number, "once", result.usage, result.outcome, None)
        print(json.dumps({**result.__dict__, "usage": result.usage.__dict__}, indent=2))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
