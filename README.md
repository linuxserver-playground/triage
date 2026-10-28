# triage

The LinuxServer.io v2 playground's first responder. Every new issue and pull request in the
`linuxserver-playground` org gets a reply within a minute or two: what it looks like (support, bug, upstream bug,
feature request), what is missing, what the playground's own data says, and labels from a short allowlist. Every
comment says it was written by an AI and that a maintainer will follow up. The design is
[workspace issue #6](https://github.com/linuxserver-playground/workspace/issues/6).

## How it works

1. **Notice.** Every 30 seconds it reads control's public `/api/issues.json` and `/api/pulls.json`. Control
   already sees every issue, pull request and comment in the org through its webhooks, so the bot needs no
   webhook, no open port and no credential of control's.
2. **Filter.** It skips threads opened before its first start (no backfill), threads opened by bots, threads
   labelled `no-bot` and, with `TRIAGE_AUTHORS=external`, threads opened by owners, members and collaborators. A
   thread it already answered gets another reply only when its author comments again, at most three times.
3. **Gather.** A short Claude tool loop (Sonnet 5.5) with read-only tools: the hub's `data` branch and the site's
   docs (shallow clones refreshed hourly, so the Pages rate limit is never touched), any file of an org repository,
   and an org-wide issue search. A pull request comes with its diff and its build report comment. CONVENTIONS.md
   and an index of every image sit in the cached system prompt.
4. **Post.** The model returns structured output only: whether to post, a classification, labels from the
   allowlist and the comment. The bot's own code posts it with the AI footer and sets the labels.

Issue and comment text is untrusted. One core rule in the prompt says so, and the structure does the rest: the
model has no tool that writes, so the worst a successful injection can do is one odd comment a maintainer sees.

## Identity and cost

The `lsio-playground-triage` GitHub App comments as `lsio-playground-triage[bot]`. It can read issues, pull
requests, contents and Actions, and write only comments and labels. Its private key and the Anthropic API key
live only on the server that runs it (`deploy/README.md`). The bot keeps a running total from each response's
usage and stops for the month at `TRIAGE_MONTHLY_CAP_USD` ($100); the API workspace's spend limit is the
backstop.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy triage tests && .venv/bin/pytest
```

The tests run against a fake GitHub and a scripted model client; none needs a key. A push to `main` publishes
`ghcr.io/linuxserver-playground/triage:latest`, and the updater beside it deploys it within five minutes.
