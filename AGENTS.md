# Working in triage

The playground's AI first responder for issues and pull requests (workspace issue #6). Python 3.12+, the
Anthropic SDK, httpx, SQLite. No web server: it only makes outgoing requests.

## Layout

- `triage/config.py` settings from env, every one with a default; `triage/poller.py` the loop over control's
  open lists, the filters, follow-ups, the cap and posting (the only code that writes to GitHub);
  `triage/agent.py` the prompt, the read-only tools, the tool loop and the structured output;
  `triage/research.py` the Haiku research assistant Sonnet calls (web search, web fetch, read-only GitHub over any
  public repository); `triage/pricing.py` the cost of a response; `triage/sources.py` the shallow clones of the hub's `data` branch and the site's docs, and the path-checked
  reads over them; `triage/github.py` the App's REST calls; `triage/state.py` what was seen and answered, and the
  cost of every run; `triage/__main__.py` run / once / spend.
- `tests/` pytest with a fake GitHub (`FakeGitHub`) and a scripted model (`FakeClaude`) in `conftest.py`.
- `root/` the s6 services; `deploy/` the operator files.

## Checks

```bash
pip install -e '.[dev]'
ruff check . && ruff format --check . && mypy triage tests
pytest
```

All must pass before a commit. A push to `main` is a deploy (Publish runs the checks first).

## Rules

- The model never gets a tool that writes. Posting and labelling happen in `poller.py` from the structured
  output; `Agent.triage` drops any label not in `LABELS`.
- Thread text is untrusted. It goes to the model only inside `render_thread`'s `<thread>` tag, which the core
  rule in `PRIMER` names; never put it in the system prompt.
- The web tools and the GitHub tools outside the org belong to the research assistant only. Its findings go back
  to Sonnet as untrusted tool results, never into a system prompt, and research calls are capped per thread
  (`max_research`).
- Tools read only the two local sources (through `Sources.resolve`, which refuses `..` and `.git`) and
  repositories of the org (`GitHub.repo` adds the owner). Never give a tool `/config`: it holds the App key.
- Keep the system prompt byte-stable within a run and between runs (it is cached): nothing per-thread or
  time-dependent goes into it.
- Every model run is a `runs` row with its cost, and the monthly cap is checked before each run.
- Secrets: `app.pem` and `.env` live only on the server that runs the bot. Never log the thread text or a key; log only ids, labels, cost
  and comment links.
- Commit subjects are short imperative sentences with no trailers (CONVENTIONS §8).
