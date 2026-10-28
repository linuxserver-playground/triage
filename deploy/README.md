# Deploying triage

triage runs as one container on the server that runs control, in its own compose project. It publishes no port and joins no
shared network: it only makes outgoing requests, to control's public API, GitHub and Anthropic.

## One time

1. Create the GitHub App as an org owner, from this prefilled form (no webhook; Issues and Pull requests read
   and write; Contents, Actions and Metadata read):

   ```
   https://github.com/organizations/linuxserver-playground/settings/apps/new?name=lsio-playground-triage&url=https://linuxserver-playground.github.io/site/&public=false&webhook_active=false&issues=write&pull_requests=write&contents=read&actions=read&metadata=read
   ```

   Note the App ID, generate a private key, and install the App on the org with all repositories.

2. Create an Anthropic API key in its own workspace, with a monthly spend limit above the bot's own cap
   (`TRIAGE_MONTHLY_CAP_USD`, $100 by default), e.g. $150.

3. Put both on the server, in a directory of their own (`~/triage` here):

   ```bash
   mkdir -p ~/triage/config
   mv ~/lsio-playground-triage.*.private-key.pem ~/triage/config/app.pem
   chmod 600 ~/triage/config/app.pem
   cat > ~/triage/.env <<'EOF'
   ANTHROPIC_API_KEY=<the key>
   TRIAGE_APP_ID=<the App ID>
   EOF
   chmod 600 ~/triage/.env
   ```

4. Copy `compose.yml` to `~/triage/compose.yml`, then check one real thread before the bot answers anything.
   `once` runs the model on it and prints the comment and labels it would post; it posts nothing. Run every
   compose command from `~/triage`: the compose file uses the directory it is run from.

   ```bash
   cd ~/triage
   docker compose -p triage pull
   docker compose -p triage run --rm --entrypoint /lsiopy/bin/python \
     -e TRIAGE_DATA_DIR=/config triage -m triage once docker-heimdall 6
   ```

5. Start it:

   ```bash
   docker compose -p triage up -d
   docker logs -f triage
   ```

   The first start records the time: only issues and pull requests opened after it are ever answered.

## Settings

Everything but the two secrets is a code default (`triage/config.py`). To override one, add it to `.env` and
`docker compose -p triage up -d triage` in `~/triage`:

| Variable | Default | |
| --- | --- | --- |
| `TRIAGE_AUTHORS` | `all` | `external` answers only people outside the team (owners, members and collaborators skipped) |
| `TRIAGE_DRY_RUN` | `false` | `true` runs the model and logs the comment, posting nothing |
| `TRIAGE_MONTHLY_CAP_USD` | `100` | the bot stops answering for the month once its own running total reaches this |
| `TRIAGE_MODEL` | `claude-sonnet-5-5` | |
| `TRIAGE_EFFORT` | `medium` | |
| `TRIAGE_POLL_SECONDS` | `30` | how often control's open issues and pull requests are read |
| `TRIAGE_MAX_FOLLOWUPS` | `3` | replies to the author after the first response, per thread |

## Checking on it

- `docker logs triage`: one line per thread it answers, with labels, cost and the comment link.
- `docker exec triage /lsiopy/bin/python -m triage spend`: this month's spend and the newest runs.
- A maintainer stops it on one thread with the `no-bot` label. To stop it everywhere, stop both services
  (`docker compose -p triage stop`): the updater's `up -d` would start a stopped bot
  again. `TRIAGE_DRY_RUN=true` keeps it running without posting.
