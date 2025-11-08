# Agent Starter Scripts

These Typer CLIs reuse the main `scripts/mdwb_cli.py` plumbing (settings, HTTP
client, polling) so you can compose lightweight automations without re-implementing
auth or job orchestration.

## Available commands

- `summarize_article`: capture (or reuse via `--job-id`) and print the first *N*
  sentences of the resulting Markdown.
- `generate_todos`: capture/reuse and emit TODO-style bullets (checkboxes, bullet
  lists, headings such as “Next Steps” or “Action Items”). Supports `--json`.

## Usage

```bash
# Summarize a fresh capture and persist the summary
uv run python -m scripts.agents.summarize_article summarize --url https://example.com --sentences 4 --out summary.txt

# Summarize an existing job id (skips capture)
uv run python -m scripts.agents.summarize_article summarize --job-id job_abc123 --out summary.txt

# Generate TODOs and emit JSON (saved to disk)
uv run python -m scripts.agents.generate_todos todos --url https://status.example --json --out todos.json

# Reuse a job, limit to 5 action items, and save newline text
uv run python -m scripts.agents.generate_todos todos --job-id job_abc123 --limit 5 --out todos.txt
```

All commands accept `--api-base`, `--http2/--no-http2`, `--profile`,
`--ocr-policy`, and `--out` just like the main CLI, and default to
`--reuse-session` so each command reuses a single HTTP client across submit /
poll / fetch. Disable it with `--no-reuse-session` if you need to isolate
connections. Credentials and defaults come from `.env` via `scripts/mdwb_cli.py`.
