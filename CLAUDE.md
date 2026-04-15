# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Music Assistant is an async Python music library manager that connects to streaming services and speakers, integrating with Home Assistant.

## Behaviour

- NEVER automatically reply on Github (PR's or Discussions) without explicit consent from the developer.

## Architecture

`MusicAssistant` (`mass.py`) is the central class. It owns and initializes all controllers, manages the lifecycle of providers, and exposes the event bus and task system.

**Core Controllers** (accessible via `mass.<name>`):
- `config` — JSON-based persistent config; drives provider UI generation
- `cache` — SQLite-backed cache with auto-cleanup
- `tasks` — Background task scheduling
- `music` — Library search, browsing, sync with providers
- `metadata` — Metadata enrichment (art, lyrics, etc.)
- `players` — Player discovery and device control
- `player_queues` — Queue management and playback state
- `streams` — Audio streaming and transcoding (ffmpeg)
- `webserver` — aiohttp REST + WebSocket API

**Event System:**
```python
# Publish
mass.signal_event(EventType.QUEUE_UPDATED, object_id=queue_id, data=queue_data)

# Subscribe (returns a callable that removes the listener)
unsub = mass.subscribe(callback, event_filter=(EventType.QUEUE_UPDATED,), id_filter="queue_1")
```
Callbacks can be sync or async.

**Task Management:**
```python
mass.create_task(coro_or_func, *args, task_id="my_task", abort_existing=False)
mass.call_later(delay_seconds, target, *args, task_id="debounce_key")  # cancels prior call with same task_id
```

**API Commands:**
Register a handler with `@api_command("namespace/method")` in any controller. Type hints drive OpenAPI schema generation automatically.

**Provider Interaction:**
Providers receive the `mass` instance on load. They call `mass.signal_event()` to emit events, `mass.create_task()` for background work, and `mass.config.get_provider_config(self.instance_id)` for their config. Features a provider supports are declared via `self.supported_features` (a set of enum values) — this controls which library/search/browse capabilities are exposed.

**Tests:**
Use the `mass` fixture (full instance with temp storage) for integration tests, or `mass_minimal` (config + cache only) for unit tests. Both are defined in `tests/conftest.py`.

## Development Commands

- `scripts/setup.sh` - Initial setup (venv, dependencies, pre-commit hooks). Re-run after pulling latest code.
- `pytest` - Run all tests
- `pytest tests/specific_test.py` - Run a specific test file
- `pre-commit run --all-files` - Run all pre-commit hooks
- `python -m music_assistant --log-level debug` - Run server locally (localhost:8095)
- Requires ffmpeg v6.1+ and Python 3.14+ (see `.python-version` for the pinned runtime)

Always run `pre-commit run --all-files` after a code change to ensure the new code adheres to the project standards.

## Provider Development

Providers are modular: music (sources), player (speakers), metadata (art/lyrics), plugin (extras). See `_demo_*_provider` directories for annotated templates when creating new providers.

Each provider has at least `__init__.py` (logic) and `manifest.json` (metadata/config schema).

Check `helpers/` for reusable utilities before writing new ones.

## Code Style

### Comments

Only use comments to explain complex, multi-line blocks of code. Do not comment obvious operations.

### Docstring Format

Use Sphinx-style docstrings with `:param:` syntax. For simple functions, a single-line docstring is fine.
Don't explain inner workings of the code in the docstrings (you can use inline comments for that if/when needed). The docstring should provide clarity to the caller of the function/method, not explain how it works technically/internally.

```python
def my_function(param1: str, param2: int, param3: bool = False) -> str:
    """
    Brief one-line description of the function.

    :param param1: Description of what param1 is used for.
    :param param2: Description of what param2 is used for.
    :param param3: Description of what param3 is used for.
    """
```

Do **not** use Google-style (`Args:`) or bullet-style (`- param:`) docstrings.

## Branching and PRs

- All PRs target `dev` (primary development branch). `stable` is for production releases.
- PRs labeled `bugfix` + `backport-to-stable` are automatically backported to `stable` — use only for bugs also present in `stable`.

## Debugging

MA stores its data in `$HOME/.musicassistant/`. When debugging locally:

- **Logs:** `$HOME/.musicassistant/musicassistant.log` (current), `musicassistant.log.1`, `.log.2`, etc. for older rotated logs.
- **Database:** `$HOME/.musicassistant/library.db` — query via `sqlite3`. **Only execute SELECT queries** — never write to a live database.
