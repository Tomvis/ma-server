"""
Skip the qqmusic tests when the client library cannot be imported.

qqmusic-api-python declares typing-extensions>=4.12.2 but imports the lowercase
`sentinel`, which 4.15 renamed to `Sentinel`. The import therefore raises at
collection time and aborts the whole run rather than failing one module. The
provider itself is unaffected in practice: it is only imported when configured.

Remove this file once the dependency is fixed upstream or the pin is raised.
"""

from __future__ import annotations

import importlib

collect_ignore_glob: list[str] = []

try:
    importlib.import_module("qqmusic_api")
except Exception:  # any import failure means the suite cannot collect
    collect_ignore_glob = ["*.py"]
