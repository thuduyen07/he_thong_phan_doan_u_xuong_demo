from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _require_activated_virtualenv() -> None:
    """Catch ``python server.py`` resolving outside an activated project venv."""
    configured_venv = os.getenv("VIRTUAL_ENV")
    if not configured_venv:
        return
    expected = Path(configured_venv).resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(
            "Server was started with a different Python interpreter than VIRTUAL_ENV. "
            "Run `./dev_scripts/run_local.sh` instead."
        )


_require_activated_virtualenv()

from backend.api import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standalone thesis defense web with local backend inference."
    )
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "4173")))
    args = parser.parse_args()

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
