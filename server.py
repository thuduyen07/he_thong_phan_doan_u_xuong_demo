from __future__ import annotations

import argparse
import os

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
        debug=os.getenv("DEMO_FLASK_DEBUG", "0") == "1",
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
