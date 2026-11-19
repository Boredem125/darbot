"""Standalone runner for the Wasalt collector.

Launched as a detached process by the orchestrator. Writes PID and logs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import logging

from collect.wasalt import collect

LOG_FILE = PROJECT_ROOT / ".orchestration" / "logs" / "wasalt_collect.log"
PID_FILE = PROJECT_ROOT / ".orchestration" / "logs" / "wasalt_collect.pid"


def main() -> None:
    # Write PID
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("wasalt_collect")
    logger.info("Wasalt collector started, PID=%d", os.getpid())

    try:
        stats = collect(max_sale=2000, max_rent=1000)
        logger.info("Collection finished: %s", stats)
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception:
        logger.exception("Collection failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
