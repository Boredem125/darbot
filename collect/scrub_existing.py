"""Scrub PII from already-saved Wasalt data files.

Rewrites each data/raw/wasalt/*.json.gz through strip_pii(), then verifies
no file contains the banned field keys.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from collect.wasalt import strip_pii

BANNED_KEYS = {
    "responsibleEmployeeName",
    "responsibleEmployeePhoneNumber",
    "deedNumber",
}


def scrub_file(path: Path) -> bool:
    """Rewrite a single .json.gz file through strip_pii. Returns True if modified."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("raw_property_details", {})
    # Apply the same strip_pii function used at collection time
    strip_pii(raw)

    # Rewrite
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return True


def verify_file(path: Path) -> list[str]:
    """Check a file for any remaining banned keys. Returns list of violations."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        text = f.read()

    violations = []
    for key in BANNED_KEYS:
        if key in text:
            violations.append(f"{path.name}: contains '{key}'")
    return violations


def main() -> None:
    data_dir = Path(PROJECT_ROOT / "data" / "raw" / "wasalt")
    files = sorted(data_dir.glob("*.json.gz"))
    print(f"Scrubbing {len(files)} files...")

    scrubbed = 0
    for f in files:
        try:
            scrub_file(f)
            scrubbed += 1
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR: {f.name}: {e}")

    print(f"Scrubbed {scrubbed}/{len(files)} files.")

    # Verification pass
    print("Verifying no banned keys remain...")
    all_violations: list[str] = []
    for f in files:
        try:
            violations = verify_file(f)
            all_violations.extend(violations)
        except (json.JSONDecodeError, OSError) as e:
            all_violations.append(f"{f.name}: read error: {e}")

    if all_violations:
        print(f"FAIL: {len(all_violations)} violations found:")
        for v in all_violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        print(f"PASS: 0 violations in {len(files)} files.")


if __name__ == "__main__":
    main()
