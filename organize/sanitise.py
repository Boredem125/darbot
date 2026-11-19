"""Sanitise text for the index.

Applied to listing descriptions, project text and article bodies.
Rules (from PLAN section 5 and WP-2-index brief):
  - Strip URLs, emails and phone numbers (Saudi and international, including Arabic-Indic).
  - Remove < and > (injection defence for <source> delimiters).
  - Collapse whitespace.
  - Cap descriptions at DESCRIPTION_MAX_CHARS (400) at a word boundary.
"""

from __future__ import annotations

import re

from common.schemas import DESCRIPTION_MAX_CHARS

# --- Patterns ---

# URLs: http(s) and www
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Emails
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Arabic-Indic digits: 0123456789 -> ٠-٩
_ARABIC_INDIC = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

# Phone numbers: Saudi (966), UAE (971), Oman (968), Qatar (974), plus generic international.
# Use (?<!\d) / (?!\d) instead of \b because \b fails at Arabic-letter/digit boundaries
# (Arabic letters are \w, so \b does not fire between them and digits).
_PHONE_RE = re.compile(
    r"""(?:"""
    # International with country code
    r"""\+?(?:966|971|968|974)\s*[\d\s\-()]{7,15}"""
    # Saudi local: 05x xxxx xxxx or 0xx xxx xxxx
    r"""|(?:(?<!\d)0[1-9]\d[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d))"""
    # Saudi toll-free: 800 or 8001
    r"""|(?:(?<!\d)800\d?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d))"""
    # Generic international: +CC (optional area) number
    r"""|(?:\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4})"""
    r""")""",
    re.VERBOSE,
)

# Angle brackets
_ANGLE_RE = re.compile(r"[<>]")

# Whitespace collapse
_WS_RE = re.compile(r"\s+")


def sanitise_text(text: str) -> str:
    """Sanitise text: strip URLs, emails, phones, angle brackets, collapse whitespace."""
    # First normalise Arabic-Indic digits so phone patterns can match them
    text = text.translate(_ARABIC_INDIC)
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE.sub("", text)
    text = _ANGLE_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def cap_description(text: str, max_chars: int = DESCRIPTION_MAX_CHARS) -> str:
    """Cap text at max_chars at a word boundary."""
    if len(text) <= max_chars:
        return text
    # Find the last space before the limit
    cut = text.rfind(" ", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut].rstrip()


def sanitise_description(text: str | None) -> str | None:
    """Sanitise and cap a description. Returns None for empty/None input."""
    if not text:
        return None
    result = sanitise_text(text)
    if not result:
        return None
    result = cap_description(result)
    return result if result else None
