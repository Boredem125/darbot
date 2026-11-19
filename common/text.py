"""Text normalisation for cache keys, embeddings, grounding and entity matching.

Implements caching.md section 3. Imports only the stdlib (dependency rule R1 for common/).
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------

# Arabic-Indic digits U+0660..0669
_ARABIC_INDIC = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)
# Extended Arabic-Indic digits U+06F0..06F9
_EXTENDED_INDIC = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
    "0123456789",
)
# Arabic decimal separator U+066B -> ".", Arabic thousands separator U+066C -> ","
_ARABIC_SEPS = str.maketrans("٫٬", ".,")

# Tatweel and diacritics to strip (caching.md section 3 step 3)
_TATWEEL = "ـ"
_DIACRITICS_RE = re.compile("[ً-ٰٟ]")

# Alef normalisation: hamza forms -> bare alef, alef maksura -> ya
_ALEF_MAP = str.maketrans(
    "أإآى",  # ALEF WITH HAMZA ABOVE, BELOW, MADDA ABOVE, ALEF MAKSURA
    "اااي",  # ALEF, ALEF, ALEF, YA
)

# ---------------------------------------------------------------------------
# Number words (English and Arabic) -- zero through twelve
# ---------------------------------------------------------------------------

_EN_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

# Arabic number words (masculine + feminine forms)
_AR_NUMBER_WORDS: dict[str, int] = {
    # zero
    "صفر": 0,  # صفر
    # one
    "واحد": 1,  # واحد
    "واحدة": 1,  # واحدة
    # two
    "اثنان": 2,  # اثنان
    "اثنتان": 2,  # اثنتان
    "اثنين": 2,  # اثنين
    "اثنتين": 2,  # اثنتين
    # three
    "ثلاث": 3,  # ثلاث
    "ثلاثة": 3,  # ثلاثة
    # four
    "أربع": 4,  # أربع
    "أربعة": 4,  # أربعة
    "اربع": 4,  # اربع (without hamza)
    "اربعة": 4,  # اربعة
    # five
    "خمس": 5,  # خمس
    "خمسة": 5,  # خمسة
    # six
    "ست": 6,  # ست
    "ستة": 6,  # ستة
    # seven
    "سبع": 7,  # سبع
    "سبعة": 7,  # سبعة
    # eight
    "ثمان": 8,  # ثمان
    "ثمانية": 8,  # ثمانية
    "ثماني": 8,  # ثماني
    # nine
    "تسع": 9,  # تسع
    "تسعة": 9,  # تسعة
    # ten
    "عشر": 10,  # عشر
    "عشرة": 10,  # عشرة
    # eleven
    "أحد عشر": 11,  # أحد عشر
    "احد عشر": 11,  # احد عشر
    "إحدى عشرة": 11,  # إحدى عشرة
    "احدى عشرة": 11,  # احدى عشرة
    # twelve
    "اثنا عشر": 12,  # اثنا عشر
    "اثنتا عشرة": 12,  # اثنتا عشرة
    "اثني عشر": 12,  # اثني عشر
    "اثنتي عشرة": 12,  # اثنتي عشرة
}

# ---------------------------------------------------------------------------
# Magnitude suffixes
# ---------------------------------------------------------------------------

_MAGNITUDES: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "ألف": 1_000,  # ألف
    "الف": 1_000,  # الف
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "مليون": 1_000_000,  # مليون
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "مليار": 1_000_000_000,  # مليار
}

# ---------------------------------------------------------------------------
# Currency tokens
# ---------------------------------------------------------------------------

_CURRENCY_TOKENS = {
    "sar", "sr", "riyal", "riyals",
    "ريال",  # ريال
    "ر.س",  # ر.س
}

# ---------------------------------------------------------------------------
# Negation cues (caching.md section 3)
# ---------------------------------------------------------------------------

_NEGATION_EN = {
    "not", "no", "without", "except", "excluding", "exclude",
    "other than", "non", "never",
    "isn't", "aren't", "don't", "doesn't",
    "anything but",
}

_NEGATION_AR = {
    "لا",  # لا
    "ليس",  # ليس
    "ليست",  # ليست
    "غير",  # غير
    "بدون",  # بدون
    "دون",  # دون
    "ماعدا",  # ماعدا
    "عدا",  # عدا
    "باستثناء",  # باستثناء
}

_ALL_NEGATION = _NEGATION_EN | _NEGATION_AR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalise_for_embedding(q: str) -> str:
    """Light normalisation for the embedder and L3/L4 keys (caching.md section 3).

    1. NFKC
    2. Arabic-Indic digits to ASCII; U+066B -> '.'; U+066C -> ','
    3. Collapse whitespace, strip
    """
    t = unicodedata.normalize("NFKC", q)
    t = t.translate(_ARABIC_INDIC)
    t = t.translate(_EXTENDED_INDIC)
    t = t.translate(_ARABIC_SEPS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalise_for_key(q: str) -> str:
    """Aggressive normalisation for exact cache keys (caching.md section 3).

    1. normalise_for_embedding
    2. casefold
    3. Remove tatweel, diacritics; normalise alef forms and alef maksura
    4. Canonicalise numbers (parse_numbers + rewrite)
    5. Number words to digits
    6. Currency tokens to 'sar'
    7. Strip trailing punctuation
    """
    t = normalise_for_embedding(q)
    t = t.casefold()
    # Arabic script normalisation
    t = t.replace(_TATWEEL, "")
    t = _DIACRITICS_RE.sub("", t)
    t = t.translate(_ALEF_MAP)
    # Canonicalise numbers in-place
    t = _canonicalise_numbers_in_text(t)
    # Number words to digits
    t = _replace_number_words(t)
    # Currency tokens to sar
    t = _replace_currency(t)
    # Strip trailing punctuation and whitespace
    t = t.rstrip("?!.؟ \t")
    return t


def normalise_alias(s: str) -> str:
    """Entity alias normalisation: normalise_for_key without number rewriting."""
    t = normalise_for_embedding(s)
    t = t.casefold()
    t = t.replace(_TATWEEL, "")
    t = _DIACRITICS_RE.sub("", t)
    t = t.translate(_ALEF_MAP)
    t = t.rstrip("?!.؟ \t")
    return t


def parse_numbers(text: str) -> list[Decimal]:
    """Extract all numbers from text, expanding magnitudes.

    Handles:
    - Plain integers and decimals (with optional commas as thousands separators)
    - Arabic-Indic digits (after normalise_for_embedding)
    - Magnitude suffixes: k, m, mn, million, billion, etc.
    - Number words zero through twelve (English and Arabic)
    - Excludes citation-like [n] patterns
    """
    # First apply light normalisation to handle Arabic-Indic digits
    t = normalise_for_embedding(text)
    t = t.casefold()
    # Remove tatweel and diacritics for Arabic word matching
    t = t.replace(_TATWEEL, "")
    t = _DIACRITICS_RE.sub("", t)
    t = t.translate(_ALEF_MAP)

    results: list[Decimal] = []

    # Remove citation patterns in all bracket forms so they don't get parsed as numbers
    # ASCII [n], fullwidth 【n】, 〔n〕, ［n］
    t_no_cite = re.sub(r"[\[【〔［](\d+)[\]】〕］]", " ", t)

    # Build a magnitude suffix alternation (longest first so "million" beats "m")
    mag_keys = sorted(_MAGNITUDES.keys(), key=len, reverse=True)
    mag_alt = "|".join(re.escape(k) for k in mag_keys)

    # Pattern with magnitude suffix
    num_mag_pattern = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\s*(" + mag_alt + r")\b",
        re.UNICODE,
    )

    # Plain number pattern
    num_plain_pattern = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\b",
        re.UNICODE,
    )

    # First find numbers with magnitudes (to avoid double-counting)
    mag_spans: list[tuple[int, int]] = []
    for m in num_mag_pattern.finditer(t_no_cite):
        num_str = m.group(1).replace(",", "")
        try:
            val = Decimal(num_str)
        except InvalidOperation:
            continue
        mag_str = m.group(2)
        if mag_str in _MAGNITUDES:
            val = val * _MAGNITUDES[mag_str]
        results.append(val)
        mag_spans.append((m.start(), m.end()))

    # Then find plain numbers not overlapping magnitude matches
    for m in num_plain_pattern.finditer(t_no_cite):
        if any(s <= m.start() < e for s, e in mag_spans):
            continue
        num_str = m.group(1).replace(",", "")
        try:
            val = Decimal(num_str)
        except InvalidOperation:
            continue
        results.append(val)

    # Number words (multi-word Arabic numbers first, then single words)
    # We need to work on the citation-stripped text
    # Sort by length descending so multi-word matches come first
    all_number_words = {**_EN_NUMBER_WORDS, **_AR_NUMBER_WORDS}
    for word, value in sorted(all_number_words.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.UNICODE)
        for _match in pattern.finditer(t_no_cite):
            results.append(Decimal(value))

    return results


def canonical_number(d: Decimal) -> str:
    """Canonical string for a number: integer string or decimal with no trailing zeros."""
    # Normalise to remove trailing zeros
    d = d.normalize()
    if d == d.to_integral_value():
        return str(int(d))
    return str(d)


def detect_negation(q: str) -> bool:
    """Return True if any negation cue is found as a whole word in q."""
    t = normalise_for_embedding(q).casefold()
    t = t.replace(_TATWEEL, "")
    t = _DIACRITICS_RE.sub("", t)
    t = t.translate(_ALEF_MAP)
    for cue in _ALL_NEGATION:
        # "other than" is multi-word
        if " " in cue:
            if cue in t:
                return True
        else:
            if re.search(r"\b" + re.escape(cue) + r"\b", t, re.UNICODE):
                return True
    return False


# ---------------------------------------------------------------------------
# Amenity / view token extraction for L2 scope (B9, caching.md)
# ---------------------------------------------------------------------------

# Tokens that distinguish answers even with identical filters.
# "unfurnished" must be checked before "furnished" to avoid partial match.
_AMENITY_TOKENS_EN: dict[str, str] = {
    "unfurnished": "unfurnished",
    "furnished": "furnished",
    "pool": "pool",
    "gym": "gym",
    "spa": "spa",
    "sea view": "sea view",
    "park view": "park view",
    "garden": "garden",
    "parking": "parking",
}

_AMENITY_TOKENS_AR: dict[str, str] = {
    "غير مفروش": "unfurnished",
    "مفروش": "furnished",
    "مسبح": "pool",
    "نادي رياضي": "gym",
    "جيم": "gym",
    "سبا": "spa",
    "إطلالة بحرية": "sea view",
    "اطلالة بحرية": "sea view",
    "حديقة": "garden",
    "مواقف": "parking",
    "موقف": "parking",
}


def extract_amenity_tokens(text: str) -> tuple[str, ...]:
    """Extract a sorted tuple of canonical amenity/view tokens from normalised text.

    The input should be the output of ``normalise_for_key(query)`` so that
    Arabic diacritics and alef forms are already normalised.  The function is
    pure and deterministic.

    "unfurnished" is checked before "furnished" so that "unfurnished" is not
    mis-detected as containing "furnished".
    """
    t = text  # already normalised by caller
    found: set[str] = set()
    has_unfurnished = False

    # Check longest tokens first within each language to avoid partial matches.
    # Because we sort by length descending, "unfurnished" is checked before "furnished",
    # and "غير مفروش" (unfurnished) is checked before "مفروش" (furnished).
    for token, canonical in sorted(_AMENITY_TOKENS_AR.items(), key=lambda x: len(x[0]), reverse=True):
        if token in t:
            found.add(canonical)
            if canonical == "unfurnished":
                has_unfurnished = True

    for token, canonical in sorted(_AMENITY_TOKENS_EN.items(), key=lambda x: len(x[0]), reverse=True):
        if _whole_word_in(token, t):
            found.add(canonical)
            if canonical == "unfurnished":
                has_unfurnished = True

    # "unfurnished" contains "furnished" as a substring (and Arabic "غير مفروش"
    # contains "مفروش"). If unfurnished was found, discard furnished.
    if has_unfurnished:
        found.discard("furnished")

    return tuple(sorted(found))


def _whole_word_in(word: str, text: str) -> bool:
    """Check if word appears as a whole word in text."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text, re.UNICODE))


def arabic_ratio(text: str) -> float:
    """Fraction of letter characters that are Arabic script."""
    letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
    if not letters:
        return 0.0
    arabic = sum(
        1 for ch in letters
        if "؀" <= ch <= "ۿ"
        or "ݐ" <= ch <= "ݿ"
        or "ﭐ" <= ch <= "﷿"
        or "ﹰ" <= ch <= "﻿"
    )
    return arabic / len(letters)


# ---------------------------------------------------------------------------
# Prompt-injection guard: strip instruction-like sentences (WP-4-guard)
# ---------------------------------------------------------------------------
#
# Two structural categories (generalised, not phrase-matched):
#
# (i)  Addresses the model AND steers its output.
# (ii) Mentions instructions/rules/prompt near a steering verb (within ~5 tokens).
#
# A sentence matching either category is dropped.

# --- Category (i): model-address cues ---
_ADDR_EN = (
    r"(?:"
    r"assistant|ai\b|(?:the\s+)?chatbot"
    r"|you\s+(?:must|should|will|are\s+now|have\s+to)"
    r"|from\s+now\s+on"
    r"|IMPORTANT\s*:|NOTE\s*:|SYSTEM\s*:"
    r")"
)
_ADDR_AR = (
    r"(?:"
    r"المساعد|يجب\s+عليك|من\s+الآن|من\s+الان"
    r")"
)
_ADDR_RE = re.compile(_ADDR_EN + r"|" + _ADDR_AR, re.IGNORECASE | re.UNICODE)

# --- Category (i): output-steering cues ---
_STEER_EN = (
    r"(?:"
    r"append|add|include|write|say|print|output"
    r"|reply|respond|answer|finish|end|start|begin"
    r"|translate|switch\s+language|reveal|repeat|show"
    r")\b"
)
_STEER_OBJ_EN = (
    r"(?:"
    r"code|word|token|prompt|instructions?"
    r")\b"
)
_STEER_AR = (
    r"(?:"
    r"أضف|اكتب|اطبع|أجب|قل|أظهر|كرر|ترجم"
    r"|الرمز|الكلمة|التعليمات|الموجه"
    r")"
)
_STEER_RE = re.compile(_STEER_EN + r"|" + _STEER_OBJ_EN + r"|" + _STEER_AR, re.IGNORECASE | re.UNICODE)

# --- Category (ii): instruction/rule/prompt mention near a steering verb ---
# Allow up to ~5 tokens (non-greedy) between the two halves.
_INSTR_NOUN_EN = r"(?:instructions?|rules?|prompt|directives?|commands?)"
_INSTR_NOUN_AR = r"(?:التعليمات|تعليمات|القواعد|الموجه|موجه\s+النظام)"
_STEER_VERB_EN = (
    r"(?:ignore|disregard|forget|override|skip|bypass|delete|remove|drop"
    r"|append|add|write|say|print|output|reveal|repeat|show|follow)"
)
_STEER_VERB_AR = r"(?:تجاهل|انسى|تخطى|أضف|اكتب|اطبع|أظهر|كرر)"

_GAP = r"(?:\S+\s+){0,5}"  # up to 5 tokens between parts

_INSTRVERB_PATTERNS: list[re.Pattern[str]] = [
    # EN: verb ... noun
    re.compile(_STEER_VERB_EN + r"\s+" + _GAP + _INSTR_NOUN_EN, re.IGNORECASE | re.UNICODE),
    # EN: noun ... verb
    re.compile(_INSTR_NOUN_EN + r"\s+" + _GAP + _STEER_VERB_EN, re.IGNORECASE | re.UNICODE),
    # AR: verb ... noun
    re.compile(_STEER_VERB_AR + r"\s+" + _GAP + _INSTR_NOUN_AR, re.IGNORECASE | re.UNICODE),
    # AR: noun ... verb (or reversed order in AR sentence)
    re.compile(_INSTR_NOUN_AR + r"\s+" + _GAP + _STEER_VERB_AR, re.IGNORECASE | re.UNICODE),
]

# --- Standalone high-signal phrases (always drop) ---
_STANDALONE_CUES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        r"end\s+of\s+(?:sources|context|data)",
        r"(?:admin|developer|debug)\s+mode",
        r"new\s+(?:rule|instruction)",
        r"تعليمات\s+جديدة",
        r"وضع\s+المسؤول",
    ]
]

# Sentence boundary: EN period/exclamation/question mark or AR question mark,
# followed by optional whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟])\s+")


def strip_instructions(text: str) -> str:
    """Remove sentences that look like injected instructions.

    Splits on sentence-ending punctuation (EN + AR), tests each sentence
    against two structural categories, and drops matches.  Returns the
    cleaned text with the surviving sentences joined by a single space.

    Category (i): sentence addresses the model AND steers its output.
    Category (ii): mentions instructions/rules/prompt near a steering verb.

    Pure, deterministic, no I/O.
    """
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    kept: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if _is_instruction(s):
            continue
        kept.append(s)
    return " ".join(kept)


def _is_instruction(sentence: str) -> bool:
    """Return True if the sentence matches any injection category."""
    # Standalone high-signal phrases
    for pat in _STANDALONE_CUES:
        if pat.search(sentence):
            return True
    # Category (i): addresses model + steers output
    if _ADDR_RE.search(sentence) and _STEER_RE.search(sentence):
        return True
    # Category (ii): instruction/rule/prompt near a steering verb
    for pat in _INSTRVERB_PATTERNS:
        if pat.search(sentence):
            return True
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonicalise_numbers_in_text(text: str) -> str:
    """Replace numbers + magnitude suffixes with their canonical forms in-place."""
    mag_keys = sorted(_MAGNITUDES.keys(), key=len, reverse=True)
    mag_alt = "|".join(re.escape(k) for k in mag_keys)
    # Pattern with magnitude suffix (eats optional space between number and suffix)
    num_mag_pattern = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\s*(" + mag_alt + r")\b",
        re.UNICODE,
    )
    # Pattern for plain numbers (commas only, no space consumption)
    num_plain_pattern = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\b",
        re.UNICODE,
    )

    def _replace_mag(m: re.Match) -> str:
        num_str = m.group(1).replace(",", "")
        try:
            val = Decimal(num_str)
        except InvalidOperation:
            return m.group(0)
        mag_str = m.group(2)
        if mag_str in _MAGNITUDES:
            val = val * _MAGNITUDES[mag_str]
        return canonical_number(val)

    def _replace_plain(m: re.Match) -> str:
        num_str = m.group(1).replace(",", "")
        if "," not in m.group(1):
            return m.group(0)  # No commas, leave as-is
        try:
            val = Decimal(num_str)
        except InvalidOperation:
            return m.group(0)
        return canonical_number(val)

    # First handle numbers with magnitude suffixes
    text = num_mag_pattern.sub(_replace_mag, text)
    # Then strip commas from comma-separated numbers
    text = num_plain_pattern.sub(_replace_plain, text)
    return text


def _replace_number_words(text: str) -> str:
    """Replace number words (zero to twelve) with digit strings."""
    all_words = {**_EN_NUMBER_WORDS, **_AR_NUMBER_WORDS}
    # Sort by length descending to match multi-word first
    for word, value in sorted(all_words.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.UNICODE)
        text = pattern.sub(str(value), text)
    return text


def _replace_currency(text: str) -> str:
    """Replace currency tokens with 'sar'."""
    for token in sorted(_CURRENCY_TOKENS, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.UNICODE)
        text = pattern.sub("sar", text)
    return text
