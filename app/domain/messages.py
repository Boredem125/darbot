"""Fixed message strings. Every module imports from here; there is only one copy.

Placeholders that dg-prompt-eval-lead finalises in Wave 3.
"""

REFUSAL_EN = "I do not have information on that. I can help with property listings in Saudi Arabia and Dar Global projects."
REFUSAL_AR = "لا تتوفر لدي معلومات عن ذلك. يمكنني المساعدة في العقارات المعروضة في السعودية ومشاريع دار جلوبال."

BUSY_EN = "The assistant is busy right now. Please try again in a minute."
BUSY_AR = "المساعد مشغول حالياً. يرجى المحاولة مرة أخرى بعد دقيقة."


# ---------------------------------------------------------------------------
# Zero-hit relaxation templates (WP-4-nomatch)
# ---------------------------------------------------------------------------


def _fmt(n: int) -> str:
    return f"{n:,}"


def nomatch_relaxed_en(
    *,
    description: str,
    relaxed_field: str,
    relaxed_count: int,
    relaxed_description: str,
) -> str:
    """EN: 'No {description}. There are {count} {relaxed_description}. Try ...'"""
    hints = {
        "bedrooms": "Try asking without the bedroom count.",
        "price": "Try asking without the price range.",
        "area": "Try asking without the area constraint.",
        "district": "Try asking for the whole city.",
    }
    hint = hints.get(relaxed_field, "Try broadening your search.")
    return (
        f"No {description} in the current listings. "
        f"There are {_fmt(relaxed_count)} {relaxed_description}. "
        f"{hint}"
    )


def nomatch_relaxed_ar(
    *,
    description: str,
    relaxed_field: str,
    relaxed_count: int,
    relaxed_description: str,
) -> str:
    """AR: same structure in Arabic."""
    hints = {
        "bedrooms": "حاول البحث بدون تحديد عدد الغرف.",
        "price": "حاول البحث بدون تحديد نطاق السعر.",
        "area": "حاول البحث بدون تحديد المساحة.",
        "district": "حاول البحث في المدينة بأكملها.",
    }
    hint = hints.get(relaxed_field, "حاول توسيع نطاق البحث.")
    return (
        f"لا توجد {description} في القوائم الحالية. "
        f"يوجد {_fmt(relaxed_count)} {relaxed_description}. "
        f"{hint}"
    )


def nomatch_none_en(description: str) -> str:
    """EN fallback: nothing matched even after relaxing."""
    return f"No {description} in the current listings."


def nomatch_none_ar(description: str) -> str:
    """AR fallback."""
    return f"لا توجد {description} في القوائم الحالية."


def describe_filters_en(
    *,
    property_type: str | None = None,
    purpose: str | None = None,
    city: str | None = None,
    district: str | None = None,
    bedrooms: str | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
) -> str:
    """Build a human-readable EN description, e.g. '2-bedroom villas for sale in Riyadh under SAR 50,000'."""
    parts: list[str] = []
    if bedrooms is not None:
        parts.append(f"{bedrooms}-bedroom")
    if property_type is not None:
        parts.append(property_type + "s")
    else:
        parts.append("properties")
    if purpose is not None:
        parts.append(f"for {purpose}")
    loc_parts: list[str] = []
    if district is not None:
        loc_parts.append(district)
    if city is not None:
        loc_parts.append(city)
    if loc_parts:
        parts.append("in " + ", ".join(loc_parts))
    # Price constraint
    if price_min is not None and price_max is not None:
        parts.append(f"between SAR {_fmt(price_min)} and SAR {_fmt(price_max)}")
    elif price_max is not None:
        parts.append(f"under SAR {_fmt(price_max)}")
    elif price_min is not None:
        parts.append(f"over SAR {_fmt(price_min)}")
    # Area constraint
    if area_min is not None and area_max is not None:
        parts.append(f"between {_fmt_area(area_min)} and {_fmt_area(area_max)} sqm")
    elif area_max is not None:
        parts.append(f"under {_fmt_area(area_max)} sqm")
    elif area_min is not None:
        parts.append(f"over {_fmt_area(area_min)} sqm")
    return " ".join(parts)


# Arabic localisation maps
_PTYPE_AR: dict[str, str] = {
    "villa": "فلل",
    "apartment": "شقق",
    "townhouse": "تاون هاوس",
    "duplex": "دوبلكس",
    "floor": "أدوار",
    "land": "أراضي",
    "building": "مباني",
    "office": "مكاتب",
    "shop": "محلات",
    "warehouse": "مستودعات",
    "chalet": "شاليهات",
    "farm": "مزارع",
    "other": "عقارات",
}

_CITY_AR: dict[str, str] = {
    "Riyadh": "الرياض",
    "Jeddah": "جدة",
    "Makkah": "مكة",
    "Madinah": "المدينة",
    "Dammam": "الدمام",
    "Khobar": "الخبر",
    "Tabuk": "تبوك",
    "Abha": "أبها",
    "Taif": "الطائف",
    "Buraidah": "بريدة",
    "Hail": "حائل",
    "Jazan": "جازان",
    "Najran": "نجران",
    "Yanbu": "ينبع",
    "Al Kharj": "الخرج",
}


def _fmt_area(v: float) -> str:
    return f"{int(v):,}" if v == int(v) else f"{v:,.1f}"


def describe_filters_ar(
    *,
    property_type: str | None = None,
    purpose: str | None = None,
    city: str | None = None,
    district: str | None = None,
    bedrooms: str | None = None,
    price_min: int | None = None,
    price_max: int | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
) -> str:
    """Build an Arabic description from filters."""
    parts: list[str] = []
    if property_type is not None:
        parts.append(_PTYPE_AR.get(property_type, property_type))
    else:
        parts.append("عقارات")
    if bedrooms is not None:
        parts.append(f"بـ {bedrooms} غرف")
    if purpose == "sale":
        parts.append("للبيع")
    elif purpose == "rent":
        parts.append("للإيجار")
    loc_parts: list[str] = []
    if district is not None:
        loc_parts.append(district)
    if city is not None:
        loc_parts.append(_CITY_AR.get(city, city))
    if loc_parts:
        parts.append("في " + "، ".join(loc_parts))
    # Price
    if price_min is not None and price_max is not None:
        parts.append(f"بين {_fmt(price_min)} و {_fmt(price_max)} ريال")
    elif price_max is not None:
        parts.append(f"أقل من {_fmt(price_max)} ريال")
    elif price_min is not None:
        parts.append(f"أكثر من {_fmt(price_min)} ريال")
    # Area
    if area_min is not None and area_max is not None:
        parts.append(f"بين {_fmt_area(area_min)} و {_fmt_area(area_max)} م2")
    elif area_max is not None:
        parts.append(f"أقل من {_fmt_area(area_max)} م2")
    elif area_min is not None:
        parts.append(f"أكثر من {_fmt_area(area_min)} م2")
    return " ".join(parts)
