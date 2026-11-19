"""Generate eval/sets/templated_{dev,test}.jsonl from data/index/listings.parquet.

Seeded and deterministic: rerunning it reproduces both files byte for byte.
`sql_truth` is computed here with DuckDB using the AggregateOp semantics in
app/domain/ports.py (price ops target annual_rent_sar when purpose == rent, else
price_sar; rows with a null target are excluded). No LLM is involved.

Usage (from the repo root):
    .venv/Scripts/python.exe eval/sets/gen_templated.py           # write the files
    .venv/Scripts/python.exe eval/sets/gen_templated.py --check   # regenerate in memory, diff
"""

from __future__ import annotations

import json
import random
import sys
from decimal import Decimal
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.text import canonical_number

SETS = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "index" / "listings.parquet"
SEED = 20261119
N_DEV, N_TEST = 60, 40
AVG_OPS = {"avg_price", "median_price", "avg_area"}

# ---------------------------------------------------------------------------
# Truth (shared with gen_sets.py)
# ---------------------------------------------------------------------------


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW l AS SELECT * FROM read_parquet('{PARQUET.as_posix()}')")
    return con


def _where(f: dict) -> tuple[str, list, str]:
    target = "annual_rent_sar" if f.get("purpose") == "rent" else "price_sar"
    conds, params = ["TRUE"], []
    for key in ("purpose", "property_type", "city", "district"):
        if f.get(key) is not None:
            conds.append(f"{key} = ?")
            params.append(f[key])
    bounds = {
        "bedrooms_min": ("bedrooms", ">="), "bedrooms_max": ("bedrooms", "<="),
        "price_min_sar": (target, ">="), "price_max_sar": (target, "<="),
        "area_min_sqm": ("area_sqm", ">="), "area_max_sqm": ("area_sqm", "<="),
    }
    for key, (col, cmp) in bounds.items():
        if f.get(key) is not None:
            conds.append(f"{col} {cmp} ?")
            params.append(f[key])
    return " AND ".join(conds), params, target


def n_matched(con, f: dict) -> int:
    where, params, _ = _where(f)
    return con.execute(f"SELECT count(*) FROM l WHERE {where}", params).fetchone()[0]


def sql_truth(con, op: str, f: dict) -> tuple[int | float | None, list[str]]:
    """Return (value, entity_ids). entity_ids are the listings at the min/max (max 5)."""
    where, params, target = _where(f)
    if op == "count":
        return con.execute(f"SELECT count(*) FROM l WHERE {where}", params).fetchone()[0], []
    col = "area_sqm" if op == "avg_area" else target
    where = f"{where} AND {col} IS NOT NULL"
    if op == "avg_area":
        v = con.execute(f"SELECT round(avg({col}), 2) FROM l WHERE {where}", params).fetchone()[0]
        return (None if v is None else float(v)), []
    agg = {"avg_price": "avg", "median_price": "median", "min_price": "min", "max_price": "max"}[op]
    v = con.execute(f"SELECT round({agg}({col}))::BIGINT FROM l WHERE {where}", params).fetchone()[0]
    ids: list[str] = []
    if op in ("min_price", "max_price") and v is not None:
        ids = [r[0] for r in con.execute(
            f"SELECT id FROM l WHERE {where} AND {col} = ? ORDER BY id LIMIT 5", [*params, v]
        ).fetchall()]
    return v, ids


def canon(v: float) -> str:
    return canonical_number(Decimal(str(v)))


def case(cid, set_, split, lang, turns, *, route, filters=None, op=None, truth=None,
         entity_ids=(), refuse=False, llm=False, answer_numbers=(), kind=None, canary=None):
    return {
        "id": cid, "set": set_, "split": split, "lang": lang, "turns": list(turns), "llm": llm,
        "expect": {
            "route": route, "filters": dict(filters or {}), "op": op, "sql_truth": truth,
            "tolerance_pct": 0.5 if op in AVG_OPS else 0.0,
            "entity_ids": list(entity_ids), "refuse": refuse,
            "answer_numbers": list(answer_numbers), "kind": kind, "canary": canary,
        },
    }


def dumps(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows)


# ---------------------------------------------------------------------------
# Wording (paraphrased templates, EN and AR)
# ---------------------------------------------------------------------------

TYPE_EN = {"apartment": ["apartments", "flats"], "villa": ["villas"], "land": ["land plots", "plots of land"],
           "floor": ["floor units"], "building": ["buildings"]}
TYPE_EN_1 = {"apartment": ["apartment", "flat"], "villa": ["villa"], "land": ["plot of land"],
             "floor": ["floor unit"], "building": ["building"]}
TYPE_AR = {"apartment": "شقق", "villa": "فلل", "land": "أرض", "floor": "دور", "building": "عمارة"}
TYPE_AR_1 = {"apartment": "شقة", "villa": "فيلا", "land": "أرض", "floor": "دور", "building": "عمارة"}
CITY_AR = {"Riyadh": "الرياض", "Jeddah": "جدة", "Khobar": "الخبر", "Dammam": "الدمام", "Makkah": "مكة",
           "Abha": "أبها", "Jazan": "جازان", "Madinah": "المدينة"}
PURPOSE_EN = {"sale": ["for sale", "to buy"], "rent": ["for rent", "to rent"]}
PURPOSE_AR = {"sale": "للبيع", "rent": "للإيجار"}
NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def money_en(rng, v: int) -> str:
    if v >= 1_000_000:
        m = v / 1_000_000
        m_s = f"{m:g}"
        return rng.choice([f"{m_s}M", f"{m_s} million SAR", f"SAR {v:,}", f"{v:,} riyals"])
    k = v // 1000
    return rng.choice([f"{k}k", f"{v:,} SAR", f"SAR {v:,}"])


def money_ar(rng, v: int) -> str:
    if v >= 1_000_000:
        s = f"{v / 1_000_000:g}"
        s = rng.choice([s, s.replace(".", "٫").translate(AR_DIGITS)])
        return f"{s} مليون ريال"
    s = str(v // 1000)
    return f"{rng.choice([s, s.translate(AR_DIGITS)])} ألف ريال"


def phrase_en(rng, op: str, f: dict, rent_monthly: bool) -> str:
    t = f["property_type"]
    many, one = rng.choice(TYPE_EN[t]), rng.choice(TYPE_EN_1[t])
    purpose = rng.choice(PURPOSE_EN[f["purpose"]])
    loc = f"in {f['district']}, {f['city']}" if f.get("district") else f"in {f['city']}"
    bed = ""
    if f.get("bedrooms_min") is not None:
        n = f["bedrooms_min"]
        bed = rng.choice([f"{n}-bed", f"{n} bedroom", f"{NUM_WORD.get(n, n)} bedroom", f"{n}br"])
    price = ""
    if f.get("price_max_sar") is not None:
        v = f["price_max_sar"]
        if rent_monthly:
            price = f"{rng.choice(['under', 'below'])} {v // 12:,} SAR {rng.choice(['a month', 'per month'])}"
        else:
            per = rng.choice([" a year", " per year"]) if f["purpose"] == "rent" else ""
            price = f"{rng.choice(['under', 'below', 'less than', 'up to'])} {money_en(rng, v)}{per}"
    elif f.get("price_min_sar") is not None:
        v = f["price_min_sar"]
        per = " per year" if f["purpose"] == "rent" else ""
        price = f"{rng.choice(['over', 'above', 'more than'])} {money_en(rng, v)}{per}"
    noun = f"{bed} {many}".strip()
    tail = f" {price}" if price else ""
    if op == "lookup":
        return rng.choice([
            f"{noun} {purpose} {loc}{tail}",
            f"Show me {noun} {purpose} {loc}{tail}",
            f"I'm looking for {noun} {purpose} {loc}{tail}",
            f"Any {noun} {purpose} {loc}{tail}?",
            f"Find {noun} {purpose} {loc}{tail}",
        ])
    if op == "count":
        return rng.choice([
            f"How many {noun} are {purpose} {loc}{tail}?",
            f"Number of {noun} {purpose} {loc}{tail}",
            f"how many {noun} {purpose} {loc}{tail} do you have",
        ])
    one_n = f"{bed} {one}".strip()
    rentword = "yearly rent" if f["purpose"] == "rent" else "price"
    if op == "avg_price":
        return rng.choice([
            f"What is the average {rentword} of {noun} {purpose} {loc}{tail}?",
            f"average {rentword} for {noun} {purpose} {loc}{tail}",
            f"What's the mean {rentword} of {noun} {purpose} {loc}{tail}?",
        ])
    if op == "median_price":
        return rng.choice([
            f"What is the median {rentword} of {noun} {purpose} {loc}{tail}?",
            f"median {rentword} for {noun} {purpose} {loc}{tail}",
        ])
    if op == "min_price":
        return rng.choice([
            f"cheapest {one_n} {purpose} {loc}{tail}",
            f"What is the cheapest {one_n} {purpose} {loc}{tail}?",
            f"Which {one_n} {purpose} {loc}{tail} has the lowest price?",
            f"least expensive {one_n} {purpose} {loc}{tail}",
        ])
    if op == "max_price":
        return rng.choice([
            f"most expensive {one_n} {purpose} {loc}{tail}",
            f"Which {one_n} {purpose} {loc}{tail} has the highest price?",
            f"What's the priciest {one_n} {purpose} {loc}{tail}?",
        ])
    if op == "avg_area":
        return rng.choice([
            f"What is the average size of {noun} {purpose} {loc}{tail}?",
            f"average area of {noun} {purpose} {loc}{tail} in sqm",
        ])
    raise ValueError(op)


def phrase_ar(rng, op: str, f: dict) -> str:
    many, one = TYPE_AR[f["property_type"]], TYPE_AR_1[f["property_type"]]
    purpose, city = PURPOSE_AR[f["purpose"]], CITY_AR[f["city"]]
    bed = ""
    if f.get("bedrooms_min") is not None:
        n = str(f["bedrooms_min"])
        bed = f" {rng.choice([n, n.translate(AR_DIGITS)])} غرف"
    price = ""
    if f.get("price_max_sar") is not None:
        price = f" {rng.choice(['أقل من', 'تحت'])} {money_ar(rng, f['price_max_sar'])}"
    elif f.get("price_min_sar") is not None:
        price = f" {rng.choice(['أكثر من', 'فوق'])} {money_ar(rng, f['price_min_sar'])}"
    body = f"{many}{bed} {purpose} في {city}{price}"
    body1 = f"{one}{bed} {purpose} في {city}{price}"
    if op == "lookup":
        return rng.choice([body, f"أبحث عن {body}", f"أريد {body1}"])
    if op == "count":
        return f"كم عدد {body}؟"
    if op == "avg_price":
        return rng.choice([f"ما متوسط سعر {body}؟", f"متوسط أسعار {body}"])
    if op == "median_price":
        return f"ما هو السعر الوسيط لـ {body}؟"  # known router gap: no Arabic median cue
    if op == "min_price":
        return rng.choice([f"ما هي أرخص {body1}؟", f"أرخص {body1}"])
    if op == "max_price":
        return rng.choice([f"ما هي أغلى {body1}؟", f"أغلى {body1}"])
    if op == "avg_area":
        return f"ما متوسط مساحة {body}؟"  # known router gap: may route to avg_price
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Scope sampling
# ---------------------------------------------------------------------------

TYPES = ("apartment", "villa", "land", "floor", "building")


def nice(v: float) -> int:
    """Round to a 'nice' bound: 2 significant figures, then to 50k/500/... steps."""
    if v <= 0:
        return 0
    mag = 10 ** (len(str(int(v))) - 2)
    return int(round(v / mag) * mag)


def candidate_scopes(con, rng) -> list[dict]:
    combos = con.execute(
        "SELECT purpose, property_type, city, count(*) n FROM l WHERE property_type IN "
        f"({','.join('?' * len(TYPES))}) GROUP BY ALL HAVING n >= 8 ORDER BY n DESC, purpose, property_type, city",
        list(TYPES),
    ).fetchall()
    scopes: list[dict] = []
    for purpose, ptype, city, _n in combos:
        if city not in CITY_AR:
            continue
        base = {"purpose": purpose, "property_type": ptype, "city": city}
        scopes.append(dict(base))
        where, params, target = _where(base)
        beds = con.execute(
            f"SELECT bedrooms, count(*) c FROM l WHERE {where} AND bedrooms IS NOT NULL "
            "GROUP BY 1 HAVING c >= 4 ORDER BY c DESC, bedrooms LIMIT 3", params).fetchall()
        for b, _c in beds:
            scopes.append({**base, "bedrooms_min": int(b), "bedrooms_max": int(b)})
        dists = con.execute(
            f"SELECT district, count(*) c FROM l WHERE {where} AND district IS NOT NULL "
            "GROUP BY 1 HAVING c >= 5 ORDER BY c DESC, district LIMIT 2", params).fetchall()
        for d, _c in dists:
            scopes.append({**base, "district": d})
        med = con.execute(f"SELECT median({target}) FROM l WHERE {where} AND {target} IS NOT NULL",
                          params).fetchone()[0]
        if med:
            scopes.append({**base, "price_max_sar": nice(med * rng.choice([0.8, 1.0, 1.2]))})
            scopes.append({**base, "price_min_sar": nice(med * rng.choice([0.9, 1.1]))})
            if beds:
                scopes.append({**base, "bedrooms_min": int(beds[0][0]), "bedrooms_max": int(beds[0][0]),
                               "price_max_sar": nice(med * 1.2)})
    return scopes


OP_PLAN = ["lookup"] * 30 + ["count"] * 20 + ["avg_price"] * 15 + ["median_price"] * 8 + \
    ["min_price"] * 10 + ["max_price"] * 9 + ["avg_area"] * 8


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    con = connect()
    scopes = candidate_scopes(con, rng)
    ops = list(OP_PLAN)
    rng.shuffle(ops)
    used: set[str] = set()
    rows: list[dict] = []
    for op in ops:
        lang = "ar" if rng.random() < 0.3 else "en"
        for _ in range(200):
            f = dict(rng.choice(scopes))
            if lang == "ar" and f.get("district"):  # districts have no Arabic aliases
                continue
            key = json.dumps([op, sorted(f.items())])
            if key in used:
                continue
            nm = n_matched(con, f)
            if nm < (5 if op in AVG_OPS else 3):
                continue
            if op == "avg_area":
                truth, _ = sql_truth(con, op, f)
                if truth is None:
                    continue
            used.add(key)
            break
        else:
            raise RuntimeError(f"no scope for {op}")
        monthly = (lang == "en" and f["purpose"] == "rent" and f.get("price_max_sar") is not None
                   and rng.random() < 0.5)
        if monthly:  # express the bound per month; the filter stays annual (x12)
            f["price_max_sar"] = max(12_000, (f["price_max_sar"] // 12 // 500) * 500 * 12)
        text = phrase_ar(rng, op, f) if lang == "ar" else phrase_en(rng, op, f, monthly)
        text = " ".join(text.split())
        if op == "lookup":
            rows.append({"op": None, "route": "lookup", "f": f, "lang": lang, "text": text, "truth": None, "ids": []})
        else:
            truth, ids = sql_truth(con, op, f)
            rows.append({"op": op, "route": "aggregate", "f": f, "lang": lang, "text": text, "truth": truth, "ids": ids})
    dev_rows, test_rows = rows[:N_DEV], rows[N_DEV:N_DEV + N_TEST]
    out = []
    for split, part in (("dev", dev_rows), ("test", test_rows)):
        cases = []
        llm_left = 4 if split == "dev" else 0  # dev LLM subset: 4 exact-valued aggregates
        for j, r in enumerate(part, 1):
            llm = bool(llm_left and r["op"] in ("count", "min_price", "max_price", "avg_price")
                       and r["truth"] is not None)
            if llm:
                llm_left -= 1
            cases.append(case(
                f"tpl-{split}-{j:03d}", "templated", split, r["lang"], [r["text"]],
                route=r["route"], filters=r["f"], op=r["op"], truth=r["truth"], entity_ids=r["ids"],
                llm=llm, answer_numbers=[canon(r["truth"])] if llm else [],
            ))
        out.append(cases)
    return out[0], out[1]


def main() -> int:
    dev, test = build()
    texts = {"templated_dev.jsonl": dumps(dev), "templated_test.jsonl": dumps(test)}
    if "--check" in sys.argv:
        bad = [n for n, t in texts.items() if (SETS / n).read_text(encoding="utf-8") != t]
        print("reproducible" if not bad else f"DIFFERS: {bad}")
        return 1 if bad else 0
    for name, text in texts.items():
        (SETS / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"dev={len(dev)} test={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
