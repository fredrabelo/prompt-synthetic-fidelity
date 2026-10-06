"""Rebuild the exact prompts sent to the models.

Every call used the same system prompt, the same forced tool-call schema, and a single user message made of an
optional ``KNOWN FACTS`` block (respondent information) followed by a ``CURRENT STIMULUS`` block (the eleven items).

    A  no respondent information
    B  narrative persona (prose)
    C  the same facts as one labelled field per line
    D  B with party identification and ideology mirrored (Democrat/Republican respondents only)

Usage
-----
    python3 build_prompts.py show anes_20160001 B            # print the user message
    python3 build_prompts.py show anes_20160001 D --reversed  # reversed item order
    python3 build_prompts.py verify                            # recompute data/calls.csv prompt hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

SYSTEM_PROMPT = (
    "You are simulating one survey respondent in a social-science replication study. Answer strictly as this person "
    "would, using only the facts given. For each answer, briefly explain (1 short sentence) the reasoning behind it, "
    "in character. Output ONLY the structured fields requested."
)

NARRATIVE_TEMPLATE = (
    "It is [YEAR]. You are a [AGE] year-old, [MARST], [RACETH] [GENDER] with [EDUCATION] making [INCOME] per year, "
    "living in the United States. You are [IDEO], [REGIS] [PID] who [INTEREST] pays attention to what's going on in "
    "government and politics."
)

# Facts in canonical order with the label used by condition C. B states them in the same order.
FACT_FIELDS = [
    ("year", "Survey year"), ("age", "Age"), ("marst", "Marital status"), ("raceth", "Race/ethnicity"),
    ("gender", "Gender"), ("education", "Education"), ("income", "Annual income"), ("__country", "Country"),
    ("ideo", "Political ideology"), ("regis", "Voter registration status"), ("pid", "Party identification"),
    ("interest", "Political interest"),
]
COUNTRY = "United States"

PID_MIRROR = {"Democrat": "Republican", "Republican": "Democrat"}
IDEO_MIRROR = {
    "extremely liberal": "extremely conservative", "liberal": "conservative",
    "slightly liberal": "slightly conservative", "moderate": "moderate",
    "slightly conservative": "slightly liberal", "conservative": "liberal",
    "extremely conservative": "extremely liberal",
}


def load_instrument() -> list[dict]:
    return json.loads((DATA / "instrument.json").read_text())["items"]


def stimulus(items: list[dict], reverse: bool = False) -> str:
    items = items[::-1] if reverse else items
    lines = [f"- [{it['key']}] (scale 0-100): {it['text']}" for it in items]
    return "Answer each of the following, using the exact synthetic_question_key as given:\n" + "\n".join(lines)


def mirrored(row: dict) -> dict | None:
    """Party/ideology mirror for condition D; None when the respondent is not eligible (Independents)."""
    if row["pid"] not in PID_MIRROR or row["ideo"] not in IDEO_MIRROR:
        return None
    return {**row, "pid": PID_MIRROR[row["pid"]], "ideo": IDEO_MIRROR[row["ideo"]]}


def narrative(row: dict) -> str:
    text = NARRATIVE_TEMPLATE
    for key in ("year", "age", "marst", "raceth", "gender", "education", "income", "ideo", "regis", "pid", "interest"):
        text = text.replace(f"[{key.upper()}]", str(row[key]))
    return text


def structured(row: dict) -> str:
    return "\n".join(f"{label}: {COUNTRY if key == '__country' else row[key]}" for key, label in FACT_FIELDS)


def user_message(row: dict, condition: str, reverse: bool = False) -> str:
    items = load_instrument()
    block = f"CURRENT STIMULUS:\n{stimulus(items, reverse)}"
    if condition == "A":
        return block
    if condition == "B":
        facts = narrative(row)
    elif condition == "C":
        facts = structured(row)
    elif condition == "D":
        m = mirrored(row)
        if m is None:
            raise ValueError("condition D is defined only for Democrat/Republican respondents")
        facts = narrative(m)
    else:
        raise ValueError(condition)
    return f"KNOWN FACTS:\n{facts}\n\n{block}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify() -> int:
    """Recompute the prompt hash of every call in data/calls.csv from respondents.csv."""
    respondents = pd.read_csv(DATA / "respondents.csv", dtype=str).set_index("external_key")
    calls = pd.read_csv(DATA / "calls.csv", usecols=["external_key", "condition", "order", "prompt_sha256"])
    calls = calls[["external_key", "condition", "order", "prompt_sha256"]]
    cache: dict[tuple, str] = {}
    bad = 0
    for key, cond, order, expected in calls.itertuples(index=False):
        k = (key, cond, order)
        if k not in cache:
            cache[k] = sha256(user_message(respondents.loc[key].to_dict(), cond, order == "R"))
        bad += cache[k] != expected
    print(f"{len(calls):,} calls, {len(cache):,} distinct prompts, {bad} hash mismatches")
    return int(bad > 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("external_key")
    s.add_argument("condition", choices=list("ABCD"))
    s.add_argument("--reversed", action="store_true")
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "show":
        r = pd.read_csv(DATA / "respondents.csv", dtype=str).set_index("external_key").loc[a.external_key].to_dict()
        print(f"[system]\n{SYSTEM_PROMPT}\n\n[user]\n{user_message(r, a.condition, a.reversed)}")
    else:
        sys.exit(verify())
