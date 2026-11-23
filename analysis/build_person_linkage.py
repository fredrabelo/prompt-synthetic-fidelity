"""Build person-level identifiers for the ANES 2016 and 2020 records.

The input is the public ANES 2016-2020-2024 Panel Merged File (May 19, 2026
release). Only the two public case-ID columns are read. The output contains
the release's external record key and a stable person key; it contains no
survey responses or additional respondent attributes.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


EXPECTED_RECORDS = 7_526
EXPECTED_PEOPLE = 6_027
EXPECTED_LINKED_PEOPLE = 1_499


def _case_number(value: object, year: int) -> str | None:
    """Return the release's case-ID string, or None for a missing code."""
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw in {"-1", "-2", "-3", "-8", "-9"}:
        return None
    raw = re.sub(r"\.0+$", "", raw)
    raw = raw.replace(",", "")
    if not raw.isdigit():
        raise ValueError(f"Unexpected {year} case identifier: {value!r}")
    number = int(raw)
    if year == 2016:
        if not 0 <= number <= 9999:
            raise ValueError(f"2016 case identifier is outside the expected range: {value!r}")
        return f"{number:04d}"
    if not 200000 <= number <= 299999:
        raise ValueError(f"2020 case identifier is outside the expected range: {value!r}")
    return str(number)


def _read_panel(path: Path) -> pd.DataFrame:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in the ANES archive, found {len(members)}")
            with archive.open(members[0]) as stream:
                return pd.read_csv(stream, dtype=str, low_memory=False)
    return pd.read_csv(path, dtype=str, low_memory=False)


def build_linkage(panel_path: Path, respondents_path: Path) -> pd.DataFrame:
    panel = _read_panel(panel_path)
    required = {"V160001", "V200001"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"ANES panel file is missing required columns: {sorted(missing)}")

    panel = panel.loc[:, ["V160001", "V200001"]].copy()
    panel["case_2016"] = panel["V160001"].map(lambda x: _case_number(x, 2016))
    panel["case_2020"] = panel["V200001"].map(lambda x: _case_number(x, 2020))
    paired = panel.dropna(subset=["case_2016", "case_2020"])
    if paired["case_2016"].duplicated().any() or paired["case_2020"].duplicated().any():
        raise ValueError("The ANES panel file does not define a one-to-one 2016/2020 case linkage")
    link_2020_to_2016 = dict(zip(paired["case_2020"], paired["case_2016"]))

    respondents = pd.read_csv(respondents_path, dtype={"external_key": str})
    if "external_key" not in respondents:
        raise ValueError("respondents.csv must contain external_key")
    if respondents["external_key"].duplicated().any():
        raise ValueError("respondents.csv contains duplicate external keys")

    rows: list[dict[str, str]] = []
    unmatched_2020: list[str] = []
    for key in respondents["external_key"]:
        match = re.fullmatch(r"anes_(2016|2020)(\d{4,6})", key)
        if match is None:
            raise ValueError(f"Unexpected external key format: {key!r}")
        year, case = int(match.group(1)), match.group(2)
        if year == 2016:
            person_key = f"anes_2016{case}"
        else:
            linked_case = link_2020_to_2016.get(case)
            if linked_case is None:
                person_key = key
                unmatched_2020.append(key)
            else:
                person_key = f"anes_2016{linked_case}"
        rows.append({"external_key": key, "person_id": person_key})

    linkage = pd.DataFrame(rows)
    n_records = len(linkage)
    n_people = linkage["person_id"].nunique()
    n_linked = int((linkage.groupby("person_id")["external_key"].nunique() > 1).sum())
    if (n_records, n_people, n_linked) != (
        EXPECTED_RECORDS,
        EXPECTED_PEOPLE,
        EXPECTED_LINKED_PEOPLE,
    ):
        raise ValueError(
            "Linkage audit failed: expected 7,526 records, 6,027 people, and "
            f"1,499 linked people; observed {n_records}, {n_people}, and {n_linked}. "
            "Check that the May 19, 2026 ANES panel release and case-ID columns are used."
        )
    return linkage.sort_values("external_key").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="Official ANES panel CSV or ZIP")
    parser.add_argument(
        "--respondents",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "respondents.csv",
        help="Frozen release respondent profiles",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "person_linkage.csv",
        help="Output path for the audited linkage table",
    )
    args = parser.parse_args()
    linkage = build_linkage(args.panel, args.respondents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    linkage.to_csv(args.out, index=False)
    print(
        f"Wrote {len(linkage):,} records for {linkage.person_id.nunique():,} people; "
        f"{(linkage.groupby('person_id').external_key.nunique() > 1).sum():,} people have both waves."
    )


if __name__ == "__main__":
    main()
