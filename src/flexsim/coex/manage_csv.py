"""Read and update manage.csv, the one-row-per-combo status table of a coex campaign."""

from __future__ import annotations

import csv
import os
import sys

from flexsim.coex.layout import COMBO_KEY_FIELDS, combo_key_from_dict


MANAGE_FIELDS = COMBO_KEY_FIELDS + [
    "mu_coex_FLEX",
    "isSubmitted",
    "isRan",
    "isAnalyzed",
    "mu_coex_FITTED",
    "mu_coex_FITTED_error",
    "RequestForAdditionalData",
    "combo_path",
]


# ---------------------------------------------------------------------------
# manage.csv helpers
# ---------------------------------------------------------------------------

def read_manage(manage_path: str) -> list[dict]:
    if not os.path.isfile(manage_path):
        return []
    with open(manage_path, "r", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows: list[dict] = []
    for line_no, row in enumerate(raw_rows, start=2):
        row = {str(k).lstrip("\ufeff"): v for k, v in row.items()}
        row.setdefault("mu_coex_FITTED_error", "")
        row.setdefault("combo_path", "")
        missing = [f for f in COMBO_KEY_FIELDS if f not in row or row[f] is None]
        if missing:
            print(
                f"[manage_csv] WARNING: skipping malformed manage.csv line {line_no} "
                f"(missing {missing})",
                file=sys.stderr,
            )
            continue
        rows.append(row)
    return rows


def write_manage(manage_path: str, rows: list[dict]) -> None:
    """Rewrite manage.csv from the full row list. Rows are never deleted."""
    if os.path.isfile(manage_path):
        prior_count = len(read_manage(manage_path))
        if len(rows) < prior_count:
            raise RuntimeError(
                f"Refusing to write {manage_path}: {len(rows)} rows would replace "
                f"{prior_count} existing rows. manage.csv rows are never deleted."
            )
    with open(manage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANAGE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANAGE_FIELDS})


def _combo_field_match(row_val, combo_val) -> bool:
    """Match manage.csv strings to job dict values (exact or numeric)."""
    if row_val == combo_val:
        return True
    rs = str(row_val).strip()
    cs = str(combo_val).strip()
    if rs == cs:
        return True
    try:
        return float(rs) == float(cs)
    except (TypeError, ValueError):
        return False


def find_manage_row(rows: list[dict], combo: dict) -> int | None:
    """Return index of the manage row matching this combo, or None."""
    for i, row in enumerate(rows):
        if all(_combo_field_match(row.get(f, ""), combo.get(f, "")) for f in COMBO_KEY_FIELDS):
            return i
    return None


def update_manage_field(manage_path: str, combo: dict, updates: dict):
    """Update specific fields on the matching combo row, only if currently empty
    (for timestamp fields) or always (for numeric fields like RequestForAdditionalData)."""
    rows = read_manage(manage_path)
    idx = find_manage_row(rows, combo)
    if idx is None:
        print(f"[analyzer] WARNING: no manage row found for {combo}", file=sys.stderr)
        return
    for field, value in updates.items():
        # Timestamp fields: only write if empty (first writer wins)
        if field in ("isRan", "isAnalyzed") and rows[idx].get(field, ""):
            continue
        rows[idx][field] = value
    write_manage(manage_path, rows)


def append_manage_rows(manage_path: str, new_rows: list[dict]) -> int:
    """Append only new combo rows; never modify or drop existing manage.csv rows."""
    existing_rows = read_manage(manage_path)
    existing_keys = {combo_key_from_dict(row) for row in existing_rows}
    to_add = [row for row in new_rows if combo_key_from_dict(row) not in existing_keys]
    write_manage(manage_path, existing_rows + to_add)
    return len(to_add)
