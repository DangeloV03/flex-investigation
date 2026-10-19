"""manage.csv read / write / append, including the 2026-10-16 wipe regression."""

from __future__ import annotations

import pytest

from flexsim.coex.manage_csv import append_manage_rows, find_manage_row, read_manage, write_manage
from coex_helpers import COMBO_DEFAULTS, default_manage_row, write_raw_manage


# ============================================================
# Analyzer — manage.csv helpers
# ============================================================

class TestManageCsv:

    def test_read_write_roundtrip(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        row = default_manage_row()
        write_raw_manage(path, [row])
        rows = read_manage(path)
        assert len(rows) == 1
        assert rows[0]["epsilon"] == "-2.0"

    def test_read_missing_file_returns_empty(self, tmp_path):
        assert read_manage(str(tmp_path / "nope.csv")) == []

    def test_find_manage_row_found(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        r1 = default_manage_row(epsilon=-2.0)
        r2 = default_manage_row(epsilon=-1.9)
        write_raw_manage(path, [r1, r2])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -1.9}
        idx = find_manage_row(rows, combo)
        assert idx == 1

    def test_find_manage_row_not_found(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        write_raw_manage(path, [default_manage_row()])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -1.5}
        assert find_manage_row(rows, combo) is None

    def test_find_manage_row_numeric_match(self, tmp_path):
        """Row stored as string '-2.0' should match float -2.0."""
        path = str(tmp_path / "manage.csv")
        write_raw_manage(path, [default_manage_row(epsilon="-2.0")])
        rows = read_manage(path)
        combo = {**COMBO_DEFAULTS, "epsilon": -2.0}
        assert find_manage_row(rows, combo) == 0

    def test_write_manage_all_fields(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        rows = [default_manage_row(mu_coex_FITTED="-3.5", isAnalyzed="2025-01-01")]
        write_manage(path, rows)
        rows2 = read_manage(path)
        assert rows2[0]["mu_coex_FITTED"] == "-3.5"
        assert rows2[0]["isAnalyzed"] == "2025-01-01"

    def test_append_rows_keeps_existing_fits(self, tmp_path):
        """Regression (2026-10-16 report): re-running a generator on an existing
        ledger used to blank mu_coex_FITTED via a stale column list."""
        path = str(tmp_path / "manage.csv")
        write_manage(path, [default_manage_row(mu_coex_FITTED="-3.5", mu_coex_FITTED_error="0.01")])
        added = append_manage_rows(path, [default_manage_row(epsilon=-1.9)])
        rows = read_manage(path)
        assert added == 1 and len(rows) == 2
        assert rows[0]["mu_coex_FITTED"] == "-3.5"
        assert rows[0]["mu_coex_FITTED_error"] == "0.01"

    def test_write_refuses_to_drop_rows(self, tmp_path):
        path = str(tmp_path / "manage.csv")
        write_manage(path, [default_manage_row(), default_manage_row(epsilon=-1.9)])
        with pytest.raises(RuntimeError, match="never deleted"):
            write_manage(path, [default_manage_row()])
