import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_formal_integrity_manifest_is_complete():
    payload = json.loads(
        (ROOT / "results" / "tables" / "formal_integrity.json").read_text()
    )
    assert payload["status"] == "pass"
    assert payload["run_count"] == 18
    assert payload["expected_test_batches"] == 19


def test_table_ii_has_six_settings_and_fa_column():
    with (ROOT / "results" / "tables" / "table_ii.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert "FA_max_increase" in rows[0]
    assert rows[0]["setting"] == "standard"


def test_paired_bootstrap_has_reproducible_direction_result():
    payload = json.loads(
        (ROOT / "results" / "tables" / "paired_forward_reverse.json").read_text()
    )
    assert len(payload["rows"]) == 3
    assert all(row["replicates"] == 2000 for row in payload["rows"])
    assert all(row["conversation_count"] == 232 for row in payload["rows"])
    assert all(row["estimate_reverse_minus_forward"] > 0 for row in payload["rows"])
