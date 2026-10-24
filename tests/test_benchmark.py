"""Tests for src/benchmark.py (rediscovery verdicts, no network)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import benchmark
from src.benchmark import NOT_IDENTIFIABLE, evaluate_benchmark


def ranked_frame(entries: list[tuple[str, str, str]]) -> pd.DataFrame:
    """entries: (target_id, description, host_taxname), rank = position + 1."""
    return pd.DataFrame([
        {"rank": i + 1, "target_id": target, "description": description,
         "host_taxname": host}
        for i, (target, description, host) in enumerate(entries)
    ])


def known_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


LCC_ROW = {
    "name": "LCC", "uniprot_id": "G9BY57",
    "organism": "uncultured bacterium (leaf-branch compost metagenome)",
    "notes": "control",
}
PHL7_ROW = {
    "name": "PHL7", "uniprot_id": "TODO_VERIFY",
    "organism": "plant-compost metagenome (unidentified)",
    "notes": "control",
}

LCC_HIT = ("AF-G9BY57-F1-model_v6", "Cutinase", "uncultured bacterium")
PHL7_HIT = ("AF-A0AA82WPD4-F1-model_v6", "Polyester hydrolase",
            "unidentified bacterium")
FILLER = ("AF-Q00001-F1-model_v6", "Lipase", "Escherichia coli")


def test_lcc_found_by_verified_accession():
    ranked = ranked_frame([FILLER, LCC_HIT, FILLER])
    result = evaluate_benchmark(ranked, known_frame([LCC_ROW]), top_n=10, required=["LCC"])
    assert result.status == "pass"
    row = result.table.iloc[0]
    assert bool(row["found"]) is True
    assert row["rank"] == 2
    assert row["identified_by"] == "uniprot:G9BY57"


def test_phl7_without_accession_is_inconclusive():
    ranked = ranked_frame([FILLER, PHL7_HIT])
    result = evaluate_benchmark(ranked, known_frame([PHL7_ROW]), top_n=10,
                                required=["PHL7"])
    assert result.status == "inconclusive"
    assert result.passed is False
    assert result.table.iloc[0]["status"] == NOT_IDENTIFIABLE
    assert "PHL7" in result.detail


def test_missing_control_inside_top_n_fails_with_rank():
    ranked = ranked_frame([FILLER] * 12 + [LCC_HIT])
    result = evaluate_benchmark(ranked, known_frame([LCC_ROW]), top_n=10,
                                required=["LCC"])
    assert result.status == "fail"
    assert result.table.iloc[0]["rank"] == 13
    assert "rank 13" in result.detail


def test_absent_control_with_accession_fails():
    ranked = ranked_frame([FILLER])
    result = evaluate_benchmark(ranked, known_frame([LCC_ROW]), top_n=10,
                                required=["LCC"])
    assert result.status == "fail"
    assert result.table.iloc[0]["status"] == "not_found"
    assert "absent" in result.detail


def test_both_controls_in_top_n_pass():
    ranked = ranked_frame([LCC_HIT, FILLER, PHL7_HIT, FILLER])
    known = known_frame([
        LCC_ROW,
        {**PHL7_ROW, "uniprot_id": "A0AA82WPD4"},
    ])
    result = evaluate_benchmark(ranked, known, top_n=10, required=["LCC", "PHL7"])
    assert result.status == "pass"
    assert result.passed is True
    assert result.table["in_top_n"].all()


def test_name_and_organism_matching_are_reported():
    ranked = ranked_frame([
        ("AF-Q1-F1-model_v6", "PHL7 polyester hydrolase", "Pseudomonas sp."),
    ])
    result = evaluate_benchmark(ranked, known_frame([PHL7_ROW]), top_n=10,
                                required=["PHL7"])
    assert result.status == "pass"
    assert result.table.iloc[0]["identified_by"] == "description:PHL7"


def test_no_required_controls_is_inconclusive():
    result = evaluate_benchmark(ranked_frame([]), known_frame([LCC_ROW]),
                                top_n=10, required=[])
    assert result.status == "inconclusive"


def test_load_known_enzymes_rejects_missing_columns(tmp_path):
    path = tmp_path / "known.csv"
    path.write_text("name,uniprot_id\nLCC,G9BY57\n")
    with pytest.raises(ValueError, match="missing columns"):
        benchmark.load_known_enzymes(path)


def test_real_control_table_has_verified_lcc_and_pending_phl7():
    path = Path(__file__).resolve().parents[1] / "data" / "known_pet_hydrolases.csv"
    known = benchmark.load_known_enzymes(path).set_index("name")
    assert known.loc["LCC", "uniprot_id"] == "G9BY57"
    assert known.loc["PHL7", "uniprot_id"].startswith("TODO")
    assert benchmark._clean_identifier(known.loc["PHL7", "uniprot_id"]) is None
