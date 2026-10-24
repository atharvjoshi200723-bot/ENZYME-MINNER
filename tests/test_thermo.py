"""Tests for src/thermo.py: genus matching, unknown hosts, ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import thermo

THERMOPHILES = {"thermus": 75.0, "thermotoga": 80.0, "geobacillus": 60.0}


def test_load_thermophiles_reads_curated_table(tmp_path):
    path = tmp_path / "thermophiles.csv"
    path.write_text(
        "genus,approx_T_opt_C,source\n"
        "Thermus,75,doi:1\n"
        "Thermotoga,80,doi:1\n"
    )
    assert thermo.load_thermophiles(path) == {"thermus": 75.0, "thermotoga": 80.0}


def test_load_thermophiles_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,temp\nThermus,75\n")
    with pytest.raises(ValueError, match="genus"):
        thermo.load_thermophiles(path)


def test_load_real_curated_table():
    path = Path(__file__).resolve().parents[1] / "data" / "thermophiles.csv"
    table = thermo.load_thermophiles(path)
    assert table["thermus"] == 75.0
    assert table["thermobifida"] == 55.0
    assert all(value > 0 for value in table.values())


@pytest.mark.parametrize("taxname, genus", [
    ("Thermus thermophilus JL-18", "Thermus"),
    ("thermotoga maritima", "thermotoga"),
    ("marine metagenome", "marine"),
    (None, None),
    ("", None),
    (float("nan"), None),
])
def test_genus_from_taxname(taxname, genus):
    assert thermo.genus_from_taxname(taxname) == genus


def test_unknown_host_is_kept_and_flagged():
    hits = pd.DataFrame({
        "target_id": ["AF-1", "AF-2"],
        "taxname": ["Thermus thermophilus", None],
        "mean_plddt": [90.0, 95.0],
    })
    ranked = thermo.rank_by_host(hits, THERMOPHILES, unknown_host_score=0.0)

    assert len(ranked) == 2
    unknown = ranked[ranked["target_id"] == "AF-2"].iloc[0]
    assert unknown["host_status"] == thermo.HOST_UNKNOWN
    assert unknown["thermo_score"] == 0.0
    assert "unknown host" in unknown["host_note"]
    # Unknown host ranks below the thermophile even though its pLDDT is higher.
    assert ranked.loc[0, "target_id"] == "AF-1"
    assert ranked["rank"].tolist() == [1, 2]


def test_unmatched_genus_is_kept_and_flagged():
    hits = pd.DataFrame({"target_id": ["AF-E"], "taxname": ["Escherichia coli"]})
    ranked = thermo.rank_by_host(hits, THERMOPHILES)
    assert ranked.loc[0, "host_status"] == thermo.HOST_UNMATCHED
    assert ranked.loc[0, "host_genus"] == "Escherichia"


def test_matching_is_case_insensitive():
    hits = pd.DataFrame({"target_id": ["AF-T"], "taxname": ["THERMUS sp."]})
    ranked = thermo.rank_by_host(hits, THERMOPHILES)
    assert ranked.loc[0, "host_status"] == thermo.HOST_MATCHED
    assert ranked.loc[0, "approx_T_opt_C"] == 75.0


def test_missing_taxname_column_does_not_crash():
    hits = pd.DataFrame({"target_id": ["AF-1", "AF-2"]})
    ranked = thermo.rank_by_host(hits, THERMOPHILES)
    assert ranked["host_status"].tolist() == [thermo.HOST_UNKNOWN] * 2


def test_empty_table_works():
    ranked = thermo.rank_by_host(pd.DataFrame({"target_id": []}), THERMOPHILES)
    assert ranked.empty


def test_ranking_uses_plddt_as_tiebreaker():
    hits = pd.DataFrame({
        "target_id": ["AF-A", "AF-B"],
        "taxname": ["Thermus a", "Thermus b"],
        "mean_plddt": [85.0, 95.0],
    })
    ranked = thermo.rank_by_host(hits, THERMOPHILES)
    assert ranked["target_id"].tolist() == ["AF-B", "AF-A"]
