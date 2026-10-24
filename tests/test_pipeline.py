"""Tests for src/pipeline.py, including the fully offline demo mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from Bio.PDB import PDBIO, StructureBuilder
import numpy as np

from src import pipeline
from tests.conftest import build_structure, triad_residues

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path: Path) -> Path:
    """Config that writes into tmp_path and points at the real curated table."""
    config = {
        "seed": {
            "pdb_id": "5XJH",
            "chain": "A",
            "cache_path": str(tmp_path / "seed.pdb"),
            "download_url": "https://example.invalid/{pdb_id}.pdb",
        },
        "triad": {
            "ser": 160, "his": 237, "asp": 206,
            "max_ser_his_dist_angstrom": 4.5,
            "max_asp_his_dist_angstrom": 4.5,
        },
        "foldseek": {
            "server_url": "https://search.foldseek.test",
            "endpoints": {
                "submit": "/api/ticket",
                "ticket": "/api/ticket/{ticket_id}",
                "result": "/api/result/{ticket_id}/{db_index}",
                "databases": "/api/databases",
            },
            "databases": ["afdb50"],
            "mode": "3diaa",
            "max_hits": 500,
            "min_probability": 0.8,
            "cache_dir": str(tmp_path / "foldseek"),
        },
        "plddt": {
            "pocket_radius_angstrom": 8.0,
            "min_mean_plddt": 80.0,
            "api_url": "https://af.test/prediction/{accession}",
            "cache_dir": str(tmp_path / "af"),
        },
        "thermo": {
            "data_file": str(ROOT / "data" / "thermophiles.csv"),
            "unknown_host_score": 0.0,
            "match_column": "genus",
            "temperature_column": "approx_T_opt_C",
        },
        "output": {
            "results_csv": str(tmp_path / "results" / "ranked.csv"),
            "funnel_json": str(tmp_path / "results" / "funnel.json"),
            "top_n": 20,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_demo_pipeline_runs_with_no_network(tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("demo mode must not use the network")

    monkeypatch.setattr(pipeline.requests, "get", no_network)
    monkeypatch.setattr("src.plddt.requests.Session", no_network)

    result = pipeline.run_pipeline(write_config(tmp_path), demo=True)

    assert result.demo is True
    assert result.funnel == {
        "foldseek_hits": 10,
        "fold_match": 10,
        "structure_available": 10,
        "triad_intact": 6,
        "confident_pocket": 5,
        "final_shortlist": 5,
        "dropped_no_structure": 0,
        "dropped_triad": 4,
        "dropped_low_plddt": 1,
    }
    assert len(result.hits) == 5
    assert result.hits["target_id"].str.startswith("DEMO-").all()
    assert result.results_csv == tmp_path / "results" / "demo_ranked_candidates.csv"
    assert result.results_csv.exists()
    assert (tmp_path / "results" / "demo_funnel.json").exists()

    csv = pd.read_csv(result.results_csv)
    assert len(csv) == 5
    assert csv["rank"].tolist() == [1, 2, 3, 4, 5]
    # Thermus (75 C) ranks first; the 60 pLDDT metagenome hit was filtered out.
    assert csv.loc[0, "target_id"] == "DEMO-THERMUS-01"
    assert csv.loc[0, "matched_genus"] == "Thermus"
    assert "DEMO-METAGENOME-10" not in set(csv["target_id"])


def test_demo_rows_record_real_triad_and_plddt_measurements(tmp_path):
    result = pipeline.run_pipeline(write_config(tmp_path), demo=True)
    top = result.hits.iloc[0]

    assert top["ser_pos"] == 3 and top["his_pos"] == 9 and top["asp_pos"] == 6
    assert top["ser_his_dist"] == pytest.approx(2.8)
    assert top["asp_his_dist"] == pytest.approx(1.2)
    assert top["mean_plddt"] == pytest.approx(95.0)
    assert top["triad_status"] == "pass"
    assert top["plddt_status"] == "pass"


def test_missing_probability_column_keeps_all_hits():
    hits = pd.DataFrame({"target_id": ["A", "B"]})
    assert len(pipeline._fold_matches(hits, 0.8)) == 2


def test_select_query_prefers_the_seed_chain():
    hits = pd.DataFrame({
        "query": ["job_A", "job_B", "job_B"],
        "target_id": ["a1", "b1", "b2"],
    })
    selected = pipeline._select_query(hits, "A")
    assert selected["target_id"].tolist() == ["a1"]


def test_evaluate_hits_integrates_triad_and_plddt():
    hits = pd.DataFrame([{
        "target_id": "AF-TEST-1",
        "description": "test",
        "database": "afdb50",
        "probability": 0.99,
        "evalue": 1e-20,
        "seq_identity": 50.0,
        "qaln": "MKSEFDGIHL",
        "taln": "MKSEFDGIHL",
        "qstart": 1,
        "tstart": 1,
        "taxname": "Thermus thermophilus",
    }, {
        "target_id": "MGYP-NO-STRUCTURE",
        "description": "no model",
        "database": "mgnify_esm30",
        "probability": 0.95,
        "evalue": 1e-10,
        "seq_identity": 40.0,
        "qaln": "MKSEFDGIHL",
        "taln": "MKSEFDGIHL",
        "qstart": 1,
        "tstart": 1,
        "taxname": None,
    }])

    structure = build_structure(triad_residues(ser_resnum=3, his_resnum=9, asp_resnum=6))
    for residue in structure[0]["A"]:
        for atom in residue:
            atom.set_bfactor(90.0)

    # Build just the sections evaluate_hits reads.
    config = {
        "seed": {"chain": "A"},
        "triad": {"max_ser_his_dist_angstrom": 4.5,
                  "max_asp_his_dist_angstrom": 4.5},
        "plddt": {"pocket_radius_angstrom": 8.0, "min_mean_plddt": 80.0},
        "foldseek": {"min_probability": 0.8},
    }

    kept, counters = pipeline.evaluate_hits(
        hits, {"ser": 3, "his": 9, "asp": 6}, config,
        structure_provider=lambda tid: structure if tid == "AF-TEST-1" else None,
    )

    assert counters["foldseek_hits"] == 2
    assert counters["fold_match"] == 2
    assert counters["structure_available"] == 1
    assert counters["triad_intact"] == 1
    assert counters["confident_pocket"] == 1
    assert counters["final_shortlist"] == 1
    assert counters["dropped_no_structure"] == 1
    assert len(kept) == 1
    assert kept.loc[0, "mean_plddt"] == pytest.approx(90.0)


def test_query_positions_account_for_pdb_offset(tmp_path):
    """5XJH-style numbering: chain residues 30..292 -> query indices 1..263."""
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure("seed")
    builder.init_model(0)
    builder.init_chain("A")
    for resnum in range(30, 293):
        builder.init_residue("GLY", " ", resnum, " ")
        builder.init_atom("CA", np.array([float(resnum), 0.0, 0.0]),
                          1.0, 1.0, " ", "CA", resnum, "C")
    seed_path = tmp_path / "seed.pdb"
    io = PDBIO()
    io.set_structure(builder.get_structure())
    io.save(str(seed_path))

    positions = pipeline.query_triad_positions(
        seed_path, "A", {"ser": 160, "his": 237, "asp": 206}
    )
    assert positions == {"ser": 131, "his": 208, "asp": 177}
    assert pipeline.seed_chain_length(seed_path, "A") == 263


def test_query_positions_fail_on_missing_triad_residue(tmp_path):
    structure = build_structure(triad_residues())
    seed_path = tmp_path / "seed.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(seed_path))
    with pytest.raises(pipeline.PipelineError, match="not present"):
        pipeline.query_triad_positions(
            seed_path, "A", {"ser": 25, "his": 999, "asp": 28}
        )


def test_query_length_mismatch_is_fatal():
    hits = pd.DataFrame({"qlen": [262]})
    with pytest.raises(pipeline.PipelineError, match="does not match"):
        pipeline.check_query_length(hits, 263)
