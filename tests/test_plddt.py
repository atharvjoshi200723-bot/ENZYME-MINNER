"""Tests for src/plddt.py: pLDDT parsing, pocket selection, download cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import plddt
from tests.conftest import build_structure

# Triad at 25/27/28; residue 26 sits ~2 A from the Ser OG and residue 40 is
# far away, so the pocket should be exactly {25, 26, 27, 28}.
TRIAD = {"ser": 25, "his": 27, "asp": 28}


def pocket_structure(bfactors: dict[int, float]):
    residues = [
        (25, "SER", {"OG": (0.0, 0.0, 0.0), "CB": (-1.4, 0.3, 0.0)}),
        (26, "GLY", {"CA": (5.0, 0.0, 0.0)}),
        (27, "HIS", {"NE2": (2.6, 0.0, 0.0), "ND1": (-2.6, 0.0, 0.0)}),
        (28, "ASP", {"OD1": (-6.0, 0.0, 0.0), "OD2": (-3.6, 0.0, 0.0)}),
        (40, "LEU", {"CA": (40.0, 0.0, 0.0)}),
    ]
    structure = build_structure(residues)
    for residue in structure[0]["A"]:
        number = residue.get_id()[1]
        for atom in residue:
            atom.set_bfactor(bfactors.get(number, 50.0))
    return structure


def test_pocket_includes_triad_and_close_residues():
    structure = pocket_structure({})
    pocket = plddt.pocket_residues(structure, TRIAD, radius_angstrom=8.0)
    assert pocket == {25, 26, 27, 28}


def test_radius_controls_pocket_size():
    structure = pocket_structure({})
    pocket = plddt.pocket_residues(structure, TRIAD, radius_angstrom=1.0)
    assert pocket == {25, 27, 28}


def test_mean_plddt_is_atom_average():
    structure = pocket_structure({25: 90.0, 26: 80.0, 27: 70.0, 28: 60.0, 40: 0.0})
    pocket = plddt.pocket_residues(structure, TRIAD, radius_angstrom=8.0)
    # Residues have 2, 1, 2 and 2 atoms respectively -> (180 + 80 + 140 + 120) / 7
    assert plddt.mean_plddt(structure, pocket) == pytest.approx(520 / 7)


def test_pocket_below_threshold_fails():
    structure = pocket_structure({25: 79.0, 26: 79.0, 27: 79.0, 28: 79.0})
    result = plddt.assess_pocket_plddt(
        "hit_low", structure, TRIAD, radius_angstrom=8.0, min_mean_plddt=80.0
    )
    assert result.status == "below_threshold"
    assert result.plddt_pass is False
    assert result.mean_plddt == pytest.approx(79.0)
    assert "79.0" in result.reason and "80.0" in result.reason


def test_threshold_is_inclusive():
    structure = pocket_structure({25: 80.0, 26: 80.0, 27: 80.0, 28: 80.0})
    result = plddt.assess_pocket_plddt(
        "hit_edge", structure, TRIAD, radius_angstrom=8.0, min_mean_plddt=80.0
    )
    assert result.status == "pass"
    assert result.plddt_pass is True


def test_missing_triad_residue_is_reported_not_raised():
    structure = pocket_structure({})
    result = plddt.assess_pocket_plddt(
        "hit_missing", structure, {"ser": 25, "his": 27, "asp": 999},
        radius_angstrom=8.0, min_mean_plddt=80.0,
    )
    assert result.status == "error"
    assert result.plddt_pass is False
    assert "999" in result.reason


def test_unmapped_triad_position_is_an_error():
    structure = pocket_structure({})
    result = plddt.assess_pocket_plddt(
        "hit_gap", structure, {"ser": 25, "his": None, "asp": 28},
        radius_angstrom=8.0, min_mean_plddt=80.0,
    )
    assert result.status == "error"
    assert "not mapped" in result.reason


# ---------------------------------------------------------------------------
# Target id / download handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_id, expected", [
    ("AF-A0A0G3BI90-F1-model_v6", "A0A0G3BI90"),
    ("AF-E4WTR0-F1-model_v6", "E4WTR0"),
    ("MGYP000123456789", None),
    ("AF-", None),
])
def test_accession_parsing(target_id, expected):
    assert plddt.accession_from_target_id(target_id) == expected


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = ""

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request {url}")
        return self.responses.pop(0)


def make_settings(tmp_path: Path) -> plddt.AlphaFoldSettings:
    return plddt.AlphaFoldSettings(
        api_url_template="https://af.test/api/prediction/{accession}",
        cache_dir=tmp_path / "af",
        timeout_sec=5,
        retries=1,
        min_request_interval_sec=0.0,
    )


def test_download_uses_prediction_endpoint_and_cache(tmp_path):
    session = FakeSession([
        FakeResponse(json_data=[{"pdbUrl": "https://af.test/files/model.pdb"}]),
        FakeResponse(content=b"HEADER test\n"),
    ])
    path = plddt.download_hit_structure("AF-P12345-F1-model_v6", make_settings(tmp_path),
                                        session=session, sleep=lambda _s: None)
    assert path is not None and path.exists()
    assert session.calls[0].endswith("/api/prediction/P12345")
    assert session.calls[1] == "https://af.test/files/model.pdb"

    # Second call must be served from disk.
    second = plddt.download_hit_structure(
        "AF-P12345-F1-model_v6", make_settings(tmp_path),
        session=FakeSession([]), sleep=lambda _s: None,
    )
    assert second == path


def test_download_returns_none_for_non_alphafold_target(tmp_path):
    session = FakeSession([])
    result = plddt.download_hit_structure("MGYP0001", make_settings(tmp_path),
                                          session=session, sleep=lambda _s: None)
    assert result is None
    assert session.calls == []


def test_download_returns_none_when_api_has_no_entry(tmp_path):
    session = FakeSession([FakeResponse(status_code=404)])
    result = plddt.download_hit_structure("AF-Q9XYZ1-F1-model_v6",
                                          make_settings(tmp_path),
                                          session=session, sleep=lambda _s: None)
    assert result is None
