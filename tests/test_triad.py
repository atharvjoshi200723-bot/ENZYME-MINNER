"""Tests for src/triad.py: alignment mapping, residue types and geometry."""

from __future__ import annotations

import pytest

from src import triad
from tests.conftest import (
    QALN,
    QSTART,
    TALN,
    TRIAD_IN_ALIGNMENT,
    TSTART,
    build_structure,
    triad_residues,
)

THRESHOLDS = {
    "max_ser_his_dist_angstrom": 4.5,
    "max_asp_his_dist_angstrom": 4.5,
}


# ---------------------------------------------------------------------------
# Alignment mapping
# ---------------------------------------------------------------------------

def test_mapping_uses_alignment_offsets():
    """Triad positions map through gaps in both sequences, with offsets."""
    mapped = triad.map_triad_positions(
        QALN, TALN, {"ser": 14, "his": 17, "asp": 18},
        query_start=QSTART, target_start=TSTART,
    )
    assert mapped == {"ser": 25, "his": 27, "asp": 28}


def test_mapping_landing_in_target_gap_is_none():
    """Query residue 16 (Asp in the sample alignment) has no target residue."""
    mapped = triad.map_triad_positions(
        QALN, TALN, {"asp": 16}, query_start=QSTART, target_start=TSTART,
    )
    assert mapped == {"asp": None}


def test_mapping_outside_aligned_region_is_none():
    mapped = triad.map_triad_positions(
        QALN, TALN, {"ser": 5}, query_start=QSTART, target_start=TSTART,
    )
    assert mapped == {"ser": None}


def test_mapping_without_gaps_and_with_offset():
    mapped = triad.map_triad_positions(
        "ACDEFG", "ACDEFG", {"ser": 160}, query_start=158, target_start=5,
    )
    assert mapped == {"ser": 7}


def test_mapping_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="different lengths"):
        triad.map_triad_positions("ACDE", "ACD", {"ser": 1})


def test_mapping_rejects_zero_start():
    with pytest.raises(ValueError, match="1-indexed"):
        triad.map_triad_positions("ACDE", "ACDE", {"ser": 1}, query_start=0)


# ---------------------------------------------------------------------------
# Residue types
# ---------------------------------------------------------------------------

def test_residue_types_pass_on_correct_triad(passing_structure):
    mapped = {"ser": 25, "his": 27, "asp": 28}
    assert triad.check_triad_residue_types(passing_structure, mapped) == {
        "ser": True, "his": True, "asp": True,
    }


def test_wrong_residue_type_is_false():
    structure = build_structure(triad_residues(ser_resname="ALA"))
    mapped = {"ser": 25, "his": 27, "asp": 28}
    result = triad.check_triad_residue_types(structure, mapped)
    assert result["ser"] is False
    assert result["his"] is True


def test_missing_residue_and_gap_are_false(passing_structure):
    mapped = {"ser": 25, "his": 999, "asp": None}
    result = triad.check_triad_residue_types(passing_structure, mapped)
    assert result == {"ser": True, "his": False, "asp": False}


def test_missing_chain_is_false_not_a_crash(passing_structure):
    mapped = {"ser": 25, "his": 27, "asp": 28}
    result = triad.check_triad_residue_types(passing_structure, mapped, chain_id="Z")
    assert result == {"ser": False, "his": False, "asp": False}


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def test_distances_use_closer_asp_oxygen(passing_structure):
    distances = triad.measure_triad_distances(
        passing_structure, {"ser": 25, "his": 27, "asp": 28}
    )
    assert distances.ser_his == pytest.approx(2.6)
    assert distances.asp_his == pytest.approx(1.0)
    assert distances.asp_atom == "OD2"


def test_distance_picks_od1_when_it_is_closer():
    structure = build_structure(triad_residues(asp_od1=(-7.0, 0.0, 0.0)))
    # OD1 is 4.4 A and OD2 1.0 A away from ND1 -> OD2 stays the closer one.
    distances = triad.measure_triad_distances(
        structure, {"ser": 25, "his": 27, "asp": 28}
    )
    assert distances.asp_atom == "OD2"

    structure = build_structure(
        triad_residues(asp_od1=(-3.0, 0.0, 0.0), asp_od2=(-8.0, 0.0, 0.0))
    )
    distances = triad.measure_triad_distances(
        structure, {"ser": 25, "his": 27, "asp": 28}
    )
    assert distances.asp_atom == "OD1"
    assert distances.asp_his == pytest.approx(0.4)


def test_single_asp_oxygen_is_allowed():
    """OD1/OD2 are equivalent carboxylate oxygens, so one is enough."""
    structure = build_structure(triad_residues(asp_od1=None))
    distances = triad.measure_triad_distances(
        structure, {"ser": 25, "his": 27, "asp": 28}
    )
    assert distances.asp_atom == "OD2"


# ---------------------------------------------------------------------------
# Missing atoms: raise, never fall back
# ---------------------------------------------------------------------------

def test_missing_ser_og_raises_and_mentions_atom():
    structure = build_structure(triad_residues(include_ser_og=False))
    with pytest.raises(triad.MissingAtomError, match="OG"):
        triad.measure_triad_distances(
            structure, {"ser": 25, "his": 27, "asp": 28}
        )


def test_missing_ser_og_does_not_substitute_backbone_atom():
    """A Ser with CB alone must fail; nothing may stand in for OG."""
    structure = build_structure(triad_residues(include_ser_og=False))
    result = triad.assess_triad(
        "hit_no_og", QALN, TALN, QSTART, TSTART, structure,
        TRIAD_IN_ALIGNMENT, THRESHOLDS,
    )
    assert result.triad_intact is False
    assert result.distances is None
    assert "OG" in result.failure_reason


def test_missing_both_asp_oxygens_raises():
    structure = build_structure(triad_residues(asp_od1=None, asp_od2=None))
    with pytest.raises(triad.MissingAtomError, match="OD1|OD2"):
        triad.measure_triad_distances(
            structure, {"ser": 25, "his": 27, "asp": 28}
        )


def test_missing_residue_raises():
    with pytest.raises(triad.MissingResidueError):
        triad.measure_triad_distances(
            build_structure(triad_residues()),
            {"ser": 25, "his": 27, "asp": 404},
        )


def test_gap_position_raises_value_error(passing_structure):
    with pytest.raises(ValueError, match="no target residue"):
        triad.measure_triad_distances(
            passing_structure, {"ser": 25, "his": None, "asp": 28}
        )


# ---------------------------------------------------------------------------
# assess_triad: the combined entry point used by the pipeline
# ---------------------------------------------------------------------------

def test_assess_triad_passes(passing_structure):
    result = triad.assess_triad(
        "hit_pass", QALN, TALN, QSTART, TSTART, passing_structure,
        TRIAD_IN_ALIGNMENT, THRESHOLDS,
    )
    assert result.status == "pass"
    assert result.triad_intact is True
    assert result.failure_reason == ""
    assert result.mapped_positions == {"ser": 25, "his": 27, "asp": 28}
    assert result.distances is not None
    assert result.distances.ser_his == pytest.approx(2.6)


def test_assess_triad_gap_is_not_conserved(passing_structure):
    result = triad.assess_triad(
        "hit_gap", QALN, TALN, QSTART, TSTART, passing_structure,
        {"ser": 14, "his": 16, "asp": 18}, THRESHOLDS,
    )
    assert result.status == "not_conserved"
    assert result.triad_intact is False
    assert "gap" in result.failure_reason.lower()


def test_assess_triad_wrong_type_is_not_conserved():
    structure = build_structure(triad_residues(ser_resname="ALA"))
    result = triad.assess_triad(
        "hit_ala", QALN, TALN, QSTART, TSTART, structure,
        TRIAD_IN_ALIGNMENT, THRESHOLDS,
    )
    assert result.status == "not_conserved"
    assert result.residue_types_ok["ser"] is False
    assert "Wrong or missing residue" in result.failure_reason


def test_assess_triad_missing_atom_is_flagged_not_raised():
    structure = build_structure(triad_residues(asp_od1=None, asp_od2=None))
    result = triad.assess_triad(
        "hit_no_od", QALN, TALN, QSTART, TSTART, structure,
        TRIAD_IN_ALIGNMENT, THRESHOLDS,
    )
    assert result.status == "error"
    assert result.triad_intact is False
    assert "OD1" in result.failure_reason or "OD2" in result.failure_reason


def test_assess_triad_geometry_failure():
    structure = build_structure(triad_residues(ser_og=(10.0, 0.0, 0.0)))
    result = triad.assess_triad(
        "hit_far", QALN, TALN, QSTART, TSTART, structure,
        TRIAD_IN_ALIGNMENT, THRESHOLDS,
    )
    assert result.status == "geometry_fail"
    assert result.triad_intact is False
    assert "Ser-His" in result.failure_reason


def test_assess_triad_requires_thresholds(passing_structure):
    with pytest.raises(KeyError, match="max_ser_his_dist_angstrom"):
        triad.assess_triad(
            "hit_cfg", QALN, TALN, QSTART, TSTART, passing_structure,
            TRIAD_IN_ALIGNMENT, {},
        )
