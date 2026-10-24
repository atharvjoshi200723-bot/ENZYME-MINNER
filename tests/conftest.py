"""Shared test helpers: build tiny in-memory Bio.PDB structures.

These structures contain only the atoms the triad code needs, which keeps the
tests fast and makes the expected distances obvious.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest
from Bio.PDB import StructureBuilder

# Alignment used across mapping tests.  Length 12, gaps in both sequences.
#   query  10 M K T A - S V D E Y - L 19
#   target 20 M K T A G S V - E Y F L 30
QALN = "MKTA-SVDEY-L"
TALN = "MKTAGSV-EYFL"
QSTART = 10
TSTART = 20

# Query triad positions that map onto target residues 25 / 27 / 28.
TRIAD_IN_ALIGNMENT = {"ser": 14, "his": 17, "asp": 18}

ResidueSpec = tuple[int, str, dict[str, tuple[float, float, float]]]


def build_structure(
    residues: Iterable[ResidueSpec],
    chain_id: str = "A",
    structure_id: str = "test",
):
    """Build a one-chain structure from (resnum, resname, {atom: xyz}) records."""
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure(structure_id)
    builder.init_model(0)
    builder.init_chain(chain_id)

    for resnum, resname, atoms in residues:
        builder.init_residue(resname, " ", resnum, " ")
        for atom_name, coord in atoms.items():
            # Bio.PDB expects numpy coordinates (as produced by PDB parsing).
            builder.init_atom(
                atom_name, np.array(coord, dtype=float),
                1.0, 1.0, " ", atom_name, resnum, atom_name[0],
            )

    return builder.get_structure()


def triad_residues(
    ser_resnum: int = 25,
    his_resnum: int = 27,
    asp_resnum: int = 28,
    ser_og: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ser_resname: str = "SER",
    asp_od1: tuple[float, float, float] | None = (-6.0, 0.0, 0.0),
    asp_od2: tuple[float, float, float] | None = (-3.6, 0.0, 0.0),
    include_ser_og: bool = True,
) -> list[ResidueSpec]:
    """Return residue records with a realistic, passing triad geometry.

    His NE2 (2.6 A from Ser OG) and His ND1 (-2.6, 0, 0) are fixed; the Asp
    oxygens are configurable so tests can check which oxygen is chosen.
    """
    ser_atoms: dict[str, tuple[float, float, float]] = {"CB": (-1.4, 0.3, 0.0)}
    if include_ser_og:
        ser_atoms["OG"] = ser_og

    asp_atoms: dict[str, tuple[float, float, float]] = {"CG": (-4.5, -1.4, 0.0)}
    if asp_od1 is not None:
        asp_atoms["OD1"] = asp_od1
    if asp_od2 is not None:
        asp_atoms["OD2"] = asp_od2

    return [
        (ser_resnum, ser_resname, ser_atoms),
        (
            his_resnum,
            "HIS",
            {
                "NE2": (2.6, 0.0, 0.0),
                "ND1": (-2.6, 0.0, 0.0),
                "CE1": (2.6, 1.4, 0.0),
                "CG": (-1.0, -1.5, 0.0),
            },
        ),
        (asp_resnum, "ASP", asp_atoms),
    ]


@pytest.fixture
def passing_structure():
    """Structure whose triad passes both type and geometry checks."""
    return build_structure(triad_residues())
