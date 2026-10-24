"""
triad.py
--------
Check whether a Foldseek hit still has a plausible Ser-His-Asp catalytic triad
at the positions that correspond to IsPETase Ser160 / Asp206 / His237.

Pipeline of this module
~~~~~~~~~~~~~~~~~~~~~~~
1. Walk the Foldseek alignment (qaln/taln) to translate query residue numbers
   into target residue numbers.  Gaps are tracked explicitly.
2. Look up those residue numbers in the hit's structure and confirm the residue
   types are Ser, His and Asp.
3. Measure Ser OG - His NE2 and Asp OD1/OD2 (whichever is closer) - His ND1.
4. Pass/fail against thresholds supplied by the caller (config.yaml).

Honest limits
~~~~~~~~~~~~~
* This is a *sequence-alignment + geometry* check.  A hit that passes is a
  predicted candidate for lab testing, not a validated enzyme.
* Residue numbering of the hit is assumed to match the Foldseek target sequence
  numbering.  That holds for AlphaFold DB models (numbered 1..N) but is an
  assumption for any other source; a mismatch surfaces as "residue not found"
  rather than as a silently wrong answer.
* We never substitute a different atom.  Missing atoms are errors, not guesses.
  The one deliberate exception is Asp OD1 vs OD2, which are chemically
  equivalent carboxylate oxygens.

Public API
~~~~~~~~~~
    map_triad_positions(query_aln, target_aln, query_triad_positions,
                        query_start, target_start) -> dict[str, int | None]

    check_triad_residue_types(structure, mapped_positions, chain_id)
        -> dict[str, bool]

    measure_triad_distances(structure, mapped_positions, chain_id)
        -> TriadDistances

    assess_triad(...) -> TriadResult      # the main entry point
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from Bio.PDB import Structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IsPETase uses the classic serine-hydrolase triad Ser-His-Asp.
EXPECTED_RESNAME: dict[str, str] = {
    "ser": "SER",
    "his": "HIS",
    "asp": "ASP",
}

# Catalytic atoms used for the two distance measurements.
SER_ATOM = "OG"          # nucleophile
HIS_ATOM_SER_SIDE = "NE2"  # histidine nitrogen facing the serine
HIS_ATOM_ASP_SIDE = "ND1"  # histidine nitrogen facing the aspartate
ASP_ATOMS = ("OD1", "OD2")  # equivalent carboxylate oxygens

# Threshold keys every caller must provide (values live in config.yaml).
REQUIRED_THRESHOLDS = ("max_ser_his_dist_angstrom", "max_asp_his_dist_angstrom")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TriadCheckError(Exception):
    """Base class for problems found while checking a hit."""


class MissingChainError(TriadCheckError):
    """The requested chain is not present in the structure."""


class MissingResidueError(TriadCheckError):
    """A mapped residue number is not present in the chain."""


class MissingAtomError(TriadCheckError):
    """A required catalytic atom is absent; we do not fall back to another atom."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TriadDistances:
    """Measured interatomic distances for a catalytic triad, in Angstrom."""

    ser_his: float            # Ser OG - His NE2
    asp_his: float            # closest Asp OD - His ND1
    asp_atom: str             # "OD1" or "OD2", whichever was closer
    ser_atom: str = SER_ATOM
    his_atom_ser_side: str = HIS_ATOM_SER_SIDE
    his_atom_asp_side: str = HIS_ATOM_ASP_SIDE


@dataclass
class TriadResult:
    """Outcome of the triad check for one hit."""

    hit_id: str

    # Residue numbers in the HIT that correspond to the query triad.
    mapped_positions: dict[str, Optional[int]] = field(default_factory=dict)

    # Whether each mapped residue has the expected amino-acid type.
    residue_types_ok: dict[str, bool] = field(default_factory=dict)

    distances: Optional[TriadDistances] = None

    # True only if mapped, correct types and geometry inside thresholds.
    triad_intact: bool = False

    # One of: "pass", "not_conserved", "geometry_fail", "error".
    status: str = "error"

    # Human-readable reason for failure or uncertainty ("" when passing).
    failure_reason: str = ""


# ---------------------------------------------------------------------------
# Step 1 - alignment-based residue mapping
# ---------------------------------------------------------------------------

def map_triad_positions(
    query_aln: str,
    target_aln: str,
    query_triad_positions: dict[str, int],
    query_start: int = 1,
    target_start: int = 1,
) -> dict[str, Optional[int]]:
    """
    Walk a gapped pairwise alignment and translate query triad positions into
    target residue numbers.

    Parameters
    ----------
    query_aln, target_aln : str
        Alignment strings from Foldseek (qaln / taln), same length, '-' = gap.
    query_triad_positions : dict[str, int]
        Triad role -> 1-indexed residue number in the FULL query sequence,
        e.g. {"ser": 160, "his": 237, "asp": 206}.
    query_start, target_start : int
        1-indexed positions in the full sequences where the alignment starts
        (Foldseek qstart / tstart).  Use 1 when the alignment covers the whole
        sequence from residue 1.

    Returns
    -------
    dict[str, Optional[int]]
        Triad role -> 1-indexed target residue number, or None when the query
        residue falls in a gap in the target / outside the aligned region.
        None means "triad not conserved", never "try another residue".

    Raises
    ------
    ValueError
        If the alignment strings differ in length or starts are not >= 1.
    """
    if len(query_aln) != len(target_aln):
        raise ValueError(
            "Alignment strings have different lengths: "
            f"query={len(query_aln)}, target={len(target_aln)}"
        )
    if query_start < 1 or target_start < 1:
        raise ValueError("Alignment start positions must be 1-indexed (>= 1).")

    # q_num -> target residue number (None when aligned to a target gap)
    query_to_target: dict[int, Optional[int]] = {}
    q_num = query_start
    t_num = target_start

    for q_char, t_char in zip(query_aln, target_aln):
        q_gap = q_char == "-"
        t_gap = t_char == "-"

        if not q_gap:
            query_to_target[q_num] = None if t_gap else t_num
            q_num += 1
        if not t_gap:
            t_num += 1

    mapped: dict[str, Optional[int]] = {}
    for role, resnum in query_triad_positions.items():
        if resnum not in query_to_target:
            logger.warning(
                "Query residue %d (%s) is outside the aligned region; "
                "it cannot be mapped onto this hit.", resnum, role,
            )
            mapped[role] = None
        else:
            mapped[role] = query_to_target[resnum]

    return mapped


# ---------------------------------------------------------------------------
# Step 2 - residue types in the hit structure
# ---------------------------------------------------------------------------

def _residues_by_number(structure: Structure, chain_id: str) -> dict[int, object]:
    """
    Return {residue number: residue} for one chain, ignoring hetero records.

    Numbering collisions (e.g. insertion codes producing the same number twice)
    keep the first residue and log a warning - the caller will then either match
    the expected type or fail loudly.
    """
    try:
        chain = structure[0][chain_id]
    except KeyError as exc:
        raise MissingChainError(
            f"Chain {chain_id!r} not found in structure {structure.get_id()!r}."
        ) from exc

    by_number: dict[int, object] = {}
    for residue in chain:
        hetero_flag, resnum, _icode = residue.get_id()
        if hetero_flag.strip():  # skip waters / ligands
            continue
        if resnum in by_number:
            logger.warning(
                "Chain %s has more than one residue numbered %d; using the first.",
                chain_id, resnum,
            )
            continue
        by_number[resnum] = residue
    return by_number


def check_triad_residue_types(
    structure: Structure,
    mapped_positions: dict[str, Optional[int]],
    chain_id: str = "A",
) -> dict[str, bool]:
    """
    Confirm the mapped residues are Ser / His / Asp in the hit structure.

    Returns
    -------
    dict[str, bool]
        Triad role -> True when the residue exists and has the expected name.
        A gap (None), missing chain or missing residue yields False.
    """
    try:
        residues = _residues_by_number(structure, chain_id)
    except MissingChainError as exc:
        logger.error("%s", exc)
        return {role: False for role in mapped_positions}

    result: dict[str, bool] = {}
    for role, resnum in mapped_positions.items():
        if resnum is None:
            logger.debug("Role %s: fell in an alignment gap.", role)
            result[role] = False
            continue

        residue = residues.get(resnum)
        if residue is None:
            logger.warning(
                "Role %s: residue %d not found in chain %s.", role, resnum, chain_id
            )
            result[role] = False
            continue

        found = residue.get_resname().strip()
        expected = EXPECTED_RESNAME[role]
        result[role] = found == expected
        if found != expected:
            logger.info(
                "Role %s: expected %s at %d, found %s.", role, expected, resnum, found
            )

    return result


# ---------------------------------------------------------------------------
# Step 3 - catalytic geometry
# ---------------------------------------------------------------------------

def _get_atom(residue, atom_name: str):
    """Return one atom, or raise MissingAtomError. Never substitutes."""
    if atom_name not in residue:
        raise MissingAtomError(
            f"Atom {atom_name!r} missing from {residue.get_resname()} "
            f"{residue.get_id()[1]}; no substitute atom is used."
        )
    return residue[atom_name]


def measure_triad_distances(
    structure: Structure,
    mapped_positions: dict[str, Optional[int]],
    chain_id: str = "A",
) -> TriadDistances:
    """
    Measure Ser OG - His NE2 and (closest Asp OD) - His ND1.

    Raises
    ------
    ValueError
        If any mapped position is None.
    MissingChainError, MissingResidueError, MissingAtomError
        If a required chain, residue or atom is absent.  No fallback atoms.
    """
    missing = [role for role in ("ser", "his", "asp")
               if mapped_positions.get(role) is None]
    if missing:
        raise ValueError(
            f"Cannot measure distances: no target residue for {missing}."
        )

    residues = _residues_by_number(structure, chain_id)

    def get_residue(role: str):
        resnum = mapped_positions[role]
        assert resnum is not None
        residue = residues.get(resnum)
        if residue is None:
            raise MissingResidueError(
                f"Residue {resnum} ({role}) not found in chain {chain_id}."
            )
        return residue

    ser_res = get_residue("ser")
    his_res = get_residue("his")
    asp_res = get_residue("asp")

    ser_og = _get_atom(ser_res, SER_ATOM)
    his_ne2 = _get_atom(his_res, HIS_ATOM_SER_SIDE)
    his_nd1 = _get_atom(his_res, HIS_ATOM_ASP_SIDE)

    ser_his = float(ser_og - his_ne2)

    # Asp OD1 and OD2 are equivalent oxygens; pick the closer one to His ND1.
    # If only one exists it is used as-is; if neither exists the hit fails.
    asp_atom_name: Optional[str] = None
    asp_his: Optional[float] = None
    for name in ASP_ATOMS:
        if name not in asp_res:
            continue
        distance = float(asp_res[name] - his_nd1)
        if asp_his is None or distance < asp_his:
            asp_atom_name, asp_his = name, distance

    if asp_atom_name is None or asp_his is None:
        raise MissingAtomError(
            f"Neither OD1 nor OD2 is present in Asp {asp_res.get_id()[1]}; "
            "cannot measure the Asp-His distance."
        )

    return TriadDistances(
        ser_his=ser_his,
        asp_his=asp_his,
        asp_atom=asp_atom_name,
    )


# ---------------------------------------------------------------------------
# Step 4 - combined assessment
# ---------------------------------------------------------------------------

def _threshold(thresholds: dict[str, float], key: str) -> float:
    if key not in thresholds:
        raise KeyError(
            f"Missing threshold {key!r}; expected keys are {REQUIRED_THRESHOLDS} "
            "(these come from config.yaml)."
        )
    return float(thresholds[key])


def assess_triad(
    hit_id: str,
    query_aln: str,
    target_aln: str,
    query_start: int,
    target_start: int,
    structure: Structure,
    query_triad_positions: dict[str, int],
    thresholds: dict[str, float],
    chain_id: str = "A",
) -> TriadResult:
    """
    Full triad check for one Foldseek hit.

    Never raises for bad hit data: problems are reported in
    ``TriadResult.triad_intact`` / ``status`` / ``failure_reason`` so the
    pipeline can log them and move on.

    ``thresholds`` must contain the keys in REQUIRED_THRESHOLDS.
    """
    result = TriadResult(hit_id=hit_id, status="error")

    try:
        max_ser_his = _threshold(thresholds, "max_ser_his_dist_angstrom")
        max_asp_his = _threshold(thresholds, "max_asp_his_dist_angstrom")

        mapped = map_triad_positions(
            query_aln, target_aln, query_triad_positions,
            query_start=query_start, target_start=target_start,
        )
        result.mapped_positions = mapped

        gap_roles = [role for role, pos in mapped.items() if pos is None]
        if gap_roles:
            result.status = "not_conserved"
            result.failure_reason = (
                f"Triad position(s) {gap_roles} fall in an alignment gap or "
                "outside the aligned region."
            )
            return result

        types_ok = check_triad_residue_types(structure, mapped, chain_id)
        result.residue_types_ok = types_ok
        wrong_types = [role for role, ok in types_ok.items() if not ok]
        if wrong_types:
            result.status = "not_conserved"
            result.failure_reason = (
                f"Wrong or missing residue at triad position(s): {wrong_types}."
            )
            return result

        distances = measure_triad_distances(structure, mapped, chain_id)
        result.distances = distances

    except TriadCheckError as exc:
        result.status = "error"
        result.failure_reason = str(exc)
        return result
    except ValueError as exc:
        result.status = "error"
        result.failure_reason = f"Alignment mapping failed: {exc}"
        return result

    failures = []
    if distances.ser_his > max_ser_his:
        failures.append(
            f"Ser-His {distances.ser_his:.2f} A > {max_ser_his:.2f} A"
        )
    if distances.asp_his > max_asp_his:
        failures.append(
            f"Asp-His {distances.asp_his:.2f} A > {max_asp_his:.2f} A"
        )

    if failures:
        result.status = "geometry_fail"
        result.failure_reason = "; ".join(failures)
        return result

    result.triad_intact = True
    result.status = "pass"
    return result
