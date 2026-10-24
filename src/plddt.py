"""
plddt.py
--------
Download candidate structures from the AlphaFold DB API and check how
confident the model is around the catalytic pocket.

AlphaFold models store per-residue confidence in the B-factor column, so the
"pocket pLDDT" is the mean B-factor over the triad residues plus every residue
with an atom within `pocket_radius_angstrom` of a triad atom.

Verified against the live API on 2026-10-24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    GET https://alphafold.ebi.ac.uk/api/prediction/{accession}
        -> [{"pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-...-model_v6.pdb", ...}]

We call the prediction endpoint rather than hard-coding a version, because the
served model version changes (v4 -> v6 during this project).

Honest limits
~~~~~~~~~~~~~
* pLDDT is the model's self-confidence, not evidence of activity.  A
  high-pLDDT pocket is a model we trust more, nothing else.
* Only AlphaFold DB entries can be fetched here.  MGnify-ESM30 targets have no
  AlphaFold accession, so they are reported as "no structure available" and
  cannot pass this filter in the prototype.
* Pocket residues are selected by any-atom distance to any triad atom,
  including backbone atoms.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import requests
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Structure import Structure

logger = logging.getLogger(__name__)

# "AF-A0A0G3BI90-F1-model_v6" -> "A0A0G3BI90"
AFDB_TARGET_RE = re.compile(r"^AF-([A-Z0-9]+)-F1(?:-|$)")


class AlphaFoldError(RuntimeError):
    """The AlphaFold DB API could not be used as expected."""


@dataclass
class AlphaFoldSettings:
    """AlphaFold DB access settings, normally built from config.yaml."""

    api_url_template: str
    cache_dir: Path
    timeout_sec: float = 60.0
    retries: int = 3
    min_request_interval_sec: float = 0.5

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AlphaFoldSettings":
        cfg = config["plddt"]
        return cls(
            api_url_template=cfg["api_url"],
            cache_dir=Path(cfg["cache_dir"]),
            timeout_sec=float(cfg.get("request_timeout_sec", 60)),
            retries=int(cfg.get("retries", 3)),
            min_request_interval_sec=float(cfg.get("min_request_interval_sec", 0.5)),
        )


@dataclass
class PlddtResult:
    """Pocket-confidence outcome for one hit."""

    hit_id: str
    mean_plddt: Optional[float] = None
    pocket_residue_count: int = 0
    threshold: float = 0.0
    plddt_pass: bool = False
    # "pass" | "below_threshold" | "no_structure" | "error"
    status: str = "error"
    reason: str = ""


def accession_from_target_id(target_id: str) -> Optional[str]:
    """
    Extract the UniProt accession from an AlphaFold-style Foldseek target id.

    Returns None for non-AlphaFold targets (e.g. MGnify-ESM30 models).
    """
    match = AFDB_TARGET_RE.match(target_id.strip())
    return match.group(1) if match else None


def parse_structure(path: Path, structure_id: Optional[str] = None) -> Structure:
    """Parse a .pdb / .cif file with BioPython (mmCIF chosen by suffix)."""
    path = Path(path)
    parser_id = structure_id or path.stem
    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise AlphaFoldError(f"Unsupported structure format: {path.suffix or path.name}")
    return parser.get_structure(parser_id, str(path))


def _request(
    session: requests.Session,
    url: str,
    settings: AlphaFoldSettings,
    sleep,
) -> requests.Response:
    """GET with retries. 404 is returned to the caller (meaning: no model)."""
    last_error: Optional[Exception] = None
    for attempt in range(settings.retries + 1):
        if settings.min_request_interval_sec > 0:
            sleep(settings.min_request_interval_sec)
        try:
            response = session.get(url, timeout=settings.timeout_sec)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("AlphaFold request failed (%s); retrying", exc)
        else:
            if response.status_code == 404:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                last_error = AlphaFoldError(f"HTTP {response.status_code} for {url}")
                pause = min(2.0 ** attempt, 30.0)
                if attempt < settings.retries:
                    sleep(pause)
                continue
            if response.status_code >= 400:
                raise AlphaFoldError(
                    f"AlphaFold request {url} -> HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response

        if attempt < settings.retries:
            sleep(min(2.0 ** attempt, 30.0))

    raise AlphaFoldError(f"AlphaFold request {url} failed after retries: {last_error}")


def download_hit_structure(
    target_id: str,
    settings: AlphaFoldSettings,
    session: Optional[requests.Session] = None,
    sleep=time.sleep,
) -> Optional[Path]:
    """
    Download the AlphaFold model for a Foldseek target id, using the cache.

    Returns the local .pdb path, or None when the target is not an AlphaFold
    entry (typical for MGnify-ESM30 hits).
    """
    accession = accession_from_target_id(target_id)
    if accession is None:
        logger.info("No AlphaFold accession in target id %r", target_id)
        return None

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.cache_dir / f"AF-{accession}-F1.pdb"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    session = session or requests.Session()
    url = settings.api_url_template.format(accession=accession)
    response = _request(session, url, settings, sleep)

    if response.status_code == 404:
        logger.warning("AlphaFold DB has no entry for %s", accession)
        return None

    try:
        entries = response.json()
    except ValueError as exc:
        raise AlphaFoldError(
            "AlphaFold prediction endpoint returned non-JSON; API may have changed."
        ) from exc

    if not isinstance(entries, list) or not entries:
        logger.warning("AlphaFold DB returned no prediction for %s", accession)
        return None

    pdb_url = entries[0].get("pdbUrl")
    if not pdb_url:
        raise AlphaFoldError(
            f"AlphaFold entry for {accession} has no pdbUrl; keys: "
            f"{sorted(entries[0])}"
        )

    model_response = _request(session, pdb_url, settings, sleep)

    # Write via a temporary file so an interrupted download is never cached.
    tmp_path = model_path.with_suffix(".pdb.part")
    tmp_path.write_bytes(model_response.content)
    tmp_path.replace(model_path)
    logger.info("Cached %s -> %s", accession, model_path)
    return model_path


def _chain(structure: Structure, chain_id: str):
    try:
        return structure[0][chain_id]
    except KeyError as exc:
        raise AlphaFoldError(
            f"Chain {chain_id!r} not found in structure {structure.get_id()!r}."
        ) from exc


def pocket_residues(
    structure: Structure,
    triad_numbers: dict[str, int],
    radius_angstrom: float,
    chain_id: str = "A",
) -> set[int]:
    """
    Residue numbers in the pocket: the triad plus anything within `radius`.

    Raises
    ------
    AlphaFoldError
        If a triad residue is missing from the structure.
    """
    chain = _chain(structure, chain_id)
    by_number = {}
    for residue in chain:
        if residue.get_id()[0].strip():
            continue
        by_number[residue.get_id()[1]] = residue

    missing = [num for num in triad_numbers.values() if num not in by_number]
    if missing:
        raise AlphaFoldError(f"Triad residues {missing} missing from chain {chain_id}.")

    triad_atoms = [atom for num in triad_numbers.values()
                   for atom in by_number[num]]
    pocket = set(triad_numbers.values())

    for number, residue in by_number.items():
        if number in pocket:
            continue
        if any(math.dist(atom.coord, anchor.coord) <= radius_angstrom
               for atom in residue for anchor in triad_atoms):
            pocket.add(number)

    return pocket


def mean_plddt(structure: Structure, residue_numbers: set[int],
               chain_id: str = "A") -> float:
    """
    Mean B-factor (pLDDT) over all atoms of the given residues.

    Raises AlphaFoldError if none of the residues (or no atoms) are found.
    """
    chain = _chain(structure, chain_id)
    values = [
        atom.get_bfactor()
        for residue in chain
        if residue.get_id()[1] in residue_numbers
        for atom in residue
    ]
    if not values:
        raise AlphaFoldError(
            f"No atoms found for pocket residues {sorted(residue_numbers)} "
            f"in chain {chain_id}."
        )
    return float(sum(values) / len(values))


def assess_pocket_plddt(
    hit_id: str,
    structure: Structure,
    mapped_positions: dict[str, Optional[int]],
    radius_angstrom: float,
    min_mean_plddt: float,
    chain_id: str = "A",
) -> PlddtResult:
    """
    Compute pocket pLDDT and compare it with the configured threshold.

    Bad input is reported, not raised, so the pipeline can drop the hit and
    keep the funnel counts accurate.
    """
    result = PlddtResult(hit_id=hit_id, threshold=min_mean_plddt)

    triad_numbers = {}
    for role, number in mapped_positions.items():
        if number is None:
            result.status = "error"
            result.reason = f"Triad position {role!r} is not mapped; cannot build a pocket."
            return result
        triad_numbers[role] = int(number)

    try:
        pocket = pocket_residues(structure, triad_numbers, radius_angstrom, chain_id)
        result.pocket_residue_count = len(pocket)
        result.mean_plddt = mean_plddt(structure, pocket, chain_id)
    except AlphaFoldError as exc:
        result.status = "error"
        result.reason = str(exc)
        return result

    if result.mean_plddt >= min_mean_plddt:
        result.status = "pass"
        result.plddt_pass = True
    else:
        result.status = "below_threshold"
        result.reason = (
            f"Mean pocket pLDDT {result.mean_plddt:.1f} < {min_mean_plddt:.1f}"
        )
    return result
