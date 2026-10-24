"""
pipeline.py
-----------
Run the whole Enzyme Miner funnel:

    seed structure -> Foldseek search -> triad check -> pocket pLDDT
                   -> thermophile ranking -> ranked CSV + funnel counts

What is real
~~~~~~~~~~~~
* The Foldseek search and AlphaFold download are live HTTP calls (cached).
* Triad mapping, geometry and pLDDT are computed from real coordinates.
* Host ranking uses the curated table in data/thermophiles.csv.

What this does NOT prove
~~~~~~~~~~~~~~~~~~~~~~~
The output is a ranked list of PREDICTED candidates for lab testing.  It is
not a list of validated enzymes, and no activity, expression or stability
claim is made.  MGnify-ESM30 hits are searched but cannot be evaluated here
because only AlphaFold DB entries have a structure download endpoint; they are
counted as "no structure" rather than silently ignored.

Demo mode
~~~~~~~~~
`run_pipeline(demo=True)` never touches the network.  It builds a small,
clearly synthetic set of hits (ids starting with "DEMO-") with in-memory
coordinates and runs the same triad/pLDDT/ranking code on them.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import requests
import yaml
from Bio.PDB import PDBParser, StructureBuilder
from Bio.PDB.Structure import Structure

from src import plddt, thermo
from src.foldseek_client import FoldseekClient, FoldseekSettings
from src.triad import TriadResult, assess_triad

logger = logging.getLogger(__name__)

# Columns written to the ranked CSV, in order.
OUTPUT_COLUMNS = [
    "rank",
    "target_id",
    "description",
    "database",
    "probability",
    "evalue",
    "seq_identity",
    "host_taxname",
    "host_genus",
    "matched_genus",
    "approx_T_opt_C",
    "thermo_score",
    "host_status",
    "host_note",
    "ser_pos",
    "his_pos",
    "asp_pos",
    "ser_his_dist",
    "asp_his_dist",
    "mean_plddt",
    "pocket_residue_count",
    "triad_status",
    "plddt_status",
    "final_status",
    "note",
]


class PipelineError(RuntimeError):
    """The pipeline cannot continue (bad config, missing seed, numbering mismatch)."""


@dataclass
class PipelineResult:
    """Final table plus the funnel counts that explain how it was reached."""

    hits: pd.DataFrame
    funnel: dict[str, int]
    demo: bool = False
    results_csv: Optional[Path] = None
    funnel_json: Optional[Path] = None


# ---------------------------------------------------------------------------
# Config and seed
# ---------------------------------------------------------------------------

def _resolve_paths(config: dict[str, Any], base: Path) -> dict[str, Any]:
    """Make relative paths in the config relative to the config file's folder."""
    relative_keys = [
        ("seed", "cache_path"),
        ("foldseek", "cache_dir"),
        ("plddt", "cache_dir"),
        ("thermo", "data_file"),
        ("output", "results_csv"),
        ("output", "funnel_json"),
        ("benchmark", "known_enzymes_file"),
        ("benchmark", "results_csv"),
    ]
    for section, key in relative_keys:
        value = config.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            config[section][key] = str(base / value)
    return config


def load_config(path: Path | str = "config.yaml") -> dict[str, Any]:
    """Read config.yaml. Every threshold and external name lives there."""
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"Config file not found: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    return _resolve_paths(config, path.resolve().parent)


def ensure_seed_structure(config: Mapping[str, Any]) -> Path:
    """Download the seed PDB once and cache it (5XJH by default)."""
    seed = config["seed"]
    path = Path(seed["cache_path"])
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = seed["download_url"].format(pdb_id=seed["pdb_id"])
    logger.info("Downloading seed %s from %s", seed["pdb_id"], url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def query_triad_positions(
    seed_path: Path,
    chain_id: str,
    triad_numbers: Mapping[str, int],
) -> dict[str, int]:
    """
    Translate PDB residue numbers into 1-based positions in the chain sequence.

    Foldseek numbers the query from 1 along the structure's residue order, not
    by the author numbering in the PDB file.  For 5XJH chain A (residues
    30..292) Ser160/Asp206/His237 become 131/177/208.
    """
    structure = PDBParser(QUIET=True).get_structure("seed", str(seed_path))
    try:
        chain = structure[0][chain_id]
    except KeyError as exc:
        raise PipelineError(
            f"Seed chain {chain_id!r} not found in {seed_path}."
        ) from exc

    ordinal: dict[int, int] = {}
    for residue in chain:
        if residue.get_id()[0].strip():
            continue
        ordinal.setdefault(residue.get_id()[1], len(ordinal) + 1)

    missing = [number for number in triad_numbers.values() if number not in ordinal]
    if missing:
        raise PipelineError(
            f"Triad residues {missing} are not present in seed chain {chain_id}."
        )

    return {role: ordinal[int(number)] for role, number in triad_numbers.items()}


def seed_chain_length(seed_path: Path, chain_id: str) -> int:
    """Number of amino-acid residues Foldseek should see for the query chain."""
    structure = PDBParser(QUIET=True).get_structure("seed", str(seed_path))
    return sum(1 for residue in structure[0][chain_id]
               if not residue.get_id()[0].strip())


def check_query_length(hits: pd.DataFrame, expected_length: int) -> None:
    """
    Guard the numbering assumption.

    If Foldseek's query length differs from our residue count, the translated
    triad positions would silently point at the wrong residues, so stop.
    """
    if hits.empty or "qlen" not in hits.columns:
        return
    observed = int(hits["qlen"].dropna().iloc[0]) if hits["qlen"].notna().any() else 0
    if observed and observed != expected_length:
        raise PipelineError(
            "Foldseek query length "
            f"({observed}) does not match the seed chain's residue count "
            f"({expected_length}). Triad numbering cannot be translated safely."
        )


# ---------------------------------------------------------------------------
# Evaluation of Foldseek hits
# ---------------------------------------------------------------------------

def _select_query(hits: pd.DataFrame, chain_id: str) -> pd.DataFrame:
    """Keep the alignments belonging to the seed chain, if Foldseek labelled them."""
    if "query" not in hits.columns:
        return hits
    wanted = f"job_{chain_id}"
    selected = hits[hits["query"] == wanted]
    if selected.empty:
        logger.warning("No alignments labelled %s; evaluating all queries.", wanted)
        return hits
    return selected


def _fold_matches(hits: pd.DataFrame, min_probability: float) -> pd.DataFrame:
    """Structural-similarity cutoff for the funnel (not for ranking)."""
    if "probability" not in hits.columns:
        return hits
    probabilities = pd.to_numeric(hits["probability"], errors="coerce")
    return hits[probabilities >= min_probability]


def _base_row(row: pd.Series) -> dict[str, Any]:
    return {
        "target_id": row.get("target_id"),
        "description": row.get("description", ""),
        "database": row.get("database", ""),
        "probability": row.get("probability"),
        "evalue": row.get("evalue"),
        "seq_identity": row.get("seq_identity"),
        # thermo.annotate_hosts turns this into host_taxname/genus/score.
        "taxname": row.get("taxname"),
    }


def evaluate_hits(
    hits: pd.DataFrame,
    triad_query_positions: Mapping[str, int],
    config: Mapping[str, Any],
    structure_provider: Callable[[str], Optional[Structure]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Run triad and pLDDT checks over fold-match rows.

    `structure_provider` returns a parsed structure or None when no model is
    available.  It is injected so demo mode can supply synthetic structures
    and the real mode can download AlphaFold models.
    """
    triad_cfg = config["triad"]
    plddt_cfg = config["plddt"]
    thresholds = {
        "max_ser_his_dist_angstrom": triad_cfg["max_ser_his_dist_angstrom"],
        "max_asp_his_dist_angstrom": triad_cfg["max_asp_his_dist_angstrom"],
    }
    chain_id = config["seed"]["chain"]
    radius = float(plddt_cfg["pocket_radius_angstrom"])
    min_plddt = float(plddt_cfg["min_mean_plddt"])

    counters = {
        "foldseek_hits": int(len(hits)),
        "fold_match": 0,
        "structure_available": 0,
        "triad_intact": 0,
        "confident_pocket": 0,
        "final_shortlist": 0,
        "dropped_no_structure": 0,
        "dropped_triad": 0,
        "dropped_low_plddt": 0,
    }
    kept: list[dict[str, Any]] = []

    fold_matches = _fold_matches(hits, float(config["foldseek"]["min_probability"]))
    counters["fold_match"] = int(len(fold_matches))

    for _, row in fold_matches.iterrows():
        target_id = str(row["target_id"])
        record = _base_row(row)

        if not row.get("qaln") or not row.get("taln"):
            counters["dropped_triad"] += 1
            record.update(triad_status="error", plddt_status="not_run",
                          final_status="dropped",
                          note="Foldseek alignment strings missing")
            continue

        structure = structure_provider(target_id)
        if structure is None:
            counters["dropped_no_structure"] += 1
            record.update(triad_status="not_run", plddt_status="no_structure",
                          final_status="dropped",
                          note="no AlphaFold DB structure available")
            continue
        counters["structure_available"] += 1

        triad_result: TriadResult = assess_triad(
            hit_id=target_id,
            query_aln=str(row["qaln"]),
            target_aln=str(row["taln"]),
            query_start=int(row["qstart"]),
            target_start=int(row["tstart"]),
            structure=structure,
            query_triad_positions=dict(triad_query_positions),
            thresholds=thresholds,
            chain_id=chain_id,
        )

        record.update(
            triad_status=triad_result.status,
            ser_pos=triad_result.mapped_positions.get("ser"),
            his_pos=triad_result.mapped_positions.get("his"),
            asp_pos=triad_result.mapped_positions.get("asp"),
            ser_his_dist=(
                triad_result.distances.ser_his if triad_result.distances else None
            ),
            asp_his_dist=(
                triad_result.distances.asp_his if triad_result.distances else None
            ),
        )

        if not triad_result.triad_intact:
            counters["dropped_triad"] += 1
            record.update(plddt_status="not_run", final_status="dropped",
                          note=triad_result.failure_reason)
            continue
        counters["triad_intact"] += 1

        plddt_result = plddt.assess_pocket_plddt(
            hit_id=target_id,
            structure=structure,
            mapped_positions=triad_result.mapped_positions,
            radius_angstrom=radius,
            min_mean_plddt=min_plddt,
            chain_id=chain_id,
        )
        record.update(
            mean_plddt=plddt_result.mean_plddt,
            pocket_residue_count=plddt_result.pocket_residue_count,
            plddt_status=plddt_result.status,
        )

        if not plddt_result.plddt_pass:
            counters["dropped_low_plddt"] += 1
            record.update(final_status="dropped", note=plddt_result.reason)
            continue

        counters["confident_pocket"] += 1
        record.update(final_status="kept", note="")
        kept.append(record)

    counters["final_shortlist"] = len(kept)
    return pd.DataFrame(kept), counters


def rank_and_write(
    kept: pd.DataFrame,
    counters: dict[str, int],
    config: Mapping[str, Any],
    demo: bool,
) -> PipelineResult:
    """Rank kept hits by host heat tolerance and write CSV + funnel JSON."""
    thermo_cfg = config["thermo"]
    thermophiles = thermo.load_thermophiles(
        Path(thermo_cfg["data_file"]),
        match_column=thermo_cfg.get("match_column", "genus"),
        temperature_column=thermo_cfg.get("temperature_column", "approx_T_opt_C"),
    )

    if kept.empty:
        ranked = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        ranked = thermo.rank_by_host(
            kept, thermophiles, float(thermo_cfg["unknown_host_score"])
        )

    for column in OUTPUT_COLUMNS:
        if column not in ranked.columns:
            ranked[column] = None
    ranked = ranked[OUTPUT_COLUMNS]

    if demo:
        csv_path = Path(config["output"]["results_csv"]).with_name(
            "demo_ranked_candidates.csv"
        )
    else:
        csv_path = Path(config["output"]["results_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(csv_path, index=False)

    funnel_path = Path(config["output"]["funnel_json"])
    if demo:
        funnel_path = funnel_path.with_name("demo_funnel.json")
    funnel_path.parent.mkdir(parents=True, exist_ok=True)
    funnel_path.write_text(json.dumps(counters, indent=2))

    logger.info("Funnel: %s", counters)
    logger.info("Wrote %d ranked candidates to %s", len(ranked), csv_path)
    return PipelineResult(hits=ranked, funnel=counters, demo=demo,
                          results_csv=csv_path, funnel_json=funnel_path)


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: Path | str = "config.yaml",
    demo: bool = False,
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> PipelineResult:
    """Run the full funnel (or the offline demo)."""
    config = load_config(config_path)
    if demo:
        return run_demo(config)

    seed_path = ensure_seed_structure(config)
    chain_id = config["seed"]["chain"]
    triad_cfg = config["triad"]
    triad_numbers = {
        "ser": triad_cfg["ser"], "his": triad_cfg["his"], "asp": triad_cfg["asp"],
    }
    query_positions = query_triad_positions(seed_path, chain_id, triad_numbers)
    logger.info("Seed triad at query positions: %s", query_positions)

    foldseek_settings = FoldseekSettings.from_config(config)
    client = FoldseekClient(foldseek_settings)
    hits = client.search(
        seed_path,
        config["foldseek"]["databases"],
        mode=config["foldseek"]["mode"],
        max_hits=int(config["foldseek"]["max_hits"]),
        use_cache=use_cache,
    )
    hits = _select_query(hits, chain_id)
    check_query_length(hits, seed_chain_length(seed_path, chain_id))
    if limit is not None:
        hits = hits.head(limit).reset_index(drop=True)

    af_settings = plddt.AlphaFoldSettings.from_config(config)
    cache: dict[str, Optional[Structure]] = {}

    def provide(target_id: str) -> Optional[Structure]:
        if target_id not in cache:
            try:
                path = plddt.download_hit_structure(target_id, af_settings)
            except (plddt.AlphaFoldError, OSError) as exc:
                logger.warning("AlphaFold download failed for %s: %s", target_id, exc)
                cache[target_id] = None
            else:
                if path is None:
                    cache[target_id] = None
                else:
                    try:
                        cache[target_id] = plddt.parse_structure(path)
                    except Exception as exc:  # malformed model: flag, do not crash
                        logger.warning("Could not parse %s: %s", path, exc)
                        cache[target_id] = None
        return cache[target_id]

    kept, counters = evaluate_hits(hits, query_positions, config, provide)
    return rank_and_write(kept, counters, config, demo=False)


# ---------------------------------------------------------------------------
# Demo mode (offline, synthetic)
# ---------------------------------------------------------------------------

DEMO_SERIES = [
    # (id, taxname, variant, mean pLDDT, probability)
    ("DEMO-THERMUS-01", "Thermus thermophilus", "pass", 95.0, 0.99),
    ("DEMO-THERMOBIFIDA-02", "Thermobifida cellulosilytica", "pass", 90.0, 0.98),
    ("DEMO-GEOBACILLUS-03", "Geobacillus stearothermophilus", "pass", 85.0, 0.97),
    ("DEMO-CALDIMONAS-04", "Caldimonas brevitalea", "pass", 88.0, 0.96),
    ("DEMO-IDEONELLA-05", "Ideonella sakaiensis", "wrong_type", 92.0, 0.95),
    ("DEMO-UNKNOWN-06", None, "pass", 97.0, 0.94),
    ("DEMO-THERMOTOGA-07", "Thermotoga maritima", "geometry_fail", 91.0, 0.93),
    ("DEMO-PYROCOCCUS-08", "Pyrococcus furiosus", "gap", 89.0, 0.92),
    ("DEMO-BACILLUS-09", "Bacillus subtilis", "missing_atom", 84.0, 0.91),
    ("DEMO-METAGENOME-10", "marine metagenome", "pass", 60.0, 0.90),
]

# Query alignment used by the demo.  Positions 3/6/9 are the synthetic triad,
# mirroring Ser/Asp/His of the seed.
DEMO_QALN = "MKSEFDGIHL"
DEMO_TALN = "MKSEFDGIHL"
DEMO_TALN_GAP = "MKSEF-GIHL"
DEMO_QUERY_POSITIONS = {"ser": 3, "his": 9, "asp": 6}


def _demo_structure(variant: str, mean_bfactor: float) -> Structure:
    """Build a tiny synthetic structure with the requested triad geometry."""
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure("demo")
    builder.init_model(0)
    builder.init_chain("A")

    ser_atoms: dict[str, tuple[float, float, float]] = {"CB": (-1.4, 0.3, 0.0)}
    if variant != "missing_atom":
        # "geometry_fail" moves the nucleophile out of range.
        ser_atoms["OG"] = (10.0, 0.0, 0.0) if variant == "geometry_fail" else (0.0, 0.0, 0.0)

    residues = [
        (2, "GLY", {"CA": (1.0, 0.0, 0.0)}),
        (3, "ALA" if variant == "wrong_type" else "SER", ser_atoms),
        (4, "GLY", {"CA": (1.5, 0.0, 0.0)}),
        (5, "GLY", {"CA": (-1.0, 0.0, 0.0)}),
        (6, "ASP", {"OD1": (-5.2, 0.0, 0.0), "OD2": (-4.0, 0.0, 0.0)}),
        (8, "GLY", {"CA": (-1.5, 0.0, 0.0)}),
        (9, "HIS", {"NE2": (2.8, 0.0, 0.0), "ND1": (-2.8, 0.0, 0.0)}),
        (10, "GLY", {"CA": (2.5, 0.0, 0.0)}),
    ]

    for resnum, resname, atoms in residues:
        builder.init_residue(resname, " ", resnum, " ")
        for atom_name, coord in atoms.items():
            builder.init_atom(
                atom_name, np.array(coord, dtype=float),
                mean_bfactor, 1.0, " ", atom_name, resnum, atom_name[0],
            )
    return builder.get_structure()


def build_demo_hits() -> tuple[pd.DataFrame, dict[str, Structure]]:
    """
    Create the synthetic hits table and their in-memory structures.

    The ids are prefixed DEMO- and the numbers are made up by this function;
    nothing here comes from a real Foldseek search.
    """
    rows = []
    structures: dict[str, Structure] = {}
    for target_id, taxname, variant, mean_plddt, probability in DEMO_SERIES:
        structures[target_id] = _demo_structure(variant, mean_plddt)
        rows.append({
            "query": "job_A",
            "target_id": target_id,
            "description": f"synthetic demo hit ({variant})",
            "database": "demo",
            "probability": probability,
            "tm_score": None,
            "evalue": 1e-30,
            "score": 1000.0,
            "seq_identity": 60.0,
            "aln_length": 10,
            "qstart": 1,
            "qend": 10,
            "tstart": 1,
            "tend": 10,
            "qlen": len(DEMO_QALN),
            "tlen": len(DEMO_TALN),
            "qaln": DEMO_QALN,
            "taln": DEMO_TALN_GAP if variant == "gap" else DEMO_TALN,
            "taxid": None,
            "taxname": taxname,
        })
    return pd.DataFrame(rows), structures


def run_demo(config: Mapping[str, Any]) -> PipelineResult:
    """Offline run over synthetic hits; no network, no downloaded files."""
    hits, structures = build_demo_hits()
    kept, counters = evaluate_hits(
        hits, DEMO_QUERY_POSITIONS, config,
        structure_provider=lambda target_id: structures.get(target_id),
    )
    return rank_and_write(kept, counters, config, demo=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Enzyme Miner pipeline.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true", help="offline synthetic run")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N fold matches (debugging)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore cached Foldseek/AlphaFold responses")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result = run_pipeline(args.config, demo=args.demo, limit=args.limit,
                          use_cache=not args.no_cache)
    print(json.dumps(result.funnel, indent=2))
    print(f"Wrote {result.results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
