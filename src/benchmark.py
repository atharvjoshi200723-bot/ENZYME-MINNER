"""
benchmark.py
------------
Rediscovery test: does the funnel rank known heat-tolerant PET hydrolases
(LCC, PHL7) near the top when seeded with IsPETase?

How hits are identified
~~~~~~~~~~~~~~~~~~~~~~~
Each row in data/known_pet_hydrolases.csv is looked up in the final ranked
list using, in order:
  1. a verified UniProt accession inside the Foldseek target id
     (e.g. "AF-G9BY57-F1-model_v6"),
  2. the enzyme name as a whole word in the AlphaFold description,
  3. the source organism inside the hit's host taxonomy (weak; only used when
     the organism string is specific).

The benchmark gets no special treatment: it reads the same ranked CSV the
pipeline produces and never injects, boosts or re-orders candidates.

Honest status
~~~~~~~~~~~~~
* "pass"          every required control was found within top N.
* "fail"          a required control was identifiable but ranked outside top N
                  (or was absent from the results).
* "inconclusive"  a required control cannot be identified from the available
                  metadata. This is a real limitation, not a pass.

As of 2026-10-24 PHL7 has no confirmed wild-type UniProt accession (8BRB's
A0AA82WPD4 belongs to the S131A mutant; wild-type structure 7NEI is listed as
"unidentified"). Unless a hit description names it, the run is inconclusive.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from src.pipeline import PipelineResult, load_config, run_pipeline

logger = logging.getLogger(__name__)

NOT_IDENTIFIABLE = "not_identifiable"


@dataclass
class BenchmarkResult:
    """Per-control table plus the overall verdict."""

    table: pd.DataFrame
    top_n: int
    status: str                 # "pass" | "fail" | "inconclusive"
    passed: bool
    detail: str
    results_csv: Optional[Path] = None


def _clean_identifier(value: Any) -> Optional[str]:
    """Return None for empty/placeholder identifiers such as TODO_VERIFY."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.upper().startswith("TODO"):
        return None
    return text


def load_known_enzymes(path: Path) -> pd.DataFrame:
    """Read the control table and drop placeholder identifiers."""
    known = pd.read_csv(path)
    required_columns = {"name", "uniprot_id", "organism", "notes"}
    missing = required_columns - set(known.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return known


def _find_accession(ranked: pd.DataFrame, accession: str) -> Optional[pd.Series]:
    """Look for AF-<accession>-F1... in the target id column."""
    pattern = re.compile(
        rf"(?:^|[|;\s])AF-{re.escape(accession)}(?:-F\d+|-model)", re.IGNORECASE
    )
    for _, row in ranked.iterrows():
        target = str(row.get("target_id", ""))
        if pattern.search(target):
            return row
    return None


def _find_name(ranked: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Whole-word match of the enzyme name in the description column."""
    pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for _, row in ranked.iterrows():
        if pattern.search(str(row.get("description", ""))):
            return row
    return None


def _organism_is_specific(organism: Optional[str]) -> bool:
    """True when an organism string can identify a hit without false matches."""
    if not organism:
        return False
    text = organism.strip().lower()
    return len(text) >= 6 and "unknown" not in text and "metagenome" not in text


def _find_organism(ranked: pd.DataFrame, organism: str) -> Optional[pd.Series]:
    """
    Match a specific organism string against the hit's host taxonomy.

    Skipped for generic strings ("unknown", "metagenome") because those would
    match unrelated hits and inflate the benchmark.
    """
    if not _organism_is_specific(organism):
        return None

    text = organism.strip().lower()
    for _, row in ranked.iterrows():
        host = str(row.get("host_taxname", "")).strip().lower()
        if host and (text in host or host in text):
            return row
    return None


def _lookup(ranked: pd.DataFrame, enzyme: pd.Series) -> tuple[Optional[pd.Series], str]:
    """Return (ranked row, method) or (None, reason)."""
    accession = _clean_identifier(enzyme.get("uniprot_id"))
    if accession:
        row = _find_accession(ranked, accession)
        if row is not None:
            return row, f"uniprot:{accession}"

    name = str(enzyme.get("name", "")).strip()
    if name:
        row = _find_name(ranked, name)
        if row is not None:
            return row, f"description:{name}"

    organism = _clean_identifier(enzyme.get("organism"))
    if organism and _organism_is_specific(organism):
        row = _find_organism(ranked, organism)
        if row is not None:
            return row, f"organism:{organism}"

    # Without an accession or a usable organism, absence is not evidence of
    # failure: we simply cannot tell whether the enzyme is in the list.
    if accession is None and not _organism_is_specific(organism):
        return None, NOT_IDENTIFIABLE
    return None, "not_found"


def evaluate_benchmark(
    ranked: pd.DataFrame,
    known: pd.DataFrame,
    top_n: int,
    required: Sequence[str],
) -> BenchmarkResult:
    """Compare the ranked list with the control table and decide pass/fail."""
    rows = []
    for _, enzyme in known.iterrows():
        row, method = _lookup(ranked, enzyme)
        if row is None:
            rows.append({
                "name": enzyme["name"],
                "found": False,
                "rank": None,
                "target_id": "",
                "identified_by": "",
                "in_top_n": False,
                "status": method,
            })
            continue

        rank_number = int(row["rank"]) if pd.notna(row.get("rank")) else None
        rows.append({
            "name": enzyme["name"],
            "found": True,
            "rank": rank_number,
            "target_id": str(row.get("target_id", "")),
            "identified_by": method,
            "in_top_n": bool(rank_number is not None and rank_number <= top_n),
            "status": "found",
        })

    table = pd.DataFrame(rows)

    required_rows = table[table["name"].isin(list(required))]
    if required_rows.empty:
        status = "inconclusive"
        detail = "No required controls are listed in the known-enzyme table."
    elif (required_rows["status"] == NOT_IDENTIFIABLE).any():
        unresolved = required_rows[
            required_rows["status"] == NOT_IDENTIFIABLE
        ]["name"].tolist()
        status = "inconclusive"
        detail = (
            f"{', '.join(unresolved)} cannot be identified from the available "
            "metadata (no verified UniProt accession and no name/organism match), "
            f"so the top-{top_n} rediscovery claim cannot be evaluated."
        )
    elif required_rows["in_top_n"].all():
        status = "pass"
        detail = (
            f"All required controls ({', '.join(required_rows['name'])}) appear "
            f"in the top {top_n}."
        )
    else:
        status = "fail"
        missing = required_rows[~required_rows["in_top_n"]]
        parts = []
        for _, row in missing.iterrows():
            if not row["found"]:
                parts.append(f"{row['name']} absent")
            else:
                parts.append(f"{row['name']} at rank {row['rank']}")
        detail = (
            f"Required controls missing from the top {top_n}: " + "; ".join(parts)
        )

    return BenchmarkResult(table=table, top_n=top_n, status=status,
                           passed=status == "pass", detail=detail)


def run_benchmark(
    config_path: Path | str = "config.yaml",
    demo: bool = False,
    limit: Optional[int] = None,
    ranked_csv: Optional[Path | str] = None,
    pipeline_result: Optional[PipelineResult] = None,
) -> BenchmarkResult:
    """
    Run the pipeline (or reuse its output) and evaluate the controls.

    `ranked_csv` benchmarks an existing ranked table without re-running the
    search; it must contain the same columns as results/ranked_candidates.csv.
    """
    config = load_config(config_path)
    bench_cfg = config["benchmark"]

    if pipeline_result is not None:
        ranked = pipeline_result.hits
    elif ranked_csv is not None:
        ranked = pd.read_csv(ranked_csv)
    else:
        ranked = run_pipeline(config_path, demo=demo, limit=limit).hits

    known = load_known_enzymes(Path(bench_cfg["known_enzymes_file"]))
    result = evaluate_benchmark(
        ranked,
        known,
        top_n=int(bench_cfg["top_n"]),
        required=bench_cfg.get("required_enzymes", []),
    )

    out_path = Path(bench_cfg["results_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_path, index=False)
    result.results_csv = out_path

    logger.info("Benchmark status: %s", result.status)
    logger.info("%s", result.detail)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PET hydrolase rediscovery benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--demo", action="store_true",
                        help="evaluate the synthetic demo table (sanity check only)")
    parser.add_argument("--ranked-csv", default=None,
                        help="benchmark an existing ranked CSV instead of searching")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result = run_benchmark(args.config, demo=args.demo, limit=args.limit,
                           ranked_csv=args.ranked_csv)
    print(result.table.to_string(index=False))
    print(json.dumps({"status": result.status, "passed": result.passed,
                      "detail": result.detail}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
