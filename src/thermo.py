"""
thermo.py
---------
Rank hits by the likely growth temperature of their source organism.

Foldseek reports the host as `taxname` for databases that carry taxonomy
(AlphaFold/UniProt50 does; MGnify-ESM30 does not).  We take the genus, look it
up in data/thermophiles.csv and use the curated optimum growth temperature as
a ranking score.

This is a ranking signal, not a filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A thermophile-derived enzyme is a *clue* that it may be more stable at high
  temperature.  It is not proof: stability depends on the specific protein,
  and many thermostable PET hydrolases (e.g. LCC) come from compost
  metagenomes with no cultivated host.
* Hits with no host information are kept and scored at
  `thermo.unknown_host_score`, so they sort below any matched thermophile but
  are never silently discarded.
* A known mesophile and an unmatched genus both score 0.0 here; they stay in
  the table with different status flags so the dashboard can explain why.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Status values written to the host_status column.
HOST_MATCHED = "matched_thermophile"
HOST_UNMATCHED = "unmatched_host"
HOST_UNKNOWN = "unknown_host"

HOST_COLUMNS = [
    "host_taxname",
    "host_genus",
    "matched_genus",
    "approx_T_opt_C",
    "thermo_score",
    "host_status",
    "host_note",
]


def load_thermophiles(
    path: Path,
    match_column: str = "genus",
    temperature_column: str = "approx_T_opt_C",
) -> dict[str, float]:
    """
    Load the curated thermophile table as {lowercase genus: T_opt_C}.

    The file is expected to have at least the genus and temperature columns
    plus a `source` column documenting where the numbers came from.
    """
    table = pd.read_csv(path)
    for column in (match_column, temperature_column):
        if column not in table.columns:
            raise ValueError(
                f"{path} is missing column {column!r}; found {list(table.columns)}"
            )

    table = table.dropna(subset=[match_column, temperature_column])
    result: dict[str, float] = {}
    for genus, temperature in zip(table[match_column], table[temperature_column]):
        key = str(genus).strip().lower()
        value = float(temperature)
        # If a genus appears twice, keep the highest documented optimum.
        result[key] = max(result.get(key, value), value)

    if not result:
        raise ValueError(f"{path} contains no usable thermophile rows.")
    return result


def genus_from_taxname(taxname: Any) -> Optional[str]:
    """'Thermus thermophilus JL-18' -> 'Thermus'; missing/NaN -> None."""
    if taxname is None or (isinstance(taxname, float) and pd.isna(taxname)):
        return None
    text = str(taxname).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return None
    return text.split()[0]


def annotate_hosts(
    hits: pd.DataFrame,
    thermophiles: Mapping[str, float],
    unknown_host_score: float = 0.0,
) -> pd.DataFrame:
    """
    Add host columns and a thermo_score to a hits table.

    Expected input column: `taxname` (may be absent, NaN or empty).
    Returns a copy; ordering is left to rank_by_host.
    """
    frame = hits.copy()
    if "taxname" not in frame.columns:
        frame["taxname"] = None

    rows = []
    for taxname in frame["taxname"]:
        genus = genus_from_taxname(taxname)
        if genus is None:
            rows.append({
                "host_taxname": "",
                "host_genus": "",
                "matched_genus": "",
                "approx_T_opt_C": None,
                "thermo_score": float(unknown_host_score),
                "host_status": HOST_UNKNOWN,
                "host_note": "unknown host: scored below thermophiles, kept",
            })
            continue

        key = genus.lower()
        if key in thermophiles:
            temperature = float(thermophiles[key])
            rows.append({
                "host_taxname": str(taxname),
                "host_genus": genus,
                "matched_genus": genus,
                "approx_T_opt_C": temperature,
                "thermo_score": temperature,
                "host_status": HOST_MATCHED,
                "host_note": "curated genus optimum; clue for stability, not proof",
            })
        else:
            rows.append({
                "host_taxname": str(taxname),
                "host_genus": genus,
                "matched_genus": "",
                "approx_T_opt_C": None,
                "thermo_score": float(unknown_host_score),
                "host_status": HOST_UNMATCHED,
                "host_note": "genus not in thermophile table; no heat-tolerance signal",
            })

    host_frame = pd.DataFrame(rows, columns=HOST_COLUMNS, index=frame.index)
    return pd.concat([frame, host_frame], axis=1)


def rank_by_host(
    hits: pd.DataFrame,
    thermophiles: Mapping[str, float],
    unknown_host_score: float = 0.0,
) -> pd.DataFrame:
    """
    Annotate hosts and sort by thermo_score (desc), then pLDDT (desc).

    pLDDT is used only to break ties; hits without a pLDDT column are fine.
    """
    frame = annotate_hosts(hits, thermophiles, unknown_host_score)

    sort_columns = ["thermo_score"]
    ascending = [False]
    if "mean_plddt" in frame.columns:
        sort_columns.append("mean_plddt")
        ascending.append(False)

    frame = frame.sort_values(sort_columns, ascending=ascending,
                              na_position="last").reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame
