# Enzyme Miner (hackathon prototype)

Given IsPETase (PDB **5XJH**, catalytic triad Ser160–Asp206–His237), search
AlphaFold DB and MGnify by 3D shape with the Foldseek web server, check that
each hit still has a plausible Ser–His–Asp triad, filter on pocket pLDDT, rank
by host heat tolerance, and emit a ranked shortlist.

> **Outputs are predicted candidates for lab testing, not validated enzymes.**
> No activity, expression, or stability claim is made anywhere in this repo.

## What it actually does

```
5XJH seed ──► Foldseek web API ──► triad mapping + geometry ──► pocket pLDDT
                    │                      │                        │
              disk cache            alignment qaln/taln        AlphaFold DB model
                    └──────────────────────┴────────────┬───────────┘
                                                        ▼
                                         host-genus heat ranking
                                                        ▼
                                         results/ranked_candidates.csv
```

Funnel counts are logged and written to `results/funnel.json`:
`foldseek_hits → fold_match → structure_available → triad_intact →
confident_pocket → final_shortlist`.

All thresholds live in [`config.yaml`](config.yaml): triad distances, pocket
radius, minimum pLDDT, minimum Foldseek probability, database names, cache
paths, benchmark controls.

## Status: what is real vs demo

| Piece | Status |
|---|---|
| Foldseek search | **Real.** Endpoints and database names were verified live on 2026-10-24 (`afdb50` = AlphaFold/UniProt50 v6, `mgnify_esm30` = MGnify-ESM30 v1). Results are JSON; the parser also handles an m8 tar/gzip archive fallback. |
| Triad mapping and distances | **Real**, computed from Foldseek `qaln`/`taln` and downloaded coordinates. Unit-tested, and checked against real 5XJH (2.92 Å / 2.67 Å). |
| Pocket pLDDT | **Real**, from the AlphaFold PDB B-factor column. |
| Host ranking | **Real** matching against the curated `data/thermophiles.csv`. Small table, clearly sourced; absence of a genus is not evidence of mesophily. |
| Dashboard demo mode | **Synthetic.** Rows have `DEMO-` ids, generated coordinates and made-up hosts; the app shows a DEMO MODE banner. It runs offline and exists to exercise the code. |
| Rediscovery benchmark | **Implemented, currently inconclusive.** See below. |

## Quick start (macOS / Linux, Python 3.11+)

```bash
cd enzyme_miner
python3.11 -m venv .venv          # or: uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt

# offline sanity run (seconds, no network)
python -m src.pipeline --demo

# dashboard
streamlit run app.py
```

The dashboard opens in **Demo** mode. Switch the sidebar to **Live** and press
**Run pipeline** for a real search. Live runs download one AlphaFold model per
hit, so start with a small "Hits to evaluate" value.

### Real pipeline from the CLI

```bash
python -m src.pipeline                 # full configured run (up to 500 hits)
python -m src.pipeline --limit 60      # smaller debug run
python -m src.pipeline --no-cache      # ignore cached Foldseek/AF responses
```

Caches live under `.cache/` (Foldseek JSON, AlphaFold PDBs, seed PDB). Deleting
`.cache/` forces fresh downloads.

### Benchmark

```bash
python -m src.benchmark                          # runs the pipeline, then benchmarks
python -m src.benchmark --ranked-csv results/ranked_candidates.csv
```

Controls are read from [`data/known_pet_hydrolases.csv`](data/known_pet_hydrolases.csv).
A hit is identified by (1) a verified UniProt accession inside the Foldseek
target id, (2) the enzyme name in the AlphaFold description, or (3) a specific
source organism. The benchmark reads the same ranked CSV as the dashboard and
never boosts or re-orders anything.

**Result on 2026-10-24 (real `--limit 60` run):**

- Funnel: 60 hits → 60 fold matches → 51 structures (9 MGnify hits have no
  AlphaFold entry) → 49 triad-intact → 49 confident pockets → 49 shortlist.
- Verdict: **inconclusive**. PHL7 has no confirmed wild-type UniProt accession
  (`8BRB` / `A0AA82WPD4` is the catalysis-dead S131A mutant; wild-type PDB
  `7NEI` is listed as "unidentified"), so it cannot be identified reliably.
- LCC (UniProt **G9BY57**) is absent from the top 1000 `afdb50` hits, even
  though its AlphaFold entry exists. `afdb50` is a 50 %-identity cluster, so
  the exact accession is not guaranteed to be a search target. Accession-only
  rediscovery is therefore a weak test; see TODO below.

These are reported as-is rather than tuned into a pass.

## Tests

```bash
python -m pytest
```

81 tests, no network. They cover alignment mapping with gaps and offsets,
missing atoms (raise/flag, never substitute), pLDDT parsing and pocket
selection, unknown hosts, the offline pipeline funnel, and benchmark verdicts.

## Honest limits

- **Prediction, not validation.** pLDDT is the model's self-confidence; triad
  geometry is necessary-looking, not sufficient for catalysis. Everything must
  be tested in the lab.
- **Numbering assumption.** Hit residue numbers come from Foldseek `tstart`
  plus the alignment walk, and are looked up directly in the structure. This is
  correct for AlphaFold models (numbered 1..N) but is an assumption elsewhere.
- **MGnify blind spot.** MGnify-ESM30 hits are searched but have no AlphaFold
  accession, so this prototype cannot evaluate their triad/pLDDT. They are
  counted as `no_structure`, not silently dropped.
- **Host ranking is a clue.** Thermophile-derived enzymes are often more stable,
  but many good PET hydrolases come from compost metagenomes with no cultivated
  host. Unknown hosts are kept and ranked below matched genera.
- **Curated table is small.** `data/thermophiles.csv` has ~15 genera with
  sources; it is not a taxonomy database.
- **Public Foldseek is shared.** Jobs can queue; there is a client-side rate
  limit, retries, and on-disk caching.
- **3D viewer loads 3Dmol.js from a CDN.** The demo itself is offline, but the
  3D tab needs network the first time unless that script is already cached.
- **Benchmark identification is metadata-bound.** A known enzyme can be present
  yet unidentified (or absent because of `afdb50` clustering).

## Unverified / TODO

- PHL7 wild-type UniProt accession — **unresolved**. Do not cite `A0AA82WPD4`
  as wild type.
- ThermoPETase / FAST-PETase accessions — TODO (engineered variants).
- Sequence-based benchmark matching against Foldseek `tSeq` (e.g. ≥90 %
  identity) would fix the `afdb50`-clustering blind spot. Not implemented.
- MGnify-ESM30 structure retrieval and pLDDT — not implemented.
- The AlphaFold3 upload in the dashboard is display-only; it is not ranked.
- Full 500-hit live run was not executed during development (one download per
  hit); the 60-hit run above is the reference output in `results/`.

## Repo layout

```
enzyme_miner/
  app.py                     Streamlit dashboard (light theme)
  config.yaml                every threshold and external name
  data/thermophiles.csv      curated genera + T_opt + source
  data/known_pet_hydrolases.csv
  src/foldseek_client.py     submit / poll / download / parse + cache
  src/triad.py               alignment mapping, residue types, geometry
  src/plddt.py               AlphaFold download, pocket selection, pLDDT
  src/thermo.py              host matching and ranking
  src/pipeline.py            real + demo funnel, CSV/funnel output
  src/benchmark.py           rediscovery test and verdict
  tests/                     offline pytest suite
  results/                   generated outputs
```
