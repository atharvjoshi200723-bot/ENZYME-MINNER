"""
app.py
------
Streamlit dashboard for Enzyme Miner.

Run with:  streamlit run app.py

The dashboard starts in DEMO MODE with clearly synthetic hits so it is useful
offline.  Switch the sidebar to "Live" and press "Run pipeline" for a real
Foldseek + AlphaFold run (minutes, not seconds).

Everything shown is a predicted candidate for lab testing, not a validated
enzyme.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from Bio.PDB import PDBIO, Superimposer
from Bio.PDB.Structure import Structure

from src import plddt, pipeline

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"
CONFIG = pipeline.load_config(CONFIG_PATH)

FUNNEL_LABELS = [
    ("foldseek_hits", "Foldseek hits"),
    ("fold_match", "Fold match"),
    ("structure_available", "Structure available"),
    ("triad_intact", "Triad intact"),
    ("confident_pocket", "Confident pocket"),
    ("final_shortlist", "Final shortlist"),
]

HOST_LABELS = {
    "matched_thermophile": "thermophile genus",
    "unmatched_host": "known host, not in table",
    "unknown_host": "unknown host",
}

st.set_page_config(page_title="Enzyme Miner", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      table.ranked {border-collapse: collapse; width: 100%; font-size: 0.88rem;}
      table.ranked th {background: #F4F6F8; text-align: left; padding: 6px 8px;
                       border-bottom: 2px solid #DCE2E8; position: sticky; top: 0;}
      table.ranked td {padding: 6px 8px; border-bottom: 1px solid #EDF0F3;}
      table.ranked tr:hover {background: #FBFCFD;}
      .banner {padding: 12px 16px; border-radius: 8px; margin-bottom: 12px;
               font-size: 0.95rem;}
      .banner-demo {background: #FFF4E5; border: 1px solid #F0B860; color: #6B4A12;}
      .banner-live {background: #EAF6F1; border: 1px solid #7BC4A6; color: #14533C;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached pipeline entry points
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_demo_cached(config_path: str) -> pipeline.PipelineResult:
    return pipeline.run_pipeline(config_path, demo=True)


def run_live(config_path: Path, limit: int) -> pipeline.PipelineResult:
    return pipeline.run_pipeline(config_path, demo=False, limit=limit)


def load_result() -> pipeline.PipelineResult:
    if "result" not in st.session_state:
        st.session_state.result = run_demo_cached(str(CONFIG_PATH))
        st.session_state.result_source = "demo"
    return st.session_state.result


# ---------------------------------------------------------------------------
# Structure helpers for the 3D tab
# ---------------------------------------------------------------------------

def _residue_atoms(structure: Structure, resnum: Optional[int], names: Iterable[str],
                   chain_id: str = "A"):
    """Return the first available atom among `names` for one residue."""
    if resnum is None:
        return None
    try:
        chain = structure[0][chain_id]
    except KeyError:
        return None
    for residue in chain:
        if residue.get_id()[0].strip() or residue.get_id()[1] != resnum:
            continue
        for name in names:
            if name in residue:
                return residue[name]
    return None


def _ca_atom(structure: Structure, resnum: Optional[int], chain_id: str = "A"):
    return _residue_atoms(structure, resnum, ["CA"], chain_id)


def _pdb_text(structure: Structure) -> str:
    buffer = io.StringIO()
    io_writer = PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(buffer)
    return buffer.getvalue()


def superimpose_on_seed(
    seed: Structure,
    seed_positions: dict[str, int],
    candidate: Structure,
    candidate_positions: dict[str, int],
) -> tuple[Structure, str]:
    """
    Superimpose the candidate onto the seed using catalytic atoms.

    Uses Ser OG, His NE2/ND1 and the closest Asp oxygen.  If any is missing,
    falls back to the CA atoms of the three triad residues and says so.
    """
    fixed = [
        _residue_atoms(seed, seed_positions.get("ser"), ["OG"]),
        _residue_atoms(seed, seed_positions.get("his"), ["NE2"]),
        _residue_atoms(seed, seed_positions.get("his"), ["ND1"]),
        _residue_atoms(seed, seed_positions.get("asp"), ["OD2", "OD1"]),
    ]
    moving = [
        _residue_atoms(candidate, candidate_positions.get("ser"), ["OG"]),
        _residue_atoms(candidate, candidate_positions.get("his"), ["NE2"]),
        _residue_atoms(candidate, candidate_positions.get("his"), ["ND1"]),
        _residue_atoms(candidate, candidate_positions.get("asp"), ["OD2", "OD1"]),
    ]

    method = "catalytic atoms"
    if any(atom is None for atom in fixed + moving):
        fixed = [_ca_atom(seed, seed_positions.get(role))
                 for role in ("ser", "his", "asp")]
        moving = [_ca_atom(candidate, candidate_positions.get(role))
                  for role in ("ser", "his", "asp")]
        method = "CA atoms (a catalytic atom was missing)"

    if any(atom is None for atom in fixed + moving):
        return candidate, "not superimposed (triad atoms incomplete)"

    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    superimposer.apply(list(candidate.get_atoms()))
    return candidate, f"superimposed on {method}, RMSD {superimposer.rms:.2f} A"


def render_overlay(
    seed: Structure,
    seed_positions: dict[str, int],
    candidate: Optional[Structure],
    candidate_positions: Optional[dict[str, int]],
    uploaded: Optional[Structure] = None,
):
    """Render seed + candidate (+ optional AF3 upload) with py3Dmol."""
    view = py3Dmol.view(width=980, height=540)
    view.setBackgroundColor("#FFFFFF")

    view.addModel(_pdb_text(seed), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "#C3CAD3", "opacity": 0.85}})
    seed_triad = ",".join(str(v) for v in seed_positions.values())
    view.addStyle({"model": 0, "resi": seed_triad},
                  {"stick": {"colorscheme": "orangeCarbon", "radius": 0.22}})

    model_index = 1
    if candidate is not None:
        view.addModel(_pdb_text(candidate), "pdb")
        view.setStyle({"model": model_index},
                      {"cartoon": {"color": "#7EA6E0", "opacity": 0.9}})
        if candidate_positions:
            cand_triad = ",".join(str(v) for v in candidate_positions.values() if v)
            view.addStyle({"model": model_index, "resi": cand_triad},
                          {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}})
        model_index += 1

    if uploaded is not None:
        view.addModel(_pdb_text(uploaded), "pdb")
        view.setStyle({"model": model_index},
                      {"cartoon": {"color": "#8BC9A3", "opacity": 0.9}})
        model_index += 1

    view.zoomTo()
    components.html(view._make_html(), height=560)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🧬 Enzyme Miner")
st.sidebar.caption(
    "Foldseek shape search → catalytic triad → pocket pLDDT → host ranking"
)

source = st.sidebar.radio(
    "Data source",
    ["Demo (offline synthetic)", "Live (Foldseek + AlphaFold)"],
    index=0,
    help="Demo uses synthetic hits with DEMO- ids. Live submits 5XJH to Foldseek.",
)
limit = st.sidebar.number_input(
    "Hits to evaluate (live only)", min_value=10, max_value=500, value=100, step=10,
    help="Each hit downloads an AlphaFold model; start small.",
)
if st.sidebar.button("Run pipeline", type="primary"):
    demo = source.startswith("Demo")
    with st.spinner("Running pipeline... (live runs take minutes)"):
        try:
            st.session_state.result = (
                run_demo_cached(str(CONFIG_PATH)) if demo
                else run_live(CONFIG_PATH, int(limit))
            )
            st.session_state.result_source = "demo" if demo else "live"
        except Exception as exc:  # surface the real error to the user
            st.sidebar.error(f"Pipeline failed: {exc}")

result = load_result()

st.sidebar.divider()
st.sidebar.subheader("Filters")
min_plddt = st.sidebar.slider(
    "Minimum pocket pLDDT", 0.0, 100.0,
    float(CONFIG["plddt"]["min_mean_plddt"]), step=1.0,
)
min_probability = st.sidebar.slider(
    "Minimum Foldseek probability", 0.0, 1.0,
    float(CONFIG["foldseek"]["min_probability"]), step=0.01,
)
host_options = list(HOST_LABELS)
host_filter = st.sidebar.multiselect(
    "Host status", host_options, default=host_options,
    format_func=lambda key: HOST_LABELS[key],
)
max_rows = st.sidebar.number_input("Rows to display", 5, 200,
                                   int(CONFIG["output"]["top_n"]), step=5)

# ---------------------------------------------------------------------------
# Header, banner, funnel
# ---------------------------------------------------------------------------

st.title("Enzyme Miner")
st.caption(
    "Seed: IsPETase (PDB 5XJH, Ser160–Asp206–His237). "
    "Outputs are predicted candidates for lab testing, not validated enzymes."
)

if result.demo:
    st.markdown(
        '<div class="banner banner-demo"><b>DEMO MODE</b> — every row below is '
        "synthetic (ids start with <code>DEMO-</code>). Coordinates, hosts and "
        "pLDDT values were generated locally to exercise the code offline. "
        "No real protein was searched or evaluated.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="banner banner-live"><b>LIVE RUN</b> — hits come from the '
        "Foldseek web server and structures from the AlphaFold DB. "
        "Confidence is structural plausibility, not measured activity.</div>",
        unsafe_allow_html=True,
    )

if source.startswith("Live") and result.demo:
    st.info("Showing the demo table. Press **Run pipeline** for a live search.")

funnel_cols = st.columns(len(FUNNEL_LABELS))
for column, (key, label) in zip(funnel_cols, FUNNEL_LABELS):
    column.metric(label, result.funnel.get(key, 0))

# ---------------------------------------------------------------------------
# Filtered table
# ---------------------------------------------------------------------------

hits = result.hits.copy()
if not hits.empty:
    hits = hits[hits["mean_plddt"].fillna(0) >= min_plddt]
    hits = hits[hits["probability"].fillna(0) >= min_probability]
    if host_filter:
        hits = hits[hits["host_status"].isin(host_filter)]
    hits = hits.head(int(max_rows))

tab_ranked, tab_3d, tab_about = st.tabs(
    ["Ranked candidates", "3D structure", "About & limits"]
)

with tab_ranked:
    st.subheader(f"Ranked candidates ({len(hits)})")
    if hits.empty:
        st.info("No candidates pass the current filters.")
    else:
        table = pd.DataFrame({
            "Rank": hits["rank"],
            "Target": hits["target_id"],
            "Annotation": hits["description"].fillna(""),
            "Prob.": hits["probability"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            "Host": hits["host_taxname"].replace("", "unknown"),
            "T_opt °C": hits["approx_T_opt_C"].map(
                lambda v: f"{v:.0f}" if pd.notna(v) else "—"),
            "Pocket pLDDT": hits["mean_plddt"].map(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
            "Ser–His Å": hits["ser_his_dist"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            "Asp–His Å": hits["asp_his_dist"].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            "Host signal": hits["host_status"].map(HOST_LABELS),
        })
        st.markdown(table.to_html(index=False, classes="ranked", escape=True),
                    unsafe_allow_html=True)
        st.caption(
            "Host signal is a ranking clue from the curated genus table, not "
            "proof of thermostability. Unknown hosts are kept and ranked lower."
        )

    csv_bytes = hits.to_csv(index=False).encode()
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="enzyme_miner_filtered_candidates.csv",
        mime="text/csv",
    )
    if result.results_csv:
        st.caption(f"Full ranked output: `{result.results_csv}`")

# ---------------------------------------------------------------------------
# 3D tab
# ---------------------------------------------------------------------------

with tab_3d:
    st.subheader("Seed vs selected candidate")
    st.caption(
        "The candidate is superimposed on the seed using the catalytic atoms; "
        "the seed triad is orange, the candidate triad is cyan. This shows "
        "predicted geometry only."
    )

    seed_structure: Optional[Structure] = None
    try:
        seed_path = pipeline.ensure_seed_structure(CONFIG)
        seed_structure = plddt.parse_structure(seed_path)
    except Exception as exc:
        st.warning(
            "Seed structure is not available locally and could not be downloaded "
            f"({exc}). Connect once with network access to cache 5XJH; the 3D "
            "tab needs its coordinates."
        )

    candidate_structure: Optional[Structure] = None
    candidate_positions: Optional[dict[str, int]] = None
    selected_id: Optional[str] = None

    if not hits.empty:
        options = hits["target_id"].tolist()
        selected_id = st.selectbox("Candidate", options)
        row = hits[hits["target_id"] == selected_id].iloc[0]
        candidate_positions = {
            "ser": int(row["ser_pos"]) if pd.notna(row["ser_pos"]) else None,
            "his": int(row["his_pos"]) if pd.notna(row["his_pos"]) else None,
            "asp": int(row["asp_pos"]) if pd.notna(row["asp_pos"]) else None,
        }

        try:
            if result.demo:
                _, demo_structures = pipeline.build_demo_hits()
                candidate_structure = demo_structures.get(selected_id)
            else:
                af_settings = plddt.AlphaFoldSettings.from_config(CONFIG)
                path = plddt.download_hit_structure(selected_id, af_settings)
                candidate_structure = plddt.parse_structure(path) if path else None
        except Exception as exc:
            st.warning(f"Could not load the candidate structure: {exc}")

    uploaded_structure: Optional[Structure] = None
    upload = st.file_uploader(
        "Optional: overlay an AlphaFold3 model (.pdb / .cif)",
        type=["pdb", "cif", "mmcif"],
    )
    if upload is not None:
        suffix = Path(upload.name).suffix or ".pdb"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(upload.getvalue())
            upload_path = Path(handle.name)
        try:
            uploaded_structure = plddt.parse_structure(upload_path)
            st.caption(f"Uploaded {upload.name} is displayed only; it is not ranked.")
        except Exception as exc:
            st.warning(f"Could not parse the uploaded model: {exc}")

    if seed_structure is None:
        st.info("3D view needs the cached seed structure (see warning above).")
    else:
        seed_positions = {
            "ser": int(CONFIG["triad"]["ser"]),
            "his": int(CONFIG["triad"]["his"]),
            "asp": int(CONFIG["triad"]["asp"]),
        }
        note = ""
        if candidate_structure is not None and candidate_positions is not None:
            candidate_structure, note = superimpose_on_seed(
                seed_structure, seed_positions, candidate_structure,
                {k: v for k, v in candidate_positions.items() if v is not None},
            )
            st.caption(f"Candidate {selected_id}: {note}.")
        elif selected_id:
            st.info("No structure available for this candidate; showing the seed only.")

        render_overlay(seed_structure, seed_positions, candidate_structure,
                       candidate_positions, uploaded_structure)

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------

with tab_about:
    st.subheader("What this is")
    st.markdown(
        """
**Enzyme Miner** takes a seed PET hydrolase (IsPETase, PDB 5XJH, catalytic
triad Ser160–Asp206–His237), searches AlphaFold DB and MGnify by 3D shape
through the Foldseek web server, and keeps hits that:

1. still carry a Ser/His/Asp triad at the mapped positions,
2. have catalytic distances within the configured thresholds, and
3. have a confident AlphaFold pocket (mean pLDDT ≥ 80).

Survivors are ranked by the documented growth temperature of their host
genus, with unknown hosts kept but ranked lower.
"""
    )
    st.subheader("Honest status")
    st.markdown(
        """
- The Foldseek search, AlphaFold downloads and all geometry/pLDDT numbers are
  real when you run in **Live** mode.
- **Demo** mode is synthetic: the ids say `DEMO-`, and the coordinates are
  generated locally.
- The rediscovery benchmark is currently **inconclusive**: PHL7 has no
  verified wild-type UniProt accession, and LCC (G9BY57) is missing from the
  top 1000 `afdb50` hits. See `results/benchmark.csv` and the README.
- MGnify-ESM30 hits are searched but cannot be evaluated in this prototype
  because only AlphaFold DB entries have a structure download endpoint; they
  are counted as "no structure".
"""
    )
    st.subheader("Limits")
    st.markdown(
        """
- pLDDT is model self-confidence, not activity.
- Triad geometry is necessary-looking, not sufficient for catalysis.
- Thermophile origin is a stability clue, not proof.
- Residue numbering of hits assumes Foldseek target numbering matches the
  structure (true for AlphaFold models).

**Outputs are predicted candidates for lab testing, not validated enzymes.**
"""
    )
