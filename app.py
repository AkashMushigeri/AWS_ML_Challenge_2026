"""Streamlit Web Application for amaML Business Entity Resolution.

Provides an interactive, container-safe dashboard for the existing amaML pipeline:
- Upload Source 1, 2, and 3 TSV files or specify server-side data paths.
- Execute entity resolution using SQLite blocking and pre-trained CatBoost matching.
- Live progress feedback and real-time execution logs.
- Validation checks via utils/validate_submission.py.
- TSV downloads for matching_results.tsv and candidate_pairs.tsv.
- Secure access control via AUTH_PASSWORD.
- Ephemeral processing with automatic disk cleanup.
"""

import io
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import pandas as pd
import streamlit as st

# Ensure repository root and src/ are in Python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Use the existing amaML pipeline classes and functions
from entity_resolution.config import PipelineConfig
from entity_resolution.io import read_records
from entity_resolution.model import PairModel
from entity_resolution.pipeline import EntityResolutionPipeline
from utils.validate_submission import validate as validate_submission

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="amaML Entity Resolution Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Logging Handler for Streamlit Live Logs
# ---------------------------------------------------------------------------
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler to stream pipeline logs into a Streamlit container."""

    def __init__(self, log_container):
        super().__init__()
        self.log_container = log_container
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
        if len(self.log_buffer) > 200:
            self.log_buffer.pop(0)
        self.log_container.code("\n".join(self.log_buffer), language="log")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def check_authentication() -> bool:
    """Verify user against AUTH_PASSWORD if configured."""
    expected_password = os.getenv("AUTH_PASSWORD", "").strip()
    if not expected_password:
        return True  # Open access when unset or empty

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.sidebar.markdown("### 🔒 Authentication Required")
    pwd_input = st.sidebar.text_input("Enter Access Password", type="password", key="login_pwd")
    if st.sidebar.button("Log In", key="login_btn"):
        if pwd_input == expected_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("Invalid password. Access denied.")
            return False

    st.warning("⚠️ Access restricted. Please authenticate in the sidebar to use this application.")
    return False


# ---------------------------------------------------------------------------
# Safe Chunked File Upload Handler
# ---------------------------------------------------------------------------
def save_uploaded_chunked(uploaded_file, target_path: Path) -> Path:
    """Save an uploaded file in 1MB chunks to disk without buffering all into RAM."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    uploaded_file.seek(0)
    return target_path


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    if not check_authentication():
        return

    st.title("Amazon ML Business Entity Resolution")
    st.markdown(
        "Link noisy business records across **Source 2** and **Source 3** to reference entities in **Source 1** "
        "using high-recall SQLite blocking and pre-trained gradient-boosted matching."
    )

    # Sidebar: Hyperparameters and Model Configuration
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.subheader("Model Selection")
    default_model_path = BASE_DIR / "models" / "pair_model.joblib"
    model_source = st.sidebar.radio(
        "Model",
        ("Pre-trained Model (models/pair_model.joblib)", "Upload Custom Model (.joblib)"),
        index=0,
    )

    custom_model_upload = None
    if model_source == "Upload Custom Model (.joblib)":
        custom_model_upload = st.sidebar.file_uploader("Upload .joblib model", type=["joblib"])

    st.sidebar.subheader("Inference Hyperparameters")
    threshold = st.sidebar.slider(
        "Match Probability Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.78,
        step=0.01,
        help="Pairs with match probability >= threshold are emitted as positive matches.",
    )
    max_candidates = st.sidebar.slider(
        "Max Candidates per Entity",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum candidate records retained from blocking for each Source 1 business.",
    )
    device = st.sidebar.selectbox(
        "Compute Device",
        ("cpu", "auto"),
        index=0,
        help="'cpu' executes inference deterministically on CPU; 'auto' attempts GPU if available.",
    )
    run_validator = st.sidebar.checkbox(
        "Run Submission Validator",
        value=True,
        help="Validates TSV header schema, singletons, entity coverage, and candidate subsets.",
    )

    # Main Tabs
    tab_predict, tab_info = st.tabs(["🚀 Run Prediction", "ℹ️ System Info & Privacy"])

    with tab_predict:
        st.subheader("1. Input Data")
        input_mode = st.radio(
            "Input Mode",
            ("Upload TSV Files (Drag & Drop)", "Server-Side Path (Mounted Volume / Large Datasets)"),
            horizontal=True,
        )

        s1_file, s2_file, s3_file = None, None, None
        server_path_input = ""

        if input_mode == "Upload TSV Files (Drag & Drop)":
            c1, c2, c3 = st.columns(3)
            with c1:
                s1_file = st.file_uploader("Source 1 TSV (Reference)", type=["tsv", "csv", "txt"], key="s1")
            with c2:
                s2_file = st.file_uploader("Source 2 TSV", type=["tsv", "csv", "txt"], key="s2")
            with c3:
                s3_file = st.file_uploader("Source 3 TSV", type=["tsv", "csv", "txt"], key="s3")
        else:
            server_path_input = st.text_input(
                "Server Dataset Directory",
                value="dataset/test",
                help="Path relative to application root or absolute path containing test_source1.tsv, test_source2.tsv, and test_source3.tsv.",
            )

        st.divider()
        st.subheader("2. Run Entity Resolution Pipeline")

        run_btn = st.button("Run Entity Resolution", type="primary", use_container_width=True)

        if run_btn:
            # Validate input selection
            if input_mode == "Upload TSV Files (Drag & Drop)":
                if not (s1_file and s2_file and s3_file):
                    st.error("Please upload all three files (Source 1, Source 2, and Source 3) before running.")
                    return
            else:
                input_dir = Path(server_path_input)
                if not input_dir.is_dir():
                    st.error(f"Server dataset directory not found: {input_dir}")
                    return
                for fname in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
                    if not (input_dir / fname).is_file():
                        st.error(f"Missing required file in server directory: {fname}")
                        return

            status_box = st.status("Initializing Entity Resolution Pipeline...", expanded=True)
            log_container = st.empty()
            log_handler = StreamlitLogHandler(log_container)
            log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(log_handler)

            try:
                # Use ephemeral temporary directory for isolated execution and safe cleanup
                with tempfile.TemporaryDirectory(prefix="amaml_run_") as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    data_dir = tmp_path / "data"
                    output_dir = tmp_path / "output"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    status_box.update(label="Staging input files...", state="running")
                    if input_mode == "Upload TSV Files (Drag & Drop)":
                        s1_path = save_uploaded_chunked(s1_file, data_dir / "test_source1.tsv")
                        s2_path = save_uploaded_chunked(s2_file, data_dir / "test_source2.tsv")
                        s3_path = save_uploaded_chunked(s3_file, data_dir / "test_source3.tsv")
                    else:
                        s1_path = Path(server_path_input) / "test_source1.tsv"
                        s2_path = Path(server_path_input) / "test_source2.tsv"
                        s3_path = Path(server_path_input) / "test_source3.tsv"

                    # 1. Load DataFrames using existing amaML read_records
                    status_box.update(label="Reading and validating input records...", state="running")
                    s1_df = read_records(s1_path)
                    s2_df = read_records(s2_path)
                    s3_df = read_records(s3_path)

                    # 2. Resolve Model
                    status_box.update(label="Loading pair resolution model...", state="running")
                    if model_source == "Upload Custom Model (.joblib)":
                        if not custom_model_upload:
                            st.error("Please upload a .joblib model file.")
                            return
                        custom_path = tmp_path / "custom_model.joblib"
                        save_uploaded_chunked(custom_model_upload, custom_path)
                        pair_model = PairModel.load(custom_path)
                    else:
                        if not default_model_path.is_file():
                            raise FileNotFoundError(f"Model file not found at {default_model_path}")
                        pair_model = PairModel.load(default_model_path)

                    # 3. Instantiate existing amaML Pipeline
                    config = PipelineConfig(
                        data_dir=data_dir,
                        output_dir=output_dir,
                        threshold=threshold,
                        max_candidates_per_entity=max_candidates,
                        device=device,
                    )
                    pipeline = EntityResolutionPipeline(config)

                    # 4. Execute prediction
                    status_box.update(label="Running SQLite candidate blocking and pairwise scoring...", state="running")
                    start_time = time.monotonic()
                    matching_path, candidate_path = pipeline.predict_frames(s1_df, s2_df, s3_df, pair_model)
                    elapsed = time.monotonic() - start_time

                    status_box.update(label=f"Pipeline completed in {elapsed:.1f}s!", state="complete")

                    # Read generated outputs into memory for direct download
                    with open(matching_path, "rb") as mf:
                        matching_bytes = mf.read()
                    with open(candidate_path, "rb") as cf:
                        candidate_bytes = cf.read()

                    matching_df = pd.read_csv(matching_path, sep="\t")

                    # Run submission validator if requested
                    if run_validator:
                        with st.expander("Submission Validator Report", expanded=False):
                            val_test_dir = str(s1_path.parent)
                            errors, warnings = validate_submission(
                                str(matching_path), str(candidate_path), val_test_dir, check_ids=False
                            )
                            for w in warnings:
                                st.warning(f"⚠️ {w}")
                            if errors:
                                for e in errors:
                                    st.error(f"❌ {e}")
                            else:
                                st.success("✅ Submission passed all validation checks! Safe to submit.")

                    st.success(
                        f"Successfully generated matches for {len(matching_df)} Source 1 entities in {elapsed:.1f}s!"
                    )

                    # Download Buttons
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.download_button(
                            label="📥 Download matching_results.tsv",
                            data=matching_bytes,
                            file_name="matching_results.tsv",
                            mime="text/tab-separated-values",
                            type="primary",
                            use_container_width=True,
                        )
                    with dcol2:
                        st.download_button(
                            label="📥 Download candidate_pairs.tsv",
                            data=candidate_bytes,
                            file_name="candidate_pairs.tsv",
                            mime="text/tab-separated-values",
                            use_container_width=True,
                        )

                    # Preview Results
                    st.subheader("Results Preview")
                    st.dataframe(matching_df.head(20), use_container_width=True)

            except Exception as exc:
                status_box.update(label="Error occurred", state="error")
                st.error(f"❌ Error during execution: {exc}")
                logging.getLogger("amaml_app").error("Pipeline execution failed: %s", exc, exc_info=True)
            finally:
                root_logger.removeHandler(log_handler)

    with tab_info:
        st.subheader("System Architecture & Data Privacy")
        st.markdown(
            """
            ### 🛡️ Privacy & Ephemeral Execution
            1. **Zero Public Leakage**: The `dataset/` directory is strictly excluded from version control and Docker images via `.gitignore` and `.dockerignore`.
            2. **Isolated Ephemeral Execution**: Uploaded files and temporary databases (`.blocking-*.sqlite`) are written into isolated temporary directories and automatically deleted after prediction.
            3. **Safe Logging**: The pipeline only logs operational metadata (record counts, processing rates, progress percentages) and never prints private business names, addresses, or tax IDs.
            4. **Path Traversal Protection**: Uploaded files are streamed using sanitized system filenames (`test_source1.tsv`), eliminating arbitrary file write or traversal vulnerabilities.

            ### 📈 Performance Considerations
            - **SQLite Inverted Index**: Candidate generation uses a disk-backed clustered index on `(key, ordinal)` without expanding millions of pairs into RAM dictionaries.
            - **Model Backend**: CatBoost is evaluated on CPU by default with zero GPU requirements, running within memory limits.
            """
        )
        st.info("Health Check Endpoint: `GET /_stcore/health` (HTTP 200 OK)")


if __name__ == "__main__":
    main()
