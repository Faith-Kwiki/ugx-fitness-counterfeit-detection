"""Streamlit workspace for UGX counterfeit and circulation-fitness research."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.banknote.models.fitness_multitask_multisignal_cnn import (
    CONDITION_NAMES,
    FITNESS_OUTCOMES,
    SEVERITY_NAMES,
    FitnessMultiTaskMultiSignalCNN,
    joint_outcome_predictions,
    joint_outcome_probabilities,
)
from src.banknote.preprocessing.prepare_streams import resize_preserving_aspect
from src.banknote.preprocessing.transforms import extract_edges, extract_rgb, extract_texture


ROOT = Path(__file__).resolve().parent
RUN_ID = "prototype-fitness-policy-seed42-20261127"
RUN_DIR = ROOT / "runs" / "fitness" / RUN_ID
DEFAULT_CHECKPOINT = RUN_DIR / "best_checkpoint.pth"
RESULTS_JSON = RUN_DIR / "test_evaluation.json"
SUMMARY_JSON = RUN_DIR / "summary.json"
PRELABEL_CSV = ROOT / "annotations" / "prototype_fitness_prelabels.csv"
REVIEW_TEMPLATE_CSV = ROOT / "annotations" / "prototype_fitness_review_template.csv"
ROADMAP_MD = ROOT / "experiments" / "ROADMAP_STATUS.md"
RULE_MD = ROOT / "annotations" / "prototype_fitness_rule.md"

OUTCOME_DISPLAY = {
    "counterfeit": "Counterfeit",
    "genuine_fit": "Genuine and fit for circulation",
    "genuine_unfit": "Genuine and unfit for circulation",
}
SEVERITY_DISPLAY = {
    "none": "No visible sign",
    "mild": "Mild indicator",
    "moderate": "Moderate indicator",
    "severe": "Severe indicator",
}


st.set_page_config(
    page_title="UGX Banknote Lab",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink: #142b3b; --muted: #607582; --teal: #147d78; --gold: #e8b54a; }
      .stApp { background: #f5f8f8; color: var(--ink); }
      [data-testid="stHeader"] { background: rgba(245,248,248,.92); }
      [data-testid="stSidebar"] { background: #102b3a; }
      [data-testid="stSidebar"] * { color: #edf5f5; }
      [data-testid="stSidebar"] hr { border-color: #315061; }
      .hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg,#123344,#147d78); margin-bottom: 1.2rem; }
      .hero h1 { color: white; margin: 0; letter-spacing: -.03em; }
      .hero p { color: #d7e9e8; margin: .4rem 0 0; font-size: 1.03rem; }
      .soft-card { padding: 1rem 1.1rem; border: 1px solid #dce7e8; border-radius: 14px;
        background: white; height: 100%; }
      .eyebrow { color: #8dd2c6; text-transform: uppercase; letter-spacing: .12em;
        font-size: .75rem; font-weight: 700; }
      .muted { color: #607582; }
      .decision { padding: 1.1rem 1.25rem; border-radius: 14px; color: #123344;
        background: #e5f3ef; border-left: 5px solid #147d78; }
      .warning { padding: .9rem 1rem; border-radius: 12px; background: #fff5db;
        border-left: 4px solid #e8b54a; color: #563f0b; }
      div[data-testid="stMetric"] { background: white; border: 1px solid #dce7e8;
        border-radius: 12px; padding: .7rem .9rem; }
      .small-note { font-size: .84rem; color: #607582; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_metrics() -> dict:
    return read_json(RESULTS_JSON)


def load_summary() -> dict:
    return read_json(SUMMARY_JSON)


def load_dataset_dir() -> Path | None:
    value = st.session_state.get("dataset_dir", "").strip()
    if value:
        path = Path(value).expanduser()
        return path if path.is_dir() else None
    candidate_roots = [ROOT / "data" / "raw"]
    desktop_copy = Path.home() / "OneDrive" / "Desktop" / ROOT.name
    if desktop_copy.resolve() != ROOT.resolve():
        candidate_roots.append(desktop_copy / "data" / "raw")
    for candidate_root in candidate_roots:
        try:
            matches = list(candidate_root.glob("Ugandan banknote*/labelled_ugandan_banknote_dataset_clean"))
            if len(matches) == 1:
                return matches[0]
        except OSError:
            continue
    return None


def resolve_image_path(row: dict | pd.Series) -> Path | None:
    raw = str(row.get("image_path", "") or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    candidate = (ROOT / candidate).resolve()
    if candidate.is_file():
        return candidate
    dataset_dir = load_dataset_dir()
    normalized = raw.replace("\\", "/")
    if dataset_dir and "/splits/" in normalized:
        tail = normalized.split("/splits/", 1)[1]
        candidate = dataset_dir / "splits" / Path(tail)
        if candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner="Loading the multi-signal model…")
def load_model(checkpoint_bytes: bytes, device_name: str):
    device = torch.device(device_name)
    # PyTorch 2.6+ treats weights_only as the default. This project's
    # checkpoints store torch.__version__ as a TorchVersion object, so allow
    # that specific safe metadata type while keeping weights_only enabled.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=device, weights_only=True
        )
    model = FitnessMultiTaskMultiSignalCNN().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, device


def load_checkpoint(checkpoint_path: Path, uploaded_checkpoint) -> tuple[bytes | None, str | None]:
    if uploaded_checkpoint is not None:
        return uploaded_checkpoint.getvalue(), uploaded_checkpoint.name
    if checkpoint_path.is_file():
        return checkpoint_path.read_bytes(), str(checkpoint_path)
    return None, None


def image_to_streams(image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file could not be read as an image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = resize_preserving_aspect(rgb, target_size=224)
    edges = extract_edges(rgb)
    texture = extract_texture(rgb)
    tensors = (
        torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0),
        torch.from_numpy(np.ascontiguousarray(edges[None, :, :])).float().div_(255.0).unsqueeze(0),
        torch.from_numpy(np.ascontiguousarray(texture[None, :, :])).float().div_(255.0).unsqueeze(0),
    )
    return rgb, *tensors


def predict(model, device, image_bytes: bytes) -> dict:
    display_image, rgb, edges, texture = image_to_streams(image_bytes)
    rgb, edges, texture = (value.to(device) for value in (rgb, edges, texture))
    with torch.no_grad():
        outputs = model(rgb, edges, texture)
        outcome_probs = joint_outcome_probabilities(
            outputs["authenticity_logits"], outputs["fitness_logits"]
        )[0].cpu().numpy()
        outcome_index = int(
            joint_outcome_predictions(
                outputs["authenticity_logits"], outputs["fitness_logits"]
            )[0].item()
        )
        authenticity_probs = outputs["authenticity_logits"].softmax(dim=1)[0].cpu().numpy()
        fitness_probs = outputs["fitness_logits"].softmax(dim=1)[0].cpu().numpy()
        condition_probs = outputs["condition_logits"].sigmoid()[0].cpu().numpy()
        severity_probs = outputs["severity_logits"].softmax(dim=2)[0].cpu().numpy()
    return {
        "display_image": display_image,
        "outcome": FITNESS_OUTCOMES[outcome_index],
        "outcome_probabilities": outcome_probs,
        "authenticity_probabilities": authenticity_probs,
        "fitness_probabilities": fitness_probs,
        "condition_probabilities": dict(zip(CONDITION_NAMES, condition_probs.tolist())),
        "severity_probabilities": {
            name: severity_probs[index].tolist()
            for index, name in enumerate(CONDITION_NAMES)
        },
    }


def show_hero(title: str, subtitle: str, eyebrow: str = "UGX banknote lab"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_prototype_notice():
    st.markdown(
        '<div class="warning"><strong>Research prototype.</strong> The model was trained and evaluated against low-confidence, rule-derived labels. It is not a Bank of Uganda sorter or authentication decision. Confirm important decisions with a trained reviewer.</div>',
        unsafe_allow_html=True,
    )


def page_dashboard(metrics: dict, summary: dict):
    show_hero("UGX Banknote Lab", "A single workspace for note assessment, label review, and experiment results.")
    show_prototype_notice()
    st.write("")
    outcomes = metrics.get("classes", {})
    unfit = outcomes.get("genuine_unfit", {})
    cols = st.columns(4)
    cols[0].metric("Held-out accuracy", f"{metrics.get('accuracy', 0):.1%}" if metrics else "—")
    cols[1].metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}" if metrics else "—")
    cols[2].metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', 0):.1%}" if metrics else "—")
    cols[3].metric("Unfit test examples", str(unfit.get("support", "—")))

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Final outcome performance")
        if outcomes:
            frame = pd.DataFrame(
                [
                    {
                        "Outcome": OUTCOME_DISPLAY.get(name, name),
                        "Precision": value.get("precision", 0),
                        "Recall": value.get("recall", 0),
                        "F1": value.get("f1", 0),
                        "Examples": value.get("support", 0),
                    }
                    for name, value in outcomes.items()
                ]
            )
            st.dataframe(
                frame.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}"}),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Evaluation results are not present yet.")
        st.caption("The unfit class has only 15 held-out examples, so its score can move substantially with a few corrected labels.")
    with right:
        st.subheader("Confusion matrix")
        matrix = metrics.get("confusion_matrix")
        if matrix:
            labels = ["Counterfeit", "Genuine fit", "Genuine unfit"]
            chart = pd.DataFrame(matrix, index=labels, columns=labels)
            st.dataframe(chart, width="stretch")
            st.caption("Rows are the prelabel outcome; columns are the model prediction.")
        st.subheader("Current training run")
        st.write(f"**Selected epoch:** {summary.get('best_epoch', '—')}")
        st.write(f"**Run:** `{RUN_ID}`")
        st.write("**Signals:** RGB, edges, and texture")
        st.write("**Decision policy:** counterfeit first; then genuine fit/unfit")

    st.subheader("Condition reason heads")
    sev = metrics.get("severity_by_condition", {})
    rows = [
        {
            "Condition": name.title(),
            "Severity macro F1": values.get("macro_f1", 0),
            "Reviewed proxy examples": values.get("count", 0),
        }
        for name, values in sev.items()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows).style.format({"Severity macro F1": "{:.3f}"}), hide_index=True, width="stretch")
        st.caption("Severity scores are too weak for the app to treat phrases such as ‘heavily creased’ as confirmed findings.")


def page_assess(checkpoint_path: Path):
    show_hero("Assess a note", "Upload a clear photo to see the prototype's joint decision and condition signals.")
    show_prototype_notice()
    uploaded_checkpoint = st.file_uploader("Optional: choose a model checkpoint", type=["pth"], help="Use this if the local checkpoint path is not available.")
    checkpoint_bytes, checkpoint_source = load_checkpoint(checkpoint_path, uploaded_checkpoint)
    if checkpoint_bytes is None:
        st.warning("The trained checkpoint is not in this copy of the project. Set its path in the sidebar or upload the .pth file.")
        return
    uploaded_image = st.file_uploader("Upload a banknote photo", type=["jpg", "jpeg", "png", "webp"], key="note_image")
    if uploaded_image is None:
        st.info("For the clearest result, photograph one note in good light and include the whole note in the frame.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, device = load_model(checkpoint_bytes, device)
        result = predict(model, device, uploaded_image.getvalue())
    except Exception as exc:
        st.error(f"Could not assess this image: {exc}")
        return
    left, right = st.columns([1, 1.2])
    with left:
        st.image(result["display_image"], caption=uploaded_image.name, width="stretch")
        st.caption(f"Checkpoint: {Path(checkpoint_source).name if checkpoint_source else 'uploaded model'} · {device.upper()}")
    with right:
        outcome = result["outcome"]
        st.markdown(f'<div class="decision"><div class="eyebrow" style="color:#147d78">Prototype suggestion</div><h2>{OUTCOME_DISPLAY[outcome]}</h2></div>', unsafe_allow_html=True)
        st.write("")
        confidence = float(result["outcome_probabilities"][FITNESS_OUTCOMES.index(outcome)])
        st.metric("Score for this outcome", f"{confidence:.1%}")
        st.caption("This score is not calibrated as a real-world probability.")
        st.subheader("Other outcome scores")
        for index, name in enumerate(FITNESS_OUTCOMES):
            st.progress(float(result["outcome_probabilities"][index]), text=f"{OUTCOME_DISPLAY[name]} · {result['outcome_probabilities'][index]:.1%}")

    st.subheader("Condition indicators")
    st.caption("These are model signals from low-confidence labels. Use them as review prompts, not confirmed damage findings.")
    cols = st.columns(4)
    for index, name in enumerate(CONDITION_NAMES):
        probability = result["condition_probabilities"][name]
        severity_index = int(np.argmax(result["severity_probabilities"][name]))
        severity = SEVERITY_NAMES[severity_index]
        with cols[index]:
            st.metric(name.title(), f"{probability:.0%} signal")
            st.write(SEVERITY_DISPLAY[severity])
            st.progress(float(probability))
    st.info("The final outcome follows the authenticity-first rule. When authenticity is predicted counterfeit, the fit/unfit head does not change the displayed outcome.")


def page_review():
    show_hero("Review labels", "Check the existing prelabels, correct a row, and download your reviewed copy.")
    show_prototype_notice()
    if "review_frame" not in st.session_state:
        if PRELABEL_CSV.is_file():
            st.session_state.review_frame = pd.read_csv(PRELABEL_CSV, keep_default_na=False)
        else:
            st.session_state.review_frame = pd.read_csv(REVIEW_TEMPLATE_CSV, keep_default_na=False) if REVIEW_TEMPLATE_CSV.is_file() else pd.DataFrame()
    frame = st.session_state.review_frame
    if frame.empty:
        st.error("No prelabel or review template CSV was found in annotations/.")
        return

    split_col, outcome_col, find_col = st.columns([1, 1, 1.5])
    split_options = ["All splits", *sorted(frame["split"].dropna().astype(str).unique())]
    outcome_options = ["All outcomes", *sorted(frame["fitness_outcome"].dropna().astype(str).unique())]
    selected_split = split_col.selectbox("Split", split_options)
    selected_outcome = outcome_col.selectbox("Outcome", outcome_options)
    query = find_col.text_input("Find image ID", placeholder="UGX_genuine_...")
    view = frame.copy()
    if selected_split != "All splits":
        view = view[view["split"].astype(str) == selected_split]
    if selected_outcome != "All outcomes":
        view = view[view["fitness_outcome"].astype(str) == selected_outcome]
    if query:
        view = view[view["image_id"].astype(str).str.contains(query, case=False, na=False)]
    if view.empty:
        st.info("No rows match those filters.")
        return
    ids = view["image_id"].astype(str).tolist()
    image_id = st.selectbox("Choose an image", ids)
    row = frame.loc[frame["image_id"].astype(str) == image_id].iloc[0].to_dict()
    image_path = resolve_image_path(row)
    image_col, edit_col = st.columns([1, 1.25])
    with image_col:
        if image_path:
            st.image(str(image_path), caption=image_id, width="stretch")
        else:
            st.warning("The source image is not available at the recorded path. Set its folder in the sidebar.")
        st.write(f"**Split:** {row.get('split','—')}  ·  **Existing authenticity:** {row.get('existing_authenticity_label','—')}")
        st.caption(str(row.get("prelabel_notes", ""))[:700])
    editable = [
        "reviewed_authenticity", "fitness_outcome",
        "soiled_presence", "soiled_severity", "creased_presence", "creased_severity",
        "torn_presence", "torn_severity", "folded_presence", "folded_severity",
        "adhesive_or_fasteners", "moisture_damage", "defacement", "important_details_obscured",
        "redemption_size_at_least_two_thirds", "redemption_signature_complete",
        "redemption_denomination_complete", "redemption_serial_complete", "reviewer", "prelabel_confidence",
    ]
    editable = [column for column in editable if column in frame.columns]
    with edit_col:
        st.subheader("Review this row")
        updated = st.data_editor(
            pd.DataFrame([row])[editable],
            hide_index=True,
            width="stretch",
            key=f"review_{image_id}",
            column_config={
                "fitness_outcome": st.column_config.SelectboxColumn(
                    "Final outcome",
                    options=["counterfeit", "genuine_fit", "genuine_unfit", "review_required"],
                    required=True,
                ),
                "prelabel_confidence": st.column_config.SelectboxColumn(
                    "Confidence", options=["low", "medium", "high"],
                ),
            },
        )
        if st.button("Save this review", type="primary", key=f"save_{image_id}"):
            for column in editable:
                frame.loc[frame["image_id"].astype(str) == image_id, column] = updated.iloc[0][column]
            st.session_state.review_frame = frame
            st.session_state.review_saved = True
            st.rerun()
    if st.session_state.get("review_saved"):
        st.success("Your changes are saved in this Streamlit session. Download the reviewed CSV to keep them.")
    csv_data = frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download reviewed labels", csv_data, file_name="prototype_fitness_reviewed.csv", mime="text/csv")
    st.caption("Edits stay in this browser session until downloaded. The source CSV and GitHub copy are not changed by the app.")


def page_metrics(metrics: dict, summary: dict):
    show_hero("Experiment results", "Validation-selected prototype results and the held-out proxy evaluation.")
    show_prototype_notice()
    if not metrics:
        st.info("No test evaluation file is available in the current project copy.")
        return
    st.subheader("Three-outcome test evaluation")
    cols = st.columns(4)
    cols[0].metric("Examples evaluated", str(metrics.get("count", "—")))
    cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    cols[2].metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
    cols[3].metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', 0):.1%}")
    st.caption("This one-time test report was not used for checkpoint selection. Targets are weak rule-based prelabels.")
    left, right = st.columns(2)
    with left:
        st.subheader("Per-outcome scores")
        rows = []
        for name, values in metrics.get("classes", {}).items():
            rows.append({
                "Outcome": OUTCOME_DISPLAY.get(name, name),
                "Examples": values.get("support", 0),
                "Precision": values.get("precision", 0),
                "Recall": values.get("recall", 0),
                "F1": values.get("f1", 0),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}"}), hide_index=True, width="stretch")
    with right:
        st.subheader("Confusion matrix")
        matrix = metrics.get("confusion_matrix", [])
        labels = ["Counterfeit", "Genuine fit", "Genuine unfit"]
        if matrix:
            st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), width="stretch")
    st.subheader("Condition severity heads")
    sev = metrics.get("severity_by_condition", {})
    if sev:
        df = pd.DataFrame([
            {"Condition": name.title(), "Macro F1": values.get("macro_f1", 0), "Examples": values.get("count", 0)}
            for name, values in sev.items()
        ])
        st.bar_chart(df.set_index("Condition")["Macro F1"], y_label="Macro F1", height=250)
        st.dataframe(df.style.format({"Macro F1": "{:.3f}"}), hide_index=True, width="stretch")
    st.subheader("Training history")
    history_path = RUN_DIR / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        if {"epoch", "val_accuracy", "val_macro_f1"}.issubset(history.columns):
            st.line_chart(history.set_index("epoch")[["val_accuracy", "val_macro_f1"]], height=260)
            st.caption(f"Selected epoch: {summary.get('best_epoch', '—')}")
    for title, path in (("Evaluation JSON", RESULTS_JSON), ("Predictions CSV", RUN_DIR / "test_predictions.csv"), ("Run history CSV", history_path)):
        if path.is_file():
            st.download_button(f"Download {title}", path.read_bytes(), file_name=path.name, key=f"download_{path.name}")


def page_project():
    show_hero("Project and roadmap", "The rule, model status, and next research steps in one place.")
    show_prototype_notice()
    st.subheader("Prototype decision rule")
    if RULE_MD.is_file():
        st.markdown(RULE_MD.read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Roadmap status")
    if ROADMAP_MD.is_file():
        st.markdown(ROADMAP_MD.read_text(encoding="utf-8"))
    st.download_button("Download roadmap", ROADMAP_MD.read_bytes(), file_name=ROADMAP_MD.name, mime="text/markdown")


def main():
    st.sidebar.markdown("<div class='eyebrow'>Research workspace</div><h2 style='color:white'>UGX Banknote Lab</h2>", unsafe_allow_html=True)
    st.sidebar.caption("Counterfeit detection · note condition · circulation fitness")
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Assess a note", "Review labels", "Metrics", "Project roadmap"],
        label_visibility="collapsed",
    )
    with st.sidebar.expander("Local project settings"):
        default_checkpoint = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else ""
        checkpoint_text = st.text_input("Model checkpoint", value=st.session_state.get("checkpoint_path", default_checkpoint))
        st.session_state.checkpoint_path = checkpoint_text
        dataset_default = os.environ.get("UGX_DATASET_DIR", str(load_dataset_dir() or ""))
        dataset_text = st.text_input("Labelled dataset folder (optional)", value=st.session_state.get("dataset_dir") or dataset_default, help="Set this if the dataset is outside this project copy.")
        st.session_state.dataset_dir = dataset_text
        st.caption("Settings are used locally in this browser session.")
    st.sidebar.divider()
    st.sidebar.caption("Prototype · labels require human review")

    metrics = load_metrics()
    summary = load_summary()
    if page == "Overview":
        page_dashboard(metrics, summary)
    elif page == "Assess a note":
        checkpoint = Path(st.session_state.get("checkpoint_path", str(DEFAULT_CHECKPOINT))).expanduser()
        page_assess(checkpoint)
    elif page == "Review labels":
        page_review()
    elif page == "Metrics":
        page_metrics(metrics, summary)
    else:
        page_project()


if __name__ == "__main__":
    main()
