
"""Exoplanet Hunter: an educational Kepler-signal classification application."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


SEED = 42
TRAIN_FRACTION = 0.8
LEARNING_RATE = 0.1
N_ITER = 2000

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "weights.npz"
DATA_PATH = ROOT / "data" / "kepler_data.csv"

PALETTE = {
    "background": "#F7F5F2",
    "surface": "#EFEBE4",
    "surface_alt": "#E8E2D9",
    "text": "#1F2A2E",
    "muted": "#5C6B73",
    "subtle": "#78858B",
    "border": "#D9D2C8",
    "accent": "#3A6B6B",
    "accent_dark": "#2F5D5D",
    "confirmed": "#4A7C59",
    "warm": "#C4A35A",
    "negative": "#A85C4A",
    "grid": "#D9D2C8",
}

FEATURE_INFO = {
    "koi_period": ("Orbital period", "days", "Time between successive transits."),
    "koi_duration": ("Transit duration", "hours", "Length of time the transit lasts."),
    "koi_depth": ("Transit depth", "ppm", "Decrease in stellar brightness during transit."),
    "koi_prad": ("Planet radius", "Earth radii", "Estimated radius of the candidate planet."),
    "koi_teq": ("Equilibrium temperature", "K", "Estimated equilibrium temperature."),
    "koi_insol": ("Insolation flux", "Earth flux", "Stellar energy received relative to Earth."),
    "koi_model_snr": ("Transit signal-to-noise", "", "Signal-to-noise ratio of the transit model."),
    "koi_steff": ("Stellar temperature", "K", "Effective temperature of the host star."),
    "koi_slogg": ("Stellar surface gravity", "log10(cm/s²)", "Logarithmic surface gravity of the host star."),
    "koi_srad": ("Stellar radius", "solar radii", "Estimated host-star radius."),
    "koi_impact": ("Impact parameter", "", "Projected transit chord relative to stellar radius."),
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probabilities, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def transform(raw: np.ndarray, features: list[str], log_features: list[str]) -> np.ndarray:
    values = np.asarray(raw, dtype=float).copy()
    for name in log_features:
        values[:, features.index(name)] = np.log1p(values[:, features.index(name)])
    return values


def add_bias(values: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((values.shape[0], 1)), values])


def gradient_ascent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    iterations: int = N_ITER,
) -> tuple[np.ndarray, list[float]]:
    weights = np.zeros(x.shape[1], dtype=float)
    losses = []
    for _ in range(iterations):
        probabilities = sigmoid(x @ weights)
        gradient = x.T @ (y - probabilities) / len(y)
        weights += learning_rate * gradient
        losses.append(log_loss(y, probabilities))
    return weights, losses


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def format_number(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def format_feature(name: str) -> str:
    return FEATURE_INFO.get(name, (name.replace("_", " ").title(), "", ""))[0]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
            font-family: "Segoe UI", system-ui, sans-serif !important;
            text-rendering: auto;
            -webkit-font-smoothing: auto;
            -moz-osx-font-smoothing: auto;
        }
        [data-testid="stAppViewContainer"] {
            background: #F7F5F2;
            color: #1F2A2E;
        }
        [data-testid="stHeader"] { background: rgba(247,245,242,0.96); }
        [data-testid="stSidebar"] { background: #EFEBE4; }
        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3 {
            color: #1F2A2E !important;
            letter-spacing: -0.02em;
        }
        h1 {
            font-family: Georgia, "Times New Roman", serif !important;
            font-weight: 500 !important;
            font-size: clamp(2.7rem, 6vw, 4.7rem) !important;
            line-height: 1.05 !important;
        }
        h2 { font-size: clamp(1.65rem, 3vw, 2.25rem) !important; }
        h3 { font-size: 1.12rem !important; }
        p, li, label {
            color: #243033;
            line-height: 1.65;
        }
        [data-testid="stCaptionContainer"], .note, .footer-note, .hero-subtitle, .section-intro {
            color: #4E5C63;
        }
        a { color: #2F5D5D !important; }
        .hero-subtitle {
            max-width: 760px;
            font-size: 1.08rem;
            line-height: 1.7;
            margin-top: .8rem;
        }
        .section-index {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem 1.3rem;
            border-top: 1px solid #C9C1B6;
            border-bottom: 1px solid #C9C1B6;
            padding: .95rem 0;
            margin: 1.8rem 0 2.8rem;
        }
        .section-index a {
            text-decoration: none;
            color: #4E5C63 !important;
            font-size: .88rem;
        }
        .section-index a:hover {
            color: #2F5D5D !important;
            text-decoration: underline;
            text-underline-offset: 4px;
        }
        .section-kicker {
            color: #2F5D5D;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }
        .section-intro {
            max-width: 760px;
            margin: -.2rem 0 1.5rem;
        }
        .rule {
            border: 0;
            border-top: 1px solid #C9C1B6;
            margin: 3.4rem 0 2.5rem;
        }
        .metric {
            border-top: 2px solid #4A7C59;
            padding: .85rem .2rem .4rem;
        }
        .metric-value {
            color: #1F2A2E;
            font-size: 1.8rem;
            font-weight: 650;
            letter-spacing: -.03em;
        }
        .metric-label {
            color: #4E5C63;
            font-size: .76rem;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .note, .footer-note { font-size: .84rem; line-height: 1.6; }
        .result-panel {
            background: #EFEBE4;
            border: 1px solid #C9C1B6;
            border-radius: 4px;
            padding: 1.1rem 1.25rem;
        }
        .result-value {
            color: #2F5D5D;
            font-size: 2rem;
            font-weight: 650;
        }
        .data-reference {
            background: #EFEBE4;
            border-left: 3px solid #3A6B6B;
            color: #1F2A2E;
            font-family: Georgia, "Times New Roman", serif;
            font-size: .92rem;
            line-height: 1.7;
            padding: .85rem 1rem;
        }
        [data-testid="stDataFrame"],
        [data-testid="stDataFrameResizable"],
        [data-testid="stTable"] {
            background: #F7F5F2 !important;
            border: 1.5px solid #2F5D5D !important;
            border-radius: 4px !important;
            overflow: hidden;
        }
        [data-testid="stDataFrame"] [data-testid="glideDataEditor"],
        [data-testid="stDataFrame"] canvas {
            font-family: "Segoe UI", system-ui, sans-serif !important;
        }
        [data-testid="stTable"] table {
            border-collapse: collapse !important;
            width: 100% !important;
            background: #F7F5F2;
        }
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            border: 1px solid #C9C1B6 !important;
            color: #1F2A2E !important;
            padding: 8px 10px !important;
            font-size: 14px !important;
            font-weight: 450;
        }
        [data-testid="stTable"] th {
            background: #E8E2D9 !important;
            color: #1F2A2E !important;
            font-weight: 650 !important;
            border-bottom: 2px solid #2F5D5D !important;
        }
        [data-testid="stMetric"] {
            background: #EFEBE4;
            border: 1px solid #C9C1B6;
            border-radius: 4px;
            padding: .8rem 1rem;
        }
        [data-testid="stMetricValue"] { color: #1F2A2E; }
        [data-testid="stMetricLabel"] { color: #4E5C63; }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid #2F5D5D;
            background: #2F5D5D;
            color: #ffffff;
            border-radius: 3px;
            font-weight: 600;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: #3A6B6B;
            background: #3A6B6B;
            color: #fff;
        }
        input, textarea, [data-baseweb="select"] > div {
            background: #EFEBE4 !important;
            color: #1F2A2E !important;
        }
        @media (max-width: 700px) {
            .block-container { padding: 1.4rem 1rem 3rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section(number: str, title: str, anchor: str, intro: str = "") -> None:
    st.markdown(f'<div id="{anchor}"></div><div class="section-kicker">{number}</div>', unsafe_allow_html=True)
    st.header(title)
    if intro:
        st.markdown(f'<p class="section-intro">{intro}</p>', unsafe_allow_html=True)


def metric_block(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-label">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def make_transit_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(PALETTE["background"])
    for ax in axes:
        ax.set_facecolor(PALETTE["background"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(PALETTE["border"])
    star = Circle((0.48, 0.52), 0.28, facecolor="#d8c6a7", edgecolor="#eee5d3", lw=1.2)
    planet = Circle((0.77, 0.52), 0.075, facecolor="#789987", edgecolor="#b7cdbd", lw=1)
    axes[0].add_patch(star)
    axes[0].add_patch(planet)
    axes[0].annotate("star", (0.48, 0.2), color=PALETTE["muted"], ha="center", fontsize=9)
    axes[0].annotate("planet", (0.77, 0.35), color=PALETTE["accent"], ha="center", fontsize=9)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Transit geometry", color=PALETTE["text"], loc="left", fontsize=10)

    time = np.linspace(-2, 2, 300)
    flux = np.ones_like(time)
    transit = (np.abs(time) < 0.55)
    flux[transit] -= 0.012
    edge = (np.abs(time) < 0.68) & ~transit
    flux[edge] -= 0.012 * (0.68 - np.abs(time[edge])) / 0.13
    axes[1].plot(time, flux, color=PALETTE["accent"], lw=2)
    axes[1].axhline(1, color=PALETTE["border"], lw=1, ls="--")
    axes[1].set_title("Illustrative stellar brightness", color=PALETTE["text"], loc="left", fontsize=10)
    axes[1].set_xlabel("Time relative to transit", color=PALETTE["muted"], fontsize=8)
    axes[1].set_ylabel("Relative brightness", color=PALETTE["muted"], fontsize=8)
    axes[1].grid(color=PALETTE["grid"], alpha=.45, lw=.6)
    fig.tight_layout()
    return fig


def plot_confusion(metrics: dict) -> plt.Figure:
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.imshow(
        matrix,
        cmap=LinearSegmentedColormap.from_list("sage", [PALETTE["surface"], PALETTE["accent"]]),
        aspect="auto",
    )
    ax.set_xticks([0, 1], ["Predicted\nfalse positive", "Predicted\nconfirmed"], color=PALETTE["muted"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual\nfalse positive", "Actual\nconfirmed"], color=PALETTE["muted"], fontsize=8)
    ax.set_xlabel("Model prediction", color=PALETTE["muted"])
    ax.set_ylabel("Actual class", color=PALETTE["muted"])
    ax.tick_params(length=0)
    for (row, col), value in np.ndenumerate(matrix):
        row_total = matrix[row].sum()
        share = value / row_total if row_total else 0
        ax.text(col, row, f"{value:,}\n{share:.0%} of row", ha="center", va="center",
                color=PALETTE["text"], fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    fig.tight_layout()
    return fig


def plot_probability_histogram(probabilities: np.ndarray, labels: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    bins = np.linspace(0, 1, 21)
    ax.hist(probabilities[labels == 0], bins=bins, alpha=.78, color=PALETTE["negative"], label="False positive")
    ax.hist(probabilities[labels == 1], bins=bins, alpha=.78, color=PALETTE["confirmed"], label="Confirmed")
    ax.axvline(.5, color=PALETTE["warm"], lw=1.2, ls="--", label="0.50 threshold")
    ax.set_xlabel("Predicted probability of confirmed", color=PALETTE["muted"])
    ax.set_ylabel("Number of signals", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.grid(axis="y", color=PALETTE["grid"], alpha=.4)
    ax.legend(frameon=False, labelcolor=PALETTE["muted"], fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    fig.tight_layout()
    return fig


def plot_weights(weights: np.ndarray, features: list[str]) -> plt.Figure:
    values = weights[1:]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    colors = [PALETTE["confirmed"] if values[i] >= 0 else PALETTE["negative"] for i in order]
    ax.barh([format_feature(features[i]) for i in order], values[order], color=colors, height=.62)
    ax.axvline(0, color=PALETTE["muted"], lw=.8)
    ax.set_xlabel("Learned coefficient (scaled feature)", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.grid(axis="x", color=PALETTE["grid"], alpha=.35)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def load_project(data_stamp: float, model_stamp: float) -> dict:
    with np.load(MODEL_PATH, allow_pickle=False) as archive:
        weights = archive["w"].astype(float)
        mean = archive["mean"].astype(float)
        std = archive["std"].astype(float)
        features = archive["features"].astype(str).tolist()
        log_features = archive["log_features"].astype(str).tolist()
        defaults = archive["defaults"].astype(float)

    frame = pd.read_csv(DATA_PATH, comment="#")
    required = set(features + ["koi_disposition"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if len(weights) != len(features) + 1:
        raise ValueError("Saved weight count does not match the saved feature list.")
    if len(mean) != len(features) or len(std) != len(features) or len(defaults) != len(features):
        raise ValueError("Saved scaling/default arrays do not match the feature count.")
    if np.any(std == 0):
        raise ValueError("Saved standard deviations contain zero values.")
    if any(name not in features for name in log_features):
        raise ValueError("Saved log-transformation list contains an unknown feature.")

    dispositions = frame["koi_disposition"].value_counts(dropna=False).to_dict()
    labelled = frame[frame["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    labelled["label"] = (labelled["koi_disposition"] == "CONFIRMED").astype(int)
    clean = labelled[features + ["label"]].dropna()
    if clean.empty:
        raise ValueError("No labelled rows remain after filtering missing feature values.")

    raw = clean[features].to_numpy(dtype=float)
    labels = clean["label"].to_numpy(dtype=int)
    permutation = np.random.RandomState(SEED).permutation(len(raw))
    split_at = int(TRAIN_FRACTION * len(permutation))
    train_idx, test_idx = permutation[:split_at], permutation[split_at:]
    x_train_raw, x_test_raw = raw[train_idx], raw[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    x_train_transformed = transform(x_train_raw, features, log_features)
    x_test_transformed = transform(x_test_raw, features, log_features)
    computed_mean = x_train_transformed.mean(axis=0)
    computed_std = x_train_transformed.std(axis=0)
    computed_std[computed_std == 0] = 1.0

    scaling_match = bool(
        np.allclose(computed_mean, mean, rtol=1e-7, atol=1e-9)
        and np.allclose(computed_std, std, rtol=1e-7, atol=1e-9)
    )

    x_train = add_bias((x_train_transformed - mean) / std)
    x_test = add_bias((x_test_transformed - mean) / std)
    train_probabilities = sigmoid(x_train @ weights)
    test_probabilities = sigmoid(x_test @ weights)
    train_metrics = classification_metrics(y_train, (train_probabilities >= .5).astype(int))
    test_metrics = classification_metrics(y_test, (test_probabilities >= .5).astype(int))

    reproduced_weights, losses = gradient_ascent(x_train, y_train)
    weight_match = bool(np.allclose(reproduced_weights, weights, rtol=1e-5, atol=1e-7))

    candidate_frame = frame[frame["koi_disposition"] == "CANDIDATE"].copy()
    candidate_frame = candidate_frame.dropna(subset=features)
    if not candidate_frame.empty:
        candidate_raw = candidate_frame[features].to_numpy(dtype=float)
        candidate_transformed = transform(candidate_raw, features, log_features)
        candidate_x = add_bias((candidate_transformed - mean) / std)
        candidate_frame["model_probability"] = sigmoid(candidate_x @ weights)
        name_columns = [c for c in ("kepoi_name", "kepler_name") if c in candidate_frame.columns]
        candidate_frame["display_name"] = (
            candidate_frame[name_columns[0]].fillna("Unnamed candidate").astype(str)
            if name_columns
            else [f"Candidate {i + 1}" for i in range(len(candidate_frame))]
        )
        candidate_frame = candidate_frame.sort_values("model_probability", ascending=False)

    name_columns = [c for c in ("kepoi_name", "kepler_name") if c in clean.columns]
    preview_columns = name_columns[:1] + ["label"] + features[:4]
    preview = clean[preview_columns].head(10).copy()

    return {
        "frame": frame,
        "dispositions": dispositions,
        "features": features,
        "log_features": log_features,
        "weights": weights,
        "mean": mean,
        "std": std,
        "defaults": defaults,
        "clean": clean,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline": max(float(y_test.mean()), 1 - float(y_test.mean())),
        "losses": losses,
        "scaling_match": scaling_match,
        "weight_match": weight_match,
        "candidates": candidate_frame,
        "preview": preview,
        "data_rows": len(frame),
        "clean_rows": len(clean),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "data_stamp": data_stamp,
        "model_stamp": model_stamp,
    }


st.set_page_config(page_title="Exoplanet Hunter", page_icon="✦", layout="wide")
inject_styles()

st.title("Exoplanet Hunter")
st.markdown(
    """<p class="hero-subtitle" style="font-size: 1.25rem; line-height: 1.7; font-weight: 400; color: #222222; margin-top: 1.5rem; margin-bottom: 0.5rem;">
    Can a measured dip in a star's brightness be distinguished from a false alarm? This project uses Kepler Objects of Interest data and a logistic regression model implemented from scratch with NumPy to explore that question.
    </p>""",
    unsafe_allow_html=True
)
st.markdown(
    """
    <nav class="section-index" aria-label="Page sections">
      <a href="#overview">Overview</a><a href="#data">Data</a><a href="#model">Model</a>
      <a href="#results">Results</a><a href="#try-it">Try It</a><a href="#candidates">Candidates</a>
    </nav>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(f"Saved model file not found: `{MODEL_PATH.relative_to(ROOT)}`")
    st.stop()
if not DATA_PATH.is_file():
    st.error(f"Dataset file not found: `{DATA_PATH.relative_to(ROOT)}`")
    st.stop()

try:
    data = load_project(DATA_PATH.stat().st_mtime, MODEL_PATH.stat().st_mtime)
except Exception as error:
    st.error(f"The project files could not be loaded: {error}")
    st.stop()

features = data["features"]
weights = data["weights"]
test_metrics = data["test_metrics"]

section("01", "The scientific question", "overview",
        "A transit occurs when a planet passes in front of its host star, producing a small, measurable dip in brightness.")
left, right = st.columns([1.05, 1], gap="large")
with left:
    st.markdown(
        "Kepler monitored the brightness of stars over time. A repeating dip can be evidence of an orbiting planet, "
        "but other astrophysical and instrumental effects can produce signals that resemble transits. "
        "The task here is a binary classification exercise: distinguish previously confirmed objects from signals "
        "catalogued as false positives."
    )
    st.markdown(
        '<p class="note">The diagram is conceptual. It illustrates the geometry of a transit and its corresponding '
        'brightness dip; it is not an observed light curve from an individual Kepler target.</p>',
        unsafe_allow_html=True,
    )
with right:
    st.pyplot(make_transit_figure(), use_container_width=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

section("02", "The evidence", "data",
        "The application reads the local Kepler Objects of Interest table and uses the same labelled observations as the notebook.")
frame = data["frame"]
disp = data["dispositions"]
total_col1, total_col2, total_col3 = st.columns(3)
with total_col1:
    metric_block("Rows in source file", f"{data['data_rows']:,}")
with total_col2:
    metric_block("Rows retained for modelling", f"{data['clean_rows']:,}")
with total_col3:
    metric_block("Input features", str(len(features)))
st.write("")
class_counts = pd.Series({
    "Confirmed": int((data["clean"]["label"] == 1).sum()),
    "False positive": int((data["clean"]["label"] == 0).sum()),
})
fig, ax = plt.subplots(figsize=(7, 2.1))
fig.patch.set_facecolor(PALETTE["background"])
ax.set_facecolor(PALETTE["background"])
left_pos = 0
for label, color in [("Confirmed", PALETTE["confirmed"]), ("False positive", PALETTE["negative"])]:
    value = int(class_counts[label])
    ax.barh([0], [value], left=left_pos, color=color, height=.38, label=f"{label}: {value:,}")
    if value:
        ax.text(left_pos + value / 2, 0, f"{value:,}", ha="center", va="center", color="#ffffff", fontweight="bold")
    left_pos += value
ax.set_yticks([])
ax.set_xlabel("Rows retained after filtering and dropping missing feature values", color=PALETTE["muted"], fontsize=8)
ax.tick_params(axis="x", colors=PALETTE["muted"], labelsize=8)
ax.legend(frameon=False, labelcolor=PALETTE["muted"], ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.35))
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("Source disposition counts"):
    disposition_table = pd.DataFrame(
        [{"Disposition": str(key), "Rows": int(value)} for key, value in disp.items()]
    ).sort_values("Rows", ascending=False)
    st.dataframe(disposition_table, hide_index=True, use_container_width=True)

st.subheader("Features used by the model")
feature_table = pd.DataFrame([
    {
        "Feature": name,
        "Description": FEATURE_INFO.get(name, (name, "", ""))[0],
        "Unit": FEATURE_INFO.get(name, ("", "", ""))[1],
        "Log-transformed": "Yes" if name in data["log_features"] else "No",
    }
    for name in features
])
st.dataframe(feature_table, hide_index=True, use_container_width=True)
st.caption("Rows with missing values in any selected feature are excluded. Candidate objects are not used to train or evaluate the classifier.")
with st.expander("Preview of retained labelled observations"):
    st.dataframe(data["preview"], hide_index=True, use_container_width=True)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

section("03", "The model", "model",
        "A binary logistic regression classifier maps the selected measurements to an estimated probability between zero and one.")
st.markdown(
    "The notebook implements the model using NumPy. Five skewed measurements are transformed with `log1p`; "
    "all features are then standardised using the training-set mean and standard deviation. An intercept is added "
    "before fitting the weights."
)
st.latex(r"p(y=1\mid x)=\sigma(w_0+w_1x_1+\cdots+w_nx_n)")
st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
st.markdown(
    "The weights are learned by gradient ascent on the average log-likelihood. The implementation uses a learning "
    "rate of 0.1 for 2,000 iterations. The saved parameters are loaded from `models/weights.npz`; the application "
    "does not fit a different model for its predictions."
)
st.subheader("Learned coefficients")
st.pyplot(plot_weights(weights, features), use_container_width=True)
with st.expander("Model file consistency checks"):
    st.write(f"Saved feature order matches the application feature list: **{features == data['features']}**")
    st.write(f"Saved scaling statistics match the reproduced training split: **{data['scaling_match']}**")
    st.write(f"Recomputed NumPy training weights match the saved weights: **{data['weight_match']}**")
    st.caption("The training-weight comparison uses the notebook's stated seed, split, learning rate and iteration count.")
st.markdown('<hr class="rule">', unsafe_allow_html=True)

section("04", "Evaluation", "results",
        "Performance is calculated on the held-out test partition using the saved model and a classification threshold of 0.50.")
metric_cols = st.columns(4)
for col, label, key in zip(
    metric_cols,
    ["Accuracy", "Precision", "Recall", "F1-score"],
    ["accuracy", "precision", "recall", "f1"],
):
    with col:
        metric_block(label, f"{test_metrics[key] * 100:.1f}%")
st.write("")
left, right = st.columns([.95, 1.05], gap="large")
with left:
    st.subheader("Confusion matrix")
    st.pyplot(plot_confusion(test_metrics), use_container_width=True)
    st.caption("Rows represent actual classes; columns represent model predictions.")
with right:
    st.subheader("Predicted probabilities")
    st.pyplot(
        plot_probability_histogram(data["test_probabilities"], data["y_test"]),
        use_container_width=True,
    )
    st.caption("The dashed line marks the 0.50 decision threshold.")
st.subheader("Training and test performance")
training_metrics = data["train_metrics"]
comparison = pd.DataFrame([
    {"Metric": "Accuracy", "Training": training_metrics["accuracy"], "Test": test_metrics["accuracy"]},
    {"Metric": "Precision", "Training": training_metrics["precision"], "Test": test_metrics["precision"]},
    {"Metric": "Recall", "Training": training_metrics["recall"], "Test": test_metrics["recall"]},
    {"Metric": "F1-score", "Training": training_metrics["f1"], "Test": test_metrics["f1"]},
])
st.dataframe(comparison.style.format({"Training": "{:.3f}", "Test": "{:.3f}"}), hide_index=True, use_container_width=True)
st.markdown(
    f'<p class="note">Majority-class baseline accuracy on this test split: '
    f'<strong>{data["baseline"]:.3f}</strong>. This baseline predicts the more frequent test class for every row.</p>',
    unsafe_allow_html=True,
)
with st.expander("Training loss"):
    fig, ax = plt.subplots(figsize=(7, 2.7))
    fig.patch.set_facecolor(PALETTE["background"])
    ax.set_facecolor(PALETTE["background"])
    ax.plot(data["losses"], color=PALETTE["accent"], lw=1.8)
    ax.set_xlabel("Iteration", color=PALETTE["muted"])
    ax.set_ylabel("Training log-loss", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.grid(color=PALETTE["grid"], alpha=.35)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
st.markdown(
    '<p class="note">A false negative is a confirmed object classified as a false positive. '
    'A false positive is a false-positive object classified as confirmed. Precision and recall describe '
    'different error trade-offs and should be interpreted together.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

section("05", "Try a signal", "try-it",
        "Adjust the measurements to see how the trained classifier responds. The output is a model estimate, not an astronomical confirmation.")
st.markdown("The initial values are the medians of the training observations saved by the notebook.")
defaults = data["defaults"]
input_values = {}
columns = st.columns(3)
for index, feature in enumerate(features):
    info = FEATURE_INFO.get(feature, (feature, "", ""))
    with columns[index % 3]:
        default = float(defaults[index])
        if not np.isfinite(default):
            default = 0.0
        input_values[feature] = st.number_input(
            f"{info[0]}{f' ({info[1]})' if info[1] else ''}",
            value=default,
            format="%.5f",
            help=info[2],
            key=f"input_{feature}",
        )
if st.button("Estimate signal probability", type="primary"):
    raw_input = np.array([[input_values[name] for name in features]], dtype=float)
    if np.any(raw_input < 0):
        st.error("These selected measurements must be non-negative for the notebook's log1p transformation.")
    else:
        transformed = transform(raw_input, features, data["log_features"])
        scaled = add_bias((transformed - data["mean"]) / data["std"])
        probability = float(sigmoid(scaled @ weights)[0])
        prediction = "More likely classified as confirmed" if probability >= .5 else "More likely classified as false positive"
        st.markdown(
            f'<div class="result-panel"><div class="metric-label">Estimated probability of confirmed</div>'
            f'<div class="result-value">{probability:.1%}</div><p>{html.escape(prediction)}</p></div>',
            unsafe_allow_html=True,
        )
        st.caption("This probability reflects the fitted model and its training data. It is not a discovery claim or a substitute for scientific vetting.")
st.markdown('<hr class="rule">', unsafe_allow_html=True)

section("06", "Explore unclassified candidates", "candidates",
        "The model scores rows labelled CANDIDATE in the source table. These scores are exploratory and do not change the catalogue dispositions.")
candidates = data["candidates"]
if candidates.empty:
    st.info("No candidate rows with complete values for all selected model features were found in the dataset.")
else:
    limit = st.slider("Number of highest-scoring candidates to display", min_value=5, max_value=50, value=15, step=5)
    display_columns = ["display_name", "model_probability"] + features[:4]
    candidate_view = candidates[display_columns].head(limit).copy()
    candidate_view = candidate_view.rename(columns={
        "display_name": "KOI identifier",
        "model_probability": "Model probability",
        **{name: format_feature(name) for name in features[:4]},
    })
    candidate_view["Model probability"] = candidate_view["Model probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(candidate_view, hide_index=True, use_container_width=True)
    st.caption("Candidates are ranked by the model's estimated probability of the confirmed class. A high score does not mean the object has been confirmed.")

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown('<p class="data-reference">Data Source: NASA Kepler Exoplanet Search Results, obtained through Kaggle. Original data: <a href="https://exoplanetarchive.ipac.caltech.edu/">NASA Exoplanet Archive</a>.<br><br>This project is an educational Machine Learning demonstration ;its predictions are not scientific confirmation of exoplanets.</p>', unsafe_allow_html=True)

