"""
dashboard.py — Streamlit DDoS Detection Dashboard
Run: streamlit run dashboard.py
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DDoS Neural Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme / CSS ────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Outfit:wght@300;400;600;700;900&display=swap');

:root {
    --bg:        #080c14;
    --surface:   #0d1520;
    --surface2:  #111d2e;
    --border:    #1a2d45;
    --cyan:      #00e5ff;
    --pink:      #ff2d78;
    --green:     #00ff88;
    --yellow:    #ffe600;
    --text:      #cdd9e8;
    --muted:     #4a6080;
    --font-mono: 'Share Tech Mono', monospace;
    --font-main: 'Outfit', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-main);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-main) !important; }

/* Main header */
.hero {
    background: linear-gradient(135deg, #080c14 0%, #0a1929 50%, #080c14 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 80% 50%, rgba(0,229,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 20% 50%, rgba(255,45,120,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(90deg, var(--cyan) 0%, #7b8fff 50%, var(--pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    font-family: var(--font-main);
}
.hero-sub {
    font-family: var(--font-mono);
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-grid { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--cyan); }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cyan::after  { background: var(--cyan); }
.metric-card.pink::after  { background: var(--pink); }
.metric-card.green::after { background: var(--green); }
.metric-card.yellow::after{ background: var(--yellow); }
.metric-card.purple::after{ background: #9d7fff; }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.metric-sub { font-size: 0.72rem; color: var(--muted); margin-top: 4px; }

/* Section headers */
.section-header {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--cyan);
    margin: 28px 0 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.section-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 8px var(--cyan);
}

/* Demo result card */
.result-card {
    background: var(--surface);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid var(--border);
    margin-top: 16px;
}
.result-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.result-value { font-size: 2rem; font-weight: 700; }
.result-benign { color: var(--green); }
.result-syn    { color: var(--pink); }
.result-udp    { color: var(--cyan); }

/* Badge */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid currentColor;
}

/* Plotly chart containers */
.chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 4px;
    margin-bottom: 20px;
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #ff2d7822) !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    border-radius: 8px !important;
    transition: all .2s !important;
    padding: 0.4rem 1.4rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff44, #ff2d7844) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.3) !important;
}
.stSelectbox > div > div, .stSlider > div {
    background: var(--surface2) !important;
}
[data-testid="stMarkdownContainer"] p { color: var(--text); }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Plotly theme ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Share Tech Mono, monospace", color="#cdd9e8", size=11),
    xaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
    yaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2d45", borderwidth=1),
)

COLORS = {
    "Random Forest": "#00e5ff",
    "KNN": "#ff2d78",
    "Extra Trees": "#00ff88",
    "MLP": "#ffe600",
    "XGBoost": "#9d7fff",
}

LABEL_COLORS = {"Benign": "#00ff88", "Syn": "#ff2d78", "UDP": "#00e5ff"}

# ─── Data / model cache ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def generate_data(n_samples=6000):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_classes=3,
        n_informative=20,
        n_redundant=5,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te, sc


@st.cache_resource(show_spinner=False)
def train_models(cv_folds=3):
    X_tr, X_te, y_tr, y_te, sc = generate_data()
    label_names = ["Benign", "Syn", "UDP"]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="mlogloss", verbosity=0
        ),
    }

    rows, trained = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te)
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        auc = roc_auc_score(y_te, proba, multi_class="ovr")
        cv = float(np.mean(cross_val_score(model, X_tr, y_tr, cv=cv_folds, n_jobs=-1)))
        rows.append(
            dict(
                Model=name,
                Accuracy=acc,
                Precision=prec,
                Recall=rec,
                F1=f1,
                ROC_AUC=auc,
                CV=cv,
            )
        )
        trained[name] = model

    df = pd.DataFrame(rows).set_index("Model")
    return df, trained, X_te, y_te, label_names, sc


# ─── Header ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
    <div class="hero-title">DDoSED</div>
    <div class="hero-sub">CICDDoS2019 · SYN / UDP Attack Detection · ML Evaluation Suite</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Config")
    cv_folds = st.slider("CV Folds", 2, 10, 3)
    selected_models = st.multiselect(
        "Models to compare",
        ["Random Forest", "KNN", "Extra Trees", "MLP", "XGBoost"],
        default=["Random Forest", "KNN", "Extra Trees", "MLP", "XGBoost"],
    )
    st.markdown("---")
    st.markdown("### 🎯 Demo Model")
    demo_model_name = st.selectbox(
        "Classifier", ["Random Forest", "KNN", "Extra Trees", "MLP", "XGBoost"]
    )

    st.markdown("---")
    st.markdown(
        """
    <div style='font-family:var(--font-mono);font-size:0.6rem;color:var(--muted);letter-spacing:1px'>
    DATASET<br>CICDDoS2019<br>CLASSES: Benign · Syn · UDP<br>SYNTHETIC DEMO MODE
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─── Load ────────────────────────────────────────────────────────────────────

with st.spinner("🔬 Training models..."):
    scores_df, trained, X_te, y_te, label_names, scaler = train_models(cv_folds)

scores_filtered = scores_df.loc[[m for m in selected_models if m in scores_df.index]]
best_model = scores_filtered["Accuracy"].idxmax()
best_acc = scores_filtered["Accuracy"].max()

# ─── KPI cards ───────────────────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>OVERVIEW</div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "Best Accuracy", f"{best_acc:.2%}", best_model, "cyan"),
    (
        c2,
        "Best ROC AUC",
        f"{scores_filtered['ROC_AUC'].max():.4f}",
        "weighted OvR",
        "pink",
    ),
    (
        c3,
        "Best F1 Score",
        f"{scores_filtered['F1'].max():.4f}",
        "weighted avg",
        "green",
    ),
    (c4, "Models Compared", str(len(scores_filtered)), "classifiers", "yellow"),
    (c5, "Classes", "3", "Benign · Syn · UDP", "purple"),
]
for col, label, val, sub, color in kpis:
    col.markdown(
        f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""",
        unsafe_allow_html=True,
    )

# ─── Scores table ────────────────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>SCORES TABLE</div>""",
    unsafe_allow_html=True,
)

display_df = scores_filtered.copy()
display_df.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC", "CV Score"]
st.dataframe(
    display_df.style.format("{:.4f}")
    .background_gradient(cmap="YlGnBu", axis=0)
    .highlight_max(color="#00ff8840", axis=0),
    use_container_width=True,
)

# ─── Charts row 1: Metrics comparison + CV Scores ────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>METRICS COMPARISON</div>""",
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([3, 2])

with col_left:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    fig = go.Figure()
    metric_colors = ["#00e5ff", "#ff2d78", "#00ff88", "#ffe600", "#9d7fff"]
    for m, color in zip(metrics, metric_colors):
        fig.add_trace(
            go.Bar(
                name=m.replace("_", " "),
                x=scores_filtered.index,
                y=scores_filtered[m],
                marker_color=color,
                marker_line_color="rgba(0,0,0,0)",
                opacity=0.85,
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="All Metrics by Model",
        yaxis_range=[scores_filtered[metrics].min().min() - 0.02, 1.01],
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    cv_vals = scores_filtered["CV"]
    fig2 = go.Figure(
        go.Bar(
            x=cv_vals.index,
            y=cv_vals.values,
            marker=dict(
                color=cv_vals.values,
                colorscale=[[0, "#ff2d78"], [0.5, "#9d7fff"], [1, "#00e5ff"]],
                showscale=False,
                line_color="rgba(0,0,0,0)",
            ),
            text=[f"{v:.4f}" for v in cv_vals.values],
            textposition="outside",
            textfont=dict(size=11, color="#cdd9e8"),
        )
    )
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Cross-Validation Score ({cv_folds}-fold)",
        yaxis_range=[cv_vals.min() - 0.02, 1.02],
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─── ROC Curves ──────────────────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>ROC CURVES</div>""",
    unsafe_allow_html=True,
)

fig_roc = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[f"Class: {n}" for n in label_names],
    shared_yaxes=True,
)
for class_idx, class_name in enumerate(label_names):
    for name in selected_models:
        if name not in trained:
            continue
        model = trained[name]
        proba = model.predict_proba(X_te)[:, class_idx]
        fpr, tpr, _ = roc_curve((y_te == class_idx).astype(int), proba)
        auc = roc_auc_score((y_te == class_idx).astype(int), proba)
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name}",
                line=dict(color=COLORS.get(name, "#fff"), width=2),
                legendgroup=name,
                showlegend=(class_idx == 0),
                hovertemplate=f"{name}<br>AUC={auc:.3f}<extra></extra>",
            ),
            row=1,
            col=class_idx + 1,
        )
    # diagonal
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#4a6080", dash="dash", width=1),
            showlegend=False,
        ),
        row=1,
        col=class_idx + 1,
    )

fig_roc.update_layout(
    **PLOTLY_LAYOUT,
    height=340,
    title="ROC Curves — One-vs-Rest per Class",
)
for i in range(1, 4):
    fig_roc.update_xaxes(
        title_text="FPR", row=1, col=i, gridcolor="#1a2d45", zerolinecolor="#1a2d45"
    )
    fig_roc.update_yaxes(
        title_text="TPR" if i == 1 else "",
        row=1,
        col=i,
        gridcolor="#1a2d45",
        zerolinecolor="#1a2d45",
    )
st.plotly_chart(fig_roc, use_container_width=True)

# ─── Confusion Matrices ───────────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>CONFUSION MATRICES</div>""",
    unsafe_allow_html=True,
)

visible = [m for m in selected_models if m in trained]
ncols = min(3, len(visible))
rows_n = (len(visible) + ncols - 1) // ncols

fig_cm = make_subplots(
    rows=rows_n,
    cols=ncols,
    subplot_titles=visible,
    horizontal_spacing=0.08,
    vertical_spacing=0.15,
)
for idx, name in enumerate(visible):
    r, c = divmod(idx, ncols)
    model = trained[name]
    cm = confusion_matrix(y_te, model.predict(X_te), normalize="true")
    fig_cm.add_trace(
        go.Heatmap(
            z=cm,
            x=label_names,
            y=label_names,
            colorscale=[[0, "#080c14"], [0.5, "#004080"], [1, "#00e5ff"]],
            showscale=False,
            text=[[f"{v:.2f}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=13, color="#fff"),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Rate: %{z:.3f}<extra></extra>",
        ),
        row=r + 1,
        col=c + 1,
    )

fig_cm.update_layout(
    **PLOTLY_LAYOUT,
    height=310 * rows_n,
    title="Normalised Confusion Matrices",
)
st.plotly_chart(fig_cm, use_container_width=True)

# ─── Per-class breakdown radar ────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>PER-CLASS F1 BREAKDOWN</div>""",
    unsafe_allow_html=True,
)

from sklearn.metrics import f1_score as f1_s

fig_radar = go.Figure()
theta = label_names + [label_names[0]]

for name in selected_models:
    if name not in trained:
        continue
    y_pred = trained[name].predict(X_te)
    per_class_f1 = f1_s(y_te, y_pred, average=None, zero_division=0).tolist()
    r_vals = per_class_f1 + [per_class_f1[0]]
    fig_radar.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=theta,
            fill="toself",
            name=name,
            line=dict(color=COLORS.get(name, "#fff"), width=2),
            fillcolor=COLORS.get(name, "#fff")
            .replace("#", "rgba(")
            .replace("ff", "ff,0.08)")
            if False
            else "rgba(0,0,0,0)",
            opacity=0.85,
        )
    )

fig_radar.update_layout(
    **PLOTLY_LAYOUT,
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True, range=[0, 1], gridcolor="#1a2d45", color="#4a6080"
        ),
        angularaxis=dict(gridcolor="#1a2d45", color="#cdd9e8"),
    ),
    title="F1 Score by Class (Radar)",
    height=360,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─── Live Demo ───────────────────────────────────────────────────────────────

st.markdown(
    """<div class="section-header"><div class="section-dot"></div>LIVE DEMO — CLASSIFY A SAMPLE</div>""",
    unsafe_allow_html=True,
)

demo_col, result_col = st.columns([1, 1])

with demo_col:
    st.markdown("**Pick a real test sample or randomise:**")

    sample_idx = st.slider("Sample index", 0, len(X_te) - 1, 0)
    use_random = st.button("🎲  Randomise sample")

    if use_random:
        sample_idx = int(np.random.randint(0, len(X_te)))

    sample = X_te[sample_idx]
    true_label = label_names[y_te[sample_idx]]

    # show feature mini-table (first 15 features)
    feat_df = pd.DataFrame(
        {
            "Feature": [f"feat_{i:02d}" for i in range(15)],
            "Value": [f"{v:.4f}" for v in sample[:15]],
        }
    )
    st.dataframe(feat_df, use_container_width=True, height=200)
    st.caption(f"Showing 15 of {len(sample)} features · sample #{sample_idx}")

with result_col:
    model_demo = trained[demo_model_name]
    proba_demo = model_demo.predict_proba(sample.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba_demo))
    pred_label = label_names[pred_idx]
    is_correct = pred_label == true_label

    color_map = {"Benign": "result-benign", "Syn": "result-syn", "UDP": "result-udp"}
    pred_class = color_map.get(pred_label, "")
    true_class = color_map.get(true_label, "")

    badge = "✅ CORRECT" if is_correct else "❌ WRONG"
    badge_color = "#00ff88" if is_correct else "#ff2d78"

    st.markdown(
        f"""
    <div class="result-card">
        <div class="result-label">PREDICTION — {demo_model_name}</div>
        <div class="result-value {pred_class}">{pred_label}</div>
        <div style="margin-top:10px">
            <span class="result-label">GROUND TRUTH &nbsp;</span>
            <span class="result-value {true_class}" style="font-size:1.2rem">{true_label}</span>
        </div>
        <div style="margin-top:14px;font-family:var(--font-mono);font-size:0.7rem;color:{badge_color}">{badge}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown("<br>", unsafe_allow_html=True)
    fig_prob = go.Figure()
    bar_colors = [LABEL_COLORS.get(ln, "#fff") for ln in label_names]
    fig_prob.add_trace(
        go.Bar(
            x=label_names,
            y=proba_demo,
            marker_color=bar_colors,
            text=[f"{p:.2%}" for p in proba_demo],
            textposition="outside",
            textfont=dict(size=12),
        )
    )
    fig_prob.update_layout(
        **PLOTLY_LAYOUT,
        title="Class Probabilities",
        yaxis_range=[0, 1.1],
        height=260,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown(
    """
<div style="
    margin-top: 48px;
    border-top: 1px solid #1a2d45;
    padding-top: 16px;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    color: #2a4060;
    text-transform: uppercase;
">
DDoS Neural Sentinel · CICDDoS2019 · SYN / UDP Classification · Synthetic Demo Mode
</div>
""",
    unsafe_allow_html=True,
)
