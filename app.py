# app.py
from __future__ import annotations
import io, typing as T
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import plotly.express as px
import plotly.graph_objects as go

# =========================
# UI + Sidebar Controls
# =========================
st.set_page_config(page_title="Void-Learning (Streamlit)", layout="wide")
st.title("Void-Learning Pipeline — Streamlit (with hashed-state Q-learning)")

with st.sidebar:
    st.header("Dataset")
    ds_choice = st.radio("Choose", ["Iris (built-in)", "Titanic (seaborn)", "Upload CSV"])
    st.caption("Upload: a single CSV or two files named with 'train' / 'test' anywhere in the filename.")
    st.markdown("---")

    st.header("Preprocessing")
    n_bins = st.slider("Numeric bin count", 3, 12, 6, 1)
    low_card_as_cat = st.checkbox("Treat low-cardinality numeric as categorical", True)
    low_card_threshold = st.slider("Low-cardinality threshold", 5, 100, 20, 1)
    st.markdown("---")

    st.header("Void-Learning (γ=0 Q-learning)")
    episodes = st.slider("Episodes", 5, 200, 60, 5)
    eps0 = st.slider("ε₀ (start exploration)", 0.01, 0.9, 0.30, 0.01)
    eps_min = st.slider("εmin", 0.0, 0.5, 0.02, 0.01)
    eps_decay = st.slider("ε decay per episode", 0.80, 0.999, 0.97, 0.001)
    optimistic_init = st.slider("Optimistic init (Q₀)", 0.0, 1.0, 0.10, 0.01)
    min_visits = st.slider("Min visits per state", 1, 200, 8, 1)
    margin_tau = st.slider("Margin threshold τ", 0.01, 1.0, 0.15, 0.01)
    hash_mod = st.select_slider("Hash modulus (state table size)", options=[50_000, 100_000, 200_000, 500_000, 1_000_000], value=200_000)
    st.markdown("---")

    st.header("Pruning")
    prune_thresh = st.slider("Mean token void rate ≥", 0.1, 1.0, 0.50, 0.05)
    st.markdown("---")

# =========================
# Helpers: Data Loading
# =========================
def load_builtin_iris() -> pd.DataFrame:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    return iris.frame.copy()  # includes 'target'

def load_builtin_titanic() -> pd.DataFrame | None:
    try:
        import seaborn as sns
        return sns.load_dataset("titanic")
    except Exception:
        return None

def load_uploads() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Returns (df_train, df_test, info) where df_test can be None."""
    up_files = st.file_uploader("Upload one CSV (auto split) OR 'train' + 'test' CSVs", type=["csv"], accept_multiple_files=True)
    if not up_files:
        return None, None, {}
    csv_names = [f.name for f in up_files if f.name.lower().endswith(".csv")]
    if not csv_names:
        st.error("No CSV files detected.")
        return None, None, {}
    # Detect train/test
    has_train = any("train" in n.lower() for n in csv_names)
    has_test  = any("test"  in n.lower() for n in csv_names)
    name_to_bytes = {f.name: f.read() for f in up_files}

    if has_train and has_test:
        train_name = next(n for n in csv_names if "train" in n.lower())
        test_name  = next(n for n in csv_names if "test"  in n.lower())
        df_train = pd.read_csv(io.BytesIO(name_to_bytes[train_name]))
        df_test  = pd.read_csv(io.BytesIO(name_to_bytes[test_name]))
        st.success(f"Detected train/test: {train_name} / {test_name}")
        return df_train, df_test, {"mode": "train_test", "train": train_name, "test": test_name}
    else:
        df_all = pd.read_csv(io.BytesIO(name_to_bytes[csv_names[0]]))
        st.info(f"Detected single CSV: {csv_names[0]} (will create a split)")
        return df_all, None, {"mode": "single", "file": csv_names[0]}

# =========================
# Auto-type detection (matches your Colab logic)
# =========================
def auto_detect_types(df: pd.DataFrame, target: str,
                      low_card_as_cat: bool = True, low_card_threshold: int = 20) -> tuple[list[str], list[str]]:
    cats, nums = [], []
    for c in df.columns:
        if c == target: 
            continue
        if df[c].dtype == 'O':
            cats.append(c)
        else:
            if low_card_as_cat:
                nunique = df[c].nunique(dropna=True)
                if nunique <= low_card_threshold:
                    cats.append(c)
                else:
                    nums.append(c)
            else:
                nums.append(c)
    return nums, cats

# =========================
# Prepare X/y (matches your Colab logic)
# =========================
def prepare_xy(df: pd.DataFrame, target: str, num_cols: list[str], cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df = df.dropna(subset=[target]).copy()
    y_raw = df[target].values
    X = df[num_cols + cat_cols].copy()

    # Fill numeric
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    # Fill categorical
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__NA__")

    # Label map y to 0..K-1
    if X.shape[0] == 0:
        raise ValueError("No rows left after cleaning.")
    if X[target].dtype == 'O' or y_raw.dtype.kind not in "iu":
        classes, y_idx = np.unique(y_raw, return_inverse=True)
        y = y_idx.astype(int)
        action_names = [str(c) for c in classes]
    else:
        classes = np.unique(y_raw)
        mapping = {v: i for i, v in enumerate(classes)}
        y = np.array([mapping[v] for v in y_raw], dtype=int)
        action_names = [str(v) for v in classes]

    return X.values, y, list(X.columns), action_names

# =========================
# Hashed State Encoder (from your Colab code)
# =========================
@dataclass
class HashedStateEncoder:
    n_bins: int = 6
    hash_mod: int = 200_000
    categorical_idx: T.Optional[T.List[int]] = None
    mins_: T.Optional[np.ndarray] = None
    maxs_: T.Optional[np.ndarray] = None
    cat_maps_: T.Optional[dict] = None
    num_idx_: T.Optional[T.List[int]] = None
    columns_: T.Optional[T.List[str]] = None

    def fit(self, X: np.ndarray, columns: T.List[str], categorical_idx: T.Optional[T.List[int]] = None):
        self.columns_ = list(columns)
        self.categorical_idx = set(categorical_idx or [])
        self.num_idx_ = [i for i in range(X.shape[1]) if i not in self.categorical_idx]
        if self.num_idx_:
            self.mins_ = X[:, self.num_idx_].astype(float).min(axis=0)
            self.maxs_ = X[:, self.num_idx_].astype(float).max(axis=0)
        else:
            self.mins_ = np.array([])
            self.maxs_ = np.array([])
        self.cat_maps_ = {}
        for j in self.categorical_idx:
            vals = pd.unique(X[:, j])
            self.cat_maps_[j] = {v: i for i, v in enumerate(vals)}
        return self

    def _bin_numeric(self, val: float, jnum: int) -> int:
        if self.maxs_.size == 0 or self.maxs_[jnum] == self.mins_[jnum]:
            return 0
        q = (val - self.mins_[jnum]) / (self.maxs_[jnum] - self.mins_[jnum] + 1e-12)
        return int(np.clip(np.floor(q * self.n_bins), 0, self.n_bins - 1))

    def transform_row(self, x: np.ndarray):
        tokens = []
        num_cursor = 0
        for i, xi in enumerate(x):
            if i in self.categorical_idx:
                mapped = self.cat_maps_.get(i, {}).get(xi, -1)
                tokens.append(f"{self.columns_[i]}:C{mapped}")
            else:
                xi = float(xi)
                b = self._bin_numeric(xi, num_cursor)
                tokens.append(f"{self.columns_[i]}:B{b}")
                num_cursor += 1
        key = "|".join(tokens)
        state_id = hash(key) % self.hash_mod
        return state_id, tuple(tokens)

# =========================
# Q-learning (γ=0) + predict (from your Colab code)
# =========================
@dataclass
class TQCHyperparams:
    episodes: int = 60
    eps0: float = 0.3
    eps_min: float = 0.02
    eps_decay: float = 0.97
    optimistic_init: float = 0.1
    seed: int = 42
    min_visits: int = 8
    margin: float = 0.15

@dataclass
class TQCModel:
    Q: dict
    visits: dict
    encoder: HashedStateEncoder
    class_weights: np.ndarray
    tokens_by_state: dict
    action_names: T.List[str]

def inverse_freq_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    vals, cnts = np.unique(y, return_counts=True)
    freq = np.zeros(n_classes); freq[vals] = cnts
    w = 1.0 / np.maximum(freq, 1)
    return w / w.mean()

def train_void_learning(
    X: np.ndarray, y: np.ndarray, n_classes: int,
    encoder: HashedStateEncoder, feature_cols: T.List[str],
    hp: TQCHyperparams, action_names: T.List[str],
    categorical_idx: list[int]
) -> TQCModel:
    rng = np.random.default_rng(hp.seed)
    class_weights = inverse_freq_weights(y, n_classes)
    encoder.fit(X, columns=feature_cols, categorical_idx=categorical_idx)

    Q = defaultdict(lambda: np.full(n_classes, hp.optimistic_init, dtype=float))
    visits = defaultdict(lambda: np.zeros(n_classes, dtype=int))
    tokens_by_state = {}
    idx = np.arange(len(X))
    eps = hp.eps0

    total_steps = hp.episodes * len(X)
    prog = st.progress(0)
    step = 0

    for ep in range(hp.episodes):
        rng.shuffle(idx)
        for i in idx:
            sid, toks = encoder.transform_row(X[i])
            if sid not in tokens_by_state:
                tokens_by_state[sid] = toks
            if rng.random() < eps:
                a = rng.integers(0, n_classes)
            else:
                a = int(np.argmax(Q[sid]))
            r = class_weights[y[i]] if a == y[i] else -class_weights[y[i]]
            visits[sid][a] += 1
            alpha = 1.0 / visits[sid][a]
            Q[sid][a] += alpha * (r - Q[sid][a])  # γ=0
            step += 1
        eps = max(hp.eps_min, eps * hp.eps_decay)
        prog.progress(min(100, int(100 * step / total_steps)))

    return TQCModel(
        Q=dict(Q), visits=dict(visits), encoder=encoder,
        class_weights=class_weights, tokens_by_state=tokens_by_state,
        action_names=action_names
    )

def predict(model: TQCModel, X: np.ndarray) -> np.ndarray:
    yhat = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        sid, _ = model.encoder.transform_row(X[i])
        yhat[i] = int(np.argmax(model.Q[sid])) if sid in model.Q else 0
    return yhat

# =========================
# Learnability & Void Tables (from your Colab code)
# =========================
@dataclass
class LearnabilityResult:
    learned_states: T.List[int]
    void_states: T.List[int]
    state_summary: pd.DataFrame
    feature_void_table: pd.DataFrame

def assess_learnability(model: TQCModel, hp: TQCHyperparams) -> LearnabilityResult:
    rows, learned_states, void_states = [], [], []
    for sid, q in model.Q.items():
        v = model.visits.get(sid, np.zeros_like(q))
        total_visits = int(v.sum())
        if total_visits < hp.min_visits:
            continue
        order = np.argsort(q)[::-1]
        max_q = float(q[order[0]])
        second_q = float(q[order[1]]) if len(order) > 1 else -np.inf
        margin = max_q - second_q
        is_learned = (margin >= hp.margin)
        label = "learned" if is_learned else "void"
        (learned_states if is_learned else void_states).append(sid)
        rows.append({
            "state_id": sid,
            "total_visits": total_visits,
            "max_action": int(order[0]),
            "max_action_name": model.action_names[order[0]],
            "max_Q": max_q,
            "second_Q": second_q,
            "margin": margin,
            "label": label,
            "tokens": "|".join(model.tokens_by_state[sid]),
        })
    state_summary = pd.DataFrame(rows).sort_values(["label","margin"], ascending=[True, False])

    feat_counts, feat_void = Counter(), Counter()
    for _, row in state_summary.iterrows():
        toks = row["tokens"].split("|")
        for t in toks:
            feat_counts[t] += 1
            if row["label"] == "void":
                feat_void[t] += 1
    feature_rows = []
    for t, cnt in feat_counts.items():
        vcnt = feat_void[t]
        feature_rows.append({
            "feature_token": t,
            "states_total": cnt,
            "states_void": vcnt,
            "void_rate": vcnt / cnt,
        })
    feature_void_table = pd.DataFrame(feature_rows).sort_values("void_rate", ascending=False)
    return LearnabilityResult(
        learned_states=learned_states,
        void_states=void_states,
        state_summary=state_summary.reset_index(drop=True),
        feature_void_table=feature_void_table.reset_index(drop=True),
    )

# =========================
# Plotly helpers
# =========================
def plot_cm_plotly(cm, labels, title):
    fig = px.imshow(cm, x=labels, y=labels, color_continuous_scale="Blues",
                    text_auto=True, aspect="auto")
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
    return fig

def plot_learning_curve_plotly(estimator, X_train, y_train, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=42
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(train_scores, axis=1), mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=train_sizes, y=np.mean(test_scores, axis=1), mode="lines+markers", name="CV"))
    fig.update_layout(title=title, xaxis_title="Train size", yaxis_title="Score")
    return fig

def df_to_download(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# =========================
# Model grids
# =========================
def model_grids():
    return {
        "KNN": (make_pipeline(StandardScaler(with_mean=True), KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": [3, 5, 7]}),
        "RF": (RandomForestClassifier(random_state=42),
               {"n_estimators": [150, 300], "max_depth": [None, 10]}),
        "LR": (make_pipeline(StandardScaler(with_mean=True),
                             LogisticRegression(max_iter=3000, n_jobs=None, random_state=42)),
               {"logisticregression__C": [0.5, 1.0, 2.0]}),
        "SVM": (make_pipeline(StandardScaler(with_mean=True), SVC(probability=False, random_state=42)),
                {"svc__C": [0.5, 1.0, 2.0], "svc__kernel": ["rbf", "linear"]}),
    }

def fit_and_report(X_train, X_test, y_train, y_test, label_prefix="Original"):
    grids = model_grids()
    results = []
    pb = st.progress(0)
    step, total = 0, len(grids)
    class_labels = sorted(np.unique(y_test))

    for name, (est, grid) in grids.items():
        step += 1
        with st.spinner(f"Training {name} — {label_prefix}"):
            gs = GridSearchCV(est, grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            st.plotly_chart(plot_cm_plotly(cm, class_labels, f"{name} — {label_prefix} (Confusion Matrix)"),
                            use_container_width=True)
            st.plotly_chart(plot_learning_curve_plotly(gs.best_estimator_, X_train, y_train,
                                f"Learning Curve — {name} ({label_prefix})"),
                            use_container_width=True)
            results.append({"model": name,
                            f"accuracy_{label_prefix.lower()}": acc,
                            f"macroF1_{label_prefix.lower()}": f1m})
        pb.progress(int(100 * step / total))
    return pd.DataFrame(results)

# =========================
# 1) Data loading (built-ins or upload)
# =========================
st.subheader("1) Data Loading")

if ds_choice == "Iris (built-in)":
    df = load_builtin_iris()
    st.success("Loaded Iris.")
elif ds_choice == "Titanic (seaborn)":
    df = load_builtin_titanic()
    if df is None:
        st.error("Could not load Titanic from seaborn. Use Upload CSV.")
        st.stop()
    st.success("Loaded Titanic.")
else:
    df_train_up, df_test_up, info = load_uploads()
    if df_train_up is None:
        st.info("Upload CSV(s) to continue.")
        st.stop()
    if info.get("mode") == "train_test":
        df_train = df_train_up.copy()
        df_test = df_test_up.copy()
        df = pd.concat([df_train.assign(__split__="train"), df_test.assign(__split__="test")], axis=0, ignore_index=True)
    else:
        df = df_train_up.copy()
    st.success("Upload(s) loaded.")

st.dataframe(df.head(10), use_container_width=True)

# =========================
# 2) Target & Feature selection (dynamic)
# =========================
st.subheader("2) Target & Feature Selection")
default_target = "target" if "target" in df.columns else ("survived" if "survived" in df.columns else df.columns[-1])
target_name = st.selectbox("Select target", options=df.columns.tolist(),
                           index=list(df.columns).index(default_target) if default_target in df.columns else len(df.columns)-1)

# Candidate feature set
candidate_features = [c for c in df.columns if c != target_name and c != "__split__"]
use_all_btn = st.button("Use all other columns as features")
if use_all_btn:
    selected_features = candidate_features
else:
    selected_features = st.multiselect("Select feature columns", options=candidate_features, default=candidate_features)

st.write(f"Target: `{target_name}` — #Features selected: {len(selected_features)}")

# Train/test define
if ds_choice == "Upload CSV" and "__split__" in df.columns:
    df_train = df[df["__split__"] == "train"].drop(columns=["__split__"]).reset_index(drop=True)
    df_test  = df[df["__split__"] == "test"].drop(columns=["__split__"]).reset_index(drop=True)
elif ds_choice == "Upload CSV":
    # single file: create split
    df_all = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df_all) * 0.75)
    df_train = df_all.iloc[:n_train].reset_index(drop=True)
    df_test  = df_all.iloc[n_train:].reset_index(drop=True)
else:
    # built-ins: random split from current selection
    df_all = df[selected_features + [target_name]].dropna(subset=[target_name]).copy()
    df_train, df_test = train_test_split(df_all, test_size=0.25, random_state=42, stratify=df_all[target_name])

# Type detection
num_cols, cat_cols = auto_detect_types(df_train[selected_features + [target_name]],
                                       target=target_name,
                                       low_card_as_cat=low_card_as_cat,
                                       low_card_threshold=low_card_threshold)
st.write("*Detected numeric:*", num_cols)
st.write("*Detected categorical:*", cat_cols)

# Prepare X/y by your Colab logic
X_train_np, y_train, feature_cols, action_names = prepare_xy(df_train[selected_features + [target_name]],
                                                             target_name, num_cols, cat_cols)
X_test_np,  y_test,  _,             _           = prepare_xy(df_test[selected_features + [target_name]],
                                                             target_name, num_cols, cat_cols)
n_classes = int(np.unique(y_train).size)
cat_idx = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

# =========================
# 3) Baseline classifiers (Original features)
# =========================
st.subheader("3) Baseline Classifiers (Original Features)")
# For baseline classifiers, we’ll use the integer-coded matrix as-is:
X_proc = pd.DataFrame(X_train_np, columns=feature_cols)
X_proc_test = pd.DataFrame(X_test_np, columns=feature_cols)
baseline_df = fit_and_report(X_proc, X_proc_test, y_train, y_test, label_prefix="Original")

# =========================
# 4) Void-Learning (hashed-state Q-learning, γ=0)
# =========================
st.subheader("4) Void-Learning (Hashed-State Tabular Q-learning, γ=0)")

hp = TQCHyperparams(episodes=episodes, eps0=eps0, eps_min=eps_min,
                    eps_decay=eps_decay, optimistic_init=optimistic_init,
                    min_visits=min_visits, margin=margin_tau)
encoder = HashedStateEncoder(n_bins=n_bins, hash_mod=hash_mod, categorical_idx=cat_idx)
with st.spinner("Training Void-Learning ..."):
    model = train_void_learning(X_train_np, y_train, n_classes, encoder, feature_cols, hp, action_names, categorical_idx=cat_idx)

# Evaluate RL policy on test
y_pred_rl = predict(model, X_test_np)
acc_rl = accuracy_score(y_test, y_pred_rl)
f1_rl = f1_score(y_test, y_pred_rl, average="macro")
st.write(f"RL Policy — Test Accuracy: **{acc_rl:.4f}**, Macro-F1: **{f1_rl:.4f}**")
cm_rl = confusion_matrix(y_test, y_pred_rl, labels=sorted(np.unique(y_test)))
st.plotly_chart(plot_cm_plotly(cm_rl, sorted(np.unique(y_test)), "RL Policy — Confusion Matrix"), use_container_width=True)

# Learnability + Void tables (CSV)
res = assess_learnability(model, hp)
st.write("State summary (head):"); st.dataframe(res.state_summary.head(12), use_container_width=True)
st.write("Feature-level void table (head):"); st.dataframe(res.feature_void_table.head(20), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    df_to_download(res.state_summary, "void_learning_state_summary.csv", "Download state summary (CSV)")
with c2:
    df_to_download(res.feature_void_table, "void_learning_feature_void_table.csv", "Download feature void table (CSV)")

# Quick plots (counts + top tokens)
counts = res.state_summary["label"].value_counts().reset_index()
counts.columns = ["label", "count"]
st.plotly_chart(px.bar(counts, x="label", y="count", title="Counts of learned vs void states"), use_container_width=True)

top_tok = res.feature_void_table.sort_values("void_rate", ascending=False).head(15)
st.plotly_chart(px.bar(top_tok, x="feature_token", y="void_rate", title="Top tokens by void rate").update_layout(xaxis_tickangle=45), use_container_width=True)

# =========================
# 5) Prune features (mean token void rate ≥ threshold)
# =========================
st.subheader("5) Pruning by Mean Token Void Rate")
feat_means = res.feature_void_table.assign(feature=lambda d: d["feature_token"].str.split(":", n=1, expand=True)[0]).groupby("feature")["void_rate"].mean()
pruned_features = feat_means[feat_means >= prune_thresh].index.tolist()
st.write("Mean void rate per feature:"); st.dataframe(feat_means.reset_index().rename(columns={"void_rate": "mean_void_rate"}))
if pruned_features:
    st.success(f"Pruned features (≥ {prune_thresh}): {pruned_features}")
else:
    st.info("No features exceeded the prune threshold.")

# Build pruned matrices
keep_cols = [c for c in feature_cols if c not in pruned_features]
X_train_p = pd.DataFrame(X_train_np, columns=feature_cols)[keep_cols].values
X_test_p  = pd.DataFrame(X_test_np,  columns=feature_cols)[keep_cols].values

# =========================
# 6) Classifiers on Pruned features
# =========================
st.subheader("6) Classifiers (Void-Pruned Features)")
pruned_df = fit_and_report(pd.DataFrame(X_train_p, columns=keep_cols),
                           pd.DataFrame(X_test_p, columns=keep_cols),
                           y_train, y_test, label_prefix="Pruned")

# =========================
# 7) Comparison table
# =========================
st.subheader("7) Comparison — Original vs Pruned (Accuracy / Macro-F1)")
if not baseline_df.empty and not pruned_df.empty:
    comp = pd.merge(baseline_df, pruned_df, on="model", how="outer")
    comp["Δ accuracy"] = comp["accuracy_pruned"] - comp["accuracy_original"]
    comp["Δ macroF1"] = comp["macroF1_pruned"] - comp["macroF1_original"]
    st.dataframe(comp, use_container_width=True)
    df_to_download(comp, "comparison_original_vs_pruned.csv", "Download comparison (CSV)")
else:
    st.write("Insufficient results to compare.")
