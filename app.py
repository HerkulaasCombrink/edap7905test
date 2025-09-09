# app.py
from __future__ import annotations
import io, typing as T
from dataclasses import dataclass
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(page_title="Void-Learning (γ=0 Q-learning)", layout="wide")
st.title("📦 Void-Learning — Hashed State Tabular Q-Learning (γ=0)")
st.caption("Upload CSV(s) • Choose target • Auto-detect types • Train RL • Assess learned vs void • Export CSVs")

# =========================
# Sidebar controls (match your Colab hyperparams)
# =========================
with st.sidebar:
    st.header("Upload")
    up_files = st.file_uploader("Upload one CSV (auto split) OR 'train' + 'test' CSVs", type=["csv"], accept_multiple_files=True)
    st.markdown("---")
    st.header("Type Detection")
    LOW_CARD_AS_CATEGORICAL = st.checkbox("Treat low-cardinality numeric as categorical", value=True)
    LOW_CARD_THRESHOLD = st.slider("Low-cardinality threshold", 5, 100, 20, 1)
    st.markdown("---")
    st.header("Q-learning (γ=0)")
    episodes = st.slider("Episodes", 5, 200, 60, 5)
    eps0 = st.slider("ε₀ (start exploration)", 0.01, 0.9, 0.30, 0.01)
    eps_min = st.slider("εmin", 0.0, 0.5, 0.02, 0.01)
    eps_decay = st.slider("ε decay per episode", 0.80, 0.999, 0.97, 0.001)
    optimistic_init = st.slider("Optimistic init (Q₀)", 0.0, 1.0, 0.10, 0.01)
    min_visits = st.slider("Min visits per state", 1, 200, 8, 1)
    margin_tau = st.slider("Margin threshold τ", 0.01, 1.0, 0.15, 0.01)
    n_bins = st.slider("Numeric bin count", 2, 12, 6, 1)
    hash_mod = st.select_slider("Hash modulus (state table size)", options=[50_000, 100_000, 200_000, 500_000, 1_000_000], value=200_000)

# =========================
# Helpers: detect upload mode
# =========================
def read_uploads(files) -> tuple[pd.DataFrame|None, pd.DataFrame|None, dict]:
    if not files:
        return None, None, {}
    csv = {f.name: f.read() for f in files if f.name.lower().endswith(".csv")}
    if not csv:
        return None, None, {}
    names = list(csv.keys())
    has_train = any("train" in n.lower() for n in names)
    has_test  = any("test"  in n.lower() for n in names)
    if has_train and has_test:
        train_name = next(n for n in names if "train" in n.lower())
        test_name  = next(n for n in names if "test"  in n.lower())
        df_train = pd.read_csv(io.BytesIO(csv[train_name]))
        df_test  = pd.read_csv(io.BytesIO(csv[test_name]))
        return df_train, df_test, {"mode": "train_test", "train": train_name, "test": test_name}
    # single CSV
    df_all = pd.read_csv(io.BytesIO(csv[names[0]]))
    return df_all, None, {"mode": "single", "file": names[0]}

df_train, df_test, info = read_uploads(up_files)

if df_train is None and df_test is None:
    st.info("⬆️ Upload one CSV (auto split) OR two CSVs with 'train' and 'test' in filenames.")
    st.stop()

with st.expander("Upload summary", expanded=True):
    if info.get("mode") == "train_test":
        st.success(f"Detected train/test: **{info['train']}** / **{info['test']}**")
        st.write("Train shape:", df_train.shape, " | Test shape:", df_test.shape)
    else:
        st.info(f"Detected single CSV: **{info.get('file')}** (will create a 75/25 split)")
        st.write("All data shape:", df_train.shape)

# =========================
# 1) Target selection
# =========================
df_show = df_train if info.get("mode") == "train_test" else df_train
st.subheader("1) Data & Target")
st.dataframe(df_show.head(10), use_container_width=True)

default_target = df_show.columns[-1]
target_name = st.selectbox("🎯 Select target column", options=df_show.columns.tolist(),
                           index=list(df_show.columns).index(default_target))

if target_name not in df_show.columns:
    st.error(f"Target '{target_name}' not found.")
    st.stop()

# =========================
# 2) Auto-detect numeric vs categorical (your rules)
# =========================
def auto_detect_types(df: pd.DataFrame, target: str):
    cats, nums = [], []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == 'O':
            cats.append(c)
        else:
            if LOW_CARD_AS_CATEGORICAL:
                nunique = df[c].nunique(dropna=True)
                if nunique <= LOW_CARD_THRESHOLD:
                    cats.append(c)
                else:
                    nums.append(c)
            else:
                nums.append(c)
    return nums, cats

if info.get("mode") == "train_test":
    nums, cats = auto_detect_types(df_train, target_name)
else:
    nums, cats = auto_detect_types(df_train, target_name)

st.write("*Detected numeric:*", nums)
st.write("*Detected categorical:*", cats)

# =========================
# 3) Train/test split if single CSV
# =========================
if info.get("mode") == "single":
    TRAIN_FRACTION = 0.75
    df_all = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_train = int(len(df_all) * TRAIN_FRACTION)
    df_train = df_all.iloc[:n_train].reset_index(drop=True)
    df_test  = df_all.iloc[n_train:].reset_index(drop=True)
    st.success(f"Split single CSV into train {df_train.shape} and test {df_test.shape}.")

# =========================
# 4) Build matrices (X, y) and encodings (your logic)
# =========================
def prepare_xy(df: pd.DataFrame, target: str, num_cols: T.List[str], cat_cols: T.List[str]):
    df = df.dropna(subset=[target]).copy()
    y_raw = df[target].values
    X = df[num_cols + cat_cols].copy()

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__NA__")

    if X.shape[0] == 0:
        raise ValueError("No rows left after cleaning.")
    if (y_raw.dtype == 'O') or (getattr(y_raw, "dtype", None) is not None and y_raw.dtype.kind not in "iu"):
        classes, y_idx = np.unique(y_raw, return_inverse=True)
        y = y_idx.astype(int)
        action_names = [str(c) for c in classes]
    else:
        classes = np.unique(y_raw)
        mapping = {v:i for i,v in enumerate(classes)}
        y = np.array([mapping[v] for v in y_raw], dtype=int)
        action_names = [str(v) for v in classes]

    return X.values, y, list(X.columns), action_names

X_train, y_train, feature_cols, action_names = prepare_xy(df_train, target_name, nums, cats)
X_test,  y_test,  _,             _           = prepare_xy(df_test,  target_name, nums, cats)
n_classes = int(np.unique(y_train).size)
cat_idx = [feature_cols.index(c) for c in cats if c in feature_cols]

# =========================
# 5) Encoder: hash + bin (your code)
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
# 6) Tabular Q-learning (γ=0) (your code)
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
    hp: TQCHyperparams, action_names: T.List[str]
) -> TQCModel:
    rng = np.random.default_rng(hp.seed)
    class_weights = inverse_freq_weights(y, n_classes)
    encoder.fit(X, columns=feature_cols, categorical_idx=cat_idx)

    Q = defaultdict(lambda: np.full(n_classes, hp.optimistic_init, dtype=float))
    visits = defaultdict(lambda: np.zeros(n_classes, dtype=int))
    tokens_by_state = {}
    idx = np.arange(len(X))
    eps = hp.eps0

    total = hp.episodes * len(X)
    bar = st.progress(0)
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
            Q[sid][a] += alpha * (r - Q[sid][a])  # γ = 0
            step += 1
        eps = max(hp.eps_min, eps * hp.eps_decay)
        bar.progress(int(100 * step / total))

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
# 7) Learnability assessment (your code)
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
# 8) Train + Evaluate RL
# =========================
st.subheader("3) Train RL policy (γ=0) and Evaluate")
hp = TQCHyperparams(
    episodes=episodes, eps0=eps0, eps_min=eps_min, eps_decay=eps_decay,
    optimistic_init=optimistic_init, min_visits=min_visits, margin=margin_tau
)
encoder = HashedStateEncoder(n_bins=n_bins, hash_mod=hash_mod, categorical_idx=cat_idx)

with st.spinner("Training Void-Learning (tabular Q-learning, γ=0)..."):
    model = train_void_learning(X_train, y_train, n_classes, encoder, feature_cols, hp, action_names)

y_pred = predict(model, X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")
st.success(f"✅ RL Policy — Test Accuracy: **{acc:.4f}**  |  Macro-F1: **{f1m:.4f}**")

cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))
fig_cm = px.imshow(cm, x=sorted(np.unique(y_test)), y=sorted(np.unique(y_test)),
                   text_auto=True, color_continuous_scale="Blues",
                   title="RL Policy — Confusion Matrix")
fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="True")
s
