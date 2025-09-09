# app.py
import io
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

# -------------------------
# Page + Sidebar
# -------------------------
st.set_page_config(page_title="Void-Learning Pipeline (Zoomable)", layout="wide")
st.title("Void-Learning Pipeline (Streamlit) — Zoomable Plots")

with st.sidebar:
    st.header("Dataset")
    ds_choice = st.radio("Choose dataset", ["Iris (built-in)", "Titanic (seaborn)", "Upload CSV"])
    st.markdown("---")
    st.header("Preprocessing")
    n_bins = st.slider("Numeric bin count", min_value=3, max_value=12, value=6, step=1)
    st.markdown("---")
    st.header("Void-Learning (Bandit Q-learning)")
    episodes = st.slider("Episodes", 5, 120, 40, step=5)
    epsilon = st.slider("Epsilon (exploration)", 0.01, 0.9, 0.3, step=0.01)
    alpha = st.slider("Alpha (learning rate)", 0.01, 1.0, 0.2, step=0.01)
    vmin = st.slider("Min visits per state", 1, 50, 8, step=1)
    tau = st.slider("Margin threshold τ", 0.01, 1.0, 0.2, step=0.01)
    prune_thresh = st.slider("Prune threshold (mean token void rate per feature)", 0.1, 1.0, 0.5, step=0.05)

# -------------------------
# Utilities
# -------------------------
def df_to_download(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def load_builtin_iris():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    return df  # includes 'target'

def load_builtin_titanic():
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        return df  # includes 'survived'
    except Exception:
        return None

def upload_csv():
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        return None
    return pd.read_csv(io.BytesIO(up.read()))

# ---------- Robust preprocessing/tokenisation (fixes KBins issues) ----------
def preprocess_tokenize(X: pd.DataFrame, n_bins=6):
    """
    Robust tokenisation:
      - Decide numeric vs categorical by coercion success
      - Clean inf/NaN
      - Numeric: handle all-NaN, constants; cap bins by unique count; fallback binning
      - Categorical: LabelEncoder
    Return X_proc (int-coded), encoders dict, num_cols, cat_cols
    """
    X = X.copy().replace(["None", "none", "NaN", "nan", "NULL", "null", ""], np.nan)

    num_cols, cat_cols = [], []
    for c in X.columns:
        coerced = pd.to_numeric(X[c], errors="coerce")
        is_mostly_num = (coerced.notna().mean() >= 0.8) or pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
        if is_mostly_num:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X_proc = pd.DataFrame(index=X.index)
    encoders = {}

    # Numeric
    for c in num_cols:
        col = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col.notna().sum() == 0:
            X_proc[c] = 0
            encoders[c] = {"type": "constant_all_nan"}
            continue

        med = float(col.median())
        col = col.fillna(med)
        uniq = np.unique(col.values)
        uniq_n = uniq.size

        if uniq_n == 1:
            X_proc[c] = 0
            encoders[c] = {"type": "constant", "value": float(uniq[0])}
            continue

        bins = int(max(2, min(n_bins, uniq_n)))
        try:
            kb = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
            X_proc[c] = kb.fit_transform(col.to_frame()).astype(int).ravel()
            encoders[c] = {"type": "kbin", "bins": bins, "strategy": "quantile", "kb": kb}
        except Exception:
            try:
                cats = pd.qcut(col, q=bins, duplicates="drop")
                X_proc[c] = cats.cat.codes.astype(int)
                encoders[c] = {"type": "qcut", "bins": bins, "edges": list(cats.cat.categories.astype(str))}
            except Exception:
                ranks = pd.qcut(col.rank(method="average"), q=2, duplicates="drop")
                X_proc[c] = ranks.cat.codes.astype(int)
                encoders[c] = {"type": "fallback_rank_q2"}

    # Categorical
    for c in cat_cols:
        s = X[c].astype(str).fillna("__NA__")
        le = LabelEncoder()
        X_proc[c] = le.fit_transform(s)
        encoders[c] = {"type": "labelencoder", "classes_": list(le.classes_)}

    # Ensure ints
    for c in X_proc.columns:
        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce").fillna(0).astype(int)

    return X_proc, encoders, num_cols, cat_cols

def make_state_key(row_series):
    tokens = [f"{col}:{row_series[col]}" for col in row_series.index]
    return "|".join(tokens)

def void_learning_bandit(X_proc, y, episodes=40, alpha=0.2, epsilon=0.3, vmin=8, tau=0.2):
    state_keys = X_proc.apply(make_state_key, axis=1).values
    actions = np.unique(y)
    Q, counts = {}, {}
    for s in np.unique(state_keys):
        Q[s] = {int(a): 0.0 for a in actions}
        counts[s] = {int(a): 0 for a in actions}

    rng = np.random.default_rng(42)
    prog = st.progress(0)
    total_steps = episodes * len(y)
    step = 0

    for ep in range(episodes):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        for i in idx:
            s = state_keys[i]
            yi = int(y.iloc[i]) if hasattr(y, "iloc") else int(y[i])
            if rng.random() < epsilon:
                a = int(rng.choice(actions))
            else:
                a = int(max(Q[s], key=Q[s].get))
            r = 1 if a == yi else -1
            counts[s][a] += 1
            Q[s][a] += alpha * (r - Q[s][a])
            step += 1
        prog.progress(min(100, int(100 * step / total_steps)))

    # Label states
    labels, rows = {}, []
    for s in Q:
        total_visits = sum(counts[s].values())
        qarr = np.array(list(Q[s].values()))
        if total_visits >= vmin and qarr.size >= 2:
            order = np.argsort(qarr)[::-1]
            margin = qarr[order[0]] - qarr[order[1]]
            lab = "learned" if margin >= tau else "void"
        else:
            lab, margin = "void", np.nan
        labels[s] = lab
        rows.append({"state_key": s, "total_visits": total_visits, "max_Q": float(np.nanmax(qarr)) if qarr.size else np.nan, "margin": margin, "label": lab})
    state_summary = pd.DataFrame(rows)

    # Token-level void table
    token_stats = []
    for col in X_proc.columns:
        vals = np.unique(X_proc[col])
        for v in vals:
            mask = (X_proc[col] == v).values
            s_sub = state_keys[mask]
            total = len(s_sub)
            voids = sum(1 for sk in s_sub if labels[sk] == "void")
            if total > 0:
                token_stats.append({"feature": col, "token": f"{col}:{v}", "states_total": total, "states_void": voids, "void_rate": voids / total})
    void_table = pd.DataFrame(token_stats)
    return Q, labels, void_table, state_summary

def prune_by_void_table(X_proc, void_table, threshold=0.5):
    feature_mean = void_table.groupby("feature")["void_rate"].mean()
    pruned_features = feature_mean[feature_mean >= threshold].index.tolist()
    X_pruned = X_proc.drop(columns=pruned_features) if pruned_features else X_proc.copy()
    return X_pruned, pruned_features, feature_mean

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

def plot_cm_plotly(cm, labels, title):
    fig = px.imshow(cm,
                    x=labels, y=labels,
                    color_continuous_scale="Blues",
                    text_auto=True,
                    aspect="auto")
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

def fit_and_report(X_train, X_test, y_train, y_test, label_prefix="Original"):
    grids = model_grids()
    results = []
    pb = st.progress(0)
    step, total = 0, len(grids)

    class_labels = sorted(np.unique(y_test))

    for name, (est, grid) in grids.items():
        step += 1
        with st.spinner(f"Training {name} ({label_prefix}) ..."):
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

            results.append({
                "model": name,
                f"accuracy_{label_prefix.lower()}": acc,
                f"macroF1_{label_prefix.lower()}": f1m
            })
        pb.progress(int(100 * step / total))

    return pd.DataFrame(results)

# -------------------------
# 1) DATA LOADING + DYNAMIC TARGET/FEATURES
# -------------------------
st.subheader("1) Data Loading & Selection")

if ds_choice == "Iris (built-in)":
    df = load_builtin_iris()
    st.write("Loaded built-in Iris dataset.")
elif ds_choice == "Titanic (seaborn)":
    df = load_builtin_titanic()
    if df is None:
        st.warning("Could not load Titanic from seaborn. Please use 'Upload CSV'.")
        st.stop()
    st.write("Loaded Titanic dataset from seaborn.")
else:
    df = upload_csv()
    if df is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    st.write("Uploaded CSV loaded.")

st.write("Preview:")
st.dataframe(df.head(10), use_container_width=True)

# Target selection (dynamic per dataset)
with st.expander("Select Class (Target) and Features", expanded=True):
    default_target = "target" if "target" in df.columns else ("survived" if "survived" in df.columns else df.columns[-1])
    target_name = st.selectbox("Select the class (target) column:", options=df.columns.tolist(),
                               index=list(df.columns).index(default_target) if default_target in df.columns else len(df.columns)-1)
    candidate_features = [c for c in df.columns if c != target_name]

    # Convenient button: use-all features except target
    use_all_btn = st.button("Use all other columns as features")
    if use_all_btn:
        selected_features = candidate_features
    else:
        selected_features = st.multiselect("Select feature columns:", options=candidate_features, default=candidate_features)

# Build X,y now that selections are made
y = df[target_name]
X = df[selected_features]

st.markdown(f"**Target:** `{target_name}`")
st.markdown(f"**#Features selected:** {len(selected_features)}")
st.dataframe(pd.concat([X.head(3), y.head(3)], axis=1), use_container_width=True)

# -------------------------
# 2) PREPROCESS & TOKENISE
# -------------------------
st.subheader("2) Preprocessing & Tokenisation")
X_proc, encoders, num_cols, cat_cols = preprocess_tokenize(X, n_bins=n_bins)
st.write("*Numeric columns (binned):*", num_cols)
st.write("*Categorical columns (encoded):*", cat_cols)
st.dataframe(X_proc.head(5), use_container_width=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.3, random_state=42,
    stratify=y if len(np.unique(y)) > 1 else None
)

# -------------------------
# 3) BASELINE CLASSIFIERS
# -------------------------
st.subheader("3) Baseline Classifiers (Original Features)")
baseline_df = fit_and_report(X_train, X_test, y_train, y_test, label_prefix="Original")

# -------------------------
# 4) VOID-LEARNING
# -------------------------
st.subheader("4) Void-Learning (Tabular Q-learning, contextual bandit)")
with st.spinner("Running Void-Learning ..."):
    Q, state_labels, void_table, state_summary = void_learning_bandit(
        X_proc, y, episodes=episodes, alpha=alpha, epsilon=epsilon, vmin=vmin, tau=tau
    )
st.write("State summary (head):")
st.dataframe(state_summary.head(10), use_container_width=True)

st.write("Token-level void table (head):")
st.dataframe(void_table.head(20), use_container_width=True)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    df_to_download(void_table, "void_table_tokens.csv", "Download token-level void table (CSV)")
with col_dl2:
    df_to_download(state_summary, "state_summary.csv", "Download state summary (CSV)")

# -------------------------
# 5) PRUNE BY VOID RATE
# -------------------------
st.subheader("5) Prune Features by Void Rate")
X_pruned, pruned_features, feature_mean = prune_by_void_table(X_proc, void_table, threshold=prune_thresh)
if pruned_features:
    st.success(f"Pruned features (mean void rate ≥ {prune_thresh}): {pruned_features}")
else:
    st.info("No features met the prune threshold.")

st.write("Mean void rate per feature:")
st.dataframe(feature_mean.reset_index().rename(columns={"void_rate": "mean_void_rate"}),
             use_container_width=True)

# Resplit pruned set
X_train_p, X_test_p, _, _ = train_test_split(
    X_pruned, y, test_size=0.3, random_state=42,
    stratify=y if len(np.unique(y)) > 1 else None
)

# -------------------------
# 6) CLASSIFIERS ON PRUNED
# -------------------------
st.subheader("6) Classifiers (Void-Pruned Features)")
pruned_df = fit_and_report(X_train_p, X_test_p, y_train, y_test, label_prefix="Pruned")

# -------------------------
# 7) COMPARISON TABLE
# -------------------------
st.subheader("7) Comparison — Original vs Pruned")
if not baseline_df.empty and not pruned_df.empty:
    comp = pd.merge(baseline_df, pruned_df, on="model", how="outer")
    comp["Δ accuracy"] = comp["accuracy_pruned"] - comp["accuracy_original"]
    comp["Δ macroF1"] = comp["macroF1_pruned"] - comp["macroF1_original"]
    st.dataframe(comp, use_container_width=True)
    df_to_download(comp, "comparison_original_vs_pruned.csv", "Download comparison table (CSV)")
else:
    st.write("Insufficient results to compare.")
