# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ========== UI SETUP ==========
st.set_page_config(page_title="Void-Learning Pipeline", layout="wide")
st.title("Void-Learning Pipeline (Streamlit)")

with st.sidebar:
    st.header("Dataset")
    ds_choice = st.radio("Choose dataset", ["Iris (built-in)", "Titanic (seaborn)", "Upload CSV"])
    st.markdown("---")
    st.header("Preprocessing")
    n_bins = st.slider("Numeric bin count", min_value=3, max_value=12, value=6, step=1)
    st.markdown("---")
    st.header("Void-Learning")
    episodes = st.slider("Episodes", 5, 100, 40, step=5)
    epsilon = st.slider("Epsilon (exploration)", 0.01, 0.9, 0.3, step=0.01)
    alpha = st.slider("Alpha (learning rate)", 0.01, 1.0, 0.2, step=0.01)
    vmin = st.slider("Min visits per state", 1, 30, 8, step=1)
    tau = st.slider("Margin threshold τ", 0.01, 1.0, 0.2, step=0.01)
    prune_thresh = st.slider("Prune threshold (mean token void rate per feature)", 0.1, 1.0, 0.5, step=0.05)

# ========== HELPERS ==========

def load_builtin_iris():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)
    target_name = "target"
    return X, y, target_name

def load_builtin_titanic():
    # Attempts to load from seaborn. If not available, ask user to upload.
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # choose target and basic cleaning
        df = df.dropna(subset=["survived"]).reset_index(drop=True)
        y = df["survived"].astype(int)
        X = df.drop(columns=["survived"])
        target_name = "survived"
        return X, y, target_name
    except Exception as e:
        st.warning("Could not load Titanic from seaborn. Please upload a CSV instead.")
        return None, None, None

def upload_csv():
    up = st.file_uploader("Upload CSV (target column selectable below)", type=["csv"])
    if up is None:
        return None, None, None, None
    df = pd.read_csv(io.BytesIO(up.read()))
    target_name = st.selectbox("Select target column", options=df.columns.tolist(), index=len(df.columns)-1)
    y = df[target_name]
    X = df.drop(columns=[target_name])
    return df, X, y, target_name

def split_cat_num(df):
    num_cols, cat_cols = [], []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def preprocess_tokenize(X, n_bins=6):
    """
    Robust tokenisation:
      - Coerce every column to either numeric (float) or categorical (string)
      - Clean NaNs/inf
      - For numeric: handle constants / low-unique / all-NaN
      - Bin numerics safely (cap bins by unique count; fallback to qcut)
      - Encode categoricals with LabelEncoder
    Returns:
      X_proc (int codes per column), encoders, num_cols, cat_cols
    """
    X = X.copy()

    # 1) Clean obvious NaNs and strings like "None"/"nan"
    X = X.replace(["None", "none", "NaN", "nan", "NULL", "null", ""], np.nan)

    # 2) Decide numeric vs categorical by attempt-to-coerce (robust)
    num_cols, cat_cols = [], []
    for c in X.columns:
        # try numeric coercion on a sample to decide
        coerced = pd.to_numeric(X[c], errors="coerce")
        # consider numeric if at least 80% successfully coercible OR dtype already numeric/bool
        is_mostly_num = (coerced.notna().mean() >= 0.8) or pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])
        if is_mostly_num:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X_proc = pd.DataFrame(index=X.index)
    encoders = {}

    # 3) Handle numeric columns
    for c in num_cols:
        col = pd.to_numeric(X[c], errors="coerce")
        # replace infs
        col = col.replace([np.inf, -np.inf], np.nan)
        # if all NaN → make zeros and mark as constant
        if col.notna().sum() == 0:
            # constant code 0
            X_proc[c] = 0
            encoders[c] = {"type": "constant_all_nan"}
            continue

        # fill remaining NaNs with median
        med = float(col.median())
        col = col.fillna(med)

        unique_vals = np.unique(col.values)
        uniq_n = unique_vals.size

        # constant column → single bin 0
        if uniq_n == 1:
            X_proc[c] = 0
            encoders[c] = {"type": "constant", "value": float(unique_vals[0])}
            continue

        # cap bins by #unique; need at least 2 bins
        bins = int(max(2, min(n_bins, uniq_n)))

        # try KBinsDiscretizer first
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            kb = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
            X_proc[c] = kb.fit_transform(col.to_frame()).astype(int).ravel()
            encoders[c] = {"type": "kbin", "bins": bins, "strategy": "quantile", "kb": kb}
        except Exception:
            # robust fallback: pandas qcut with duplicates dropped
            try:
                cats = pd.qcut(col, q=bins, duplicates="drop")
                X_proc[c] = cats.cat.codes.astype(int)
                encoders[c] = {"type": "qcut", "bins": bins, "edges": list(cats.cat.categories.astype(str))}
            except Exception:
                # final fallback: rank-based 2 bins
                ranks = pd.qcut(col.rank(method="average"), q=2, duplicates="drop")
                X_proc[c] = ranks.cat.codes.astype(int)
                encoders[c] = {"type": "fallback_rank_q2"}

    # 4) Handle categorical columns
    from sklearn.preprocessing import LabelEncoder
    for c in cat_cols:
        s = X[c].astype(str)  # ensure string
        s = s.fillna("__NA__")
        le = LabelEncoder()
        X_proc[c] = le.fit_transform(s)
        encoders[c] = {"type": "labelencoder", "classes_": list(le.classes_)}

    # Ensure integer dtype for all processed columns
    for c in X_proc.columns:
        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce").fillna(0).astype(int)

    return X_proc, encoders, num_cols, cat_cols

def make_state_key(row_series):
    # join as tokens "col:val" in sorted column order for stability
    tokens = [f"{col}:{row_series[col]}" for col in row_series.index]
    return "|".join(tokens)

def void_learning_bandit(X_proc, y, episodes=40, alpha=0.2, epsilon=0.3, vmin=8, tau=0.2):
    """Tabular Q-learning contextual bandit; returns Q, state_labels, void_table, state_summary."""
    # Build state keys
    state_keys = X_proc.apply(make_state_key, axis=1).values
    actions = np.unique(y)
    # Initialize Q and counts
    Q = {}
    counts = {}
    for s in np.unique(state_keys):
        Q[s] = {int(a): 0.0 for a in actions}
        counts[s] = {int(a): 0 for a in actions}

    rng = np.random.default_rng(42)
    # Train
    for ep in range(episodes):
        indices = np.arange(len(y))
        rng.shuffle(indices)
        for i in indices:
            s = state_keys[i]
            yi = int(y.iloc[i]) if hasattr(y, "iloc") else int(y[i])
            # epsilon-greedy
            if rng.random() < epsilon:
                a = int(rng.choice(actions))
            else:
                qvals = Q[s]
                a = int(max(qvals, key=qvals.get))
            # reward
            r = 1 if a == yi else -1
            counts[s][a] += 1
            Q[s][a] += alpha * (r - Q[s][a])  # gamma=0 bandit TD
    # Label states
    labels = {}
    rows = []
    for s in Q:
        total_visits = sum(counts[s].values())
        qarr = np.array(list(Q[s].values()))
        if total_visits >= vmin and len(qarr) >= 2:
            order = np.argsort(qarr)[::-1]
            margin = qarr[order[0]] - qarr[order[1]]
            lab = "learned" if margin >= tau else "void"
        else:
            lab = "void"
            margin = np.nan
        labels[s] = lab
        rows.append({
            "state_key": s,
            "total_visits": total_visits,
            "max_Q": float(np.nanmax(qarr)) if qarr.size else np.nan,
            "margin": margin,
            "label": lab
        })
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
                token_stats.append({
                    "feature": col,
                    "token": f"{col}:{v}",
                    "states_total": total,
                    "states_void": voids,
                    "void_rate": voids / total
                })
    void_table = pd.DataFrame(token_stats)

    return Q, labels, void_table, state_summary

def prune_by_void_table(X_proc, void_table, threshold=0.5):
    """Prune features whose mean token-void-rate >= threshold."""
    feature_mean = void_table.groupby("feature")["void_rate"].mean()
    pruned_features = feature_mean[feature_mean >= threshold].index.tolist()
    X_pruned = X_proc.drop(columns=pruned_features) if pruned_features else X_proc.copy()
    return X_pruned, pruned_features, feature_mean

def model_grids():
    return {
        "KNN": (make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": [3, 5, 7]}),
        "RF": (RandomForestClassifier(random_state=42),
               {"n_estimators": [100, 200], "max_depth": [None, 10]}),
        "LR": (make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(max_iter=2000, n_jobs=None, random_state=42)),
               {"logisticregression__C": [0.1, 1.0, 10.0]}),
        "SVM": (make_pipeline(StandardScaler(with_mean=False), SVC(probability=False, random_state=42)),
                {"svc__C": [0.5, 1.0, 2.0], "svc__kernel": ["rbf", "linear"]}),
    }

def fit_and_report(X_train, X_test, y_train, y_test, label_prefix="Original"):
    grids = model_grids()
    results = []
    pb = st.progress(0)
    step = 0
    total = len(grids)

    for name, (est, grid) in grids.items():
        step += 1
        with st.spinner(f"Training {name} ({label_prefix}) ..."):
            gs = GridSearchCV(est, grid, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")

            # Confusion matrix
            fig_cm, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax)
            ax.set_title(f"{name} — {label_prefix}")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            # Learning curve (using best estimator)
            fig_lc, ax2 = plt.subplots(figsize=(5, 3))
            train_sizes, train_scores, test_scores = learning_curve(
                gs.best_estimator_, X_train, y_train, cv=3, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=42
            )
            ax2.plot(train_sizes, np.mean(train_scores, axis=1), marker="o", label="Train")
            ax2.plot(train_sizes, np.mean(test_scores, axis=1), marker="o", label="CV")
            ax2.set_title(f"Learning Curve — {name} ({label_prefix})")
            ax2.set_xlabel("Train size"); ax2.set_ylabel("Score")
            ax2.legend()
            st.pyplot(fig_lc)
            plt.close(fig_lc)

            results.append({
                "model": name,
                f"accuracy_{label_prefix.lower()}": acc,
                f"macroF1_{label_prefix.lower()}": f1m
            })
        pb.progress(int(100 * step / total))

    return pd.DataFrame(results)

def df_to_download(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ========== DATA LOADING ==========
if ds_choice == "Iris (built-in)":
    X, y, target_name = load_builtin_iris()
elif ds_choice == "Titanic (seaborn)":
    X, y, target_name = load_builtin_titanic()
    if X is None:
        st.stop()
else:
    df, X, y, target_name = upload_csv()
    if X is None:
        st.info("Please upload a CSV to continue.")
        st.stop()

st.subheader("1) Data Preview")
st.write(f"Target: **{target_name}**")
st.write("Shape:", X.shape)
st.dataframe(pd.concat([X.head(5), y.head(5)], axis=1))

# ========== PREPROCESS/TOKENIZE ==========
st.subheader("2) Preprocessing & Tokenisation")
X_proc, encoders, num_cols, cat_cols = preprocess_tokenize(X, n_bins=n_bins)
st.write("*Numeric columns (binned):*", num_cols)
st.write("*Categorical columns (encoded):*", cat_cols)
st.write("Preview of processed features:")
st.dataframe(X_proc.head(5))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ========== BASELINE CLASSIFIERS ==========
st.subheader("3) Baseline Classifiers (Original Features)")
baseline_df = fit_and_report(X_train, X_test, y_train, y_test, label_prefix="Original")

# ========== VOID-LEARNING ==========
st.subheader("4) Void-Learning (Tabular Q-learning, contextual bandit)")
with st.spinner("Running Void-Learning ..."):
    Q, state_labels, void_table, state_summary = void_learning_bandit(
        X_proc, y, episodes=episodes, alpha=alpha, epsilon=epsilon, vmin=vmin, tau=tau
    )

st.write("State summary (head):")
st.dataframe(state_summary.head(10))

st.write("Token-level void table (head):")
st.dataframe(void_table.head(20))

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    df_to_download(void_table, "void_table_tokens.csv", "Download token-level void table (CSV)")
with col_dl2:
    df_to_download(state_summary, "state_summary.csv", "Download state summary (CSV)")

# ========== PRUNE FEATURES ==========
st.subheader("5) Prune Features by Void Rate")
X_pruned, pruned_features, feature_mean = prune_by_void_table(X_proc, void_table, threshold=prune_thresh)
if pruned_features:
    st.write("**Pruned features (mean void rate ≥ threshold):**", pruned_features)
else:
    st.write("No features met the prune threshold.")

st.write("Mean void rate per feature:")
st.dataframe(feature_mean.reset_index().rename(columns={"void_rate": "mean_void_rate"}))

# Split again for pruned feature set (same indices to align fair comparison)
X_train_p, X_test_p, _, _ = train_test_split(
    X_pruned, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None
)

# ========== CLASSIFIERS ON PRUNED ==========
st.subheader("6) Classifiers (Void-Pruned Features)")
pruned_df = fit_and_report(X_train_p, X_test_p, y_train, y_test, label_prefix="Pruned")

# ========== COMPARISON TABLE ==========
st.subheader("7) Comparison — Original vs Pruned")
if not baseline_df.empty and not pruned_df.empty:
    comp = pd.merge(baseline_df, pruned_df, on="model", how="outer")
    # compute deltas
    comp["Δ accuracy"] = comp["accuracy_pruned"] - comp["accuracy_original"]
    comp["Δ macroF1"] = comp["macroF1_pruned"] - comp["macroF1_original"]
    st.dataframe(comp)
    df_to_download(comp, "comparison_original_vs_pruned.csv", "Download comparison table (CSV)")
else:
    st.write("Insufficient results to compare.")
