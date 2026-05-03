"""
Streamlit dashboard for metaheuristic feature selection (GA or PSO). Expects preprocessed
NumPy arrays: X_train.npy, X_test.npy, y_train.npy, y_test.npy  (+ optional scaler.pkl).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_val_score

from pso_feature_selection import run_pso_feature_selection

# ---------------------------------------------------------------------------
# Paths — same layout as notebook: .../processed/*.npy
# Override with env STREAMLIT_DATA_DIR or place arrays next to this file under results/processed
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("STREAMLIT_DATA_DIR", APP_DIR / "results" / "processed"))

BREAST_CANCER_FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def load_arrays(data_dir: Path):
    """Load train/test arrays; raises FileNotFoundError with a clear message if missing."""
    data_dir = Path(data_dir)
    req = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    missing = [f for f in req if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {data_dir}: {missing}. Run the preprocessing cell in the notebook first."
        )
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy").ravel()
    y_test = np.load(data_dir / "y_test.npy").ravel()
    return X_train, X_test, y_train, y_test


def get_feature_names(n_features: int, data_dir: Path):
    """
    Load feature names if available; otherwise fallback to breast-cancer defaults,
    then generic names.
    """
    data_dir = Path(data_dir)
    candidates = [
        data_dir / "feature_names.npy",
        data_dir / "feature_names.txt",
        data_dir / "feature_names.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            names = np.load(path, allow_pickle=True).tolist()
        elif path.suffix == ".txt":
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            df = pd.read_csv(path)
            if "feature_name" in df.columns:
                names = df["feature_name"].astype(str).tolist()
            else:
                names = df.iloc[:, 0].astype(str).tolist()
        if len(names) == n_features:
            return names

    if n_features == len(BREAST_CANCER_FEATURE_NAMES):
        return BREAST_CANCER_FEATURE_NAMES
    return [f"Feature {i}" for i in range(n_features)]


# ---------------------------------------------------------------------------
# GA core (aligned with notebook)
# ---------------------------------------------------------------------------
def initialize_population(pop_size, n_features, method="uniform", seed=None):
    if seed is not None:
        np.random.seed(seed)
    if method == "uniform":
        return np.random.randint(0, 2, size=(pop_size, n_features))
    if method == "sparse":
        prob_one = 0.2
        return np.random.choice([0, 1], size=(pop_size, n_features), p=[1 - prob_one, prob_one])
    raise ValueError("method must be 'uniform' or 'sparse'")


def calculate_fitness(chromosome, X, y, config, seed=None):
    if seed is not None:
        np.random.seed(seed)

    selected_idx = np.where(chromosome == 1)[0]
    n_selected = len(selected_idx)
    n_total = len(chromosome)

    if n_selected == 0:
        return -10.0

    X_subset = X[:, selected_idx]
    clf = clone(config["classifier"])
    if "n_jobs" in clf.get_params(deep=False):
        clf.set_params(n_jobs=1)
    try:
        acc = np.mean(
            cross_val_score(
                clf,
                X_subset,
                y,
                cv=config["cv_folds"],
                scoring="accuracy",
                n_jobs=config.get("cv_score_n_jobs", -1),
            )
        )
    except ValueError:
        acc = 0.0

    feature_ratio = n_selected / n_total
    base_fitness = config["alpha"] * acc - config["beta"] * feature_ratio

    max_allowed = int(n_total * config["max_features_ratio"])
    penalty = 0.0
    if n_selected > max_allowed:
        excess_ratio = (n_selected - max_allowed) / n_total
        penalty = config["penalty_weight"] * excess_ratio

    return base_fitness - penalty


def calculate_fitness_and_accuracy(chromosome, X, y, config, seed=None):
    if seed is not None:
        np.random.seed(seed)

    selected_idx = np.where(chromosome == 1)[0]
    n_selected = len(selected_idx)
    n_total = len(chromosome)

    if n_selected == 0:
        return -10.0, 0.0, 0

    X_subset = X[:, selected_idx]
    clf = clone(config["classifier"])
    if "n_jobs" in clf.get_params(deep=False):
        clf.set_params(n_jobs=1)
    try:
        acc = float(
            np.mean(
                cross_val_score(
                    clf,
                    X_subset,
                    y,
                    cv=config["cv_folds"],
                    scoring="accuracy",
                    n_jobs=config.get("cv_score_n_jobs", -1),
                )
            )
        )
    except ValueError:
        acc = 0.0

    feature_ratio = n_selected / n_total
    base_fitness = config["alpha"] * acc - config["beta"] * feature_ratio

    max_allowed = int(n_total * config["max_features_ratio"])
    penalty = 0.0
    if n_selected > max_allowed:
        excess_ratio = (n_selected - max_allowed) / n_total
        penalty = config["penalty_weight"] * excess_ratio

    return base_fitness - penalty, acc, n_selected


def evaluate_population(population, X, y, config, seed=None):
    rows = [calculate_fitness_and_accuracy(ind, X, y, config, seed) for ind in population]
    fitness = np.array([r[0] for r in rows], dtype=float)
    accuracy = np.array([r[1] for r in rows], dtype=float)
    n_selected = np.array([r[2] for r in rows], dtype=int)
    return fitness, accuracy, n_selected


def roulette_wheel_selection(population, fitness):
    fitness = np.array(fitness)
    fitness = fitness - fitness.min() + 1e-6
    probs = fitness / fitness.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


def tournament_selection(population, fitness, k=3):
    selected_idx = np.random.choice(len(population), k, replace=False)
    best = selected_idx[np.argmax(fitness[selected_idx])]
    return population[best]


def single_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def uniform_crossover(parent1, parent2, prob=0.5):
    mask = np.random.rand(len(parent1)) < prob
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def bit_flip_mutation(individual, mutation_rate=0.02):
    individual = individual.copy()
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    if np.sum(individual) == 0:
        individual[np.random.randint(0, len(individual))] = 1
    return individual


def adaptive_mutation(individual, generation, max_generations, base_rate=0.1, min_rate=0.01):
    individual = individual.copy()
    mutation_rate = base_rate * (1 - generation / max_generations)
    mutation_rate = max(mutation_rate, min_rate)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    if np.sum(individual) == 0:
        individual[np.random.randint(0, len(individual))] = 1
    return individual, mutation_rate


def apply_fitness_sharing(fitness, population, sigma=0.5):
    shared = fitness.copy().astype(float)
    n = len(population)
    for i in range(n):
        niche_count = 0.0
        for j in range(n):
            dist = np.sum(population[i] != population[j]) / len(population[i])
            if dist < sigma:
                niche_count += 1.0 - (dist / sigma) ** 2
        shared[i] = fitness[i] / max(niche_count, 1.0)
    return shared


def run_ga_with_sharing(
    config,
    X,
    y,
    n_features,
    seed,
    sigma=0.5,
    on_generation=None,
):
    """
    on_generation(gen_1based, best_accuracy_history, feature_count_history, best_raw_fitness_so_far)
    called after each generation’s stats are recorded.
    """
    np.random.seed(seed)

    pop_size = config["pop_size"]
    generations = config["generations"]
    patience = config.get("early_stop_patience", generations)

    population = initialize_population(pop_size, n_features, method="uniform", seed=seed)
    best_acc_history = []
    feature_history = []
    best_solution = None
    best_fitness = -np.inf
    no_improve = 0

    for gen in range(generations):
        raw_fitness, raw_accuracy, n_selected = evaluate_population(population, X, y, config, seed)
        shared_fitness = apply_fitness_sharing(raw_fitness, population, sigma=sigma)

        best_idx = np.argmax(raw_fitness)
        best_individual = population[best_idx]

        if raw_fitness[best_idx] > best_fitness:
            best_fitness = float(raw_fitness[best_idx])
            best_solution = best_individual.copy()
            no_improve = 0
        else:
            no_improve += 1

        best_acc_history.append(float(raw_accuracy[best_idx]))
        feature_history.append(int(n_selected[best_idx]))

        if on_generation is not None:
            on_generation(
                gen + 1,
                list(best_acc_history),
                list(feature_history),
                best_fitness,
            )

        if no_improve >= patience:
            break

        new_population = [best_individual.copy()]

        while len(new_population) < pop_size:
            if config["selection"] == "tournament":
                p1 = tournament_selection(population, shared_fitness)
                p2 = tournament_selection(population, shared_fitness)
            else:
                p1 = roulette_wheel_selection(population, shared_fitness)
                p2 = roulette_wheel_selection(population, shared_fitness)

            if config["crossover"] == "single":
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = uniform_crossover(p1, p2)

            if config["mutation"] == "bitflip":
                c1 = bit_flip_mutation(c1)
                c2 = bit_flip_mutation(c2)
            else:
                c1, _ = adaptive_mutation(c1, gen, generations)
                c2, _ = adaptive_mutation(c2, gen, generations)

            new_population.extend([c1, c2])

        population = np.array(new_population[:pop_size])

    return best_solution, best_acc_history, feature_history, best_fitness


def test_accuracy_for_mask(X_tr, y_tr, X_te, y_te, mask, clf_template):
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return 0.0
    clf = clone(clf_template)
    clf.fit(X_tr[:, idx], y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te[:, idx])))


def evaluate_mask_metrics(X_tr, y_tr, X_te, y_te, mask, clf_template, malignant_label):
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return 0.0, 0.0
    clf = clone(clf_template)
    clf.fit(X_tr[:, idx], y_tr)
    y_pred = clf.predict(X_te[:, idx])
    acc = float(accuracy_score(y_te, y_pred))
    recall_malignant = float(recall_score(y_te, y_pred, pos_label=malignant_label))
    return acc, recall_malignant


def build_fig_feature_bars(mask: np.ndarray, feature_names: list[str]) -> plt.Figure:
    selected_idx = np.where(mask == 1)[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(selected_idx) == 0:
        ax.text(0.5, 0.5, "No selected features", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    labels = [f"{i} - {feature_names[i]}" for i in selected_idx]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, np.ones(len(labels)), color="#2ecc71", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Selected")
    ax.set_title("Selected feature names")
    ax.set_xlim(0, 1.2)
    ax.set_xticks([1.0])
    fig.tight_layout()
    return fig


def fmt_float(v: float, digits: int = 4) -> str:
    return f"{float(v):.{digits}f}"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Feature selection (GA / PSO)", layout="wide")
st.title("Feature selection — GA or PSO")
st.caption("Loads preprocessed arrays from the notebook pipeline (`results/processed` by default).")

data_dir_input = st.sidebar.text_input("Data directory", value=str(DEFAULT_DATA_DIR))

try:
    X_train, X_test, y_train, y_test = load_arrays(Path(data_dir_input))
    n_features = X_train.shape[1]
    feature_names = get_feature_names(n_features, Path(data_dir_input))
    st.sidebar.success(f"Loaded: {X_train.shape[0]} train / {X_test.shape[0]} test / {n_features} features")
except FileNotFoundError as e:
    st.sidebar.error(str(e))
    st.stop()

algorithm = st.sidebar.radio(
    "Optimization algorithm",
    ["GA", "PSO"],
    horizontal=True,
    help="GA: crossover/mutation/fitness sharing. PSO: binary swarm (`pso_feature_selection.py`).",
)

st.sidebar.header("Search parameters")
pop_size = st.sidebar.slider(
    "Population / swarm size",
    min_value=10,
    max_value=120,
    value=40 if algorithm == "GA" else 30,
    step=5,
)
generations = st.sidebar.number_input("Number of generations / iterations", min_value=1, max_value=200, value=30, step=1)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

if algorithm == "GA":
    st.sidebar.subheader("GA fitness weights")
    alpha = st.sidebar.number_input("Alpha (accuracy weight)", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
    beta = st.sidebar.number_input("Beta (feature penalty weight)", min_value=0.0, max_value=5.0, value=0.3, step=0.05)
    max_features_ratio = st.sidebar.slider("Max feature ratio", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
    early_stop_patience = st.sidebar.number_input(
        "Early stop patience (generations)",
        min_value=1,
        max_value=generations,
        value=min(8, int(generations)),
        step=1,
    )
    st.sidebar.subheader("Operator configuration (notebook defaults)")
    selection = st.sidebar.selectbox("Selection", ["tournament", "roulette"], index=0)
    crossover = st.sidebar.selectbox("Crossover", ["single", "uniform"], index=0)
    mutation = st.sidebar.selectbox("Mutation", ["bitflip", "adaptive"], index=0)
else:
    st.sidebar.subheader("PSO coefficients")
    c1 = st.sidebar.number_input("c1 (cognitive)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    c2 = st.sidebar.number_input("c2 (social)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
labels = sorted(np.unique(y_train).tolist())
default_malignant = 0 if 0 in labels else labels[0]
malignant_label = st.sidebar.selectbox("Malignant class label", options=labels, index=labels.index(default_malignant))

with st.sidebar.expander("Faster runs (fitness model & CV)", expanded=False):
    st.caption(
        "Each generation evaluates many random forests with cross-validation. "
        "Lower trees / fewer folds / parallel CV drastically cuts wall time (often 5–10×)."
    )
    cv_folds_ui = st.selectbox("CV folds", options=[2, 3], index=1, help="2 folds ≈ 33% fewer fits than 3.")
    rf_n_estimators = st.slider(
        "RF trees during search",
        min_value=25,
        max_value=300,
        value=50,
        step=25,
        help="300 trees ≈ 6× slower per evaluation than 50; quality often similar for ranking masks.",
    )
    rf_max_depth = st.slider(
        "RF max_depth",
        min_value=4,
        max_value=32,
        value=12,
        step=1,
        help="Caps tree depth; unlimited depth is much slower.",
    )
    parallel_cv = st.checkbox(
        "Parallel CV across folds (recommended)",
        value=True,
        help="Uses multiple CPU cores for each cross_val_score call.",
    )

run_clicked = st.sidebar.button("Run algorithm", type="primary")

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if run_clicked:
    progress = st.progress(0.0)
    status = st.empty()

    clf = RandomForestClassifier(
        n_estimators=int(rf_n_estimators),
        max_depth=int(rf_max_depth),
        min_samples_leaf=2,
        n_jobs=1,
        random_state=42,
    )
    cv_score_n_jobs = -1 if parallel_cv else 1

    def on_generation_progress(gen: int, max_gen: int):
        progress.progress(min(gen / max_gen, 1.0))

    if algorithm == "GA":
        config = {
            "algorithm": "GA",
            "random_seed": int(random_seed),
            "pop_size": int(pop_size),
            "alpha": float(alpha),
            "beta": float(beta),
            "max_features_ratio": float(max_features_ratio),
            "penalty_weight": 2.0,
            "cv_folds": int(cv_folds_ui),
            "cv_score_n_jobs": cv_score_n_jobs,
            "classifier": clf,
            "generations": int(generations),
            "early_stop_patience": int(early_stop_patience),
            "selection": selection,
            "crossover": crossover,
            "mutation": mutation,
            "rf_n_estimators": int(rf_n_estimators),
            "rf_max_depth": int(rf_max_depth),
            "parallel_cv": parallel_cv,
        }

        def on_generation_ga(gen, hist, feat_hist, best_so_far):
            on_generation_progress(gen, config["generations"])
            status.markdown(f"**Best fitness (raw so far):** {fmt_float(best_so_far)}")

        best_mask, hist, feat_hist, final_best_fitness = run_ga_with_sharing(
            config,
            X_train,
            y_train,
            n_features,
            int(random_seed),
            sigma=0.5,
            on_generation=on_generation_ga,
        )
        final_pso_loss = None
    else:
        config = {
            "algorithm": "PSO",
            "random_seed": int(random_seed),
            "pop_size": int(pop_size),
            "generations": int(generations),
            "c1": float(c1),
            "c2": float(c2),
            "cv_folds": int(cv_folds_ui),
            "cv_score_n_jobs": cv_score_n_jobs,
            "rf_n_estimators": int(rf_n_estimators),
            "rf_max_depth": int(rf_max_depth),
            "parallel_cv": parallel_cv,
        }

        def on_generation_pso(gen, hist, feat_hist, best_loss_so_far):
            on_generation_progress(gen, config["generations"])
            status.markdown(f"**Best PSO objective (1 − CV accuracy):** {fmt_float(best_loss_so_far)}")

        best_mask, hist, feat_hist, final_pso_loss = run_pso_feature_selection(
            X_train,
            y_train,
            n_features,
            clf,
            pop_size=config["pop_size"],
            max_generations=config["generations"],
            c1=config["c1"],
            c2=config["c2"],
            cv_folds=config["cv_folds"],
            cv_score_n_jobs=config["cv_score_n_jobs"],
            seed=config["random_seed"],
            on_generation=on_generation_pso,
        )
        final_best_fitness = None

    progress.progress(1.0)

    baseline_acc, baseline_recall_malignant = evaluate_mask_metrics(
        X_train,
        y_train,
        X_test,
        y_test,
        np.ones(n_features),
        clf,
        malignant_label,
    )
    selected_acc, selected_recall_malignant = evaluate_mask_metrics(
        X_train,
        y_train,
        X_test,
        y_test,
        best_mask,
        clf,
        malignant_label,
    )
    n_selected = int(np.sum(best_mask))

    fig = build_fig_feature_bars(best_mask, feature_names)
    st.pyplot(fig)
    plt.close(fig)

    st.session_state.last_run = {
        "algorithm": algorithm,
        "best_mask": best_mask,
        "hist": hist,
        "feat_hist": feat_hist,
        "final_best_fitness": final_best_fitness,
        "final_pso_loss": final_pso_loss,
        "baseline_acc": baseline_acc,
        "selected_acc": selected_acc,
        "baseline_recall_malignant": baseline_recall_malignant,
        "selected_recall_malignant": selected_recall_malignant,
        "n_selected": n_selected,
        "config": config,
        "malignant_label": malignant_label,
        "feature_names": feature_names,
    }

result = st.session_state.last_run
if result is not None:
    run_algo = result.get("algorithm", "GA")
    subset_label = f"{run_algo}-selected subset"

    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    if run_algo == "PSO" and result.get("final_pso_loss") is not None:
        m1.metric("Best CV accuracy (during search)", fmt_float(1.0 - float(result["final_pso_loss"])))
    elif result.get("final_best_fitness") is not None:
        m1.metric("Best fitness (final best raw in run)", fmt_float(float(result["final_best_fitness"])))
    else:
        m1.metric("Optimization score", "—")
    m2.metric("Test accuracy (selected features)", fmt_float(result["selected_acc"]))
    m3.metric("Features selected", f"{result['n_selected']} / {n_features}")
    m4.metric(
        f"Recall (malignant class={result['malignant_label']})",
        fmt_float(result["selected_recall_malignant"]),
    )

    hist = result.get("hist") or []
    feat_hist = result.get("feat_hist") or []
    if len(hist) > 0 and len(feat_hist) > 0:
        st.subheader("Optimization history (final run)")
        ch_left, ch_right = st.columns(2)
        with ch_left:
            df_line = pd.DataFrame(
                {"accuracy": hist},
                index=range(1, len(hist) + 1),
            )
            df_line.index.name = "generation"
            st.line_chart(df_line)
        with ch_right:
            df_feat = pd.DataFrame(
                {"features": feat_hist},
                index=range(1, len(feat_hist) + 1),
            )
            df_feat.index.name = "generation"
            st.line_chart(df_feat)

    st.subheader("Before vs after feature selection (same RF, test set)")
    cmp = pd.DataFrame(
        {
            "Setting": ["All features", subset_label],
            "Test accuracy": [fmt_float(result["baseline_acc"]), fmt_float(result["selected_acc"])],
            f"Recall malignant (class={result['malignant_label']})": [
                fmt_float(result["baseline_recall_malignant"]),
                fmt_float(result["selected_recall_malignant"]),
            ],
            "Feature count": [n_features, result["n_selected"]],
        }
    )
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    same_mask_as_all_features = bool(result["n_selected"] == n_features)
    if same_mask_as_all_features:
        st.info(
            "The optimizer used **every** feature (selected count equals total). "
            "Baseline and optimizer rows train on the **same columns**, so identical accuracy and recall are expected — not an error."
        )
    elif math.isclose(
        result["baseline_acc"],
        result["selected_acc"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        result["baseline_recall_malignant"],
        result["selected_recall_malignant"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        st.caption(
            "Metrics match with a strict subset because, on this test split, "
            "the random forest achieved the **same predictions** — the dropped features "
            "did not change classification here."
        )

    st.subheader("Run configuration")
    cfg = result["config"]
    run_seed = int(cfg.get("random_seed", random_seed))
    def _fitness_model_rows():
        rfe = cfg.get("rf_n_estimators")
        rfd = cfg.get("rf_max_depth")
        par = cfg.get("parallel_cv")
        return [
            "RF n_estimators (search)",
            "RF max_depth (search)",
            "Parallel CV",
        ], [
            rfe if rfe is not None else "—",
            rfd if rfd is not None else "—",
            "Yes" if par is True else "No" if par is False else "—",
        ]

    if cfg.get("algorithm") == "PSO":
        extra_p, extra_v = _fitness_model_rows()
        cfg_df = pd.DataFrame(
            {
                "Parameter": [
                    "Algorithm",
                    "Swarm size",
                    "Iterations",
                    "c1 (cognitive)",
                    "c2 (social)",
                    *extra_p,
                    "CV folds",
                    "Random seed",
                ],
                "Value": [
                    "PSO",
                    cfg["pop_size"],
                    cfg["generations"],
                    fmt_float(cfg["c1"]),
                    fmt_float(cfg["c2"]),
                    *extra_v,
                    cfg["cv_folds"],
                    run_seed,
                ],
            }
        )
    else:
        extra_p, extra_v = _fitness_model_rows()
        cfg_df = pd.DataFrame(
            {
                "Parameter": [
                    "Algorithm",
                    "Selection method",
                    "Crossover type",
                    "Mutation type",
                    "Population size",
                    "Generations",
                    "Early stop patience",
                    "Alpha",
                    "Beta",
                    "Max features ratio",
                    "Penalty weight",
                    *extra_p,
                    "CV folds",
                    "Random seed",
                ],
                "Value": [
                    "GA",
                    cfg["selection"],
                    cfg["crossover"],
                    cfg["mutation"],
                    cfg["pop_size"],
                    cfg["generations"],
                    cfg["early_stop_patience"],
                    fmt_float(cfg["alpha"]),
                    fmt_float(cfg["beta"]),
                    fmt_float(cfg["max_features_ratio"]),
                    fmt_float(cfg["penalty_weight"]),
                    *extra_v,
                    cfg["cv_folds"],
                    run_seed,
                ],
            }
        )
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)

    st.subheader("Selected feature names")
    selected_indices = np.where(result["best_mask"] == 1)[0]
    selected_features_df = pd.DataFrame(
        {
            "Feature index": selected_indices,
            "Feature name": [result["feature_names"][i] for i in selected_indices],
        }
    )
    st.dataframe(selected_features_df, use_container_width=True, hide_index=True)

elif not run_clicked:
    st.info('Set parameters in the sidebar and click **Run algorithm**.')
