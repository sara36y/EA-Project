"""
Binary PSO for feature selection (NumPy + scikit-learn; no DEAP required).

Objective minimized during search: (1 − mean CV accuracy).

CLI:  python pso_feature_selection.py
expects ``results/processed/X_train.npy`` and ``y_train.npy`` next to the project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def _evaluate_pso_loss(
    position: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    clf_template,
    cv_folds: int,
    cv_score_n_jobs: int = -1,
) -> float:
    selected_idx = np.where(np.asarray(position) == 1)[0]
    if len(selected_idx) == 0:
        return 1.0
    clf = clone(clf_template)
    if "n_jobs" in clf.get_params(deep=False):
        clf.set_params(n_jobs=1)
    X_sel = X[:, selected_idx]
    try:
        acc = float(
            np.mean(
                cross_val_score(
                    clf,
                    X_sel,
                    y,
                    cv=cv_folds,
                    scoring="accuracy",
                    n_jobs=cv_score_n_jobs,
                )
            )
        )
    except ValueError:
        acc = 0.0
    return float(1.0 - acc)


def _update_particle_binary(
    position: np.ndarray,
    speed: np.ndarray,
    personal_best: np.ndarray,
    global_best: np.ndarray,
    c1: float,
    c2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = len(position)
    r1 = rng.random(n_features)
    r2 = rng.random(n_features)
    cognitive = c1 * r1 * (personal_best.astype(float) - position.astype(float))
    social = c2 * r2 * (global_best.astype(float) - position.astype(float))
    new_speed = speed + cognitive + social
    prob = 1.0 / (1.0 + np.exp(-new_speed))
    new_position = np.where(rng.random(n_features) < prob, 1, 0).astype(int)
    if np.sum(new_position) == 0:
        new_position[int(rng.integers(0, n_features))] = 1
    return new_position, new_speed


def run_pso_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    clf_template,
    *,
    pop_size: int = 30,
    max_generations: int = 50,
    c1: float = 2.0,
    c2: float = 2.0,
    cv_folds: int = 3,
    cv_score_n_jobs: int = -1,
    seed: int = 42,
    on_generation=None,
) -> tuple[np.ndarray, list[float], list[int], float]:
    rng = np.random.default_rng(seed)

    particles: list[dict] = []
    for _ in range(pop_size):
        pos = rng.integers(0, 2, size=n_features, dtype=int)
        spd = rng.uniform(-1.0, 1.0, size=n_features).astype(float)
        particles.append(
            {
                "position": pos.copy(),
                "speed": spd,
                "personal_best": pos.copy(),
                "best_loss": float("inf"),
            }
        )

    best_global: np.ndarray | None = None
    best_global_loss = float("inf")
    best_acc_history: list[float] = []
    feature_history: list[int] = []

    for gen in range(max_generations):
        for p in particles:
            loss = _evaluate_pso_loss(
                p["position"], X, y, clf_template, cv_folds, cv_score_n_jobs=cv_score_n_jobs
            )
            if loss < p["best_loss"]:
                p["best_loss"] = loss
                p["personal_best"] = p["position"].copy()
            if loss < best_global_loss:
                best_global_loss = loss
                best_global = p["position"].copy()

        assert best_global is not None
        best_acc_history.append(float(1.0 - best_global_loss))
        feature_history.append(int(np.sum(best_global)))

        if on_generation is not None:
            on_generation(
                gen + 1,
                list(best_acc_history),
                list(feature_history),
                float(best_global_loss),
            )

        for p in particles:
            new_pos, new_spd = _update_particle_binary(
                p["position"],
                p["speed"],
                p["personal_best"],
                best_global,
                c1,
                c2,
                rng,
            )
            p["position"] = new_pos
            p["speed"] = new_spd

    assert best_global is not None
    return best_global.astype(int), best_acc_history, feature_history, float(best_global_loss)


def run_pso():
    base = Path(__file__).resolve().parent
    data_path = base / "results" / "processed"
    x_path, y_path = data_path / "X_train.npy", data_path / "y_train.npy"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(
            f"Missing:\n  {x_path}\n  {y_path}\nCopy preprocessing outputs here or change data_path."
        )
    X_train = np.load(x_path)
    y_train = np.load(y_path).ravel()
    n_feat = X_train.shape[1]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    mask, hist_acc, _, loss = run_pso_feature_selection(
        X_train,
        y_train,
        n_feat,
        clf,
        pop_size=30,
        max_generations=50,
        c1=2.0,
        c2=2.0,
        cv_folds=3,
        seed=42,
    )
    print("BEST FEATURE INDICES:", np.where(mask == 1)[0])
    print("NUM FEATURES:", int(np.sum(mask)))
    print("FINAL LOSS (1-acc):", f"{loss:.4f}")
    print("FINAL CV ACC:", f"{hist_acc[-1]:.4f}")

    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, len(hist_acc) + 1), hist_acc)
        plt.xlabel("Generation")
        plt.ylabel("Best CV accuracy")
        plt.title("PSO feature selection")
        plt.savefig(plots_dir / "pso_convergence.png")
        plt.close()
        print(f"Saved: {plots_dir / 'pso_convergence.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    run_pso()
