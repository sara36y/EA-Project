"""
pso_feature_selection.py — PSO Feature Selection
Fitness function aligned with the GA model:
    fitness(x) = alpha * accuracy - beta * (|S| / n) - penalty
where penalty activates when |S| > max_features_ratio * n.

This makes PSO vs GA comparisons fair and consistent.
"""

import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── Paths ──────────────────────────────────────────────────────────────────
# Project root = folder containing this file (same layout as Streamlit app: results/processed/*.npy)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results", "processed")

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_PATH, "y_test.npy"))
n_features = X_train.shape[1]

# ══════════════════════════════════════════════════════════════════════════
# SHARED CONFIG  — identical values to GA so comparisons are apples-to-apples
# ══════════════════════════════════════════════════════════════════════════
PSO_CONFIG = {
    "alpha"             : 1.0,   # weight on accuracy (same as GA)
    "beta"              : 0.3,   # penalty weight on feature ratio (same as GA)
    "max_features_ratio": 0.6,   # hard cap: max 60 % of features (same as GA)
    "penalty_weight"    : 2.0,   # penalty multiplier for exceeding cap (same as GA)
    "cv_folds"          : 3,     # cross-validation folds (same as GA)
    "n_estimators"      : 50,    # RF trees during search (same as GA)
    # PSO-specific hyper-parameters
    "pop_size"          : 30,
    "max_generations"   : 50,
    "w"                 : 0.7,   # inertia weight
    "c1"                : 2.0,   # cognitive coefficient
    "c2"                : 2.0,   # social coefficient
    "v_max"             : 6.0,   # velocity clamp
    "random_seed"       : 42,
    "early_stop_patience": 8,    # stop if no improvement for N generations
}

# ══════════════════════════════════════════════════════════════════════════
# SHARED FITNESS FUNCTION  — exact mirror of GA's calculate_fitness()
# ══════════════════════════════════════════════════════════════════════════

def calculate_fitness(chromosome, X, y, config):
    """
    Returns a scalar fitness (higher = better), matching the GA formula:

        fitness = alpha * accuracy - beta * (|S| / n) - penalty

    penalty = penalty_weight * (|S| - max_allowed) / n  if |S| > max_allowed
            = 0                                          otherwise
    """
    selected = np.where(np.array(chromosome) == 1)[0]
    n_total  = len(chromosome)

    if len(selected) == 0:
        return -10.0

    clf = RandomForestClassifier(
        n_estimators   = config["n_estimators"],
        max_depth      = 8,
        min_samples_leaf = 3,
        n_jobs         = -1,
        random_state   = 42,
    )
    acc = np.mean(cross_val_score(
        clf, X[:, selected], y,
        cv      = config["cv_folds"],
        scoring = "accuracy",
        n_jobs  = -1,
    ))

    feature_ratio = len(selected) / n_total
    base_fitness  = config["alpha"] * acc - config["beta"] * feature_ratio

    max_allowed = int(n_total * config["max_features_ratio"])
    penalty = 0.0
    if len(selected) > max_allowed:
        penalty = config["penalty_weight"] * (len(selected) - max_allowed) / n_total

    return base_fitness - penalty


# ══════════════════════════════════════════════════════════════════════════
# DEAP SETUP
# ══════════════════════════════════════════════════════════════════════════

# Guard against duplicate creator registrations (e.g., when re-running)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))   # maximise
if not hasattr(creator, "Particle"):
    creator.create("Particle", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def create_particle(seed=None):
    rng = np.random.default_rng(seed)
    particle = creator.Particle(rng.integers(0, 2, size=n_features).tolist())
    particle.speed        = rng.uniform(-1, 1, n_features)
    particle.best         = particle[:]
    particle.best_fitness = -np.inf
    return particle


def update_particle(particle, best_global, config):
    """S-shaped binary PSO velocity update."""
    w, c1, c2, v_max = config["w"], config["c1"], config["c2"], config["v_max"]
    r1 = np.random.rand(n_features)
    r2 = np.random.rand(n_features)

    cognitive = c1 * r1 * (np.array(particle.best) - np.array(particle))
    social    = c2 * r2 * (np.array(best_global)   - np.array(particle))

    particle.speed = w * particle.speed + cognitive + social
    particle.speed = np.clip(particle.speed, -v_max, v_max)

    prob      = 1.0 / (1.0 + np.exp(-particle.speed))
    particle[:] = np.where(np.random.rand(n_features) < prob, 1, 0).tolist()

    # Ensure at least one feature is selected
    if np.sum(particle) == 0:
        particle[np.random.randint(n_features)] = 1


# ══════════════════════════════════════════════════════════════════════════
# MAIN PSO LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_pso(config=None, verbose=True):
    if config is None:
        config = PSO_CONFIG

    np.random.seed(config["random_seed"])
    pop_size      = config["pop_size"]
    max_gen       = config["max_generations"]
    patience      = config["early_stop_patience"]

    population         = [create_particle() for _ in range(pop_size)]
    best_global        = None
    best_global_fitness = -np.inf
    no_improve         = 0

    fitness_history = []   # best fitness per generation
    feat_history    = []   # #features of best particle per generation

    for gen in range(max_gen):
        for particle in population:
            fit = calculate_fitness(particle, X_train, y_train, config)
            particle.fitness.values = (fit,)

            # Update personal best
            if fit > particle.best_fitness:
                particle.best         = particle[:]
                particle.best_fitness = fit

            # Update global best
            if fit > best_global_fitness:
                best_global         = particle[:]
                best_global_fitness = fit
                no_improve          = 0

        # Track improvement for early stopping
        if len(fitness_history) > 0 and best_global_fitness <= fitness_history[-1]:
            no_improve += 1
        else:
            no_improve = 0

        n_sel = int(np.sum(best_global))
        fitness_history.append(best_global_fitness)
        feat_history.append(n_sel)

        if verbose:
            print(f"Gen {gen+1:3d}/{max_gen} | Fitness: {best_global_fitness:.4f} | "
                  f"Features: {n_sel}/{n_features} | "
                  f"No-improve: {no_improve}")

        # Early stopping
        if no_improve >= patience:
            if verbose:
                print(f"  Early stop at generation {gen+1}.")
            break

        # Velocity + position update
        for particle in population:
            update_particle(particle, best_global, config)

    # ── Final evaluation on test set ──────────────────────────────────────
    selected = np.where(np.array(best_global) == 1)[0]
    clf_final = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf_final.fit(X_train[:, selected], y_train)
    test_acc = clf_final.score(X_test[:, selected], y_test)

    results = {
        "best_solution"    : best_global,
        "selected_indices" : selected.tolist(),
        "n_selected"       : len(selected),
        "n_total"          : n_features,
        "feature_reduction": round(100 * (1 - len(selected) / n_features), 1),
        "best_fitness"     : best_global_fitness,
        "test_accuracy"    : test_acc,
        "generations_run"  : len(fitness_history),
        "fitness_history"  : fitness_history,
        "feat_history"     : feat_history,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("PSO RESULTS")
        print("=" * 50)
        print(f"  Test Accuracy      : {test_acc:.4f}")
        print(f"  Features Selected  : {len(selected)} / {n_features} "
              f"({results['feature_reduction']} % reduction)")
        print(f"  Best Fitness Score : {best_global_fitness:.4f}")
        print(f"  Generations Run    : {len(fitness_history)}")
        print(f"  Selected Indices   : {selected.tolist()}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION  (standalone PSO convergence)
# ══════════════════════════════════════════════════════════════════════════

def plot_pso_convergence(results, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(results["fitness_history"], color="#E87722", linewidth=2,
             marker="o", markersize=3)
    ax1.set_title("PSO — Fitness over generations", fontsize=12)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best fitness")
    ax1.grid(True, alpha=0.3)

    ax2.plot(results["feat_history"], color="#3A86FF", linewidth=2,
             marker="s", markersize=3)
    ax2.axhline(n_features, color="gray", linestyle="--",
                linewidth=1, label="All features")
    ax2.set_title("PSO — Feature count over generations", fontsize=12)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Selected features")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
    return fig


if __name__ == "__main__":
    results = run_pso()
    PLOTS_PATH = os.path.join(BASE_DIR, "plots")
    plot_pso_convergence(
        results,
        save_path=os.path.join(PLOTS_PATH, "pso_convergence.png"),
    )
