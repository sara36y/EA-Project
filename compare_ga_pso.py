"""
compare_ga_pso.py — Fair comparative analysis: GA vs PSO

Both algorithms use the IDENTICAL fitness function:
    fitness(x) = alpha * accuracy - beta * (|S| / n) - penalty

Metrics compared:
  • Test accuracy          (higher is better)
  • Number of features selected (lower is better)
  • Feature reduction %    (higher is better)
  • Best fitness score     (higher is better)
  • Convergence speed      (generations to best solution)

Run: python compare_ga_pso.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(DATA_PATH, "y_test.npy"))
N_FEATURES = X_train.shape[1]

# ══════════════════════════════════════════════════════════════════════════
# SHARED FITNESS FUNCTION  ← single source of truth for both algorithms
# ══════════════════════════════════════════════════════════════════════════

SHARED_CONFIG = {
    "alpha"             : 1.0,
    "beta"              : 0.3,
    "max_features_ratio": 0.6,
    "penalty_weight"    : 2.0,
    "cv_folds"          : 3,
    "n_estimators"      : 50,
}


def calculate_fitness(chromosome, X, y, config):
    """
    Shared fitness function — used by BOTH GA and PSO.
    fitness = alpha * accuracy - beta * (|S| / n) - penalty
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
        cv=config["cv_folds"], scoring="accuracy", n_jobs=-1,
    ))

    feature_ratio = len(selected) / n_total
    base_fitness  = config["alpha"] * acc - config["beta"] * feature_ratio

    max_allowed = int(n_total * config["max_features_ratio"])
    penalty = 0.0
    if len(selected) > max_allowed:
        penalty = config["penalty_weight"] * (len(selected) - max_allowed) / n_total

    return base_fitness - penalty


# ══════════════════════════════════════════════════════════════════════════
# GA RUNNER  (self-contained, no external import needed)
# ══════════════════════════════════════════════════════════════════════════

def _init_pop(pop_size, n_feat, seed):
    np.random.seed(seed)
    return np.random.randint(0, 2, (pop_size, n_feat))


def _roulette(pop, fit):
    fit = fit - fit.min() + 1e-6
    return pop[np.random.choice(len(pop), p=fit / fit.sum())]


def _tournament(pop, fit, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fit[idx])]]


def _uniform_cross(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def _adaptive_mutate(ind, gen, max_gen):
    ind  = ind.copy()
    rate = max(0.1 * (1 - gen / max_gen), 0.01)
    ind[np.random.rand(len(ind)) < rate] ^= 1
    if np.sum(ind) == 0:
        ind[np.random.randint(len(ind))] = 1
    return ind


def _fitness_sharing(fitness, population, sigma=0.5):
    shared = fitness.copy().astype(float)
    for i in range(len(population)):
        niche = sum(
            1.0 - (np.sum(population[i] != population[j]) / len(population[i]) / sigma) ** 2
            for j in range(len(population))
            if np.sum(population[i] != population[j]) / len(population[i]) < sigma
        )
        shared[i] = fitness[i] / max(niche, 1.0)
    return shared


def run_ga(config, seed=42, verbose=True):
    np.random.seed(seed)
    pop_size  = config.get("pop_size", 40)
    max_gen   = config.get("generations", 30)
    patience  = config.get("early_stop_patience", 8)

    population       = _init_pop(pop_size, N_FEATURES, seed)
    best_fitness     = -np.inf
    best_solution    = None
    no_improve       = 0
    fitness_history  = []
    feat_history     = []

    for gen in range(max_gen):
        fit_vals = np.array([calculate_fitness(ind, X_train, y_train, config)
                             for ind in population])
        fit_vals = _fitness_sharing(fit_vals, population)

        best_idx = np.argmax(fit_vals)
        if fit_vals[best_idx] > best_fitness:
            best_fitness  = fit_vals[best_idx]
            best_solution = population[best_idx].copy()
            no_improve    = 0
        else:
            no_improve += 1

        fitness_history.append(best_fitness)
        feat_history.append(int(np.sum(population[best_idx])))

        if verbose:
            print(f"GA  Gen {gen+1:3d}/{max_gen} | "
                  f"Fitness: {best_fitness:.4f} | "
                  f"Features: {feat_history[-1]}/{N_FEATURES}")

        if no_improve >= patience:
            if verbose:
                print(f"  GA early stop at generation {gen+1}.")
            break

        new_pop = [population[best_idx].copy()]
        while len(new_pop) < pop_size:
            p1 = _tournament(population, fit_vals)
            p2 = _tournament(population, fit_vals)
            c1, c2 = _uniform_cross(p1, p2)
            c1 = _adaptive_mutate(c1, gen, max_gen)
            c2 = _adaptive_mutate(c2, gen, max_gen)
            new_pop.extend([c1, c2])
        population = np.array(new_pop[:pop_size])

    selected  = np.where(best_solution == 1)[0]
    clf_final = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf_final.fit(X_train[:, selected], y_train)
    test_acc = clf_final.score(X_test[:, selected], y_test)

    return {
        "algorithm"        : "GA",
        "best_solution"    : best_solution,
        "selected_indices" : selected.tolist(),
        "n_selected"       : len(selected),
        "feature_reduction": round(100 * (1 - len(selected) / N_FEATURES), 1),
        "best_fitness"     : best_fitness,
        "test_accuracy"    : test_acc,
        "generations_run"  : len(fitness_history),
        "fitness_history"  : fitness_history,
        "feat_history"     : feat_history,
    }


# ══════════════════════════════════════════════════════════════════════════
# PSO RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_pso(config, seed=42, verbose=True):
    np.random.seed(seed)
    pop_size  = config.get("pop_size", 30)
    max_gen   = config.get("max_generations", 50)
    patience  = config.get("early_stop_patience", 8)
    w, c1, c2 = config.get("w", 0.7), config.get("c1", 2.0), config.get("c2", 2.0)
    v_max     = config.get("v_max", 6.0)

    # Initialise particles as plain dicts (no DEAP dependency here)
    particles = []
    for _ in range(pop_size):
        pos   = np.random.randint(0, 2, N_FEATURES)
        speed = np.random.uniform(-1, 1, N_FEATURES)
        particles.append({
            "pos"         : pos.copy(),
            "speed"       : speed,
            "best_pos"    : pos.copy(),
            "best_fitness": -np.inf,
        })

    best_global        = None
    best_global_fitness = -np.inf
    no_improve         = 0
    fitness_history    = []
    feat_history       = []

    for gen in range(max_gen):
        for p in particles:
            fit = calculate_fitness(p["pos"], X_train, y_train, config)
            if fit > p["best_fitness"]:
                p["best_fitness"] = fit
                p["best_pos"]     = p["pos"].copy()
            if fit > best_global_fitness:
                best_global_fitness = fit
                best_global         = p["pos"].copy()
                no_improve          = 0

        if len(fitness_history) > 0 and best_global_fitness <= fitness_history[-1]:
            no_improve += 1
        else:
            no_improve = 0

        n_sel = int(np.sum(best_global))
        fitness_history.append(best_global_fitness)
        feat_history.append(n_sel)

        if verbose:
            print(f"PSO Gen {gen+1:3d}/{max_gen} | "
                  f"Fitness: {best_global_fitness:.4f} | "
                  f"Features: {n_sel}/{N_FEATURES}")

        if no_improve >= patience:
            if verbose:
                print(f"  PSO early stop at generation {gen+1}.")
            break

        for p in particles:
            r1 = np.random.rand(N_FEATURES)
            r2 = np.random.rand(N_FEATURES)
            cognitive = c1 * r1 * (p["best_pos"] - p["pos"])
            social    = c2 * r2 * (best_global   - p["pos"])
            p["speed"] = np.clip(w * p["speed"] + cognitive + social, -v_max, v_max)
            prob       = 1.0 / (1.0 + np.exp(-p["speed"]))
            p["pos"]   = np.where(np.random.rand(N_FEATURES) < prob, 1, 0)
            if np.sum(p["pos"]) == 0:
                p["pos"][np.random.randint(N_FEATURES)] = 1

    selected  = np.where(best_global == 1)[0]
    clf_final = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf_final.fit(X_train[:, selected], y_train)
    test_acc = clf_final.score(X_test[:, selected], y_test)

    return {
        "algorithm"        : "PSO",
        "best_solution"    : best_global,
        "selected_indices" : selected.tolist(),
        "n_selected"       : len(selected),
        "feature_reduction": round(100 * (1 - len(selected) / N_FEATURES), 1),
        "best_fitness"     : best_global_fitness,
        "test_accuracy"    : test_acc,
        "generations_run"  : len(fitness_history),
        "fitness_history"  : fitness_history,
        "feat_history"     : feat_history,
    }


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════

COLORS = {"GA": "#1D9E75", "PSO": "#E87722"}


def plot_comparison(ga_r, pso_r, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("GA vs PSO — Feature Selection Comparison\n(Identical fitness function for both)",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── 1. Fitness convergence ─────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(ga_r["fitness_history"],  color=COLORS["GA"],  lw=2, label="GA")
    ax1.plot(pso_r["fitness_history"], color=COLORS["PSO"], lw=2, label="PSO")
    ax1.set_title("Fitness Convergence")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. Feature count over generations ─────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(ga_r["feat_history"],  color=COLORS["GA"],  lw=2, label="GA")
    ax2.plot(pso_r["feat_history"], color=COLORS["PSO"], lw=2, label="PSO")
    ax2.axhline(N_FEATURES, color="gray", ls="--", lw=1, label="All features")
    ax2.set_title("Feature Count over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("# Features Selected")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── 3. Test Accuracy bar chart ─────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    algos = ["GA", "PSO"]
    accs  = [ga_r["test_accuracy"], pso_r["test_accuracy"]]
    bars  = ax3.bar(algos, accs, color=[COLORS["GA"], COLORS["PSO"]],
                    width=0.4, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_title("Test Accuracy")
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim(min(accs) - 0.02, 1.0)
    ax3.grid(True, alpha=0.3, axis="y")

    # ── 4. Features selected bar chart ────────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    n_sels = [ga_r["n_selected"], pso_r["n_selected"]]
    bars4  = ax4.bar(algos, n_sels, color=[COLORS["GA"], COLORS["PSO"]],
                     width=0.4, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars4, n_sels):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax4.axhline(N_FEATURES, color="gray", ls="--", lw=1, label=f"All ({N_FEATURES})")
    ax4.set_title("Features Selected")
    ax4.set_ylabel("# Features")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── 5. Best fitness score bar chart ───────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    fits = [ga_r["best_fitness"], pso_r["best_fitness"]]
    bars5 = ax5.bar(algos, fits, color=[COLORS["GA"], COLORS["PSO"]],
                    width=0.4, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars5, fits):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax5.set_title("Best Fitness Score")
    ax5.set_ylabel("Fitness")
    ax5.set_ylim(min(fits) - 0.02, max(fits) + 0.04)
    ax5.grid(True, alpha=0.3, axis="y")

    # ── 6. Summary radar / table ───────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    winner_acc  = "GA" if ga_r["test_accuracy"] >= pso_r["test_accuracy"] else "PSO"
    winner_feat = "GA" if ga_r["n_selected"]    <= pso_r["n_selected"]    else "PSO"
    winner_fit  = "GA" if ga_r["best_fitness"]  >= pso_r["best_fitness"]  else "PSO"
    winner_spd  = ("GA"  if ga_r["generations_run"]  <= pso_r["generations_run"]
                   else "PSO")

    table_data = [
        ["Metric",              "GA",
         f"{ga_r['test_accuracy']:.4f}",
         f"{pso_r['test_accuracy']:.4f}",
         winner_acc],
        ["Test Accuracy",       "",
         f"{ga_r['test_accuracy']:.4f}",
         f"{pso_r['test_accuracy']:.4f}",
         winner_acc],
        ["Features Selected",   "",
         str(ga_r["n_selected"]),
         str(pso_r["n_selected"]),
         winner_feat],
        ["Feature Reduction %", "",
         f"{ga_r['feature_reduction']}%",
         f"{pso_r['feature_reduction']}%",
         winner_feat],
        ["Best Fitness",        "",
         f"{ga_r['best_fitness']:.4f}",
         f"{pso_r['best_fitness']:.4f}",
         winner_fit],
        ["Generations Run",     "",
         str(ga_r["generations_run"]),
         str(pso_r["generations_run"]),
         winner_spd],
    ]

    col_labels  = ["Metric", "GA", "PSO", "Winner"]
    row_labels  = ["Test Accuracy", "Features Selected",
                   "Feature Reduction %", "Best Fitness", "Generations Run"]
    cell_data   = [
        [f"{ga_r['test_accuracy']:.4f}",  f"{pso_r['test_accuracy']:.4f}",  winner_acc],
        [str(ga_r["n_selected"]),          str(pso_r["n_selected"]),          winner_feat],
        [f"{ga_r['feature_reduction']}%",  f"{pso_r['feature_reduction']}%",  winner_feat],
        [f"{ga_r['best_fitness']:.4f}",    f"{pso_r['best_fitness']:.4f}",    winner_fit],
        [str(ga_r["generations_run"]),     str(pso_r["generations_run"]),     winner_spd],
    ]

    tbl = ax6.table(
        cellText   = cell_data,
        rowLabels  = row_labels,
        colLabels  = ["GA", "PSO", "Winner"],
        loc        = "center",
        cellLoc    = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.15, 1.6)

    # Colour winner cells green
    for row_idx, row in enumerate(cell_data):
        winner = row[2]
        col_idx = 0 if winner == "GA" else 1
        tbl[row_idx + 1, col_idx].set_facecolor("#d4f1e4")
        tbl[row_idx + 1, 2].set_facecolor("#fff8e1")

    ax6.set_title("Summary Table", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {save_path}")
    plt.show()
    return fig


def print_comparison_report(ga_r, pso_r):
    sep = "=" * 60
    print(f"\n{sep}")
    print("   GA vs PSO — COMPARATIVE ANALYSIS REPORT")
    print(f"   Fitness function: alpha*acc - beta*(|S|/n) - penalty")
    print(f"   alpha={SHARED_CONFIG['alpha']}, beta={SHARED_CONFIG['beta']}, "
          f"max_features_ratio={SHARED_CONFIG['max_features_ratio']}")
    print(sep)

    header = f"{'Metric':<28} {'GA':>10} {'PSO':>10} {'Winner':>8}"
    print(header)
    print("-" * 60)

    def row(label, ga_val, pso_val, higher_better=True):
        if higher_better:
            w = "GA ✓" if ga_val >= pso_val else "PSO ✓"
        else:
            w = "GA ✓" if ga_val <= pso_val else "PSO ✓"
        print(f"{label:<28} {str(ga_val):>10} {str(pso_val):>10} {w:>8}")

    row("Test Accuracy",        f"{ga_r['test_accuracy']:.4f}",
                                f"{pso_r['test_accuracy']:.4f}", higher_better=True)
    row("Features Selected",    ga_r["n_selected"],
                                pso_r["n_selected"],             higher_better=False)
    row("Feature Reduction %",  f"{ga_r['feature_reduction']}%",
                                f"{pso_r['feature_reduction']}%", higher_better=True)
    row("Best Fitness Score",   f"{ga_r['best_fitness']:.4f}",
                                f"{pso_r['best_fitness']:.4f}",  higher_better=True)
    row("Generations Run",      ga_r["generations_run"],
                                pso_r["generations_run"],        higher_better=False)

    print(sep)
    print("\nSelected Features:")
    print(f"  GA  ({ga_r['n_selected']:2d}): {ga_r['selected_indices']}")
    print(f"  PSO ({pso_r['n_selected']:2d}): {pso_r['selected_indices']}")

    overlap = set(ga_r["selected_indices"]) & set(pso_r["selected_indices"])
    print(f"\n  Common features ({len(overlap)}): {sorted(overlap)}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    GA_RUN_CONFIG = {
        **SHARED_CONFIG,
        "pop_size"           : 40,
        "generations"        : 30,
        "early_stop_patience": 8,
    }

    PSO_RUN_CONFIG = {
        **SHARED_CONFIG,
        "pop_size"           : 30,
        "max_generations"    : 50,
        "early_stop_patience": 8,
        "w"                  : 0.7,
        "c1"                 : 2.0,
        "c2"                 : 2.0,
        "v_max"              : 6.0,
        "random_seed"        : 42,
    }

    print("Running GA …")
    ga_results = run_ga(GA_RUN_CONFIG, seed=42, verbose=True)

    print("\nRunning PSO …")
    pso_results = run_pso(PSO_RUN_CONFIG, seed=42, verbose=True)

    print_comparison_report(ga_results, pso_results)

    plot_comparison(
        ga_results, pso_results,
        save_path=os.path.join(PLOTS_DIR, "ga_pso_comparison.png"),
    )
