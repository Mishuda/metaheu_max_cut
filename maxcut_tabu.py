import time
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 4)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def derive_seeds(seed):
    # Standardized seed offsets per algorithm/instance.
    return {
        "instance_seed": seed,
        "algo_seed": seed + 10_000,
        "random_seed": seed + 20_000,
        "greedy_seed": seed + 30_000,
    }


def generate_graph(n, p, weighted=True, w_low=1, w_high=10, seed=None):
    #to generate an Erdos-Renyi graph and optionally integer edge weights.
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    if weighted:
        rng = np.random.default_rng(seed)
        for u, v in G.edges():
            G[u][v]["weight"] = int(rng.integers(w_low, w_high + 1))
    return G


def cut_value(G, x, node_list=None, node_index=None):
    #calculate cut value by iterating edges: works for both weighted/unweighted graphs!
    if node_list is None:
        node_list = list(G.nodes())
    if node_index is None:
        node_index = {v: i for i, v in enumerate(node_list)}
    total = 0
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        if x[node_index[u]] != x[node_index[v]]:
            total += w
    return total

def delta_init(G, x, node_list=None, node_index=None):
    # Initial gain vector: delta[v] = F(x with v flipped) - F(x).
    if node_list is None:
        node_list = list(G.nodes())
    if node_index is None:
        node_index = {v: i for i, v in enumerate(node_list)}

    n = len(node_list)
    delta = np.zeros(n, dtype=np.int64)
    for v in node_list:
        iv = node_index[v]
        s = 0
        for u, data in G.adj[v].items():
            w = data.get("weight", 1)
            iu = node_index[u]
            s += w * (1 if x[iu] == x[iv] else -1)
        delta[iv] = s
    return delta

def tabu_search_maxcut(
    G,
    max_iters=5000,
    tenure=10,
    seed=0,
    patience=1000,
    sanity_check=False,
    check_every=100,
):
    # Tabu Search for Max-Cut.
    # - x is a 0/1 numpy vector aligned with node_list
    # - delta[v] = gain if we flip v
    # - tabu_until[v] stores the earliest iteration when v becomes admissible
    # - tenure is randomized each move (use int for [1, tenure] or tuple/list for [lo, hi])
    if isinstance(tenure, (tuple, list)):
        if len(tenure) != 2 or tenure[0] <= 0 or tenure[1] <= 0:
            raise ValueError("tenure must be a positive int or a (low, high) pair of positive ints")
    else:
        if tenure <= 0:
            raise ValueError("tenure must be a positive int")
    node_list = list(G.nodes())
    node_index = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)

    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n, dtype=int)

    F = cut_value(G, x, node_list, node_index)
    delta = delta_init(G, x, node_list, node_index)

    best_x = x.copy()
    best_F = F
    best_history = []

    tabu_until = np.zeros(n, dtype=int)
    last_improve_iter = 0

    for it in range(max_iters):
        # Choose best admissible move (aspiration allowed), break ties at random
        best_gain = -np.inf
        best_moves = []
        for v in range(n):
            gain = delta[v]
            admissible = (it >= tabu_until[v]) or (F + gain >= best_F)
            if not admissible:
                continue
            if gain > best_gain:
                best_gain = gain
                best_moves = [v]
            elif gain == best_gain:
                best_moves.append(v)

        if not best_moves:
            break

        best_v = int(rng.choice(best_moves))

        old_xv = x[best_v]
        x[best_v] = 1 - x[best_v]
        F = F + delta[best_v]
        if isinstance(tenure, (tuple, list)) and len(tenure) == 2:
            tenure_low, tenure_high = tenure
        else:
            enure_low, tenure_high = 1, tenure
        tenure_len = int(rng.integers(tenure_low, tenure_high + 1))
        tabu_until[best_v] = it + tenure_len

        # Update deltas efficiently
        delta[best_v] = -delta[best_v]
        v_node = node_list[best_v]
        for u, data in G.adj[v_node].items():
            w = data.get("weight", 1)
            iu = node_index[u]
            if x[iu] != old_xv:
                delta[iu] += 2 * w
            else:
                delta[iu] -= 2 * w

        if sanity_check and (it % check_every == 0):
            assert F == cut_value(G, x, node_list, node_index)

        if F > best_F:
            best_F = F
            best_x = x.copy()
            last_improve_iter = it

        best_history.append(best_F)
        if it - last_improve_iter >= patience:
            break

    return {
        "best_x": best_x,
        "best_F": int(best_F),
        "final_x": x.copy(),
        "final_F": int(F),
        "iters": it + 1,
        "best_history": best_history,
    }


#BASELINES
def random_search(G, num_samples=500, seed=0):
    # Random assignments; return best cut found.
    node_list = list(G.nodes())
    node_index = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)
    rng = np.random.default_rng(seed)

    best_F = -np.inf
    best_x = None
    for _ in range(num_samples):
        x = rng.integers(0, 2, size=n, dtype=int) #samples a length-n vector of 0/1 (upper bound 2 is exclusive)
        F = cut_value(G, x, node_list, node_index)
        if F > best_F:
            best_F = F
            best_x = x.copy()
    return {"best_x": best_x, "best_F": int(best_F), "final_x": best_x, "final_F": int(best_F), "iters": num_samples}


def greedy_hill_climb(G, seed=0, max_iters=2000):
    # Best-improving single-flip local search (no tabu).
    node_list = list(G.nodes())
    node_index = {v: i for i, v in enumerate(node_list)}
    n = len(node_list)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n, dtype=int)

    F = cut_value(G, x, node_list, node_index)
    delta = delta_init(G, x, node_list, node_index)

    for it in range(max_iters):
        v = int(np.argmax(delta)) #returns the index of the maximum element (the best gain flippp)
        if delta[v] <= 0:
            return {"current_x": x, "current_F": int(F), "final_x": x, "final_F": int(F), "iters": it + 1}

        old_xv = x[v]
        x[v] = 1 - x[v]
        F = F + delta[v]
        delta[v] = -delta[v]

        v_node = node_list[v]
        for u, data in G.adj[v_node].items():
            w = data.get("weight", 1)
            iu = node_index[u]
            if x[iu] != old_xv:
                delta[iu] += 2 * w
            else:
                delta[iu] -= 2 * w

    return {"current_x": x, "current_F": int(F), "final_x": x, "final_F": int(F), "iters": max_iters}


def _pad_histories(histories):
    if not histories:
        return None, None, None, 0
    max_len = max(len(h) for h in histories)
    H = np.zeros((len(histories), max_len), dtype=float)
    for i, h in enumerate(histories):
        H[i, :len(h)] = h
        if len(h) < max_len:
            H[i, len(h):] = h[-1]
    mean_curve = H.mean(axis=0)
    std_curve = H.std(axis=0)
    return H, mean_curve, std_curve, max_len


def run_experiments():
    # Experiment grid
    n_values = [50, 100, 200]
    p_values = [0.1, 0.3]
    tenures = [5, 10, 20]
    seeds = list(range(10))

    chosen_n = 100
    chosen_p = 0.3
    chosen_tenure = 10

    rows = []
    history_samples = []

    for n in n_values:
        for p in p_values:
            for seed in seeds:
                seeds_map = derive_seeds(seed)
                G = generate_graph(n, p, weighted=True, seed=seeds_map["instance_seed"])
                m = G.number_of_edges()
                total_weight = sum(data.get("weight", 1) for _, _, data in G.edges(data=True))

                # Random baseline
                t0 = time.perf_counter()
                r = random_search(G, num_samples=300, seed=seeds_map["random_seed"])
                rows.append({
                    "n": n, "p": p, "tenure": None, "seed": seed, "method": "random",
                    "best_F": r["best_F"], "final_F": r["final_F"], "iters": r["iters"],
                    "m": m, "total_weight": total_weight,
                    "seconds": time.perf_counter() - t0,
                })

                # Greedy baseline
                t0 = time.perf_counter()
                g = greedy_hill_climb(G, seed=seeds_map["greedy_seed"], max_iters=2000)
                rows.append({
                    "n": n, "p": p, "tenure": None, "seed": seed, "method": "greedy",
                    "best_F": g["final_F"], "final_F": g["final_F"], "iters": g["iters"],
                    "m": m, "total_weight": total_weight,
                    "seconds": time.perf_counter() - t0,
                })

                # Tabu Search across tenures
                for tenure in tenures:
                    algo_seed = seeds_map["algo_seed"] + 100 * tenure
                    t0 = time.perf_counter()
                    t = tabu_search_maxcut(
                        G,
                        max_iters=2000,
                        tenure=tenure,
                        seed=algo_seed,
                        patience=500,
                    )
                    rows.append({
                        "n": n, "p": p, "tenure": tenure, "seed": seed, "method": "tabu",
                        "best_F": t["best_F"], "final_F": t["final_F"], "iters": t["iters"],
                        "m": m, "total_weight": total_weight,
                        "seconds": time.perf_counter() - t0,
                    })
                    if (n == chosen_n and p == chosen_p and tenure == chosen_tenure):
                        history_samples.append(t["best_history"])

    results = pd.DataFrame(rows)
    results.to_csv("results_maxcut_tabu.csv", index=False)
    print(results.head())

    # PLOTS
    # a) mean curve + variability band (std or quantiles)
    H, mean_curve, std_curve, max_len = _pad_histories(history_samples)
    if H is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mean_curve, label="mean best_F")
        ax.fill_between(
            np.arange(max_len),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            label="±1 std",
        )
        ax.set(
            title=f"Tabu best_F vs iteration (n={chosen_n}, p={chosen_p}, tenure={chosen_tenure})",
            xlabel="iteration",
            ylabel="best_F",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig("tabu_mean_curve.png", dpi=200, bbox_inches="tight")
        plt.show()
    else:
        print("No tabu histories collected for the chosen (n, p, tenure).")

    # b) boxplot that respects tenure
    subset = results[(results["n"] == 100) & (results["p"] == 0.3) &
                     ((results["method"] != "tabu") | (results["tenure"] == 10))].copy()

    subset["label"] = subset.apply(
        lambda r: f"tabu_t{int(r['tenure'])}" if r["method"] == "tabu" else r["method"],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    subset.boxplot(column="best_F", by="label", ax=ax)
    ax.set_title("Best cut values (n=100, p=0.3)")
    ax.set_xlabel("")
    ax.set_ylabel("best_F")
    plt.suptitle("")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    fig.savefig("best_F_boxplot.png", dpi=200, bbox_inches="tight")
    plt.show()

    # c) Runtime vs n, split by p
    tabu_only = results[results["method"] == "tabu"].copy()
    tabu_only = tabu_only[tabu_only["tenure"] == 10]  # fix one tenure for readability

    grouped = tabu_only.groupby(["p", "n"])["seconds"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    for pval, dfp in grouped.groupby("p"):
        ax.plot(dfp["n"], dfp["seconds"], marker="o", label=f"p={pval}")
    ax.set(title="Mean runtime vs n (Tabu, tenure=10)",
           xlabel="n", ylabel="seconds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig("runtime_vs_n.png", dpi=200, bbox_inches="tight")
    plt.show()

    # d) Tabu-only boxplot by tenure
    tabu_subset = results[(results["method"] == "tabu") &
                          (results["n"] == 100) &
                          (results["p"] == 0.3)].copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    tabu_subset.boxplot(column="best_F", by="tenure", ax=ax)
    ax.set_title("Tabu performance vs tenure (n=100, p=0.3)")
    ax.set_xlabel("tenure")
    ax.set_ylabel("best_F")
    plt.suptitle("")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig("tabu_performance_vs_tenure.png", dpi=200, bbox_inches="tight")
    plt.show()

    return results


def run_example():
    # Run one example instance and print the partition
    seeds_map = derive_seeds(42)
    G = generate_graph(50, 0.2, weighted=True, seed=seeds_map["instance_seed"])
    res = tabu_search_maxcut(
        G,
        max_iters=3000,
        tenure=10,
        seed=seeds_map["algo_seed"],
        patience=800,
    )

    node_list = list(G.nodes())
    A = [node_list[i] for i, bit in enumerate(res["best_x"]) if bit == 0]
    B = [node_list[i] for i, bit in enumerate(res["best_x"]) if bit == 1]

    print("best cut value:", res["best_F"])
    print("|A|=", len(A), "|B|=", len(B))
    print("A:", A)
    print("B:", B)


if __name__ == "__main__":
    run_experiments()
    run_example()
