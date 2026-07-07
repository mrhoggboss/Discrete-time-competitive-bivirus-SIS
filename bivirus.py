"""
Minimal simulation / plotting utilities for the discrete-time networked
competitive bivirus SIS model.  Reproduces the figures of

    S. Gracy, Y. Xu, J. Liu, T. Basar, C. A. Uribe,
    "Networked Competitive Bivirus SIS Model - Analysis of the Discrete-Time Case".

Model:  x^l(k+1) = x^l(k) + h[ (I - X^1 - X^2) B^l - D^l ] x^l(k),  l = 1, 2.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcdefaults()
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


class SimulationConfig:
    def __init__(self, N=20, h=0.001, threshold=1.5, W=2, iterations=10000, tolerance=1e-6):
        self.N = N                    # number of nodes
        self.h = h                    # sampling parameter
        self.threshold = threshold    # threshold for the random adjacency matrices
        self.W = W                    # upper bound for the random adjacency entries
        self.iterations = iterations  # max iterations
        self.tolerance = tolerance    # convergence tolerance


def path_graph_adjacency(n, W=1):
    """Off-diagonal path-graph adjacency (added to ensure strong connectivity)."""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = np.random.uniform(0, W)
        A[i + 1, i] = np.random.uniform(0, W)
    return A


def run_simulation(x1, x2, B, delta, config):
    """Simulate the two-virus system until convergence (or config.iterations)."""
    N, h, iterations = config.N, config.h, config.iterations
    x1_history, x2_history = [x1.copy()], [x2.copy()]
    x1_avg_history, x2_avg_history = [np.average(x1)], [np.average(x2)]
    x = [x1, x2]
    for i in range(iterations):
        sum_of_x = np.diag(x[0]) + np.diag(x[1])
        x[0] = x[0] + h * ((np.eye(N) - sum_of_x) @ B[0] - np.diag(delta[0])) @ x[0]
        x[1] = x[1] + h * ((np.eye(N) - sum_of_x) @ B[1] - np.diag(delta[1])) @ x[1]
        x1_history.append(x[0].copy())
        x2_history.append(x[1].copy())
        x1_avg_history.append(np.average(x[0]))
        x2_avg_history.append(np.average(x[1]))
        if (np.linalg.norm(x1_history[-1] - x1_history[-2]) < config.tolerance and
                np.linalg.norm(x2_history[-1] - x2_history[-2]) < config.tolerance):
            break
    return {
        "x1_history": x1_history, "x2_history": x2_history,
        "x1_avg_history": x1_avg_history, "x2_avg_history": x2_avg_history,
    }


def x_bar(x1, B1, delta, config):
    """Single-virus endemic equilibrium (other virus treated as absent)."""
    N, h, iterations = config.N, config.h, config.iterations
    x_history = [x1.copy()]
    x = x1.copy()
    for _ in range(iterations):
        x = x + h * ((np.eye(N) - np.diag(x)) @ B1 - np.diag(delta)) @ x
        x_history.append(x.copy())
        if np.linalg.norm(x_history[-1] - x_history[-2]) < config.tolerance:
            break
    return x_history[-1]


def find_C(z):
    """Irreducible C with C z = z and spectral radius 1 (Theorem 7)."""
    N, = z.shape
    pi = z / np.sum(z)
    return np.outer(pi, np.ones(N))


def plot_simulation_single(x1_avg_history, x2_avg_history, yscale='linear', figsize=(8, 3)):
    """Average infection level of each virus versus time (a single panel)."""
    plt.figure(figsize=figsize)
    plt.plot(x1_avg_history, color='b', label='Virus 1')
    plt.plot(x2_avg_history, color='r', label='Virus 2')
    plt.xlabel('Time step')
    plt.ylabel(r'Avg. Inf. level, $\frac{1}{n}\sum_{i=1}^{n} x_i^{\ell}(t)$')
    plt.yscale(yscale)
    plt.ylim(0.01, 1)
    plt.xlim(left=0)
    plt.margins(x=0)
    plt.legend(loc='upper right')
    plt.show()


def plot_bivirus_graph_simple(A1, A2, x1, x2, seed=42, d0=300, r0=700, figsize=(8, 8), tol=1e-3):
    """
    Network state: node color = which virus(es) infect it (blue = virus 1,
    red = virus 2, magenta = both, white = healthy); node size grows with the
    total infection level. Edges of layer 1 in gray, layer 2 in green.
    """
    n = len(x1)
    G1 = nx.from_numpy_array(A1, create_using=nx.DiGraph)
    G2 = nx.from_numpy_array(A2, create_using=nx.DiGraph)
    pos = nx.spring_layout(G1, seed=seed)
    plt.figure(figsize=figsize)
    for i in range(n):
        total = x1[i] + x2[i]
        if abs(total) < tol:
            color = [1, 1, 1]                          # healthy
        elif abs(x2[i]) < tol and abs(x1[i]) >= tol:
            color = [0, 0, 1]                          # virus 1
        elif abs(x1[i]) < tol and abs(x2[i]) >= tol:
            color = [1, 0, 0]                          # virus 2
        else:
            color = [1, 0, 1]                          # coexistence
        nx.draw_networkx_nodes(G1, pos, nodelist=[i], node_color=[color],
                               edgecolors='black', node_size=d0 + total * r0)
    nx.draw_networkx_edges(G1, pos, edge_color='gray', width=2, alpha=0.5)
    nx.draw_networkx_edges(G2, pos, edge_color='green', width=2, alpha=0.3)
    plt.axis('off')
    plt.show()


def plot_phase_portrait(initializations, B, delta, config, figsize=(4.5, 4.5)):
    """
    Phase portrait of the average infection levels (virus 1 vs virus 2).  Each
    initialization's averaged trajectory is drawn (o = start, x = equilibrium),
    together with the least-squares line through the equilibria and the
    x1 + x2 = 1 constraint. Used to illustrate the continuum of coexistence
    equilibria (Theorem 7).
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10", max(10, len(initializations)))
    ex, ey = [], []
    for k, (x1_0, x2_0) in enumerate(initializations):
        r = run_simulation(np.asarray(x1_0, float).copy(), np.asarray(x2_0, float).copy(),
                           B, delta, config)
        xt = np.asarray(r["x1_avg_history"]); yt = np.asarray(r["x2_avg_history"])
        c = cmap(k % cmap.N)
        ax.plot(xt, yt, color=c, linewidth=1.6, alpha=0.95)
        ax.scatter([xt[0]], [yt[0]], color=c, s=30, marker="o", zorder=3)
        ax.scatter([xt[-1]], [yt[-1]], color=c, s=30, marker="x", zorder=3)
        ex.append(xt[-1]); ey.append(yt[-1])
    ex, ey = np.array(ex), np.array(ey)
    m, b0 = np.polyfit(ex, ey, 1)
    xs = np.linspace(max(0.0, ex.min() - 0.05), min(1.0, ex.max() + 0.05), 300)
    ys = m * xs + b0
    intri = (xs >= 0) & (ys >= 0) & (xs + ys <= 1)
    ax.plot(xs[intri], ys[intri], color="black", linestyle="--", linewidth=1.5, alpha=0.85, zorder=11)
    ax.plot([1.0, 0.0], [0.0, 1.0], color="black", linewidth=1.0, zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"Average infection of virus 1")
    ax.set_ylabel(r"Average infection of virus 2")
    plt.show()
