import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gravis as gv
import networkx as nx
import matplotlib.patches as mpatches
# Some useful functions for the simulation

class SimulationConfig:
    def __init__(self, N=20, h=0.1, threshold = 1, W=2, iterations=1000, tolerance=1e-3):
        self.N = N
        self.h = h
        self.threshold = threshold
        self.W = W
        self.iterations = iterations # max iterations before stopping
        self.tolerance = tolerance # tolerance for convergence

def run_simulation(x1, x2, B, delta, config):
    '''
    runs the simulation for two viruses with the given parameters
    x1: initial infection levels for virus 1
    x2: initial infection levels for virus 2
    B: transmission matrices for the two viruses, each of shape (N, N)
    delta: healing rates for the two viruses, each of shape (N,)
    config: SimulationConfig object containing simulation parameters
    '''
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
        if np.linalg.norm(x1_history[-1] - x1_history[-2]) < config.tolerance and np.linalg.norm(x2_history[-1] - x2_history[-2]) < config.tolerance:
            print(f"Main loop Converged at iteration {i}")
            break
    if i == iterations - 1:
        print("Reached max iterations and did not converge")

    return {
        "x1_history": x1_history,
        "x2_history": x2_history,
        "x1_avg_history": x1_avg_history,
        "x2_avg_history": x2_avg_history,
        "final_x1": x1_history[-1],
        "final_x2": x2_history[-1]
    }

def x_bar(x1, B1, delta, config):
    '''
        finds the single virus equilibrium point for the given parameters. Treats the other virus as nonexistent.
        x1: initial infection levels for the virus - The outcome of the simulation is independent of this value
        B1: transmission matrix for that particular virus
        delta: healing rates for the population
        config: SimulationConfig object
    '''
    N, h, iterations = config.N, config.h, config.iterations
    x_history = [x1.copy()]
    x = x1.copy()

    # Simulation loop
    for _ in range(iterations):
        x = x + h * ((np.eye(N) - np.diag(x)) @ B1 - np.diag(delta)) @ x
        # x = np.clip(x, 0, 1)  # Ensure infection levels are between 0 and 1
        assert(np.all(x >= 0) and np.all(x <= 1)), "Infection levels out of bounds"
        x_history.append(x.copy())
        if np.linalg.norm(x_history[-1] - x_history[-2]) < config.tolerance:
            print(f"x_bar Converged at iteration {_}")
            break
    if _ == iterations - 1:
        print("Reached max iterations and did not converge")
    
    return x_history[-1]

def random_parameters(config):
    '''
    generate random parameters for the simulation based on the config
    config: SimulationConfig object
    returns: a tuple of two lists, each containing the parameters for one virus
    '''
    N, W = config.N, config.W
    A1, A2 = np.random.uniform(0, W, (N, N)), np.random.uniform(0, W, (N, N))
    delta_1 = np.random.uniform(0, 1, N)
    delta_2 = np.random.uniform(0, 1, N)
    beta_1 = np.random.uniform(0, 1, N)
    beta_2 = np.random.uniform(0, 1, N)
    B1 = np.diag(beta_1) @ A1
    B2 = np.diag(beta_2) @ A2
    return [B1, B2], [delta_1, delta_2]
def plot_average_infection(x1_avg_history, x2_avg_history, title="Average Infection Levels", figsize=(6.4, 4.8)):
    plt.figure(figsize=figsize)
    plt.plot(x1_avg_history, label="Virus 1", color='b')
    plt.plot(x2_avg_history, label="Virus 2", color='r')
    plt.xlabel("Time step")
    plt.ylabel("Average Infection Level")
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    # Add final infection level labels
    plt.text(len(x1_avg_history)-1, x1_avg_history[-1], f"{x1_avg_history[-1]:.2f}", color='b', fontsize=10, va='bottom')
    plt.text(len(x2_avg_history)-1, x2_avg_history[-1], f"{x2_avg_history[-1]:.2f}", color='r', fontsize=10, va='bottom')
    plt.show()

def plot_simulation_3by3(x1_avg_histories, x2_avg_histories, yscale='log', x1_bar_avg=None, x2_bar_avg=None, figsize=(12, 12)):
    '''
    x1_avg_histories: list of 9 lists, each inner list is a histogram of average infection levels for virus 1
    x2_avg_histories: list of 9 lists, each inner list is a histogram of average infection levels for virus 2
    x1_bar: (optional) float, equilibrium value for virus 1 to plot as a horizontal line
    x2_bar: (optional) float, equilibrium value for virus 2 to plot as a horizontal line
    figsize: tuple, figure size
    '''
    iterations = len(x1_avg_histories[0]) - 1
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    idx = 0
    for row in axs:
        for col in row:
            x1_history = x1_avg_histories[idx]
            x2_history = x2_avg_histories[idx]
            idx += 1
            col.plot(x1_history, color='b')
            col.plot(x2_history, color='r')
            col.set_xlim(left=0)
            col.margins(x=0)
            if x1_bar_avg is not None:
                col.axhline(y=x1_bar_avg, color='b', linestyle='--', linewidth=1, label='x1_bar')
            if x2_bar_avg is not None:
                col.axhline(y=x2_bar_avg, color='r', linestyle='--', linewidth=1, label='x2_bar')
            col.set_ylim(0.01, 1)
            col.set_yscale(yscale)
            col.set(xlabel='Time step', ylabel='Avg. Inf. level')
            col.label_outer()
    fig.suptitle(f'Average Infection level VS Time')
    if x1_bar_avg is not None or x2_bar_avg is not None:
        handles, labels = axs[0,2].get_legend_handles_labels()
        if handles:
            axs[0,2].legend(loc='upper right')
    plt.show()

def plot_simulation_1by3(x1_avg_histories, x2_avg_histories, yscale='log', title=None, x1_bar_avg=None, x2_bar_avg=None, figsize=(8, 9)):
    '''
    x1_avg_histories: list of 3 lists, each inner list is a histogram of average infection levels for virus 1
    x2_avg_histories: list of 3 lists, each inner list is a histogram of average infection levels for virus 2
    x1_bar: (optional) float, equilibrium value for virus 1 to plot as a horizontal line
    x2_bar: (optional) float, equilibrium value for virus 2 to plot as a horizontal line
    figsize: tuple, figure size
    '''
    iterations = len(x1_avg_histories[0]) - 1
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    for idx, ax in enumerate(axs):
        x1_history = x1_avg_histories[idx]
        x2_history = x2_avg_histories[idx]
        line1, = ax.plot(x1_history, color='b', label='Virus 1')
        line2, = ax.plot(x2_history, color='r', label='Virus 2')
        ax.set_xlim(left=0)
        ax.margins(x=0)
        if x1_bar_avg is not None:
            ax.axhline(y=x1_bar_avg, color='b', linestyle='--', linewidth=1, label='x1_bar')
        if x2_bar_avg is not None:
            ax.axhline(y=x2_bar_avg, color='r', linestyle='--', linewidth=1, label='x2_bar')
        ax.set_ylim(0.01, 1)
        ax.set_yscale(yscale)
        ax.set(xlabel='Time step', ylabel='Avg. Inf. level')
        ax.label_outer()
    if title is not None:
        fig.suptitle(title)
    if x1_bar_avg is not None or x2_bar_avg is not None:
        handles, labels = axs[-1].get_legend_handles_labels()
        if handles:
            axs[0].legend(loc='upper right')
    else:
        axs[0].legend(['Virus 1', 'Virus 2'], loc='upper right')
    plt.show()

def plot_simulation_1by2(x1_avg_histories, x2_avg_histories, yscale='log', title=None, x1_bar_avg=None, x2_bar_avg=None, figsize=(8, 6)):
    '''
    x1_avg_histories: list of 2 lists, each inner list is a histogram of average infection levels for virus 1
    x2_avg_histories: list of 2 lists, each inner list is a histogram of average infection levels for virus 2
    x1_bar: (optional) float, equilibrium value for virus 1 to plot as a horizontal line
    x2_bar: (optional) float, equilibrium value for virus 2 to plot as a horizontal line
    figsize: tuple, figure size
    '''
    iterations = len(x1_avg_histories[0]) - 1
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    for idx, ax in enumerate(axs):
        x1_history = x1_avg_histories[idx]
        x2_history = x2_avg_histories[idx]
        line1, = ax.plot(x1_history, color='b', label='Virus 1')
        line2, = ax.plot(x2_history, color='r', label='Virus 2')
        ax.set_xlim(left=0)
        ax.margins(x=0)
        if x1_bar_avg is not None:
            ax.axhline(y=x1_bar_avg, color='b', linestyle='--', linewidth=1, label='x1_bar')
        if x2_bar_avg is not None:
            ax.axhline(y=x2_bar_avg, color='r', linestyle='--', linewidth=1, label='x2_bar')
        ax.set_ylim(0.01, 1)
        ax.set_yscale(yscale)
        ax.set(xlabel='Time step', ylabel='Avg. Inf. level')
        ax.label_outer()
    if title is not None:
        fig.suptitle(title)
    if x1_bar_avg is not None or x2_bar_avg is not None:
        handles, labels = axs[-1].get_legend_handles_labels()
        if handles:
            axs[0].legend(loc='upper right')
    else:
        axs[0].legend(['Virus 1', 'Virus 2'], loc='upper right')
    plt.show()

def plot_simulation_single(x1_avg_history, x2_avg_history, yscale='log', title=None, x1_bar_avg=None, x2_bar_avg=None, figsize=(8, 3)):
    """
    Plots the average infection levels for Virus 1 and Virus 2 on a single plot.

    x1_avg_history: list of average infection levels for virus 1
    x2_avg_history: list of average infection levels for virus 2
    yscale: 'log' or 'linear'
    title: optional plot title
    x1_bar_avg: (optional) float, equilibrium value for virus 1 to plot as a horizontal line
    x2_bar_avg: (optional) float, equilibrium value for virus 2 to plot as a horizontal line
    figsize: tuple, figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(x1_avg_history, color='b', label='Virus 1')
    plt.plot(x2_avg_history, color='r', label='Virus 2')
    if x1_bar_avg is not None:
        plt.axhline(y=x1_bar_avg, color='b', linestyle='--', linewidth=1, label='x1_bar')
    if x2_bar_avg is not None:
        plt.axhline(y=x2_bar_avg, color='r', linestyle='--', linewidth=1, label='x2_bar')
    plt.xlabel('Time step')
    plt.ylabel('Avg. Inf. level')
    plt.yscale(yscale)
    plt.ylim(0.01, 1)
    plt.xlim(left=0)
    plt.margins(x=0)
    if title is not None:
        plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

def check_basic_assumptions(x1, x2, B, delta, config):
    """
    Check the basic assumptions of the paper, i.e., Assumptions 1-6
    """
    # Assumption 1: initial portion of infection of both virus must be between 0 and 1, and initial portion of the healthy must be between 0 and 1
    
    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert 0 <= x1[i] <= 1, "A1, x1[i] out of bounds"
        assert 0 <= x2[i] <= 1, "A1, x2[i] out of bounds"
        assert 0 <= 1 - x1[i] - x2[i] <= 1, "A1, healthy[i] out of bounds"
    
    # Assumption 2: Non-negative B and deltas

    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert delta[0][i] >= 0, "A2, delta[0][i] negative"
        assert delta[1][i] >= 0, "A2, delta[1][i] negative"
    for i in range(config.N):
        for j in range(config.N):
            assert B[0][i][j] >= 0, "A2, B[0][i][j] negative"
            assert B[1][i][j] >= 0, "A2, B[1][i][j] negative"
    
    # Assumption 3: sampling parameter upper bound

    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert config.h * delta[0][i] < 1, "A3, h * delta[0][i] >= 1"
        assert config.h * delta[1][i] < 1, "A3, h * delta[1][i] >= 1"
    
    for i in range(config.N):
        row_sum1 = sum(B[0][i])
        row_sum2 = sum(B[1][i])
        assert config.h * (row_sum1 + row_sum2) <= 1, "A3, h * (row_sum1 + row_sum2) > 1"
    
    # Assumption 4: nonzero B, sampling parameter and nontrivial dimension
    A4_flag = True
    # This needs to be satisfied for theorems 3, 4, 5, 6, 7
    if not (config.N > 1 and config.h > 0 and np.any(B[0]) and np.any(B[1])):
        A4_flag = False
    
    if not A4_flag:
        print("Assumption 4 failed")
        print("N: ", config.N)
        print("h: ", config.h)
        print("B1: ", B[0])
        print("B2: ", B[1])
        print("delta1: ", delta[0])
        print("delta2: ", delta[1])

    # Assumption 5: irreducible B, equivalent to saying that the graphs represented by B1 and B2 are strongly connected
    B1 = nx.from_numpy_array(B[0], create_using=nx.DiGraph) 
    B2 = nx.from_numpy_array(B[1], create_using=nx.DiGraph)

    # check for strong connectivity
    assert nx.is_strongly_connected(B1), "A5, B1 is not strongly connected"
    assert nx.is_strongly_connected(B2), "A5, B2 is not strongly connected"

    # Assumption 6: stricter sampling parameter upper bound
    A6_flag = True
    # This needs to be satisfied for theorems 3, 4, 5, 6, 7
    for i in range(config.N):
        row_sum1 = sum(B[0][i])
        row_sum2 = sum(B[1][i])
        if not config.h * delta[0][i] + config.h * (row_sum1 + row_sum2) <= 1:
            A6_flag = False
        if not config.h * delta[1][i] + config.h * (row_sum1 + row_sum2) <= 1:
            A6_flag = False
        
    if not A6_flag:
        print("Assumption 6 failed")
        print("N: ", config.N)
        print("h: ", config.h)
        print("B1: ", B[0])
        print("B2: ", B[1])
        print("delta1: ", delta[0])
        print("delta2: ", delta[1])
    
    print("All assumptions satisfied")

def check_theorem_2(B, delta, config):
    """
    returns true if the model satisfies the assumptions of theorem 2
    false otherwise
    """
    spectral_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[0]) + config.h * B[0])))
    spectral_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[1]) + config.h * B[1])))
    print('spectral radius 1 is '+str(spectral_radius_1))
    print('spectral radius 2 is '+str(spectral_radius_2))
    return (spectral_radius_1 <= 1 and spectral_radius_2 <= 1)

def check_theorem_3(B, delta, config):
    """
    returns a float where
    3.1: theorem 3, virus 1 survives, virus 2 dies out
    3.2: theorem 3, virus 1 dies out, virus 2 survives
    0 otherwise
    """
    spectral_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[0]) + config.h * B[0])))
    spectral_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[1]) + config.h * B[1])))
    if (spectral_radius_1 > 1 and spectral_radius_2 <= 1):
        return 3.1
    elif (spectral_radius_1 <= 1 and spectral_radius_2 > 1):
        return 3.2
    else:
        return 0

def check_theorem_4(B, delta, config):
    """
    returns a float where:
    4.1: theorem 4, virus 1 stable, virus 2 stable
    4.2: theorem 4, virus 1 stable, virus 2 unstable
    4.3: theorem 4, virus 1 unstable, virus 2 stable
    4.4: theorem 4, virus 1 unstable, virus 2 unstable
    0 otherwise
    """

    # note that the result of this function should not depend on the initial conditions x1, x2
    x1 = 0.1 * np.ones(config.N)
    x2 = 0.1 * np.ones(config.N)
    if check_theorem_2(B, delta, config) or (check_theorem_3(B, delta, config) != 0):
        return 0
    
    x1_bar = x_bar(x1, B[0], delta[0], config)
    x2_bar = x_bar(x2, B[1], delta[1], config)
    # calculate the spectral radii used to determine the stability of endemic equilibria.
    det_radius_1 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[1]) + (np.eye(config.N) - np.diag(x1_bar)) @ B[1])))
    det_radius_2 = np.max(np.abs(np.linalg.eigvals(np.eye(config.N) - config.h * np.diag(delta[0]) + (np.eye(config.N) - np.diag(x2_bar)) @ B[0])))
    
    print('det radius 1 is '+str(det_radius_1))
    print('det radius 2 is '+str(det_radius_2))
    
    if det_radius_1 <= 1 and det_radius_2 <= 1:
        label = 4.1
        # 1) both (x_1, 0) and (0, x_2) are stable
        # 2) also exists (x_1hat, x_2hat) which is unstable (theorem 5, not validated in simulations)
    elif det_radius_1 > 1 and det_radius_2 > 1:
        label = 4.4
        # 1) both (x_1, 0) and (0, x_2) are unstable
    elif det_radius_1 <= 1 and det_radius_2 > 1:
        label = 4.2
        # 1) (x_1, 0) stable 
        # 2) (0, x_2) unstable
    elif det_radius_1 > 1 and det_radius_2 <= 1:
        label = 4.3
        # 1) (x_1, 0) unstable 
        # 2) (0, x_2) stable
    
    return label

def find_C(z):
    '''
    finds a matrix C such that:
    1. C @ z = z
    2. C is irreducible
    3. C has spectral radius 1

    input: z, the single virus equilibrium of the first virus
    '''
    N,  = z.shape
    pi = z / np.sum(z)
    C = np.outer(pi, np.ones(N))
    return C

def verify_C(z, C, tol=1e-6):
    '''
    verifies whether a given single virus equilibrium z and a matrix C satisfies the constraints given in Theorem 7
    '''
    # 1. z is a eigenvector of C with eigenvalue 1
    assert np.allclose(C @ z, z, atol=tol), "Cz != z"
    print("C @ z is: " + str(C @ z))
    print("z is: " + str(z))
    # 2. C is irreducible
    import networkx as nx
    def is_irreducible(A: np.ndarray) -> bool:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        G.remove_edges_from([(u, v) for u, v, w in G.edges(data=True) if w["weight"] == 0])
        return nx.is_strongly_connected(G)

    assert is_irreducible(C), "C is reducible"

    # 3. C has spectral radius = 1
    assert np.abs(np.max(np.abs(np.linalg.eigvals(C))) - 1) < tol, "C does not have spectral radius 1"
    print("C has spectral radius: " + str(np.max(np.abs(np.linalg.eigvals(C)))))
    print("C satisfies the assumptions of Theorem 7")

def plot_two_networks_piechart_nodes(G1, G2, x1, x2, pos1=None, pos2=None, node_size=300):
    """
    plots two networkx graphs as side-by-side subplots, with each node as a pie chart representing:
    - blue: x1[i]
    - red: x2[i]
    - green: 1 - x1[i] - x2[i] (healthy)
    Args:
        G1, G2: networkx.Graph or DiGraph
        x1, x2: array-like, distributions over nodes (must be same length as number of nodes in each graph)
        pos1, pos2: dict, optional, node positions for plotting
        node_size: int, optional, node size for plotting
    """
    import matplotlib.patches as mpatches

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    assert x1.shape == x2.shape == (len(G1),)
    assert x1.shape == (len(G2),)
    assert np.all(x1 + x2 < 1 + 1e-8), "sum of infection levels exceeds 1 for some nodes"

    if pos1 is None:
        pos1 = nx.spring_layout(G1, seed=42)
    if pos2 is None:
        pos2 = nx.spring_layout(G2, seed=42)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, G, pos in zip(axs, [G1, G2], [pos1, pos2]):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray')
        for i, (x, y) in enumerate(pos.values()):
            sizes = [x1[i], x2[i], 1 - x1[i] - x2[i]]
            colors = ['blue', 'red', 'green']
            wedges, _ = ax.pie(
                sizes,
                colors=colors,
                radius=np.sqrt(node_size)/200,
                center=(x, y),
                frame=True
            )
            for wedge in wedges:
                wedge.set_edgecolor('none')
        for i, (x, y) in enumerate(pos.values()):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8, color='white', weight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    legend_patches = [
        mpatches.Patch(color='blue', label='virus 1'),
        mpatches.Patch(color='red', label='virus 2'),
        mpatches.Patch(color='green', label='healthy')
    ]
    axs[1].legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_bivirus_graph(G1, G2, x1, x2, seed=42, d0=300, r0=700, figsize=(8, 8)):
    """
    Plots the graphical representation of the bi-virus system.

    Parameters:
    G1, G2: np.ndarray
        Adjacency matrices for graphs for virus 1 and virus 2.
    x1, x2: np.ndarray
        Arrays representing the infection proportions for each agent for virus 1 and virus 2.
    d0: float
        Default (smallest) diameter of the node.
    r0: float
        Scaling factor for the node diameter.
    """
    n = len(x1)

    G1 = nx.from_numpy_array(G1, create_using=nx.DiGraph)
    G2 = nx.from_numpy_array(G2, create_using=nx.DiGraph)

    pos = nx.spring_layout(G1, seed=seed)

    plt.figure(figsize=figsize)

    for i in range(n):
        total_infection = x1[i] + x2[i]
        if np.allclose(total_infection, 0, atol=1e-4):
            color = [1, 1, 1]  # white for healthy
            edge_color = 'black'
        else:
            red_component = x2[i] / total_infection
            blue_component = x1[i] / total_infection
            color = [red_component, 0, blue_component]
            edge_color = 'black'

        diameter = d0 + total_infection * r0
        nx.draw_networkx_nodes(G1, pos,
                               nodelist=[i],
                               node_color=[color],
                               edgecolors=edge_color,
                               node_size=diameter)

    # overlapping two networks
    nx.draw_networkx_edges(G1, pos, edge_color='gray', width=2, alpha=0.5)
    nx.draw_networkx_edges(G2, pos, edge_color='green', width=2, alpha=0.3)

    # labels = {i: f'{i}' for i in range(n)}
    # nx.draw_networkx_labels(G1, pos, labels=labels)

    # plt.title("Bi-virus system representation")
    plt.axis('off')
    plt.show()

def plot_bivirus_graph_simple(G1, G2, x1, x2, seed=42, d0=300, r0=700, figsize=(8, 8), tol=1e-3):
    """
    Plots the bi-virus system, coloring all infected nodes magenta, healthy nodes white.

    Parameters:
    G1, G2: np.ndarray
        Adjacency matrices for graphs for virus 1 and virus 2.
    x1, x2: np.ndarray
        Infection proportions for each agent for virus 1 and virus 2.
    d0: float
        Default node diameter.
    r0: float
        Scaling factor for node diameter.
    """
    n = len(x1)
    G1 = nx.from_numpy_array(G1, create_using=nx.DiGraph)
    G2 = nx.from_numpy_array(G2, create_using=nx.DiGraph)
    pos = nx.spring_layout(G1, seed=seed)
    plt.figure(figsize=figsize)

    for i in range(n):
        total_infection = x1[i] + x2[i]
        if abs(total_infection) < tol:
            color = [1, 1, 1]  # white for healthy
        elif abs(x2[i]) < tol and abs(x1[i]) >= tol:
            color = [0, 0, 1]  # blue for virus 1
        elif abs(x1[i]) < tol and abs(x2[i]) >= tol:
            color = [1, 0, 0]  # red for virus 2
        else:
            color = [1, 0, 1]  # magenta for coexistence
        diameter = d0 + total_infection * r0
        nx.draw_networkx_nodes(G1, pos,
                               nodelist=[i],
                               node_color=[color],
                               edgecolors='black',
                               node_size=diameter)

    nx.draw_networkx_edges(G1, pos, edge_color='gray', width=2, alpha=0.5)
    nx.draw_networkx_edges(G2, pos, edge_color='green', width=2, alpha=0.3)
    plt.axis('off')
    plt.show()