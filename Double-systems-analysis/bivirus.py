import numpy as np
import matplotlib.pyplot as plt

# Some useful functions for the simulation

class SimulationConfig:
    def __init__(self, N=20, h=0.1, W=2, iterations=1000, tolerance=1e-3):
        self.N = N
        self.h = h
        self.W = W
        self.iterations = iterations # max iterations before stopping
        self.tolerance = tolerance # tolerance for convergence

def run_simulation(x1, x2, B, delta, config):
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
            print(f"Converged at iteration {i}")
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

def plot_average_infection(x1_avg_history, x2_avg_history, title="Average Infection Levels"):
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

def plot_simulation_3by3(x1_avg_histories, x2_avg_histories, yscale='log'):
    '''
    x1_avg_histories: list of 9 lists, each inner list is a histogram of average infection levels for virus 1
    x2_avg_histories: list of 9 lists, each inner list is a histogram of average infection levels for virus 2

    plots a 3x3 grid of subplots, each showing the average infection levels for virus 1 and virus 2 over time under a different intial condition
    '''
    # retrieve iterations
    iterations = len(x1_avg_histories[0]) - 1

    # Plot the results  
    fig, axs = plt.subplots(nrows=3, ncols=3)

    idx = 0
    for row in axs:
        for col in row:
            x1_history = x1_avg_histories[idx]
            x2_history = x2_avg_histories[idx]
            idx += 1
            col.plot(x1_history, color='b')
            col.plot(x2_history, color='r')

            col.text(iterations, x1_history[-1], f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
            col.text(iterations, x2_history[-1], f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            
            col.set_ylim(0.01, 1)
            col.set_yscale(yscale)
            col.set(xlabel='Time step', ylabel='Avg. Infection level')
            col.label_outer()
    ax = plt.gca()
    plt.xlabel('Time step')
    plt.ylabel(f'Avg Infection level')
        
    fig.suptitle(f'Average Infection level VS Time')
    plt.show()

def check_assumptions(x1, x2, B, delta, config):
    """
    Check the assumptions of the paper
    """
    # Assumption 1: initial portion of infection of both virus must be between 0 and 1, and initial portion of the healthy must be between 0 and 1
    
    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert(0 <= x1[i] <= 1, "A1, x1[i] out of bounds")
        assert(0 <= x2[i] <= 1, "A1, x2[i] out of bounds")
        assert(0 <= 1 - x1[i] - x2[i] <= 1, "A1, healthy[i] out of bounds")
    
    # Assumption 2: Non-negative B and deltas

    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert(delta[0][i] >= 0, "A2, delta[0][i] negative")
        assert(delta[1][i] >= 0, "A2, delta[1][i] negative")
    for i in range(config.N):
        for j in range(config.N):
            assert(B[0][i][j] >= 0, "A2, B[0][i][j] negative")
            assert(B[1][i][j] >= 0, "A2, B[1][i][j] negative")
    
    # Assumption 3: sampling parameter upper bound

    # This needs to be satisfied for all theorems
    for i in range(config.N):
        assert(config.h * delta[0][i] < 1, "A3, h * delta[0][i] >= 1")
        assert(config.h * delta[1][i] < 1, "A3, h * delta[1][i] >= 1")
    
    for i in range(config.N):
        row_sum1 = sum(B[0][i])
        row_sum2 = sum(B[1][i])
        assert(config.h * (row_sum1 + row_sum2) <= 1, "A3, h * (row_sum1 + row_sum2) > 1")
    
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
    # this is already satisfied since the matrices are fully connected

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