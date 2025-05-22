import numpy as np
import matplotlib.pyplot as plt

# Some useful functions for the simulation

class SimulationConfig:
    def __init__(self, N=20, h=0.1, W=2, iterations=10000, seed=None):
        self.N = N
        self.h = h
        self.W = W
        self.iterations = iterations
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

def run_simulation(x1, x2, B, delta, config):
    N, h, iterations = config.N, config.h, config.iterations
    x1_history, x2_history = [x1.copy()], [x2.copy()]
    x1_avg_history, x2_avg_history = [np.average(x1)], [np.average(x2)]
    x = [x1, x2]
    for _ in range(iterations):
        sum_of_x = np.diag(x[0]) + np.diag(x[1])
        x[0] = x[0] + h * ((np.eye(N) - sum_of_x) @ B[0] - np.diag(delta[0])) @ x[0]
        x[1] = x[1] + h * ((np.eye(N) - sum_of_x) @ B[1] - np.diag(delta[1])) @ x[1]
        x1_history.append(x[0].copy())
        x2_history.append(x[1].copy())
        x1_avg_history.append(np.average(x[0]))
        x2_avg_history.append(np.average(x[1]))
    return {
        "x1_history": x1_history,
        "x2_history": x2_history,
        "x1_avg_history": x1_avg_history,
        "x2_avg_history": x2_avg_history,
        "final_x1": x1_history[-1],
        "final_x2": x2_history[-1]
    }

def random_parameters(config):
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

def plot_simulation_3by3(x1_avg_histories, x2_avg_histories):
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
            
            # if abs(x1_history[-1] - x2_history[-1]) > 0.1:
                # col.text(iterations, x1_history[-1], f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
                # col.text(iterations, x2_history[-1], f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            # elif x1_history[-1] > x2_history[-1]:
            #     col.text(iterations, x1_history[-1] + 0.05, f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
            #     col.text(iterations, x2_history[-1] - 0.05, f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            # elif x1_history[-1] <= x2_history[-1]:
            #     col.text(iterations, x1_history[-1] - 0.05, f"({round(x1_history[-1], 2)})", fontsize=8, color = 'b')
            #     col.text(iterations, x2_history[-1] + 0.05, f"({round(x2_history[-1], 2)})", fontsize=8, color = 'r')
            col.set_yscale('log')
            col.set(xlabel='Time step', ylabel='Avg. Infection level')
            col.label_outer()
    ax = plt.gca()
    plt.xlabel('Time step')
    plt.ylabel(f'Avg Infection level')
        
    fig.suptitle(f'Average Infection level VS Time')
    plt.show()