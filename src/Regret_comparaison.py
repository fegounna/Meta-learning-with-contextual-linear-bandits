"""Implementation of the Algorithm Meta-Represented Greedy Policy"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from regret import calculate_average_regret

np.random.seed(42)

with open("../data/X_test_80test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("../data/Y_test_80test.pkl", "rb") as f:
    Y_test = pickle.load(f)

B_trace = np.load("../data/B_ml_80test.npy")
B_mom = np.load("../data/B_ml_mom_80test.npy")

total_iterations = 30

# Algorithm : Meta-Represented Greedy Policy
cumulative_regrets_trace = np.cumsum(
    calculate_average_regret(X_test, Y_test, B_trace, total_iterations)
)

# Algorithm : Method of Moments
cumulative_regrets_mom = np.cumsum(
    calculate_average_regret(X_test, Y_test, B_mom, total_iterations)
)

# Algorithm : Random Selection of the arms
regrets_random = np.zeros(total_iterations)

for t in range(len(X_test)):
    X_test_array = np.array(X_test[t])
    Y_test_array = np.array(Y_test[t])

    n, d = X_test_array.shape
    arms = set(range(n))

    for i in range(total_iterations):
        arm = np.random.choice(list(arms))
        best_reward = np.max([Y_test_array[a] for a in arms])
        arms.remove(arm)
        regrets_random[i] += best_reward - Y_test_array[arm]

cumulative_regrets_random = np.cumsum(regrets_random) / len(X_test)

plt.figure(figsize=(12, 6))
plt.plot(
    cumulative_regrets_trace,
    label="Trace norm regularization Method",
)
plt.plot(cumulative_regrets_mom, label="Method of Moments")
plt.plot(cumulative_regrets_random, label="Random selection")
plt.xlabel("Iteration")
plt.ylabel("Average Cumulative Regret")
plt.title(
    "Comparison of the Meta-represented greedy policies with random selection with 80 users having at least 30 rated movies"
)
plt.legend()
plt.grid(True)
plt.savefig(
    "../results/Average Cumulative Regret of 5 test users with 80 users having at least 30 rated movies.png"
)
plt.show()
