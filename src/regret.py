import numpy as np


def calculate_average_regret(X_test, Y_test, B, total_iterations):
    """
    Calculate average regret for a meta learning algorithm for multiple iterations and test instances.

    Parameters:
    -----------
    - X_test:
        List of feature matrices for each test instance.
    - Y_test:
        List of reward vectors for each test instance.
    - B:
        Parameter matrix used for estimating rewards.
    - total_iterations:
        Number of iterations to run the regret calculation.

    Returns:
    --------
    - regrets:
        Array of average regret of users for each iteration.
    """
    regrets = np.zeros(total_iterations)

    for t in range(len(X_test)):
        X_test_array = np.array(X_test[t])
        Y_test_array = np.array(Y_test[t])

        n, d = X_test_array.shape

        arms = set(range(n))
        np.random.seed(42)
        arm = np.random.randint(0, n)
        arms.remove(arm)

        X_i = np.array([X_test_array[arm]])
        Y_i = np.array([Y_test_array[arm]])

        best_reward = np.max(Y_test_array)
        regrets[0] += best_reward - Y_test_array[arm]

        for i in range(1, total_iterations):
            X_iTX_i = X_i.T @ X_i
            BTX_iTX_iB = B.T @ X_iTX_i @ B
            BTX_iTY_i = B.T @ X_i.T @ Y_i

            reg = 1e-6 * np.eye(BTX_iTX_iB.shape[0])
            alpha_i = np.linalg.inv(BTX_iTX_iB + reg) @ BTX_iTY_i

            arm = max(arms, key=lambda a: np.dot(B @ alpha_i, X_test_array[a]))
            best_reward = np.max([Y_test_array[a] for a in arms])

            X_i = np.vstack([X_i, X_test_array[arm]])
            Y_i = np.append(Y_i, Y_test_array[arm])

            regrets[i] += best_reward - Y_test_array[arm]
            arms.remove(arm)
    return regrets / len(X_test)
