"""
Simulate multiple GARCH(1,1) time series, estimate model parameters via the BHHH algorithm with backtracking,
and compute statistical diagnostics for simulated returns, conditional standard deviations, and normalized returns.
"""

import numpy as np
from scipy.stats import kurtosis

# -------------------------------
# 1. Simulate GARCH(1,1) Data
# -------------------------------
def simulate_garch(T, omega, alpha, beta, seed=42):
    """
    Simulate a GARCH(1,1) time series.
    Parameters:
        T (int): Length of the time series.
        omega (float): Constant parameter.
        alpha (float): Parameter on lagged squared returns.
        beta (float): Parameter on lagged variance.
        seed (int, optional): Random seed (default is 42).
    Returns:
        y (np.ndarray): Simulated returns.
        h (np.ndarray): Simulated conditional variances.
    """
    np.random.seed(seed)
    y = np.zeros(T)
    h = np.zeros(T)
    h[0] = omega / (1 - alpha - beta)
    z = np.random.randn(T)
    y[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, T):
        h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
        y[t] = np.sqrt(h[t]) * z[t]
    return y, h

# -------------------------------
# 2. Compute Log-Likelihood and Gradients
# -------------------------------
def garch_loglik_grad(theta, y):
    """
    Compute the log-likelihood and its gradient for a GARCH(1,1) model.
    Parameters theta = [omega, alpha, beta].
    The first observation is treated as given (using a fixed initial variance).
    """
    T = len(y)
    omega, alpha, beta = theta

    # Check parameter constraints: omega > 0, alpha, beta >= 0, alpha+beta < 1.
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
        return -1e10, np.zeros(3), np.eye(3)
    
    h = np.zeros(T)
    dh_domega = np.zeros(T)
    dh_dalpha = np.zeros(T)
    dh_dbeta = np.zeros(T)
    
    h[0] = omega / (1 - alpha - beta)
    dh_domega[0] = 0.
    dh_dalpha[0] = 0.
    dh_dbeta[0] = 0.
    
    scores = np.zeros((T, 3))
    loglik = -0.5 * (np.log(2 * np.pi) + np.log(h[0]) + y[0]**2 / h[0])
    
    for t in range(1, T):
        h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
        if h[t] <= 0:
            return -1e10, np.zeros(3), np.eye(3)
        
        dh_domega[t] = 1 + beta * dh_domega[t-1]
        dh_dalpha[t] = y[t-1]**2 + beta * dh_dalpha[t-1]
        dh_dbeta[t] = h[t-1] + beta * dh_dbeta[t-1]
        
        loglik += -0.5 * (np.log(2 * np.pi) + np.log(h[t]) + y[t]**2 / h[t])
        dL_dh = 0.5 * (y[t]**2 / h[t]**2 - 1 / h[t])
        
        scores[t, 0] = dL_dh * dh_domega[t]
        scores[t, 1] = dL_dh * dh_dalpha[t]
        scores[t, 2] = dL_dh * dh_dbeta[t]
    
    grad = np.sum(scores[1:], axis=0)
    
    info_matrix = np.zeros((3, 3))
    for t in range(1, T):
        s = scores[t].reshape(3, 1)
        info_matrix += s @ s.T
        
    return loglik, grad, info_matrix

# -------------------------------
# 3. BHHH Algorithm with Backtracking Line Search
# -------------------------------
def bhhh_garch(y, initial_theta, tol=1e-6, max_iter=500):
    """
    Estimate GARCH(1,1) parameters using the BHHH algorithm with a backtracking line search.
    Parameters:
        y (np.ndarray): Time series returns.
        initial_theta (list or array): Initial parameter guess [omega, alpha, beta].
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    Returns:
        theta (np.ndarray): Estimated parameters.
        loglik_history (list): Log-likelihood values across iterations.
    """
    theta = np.array(initial_theta, dtype=float)
    loglik_history = []
    
    for it in range(max_iter):
        loglik, grad, info_matrix = garch_loglik_grad(theta, y)
        loglik_history.append(loglik)
        
        try:
            delta = np.linalg.solve(info_matrix, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(info_matrix).dot(grad)
        
        # Simple backtracking line search to retain valid parameter values.
        step_scaling = 1.0
        theta_new = theta + delta * step_scaling
        while not (theta_new[0] > 0 and theta_new[1] >= 0 and theta_new[2] >= 0 and (theta_new[1] + theta_new[2] < 1)):
            step_scaling *= 0.5
            theta_new = theta + delta * step_scaling
            if step_scaling < 1e-8:
                break
        
        if it % 20 == 0:
            print(f"Iteration {it}: logLik = {loglik:.4f}, theta = {theta}")
        
        if np.linalg.norm(delta) * step_scaling < tol:
            print(f"Convergence achieved at iteration {it}.")
            theta = theta_new
            break
        
        theta = theta_new
    
    return theta, loglik_history

# -------------------------------
# 4. Compute Conditional Variances
# -------------------------------
def compute_garch_variance(y, theta):
    """
    Compute conditional variances for a GARCH(1,1) model given returns and parameters.
    Parameters:
        y (np.ndarray): Time series returns.
        theta (list or array): Model parameters [omega, alpha, beta].
    Returns:
        h (np.ndarray): Computed conditional variances.
    """
    T = len(y)
    omega, alpha, beta = theta
    h = np.zeros(T)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, T):
        h[t] = omega + alpha * y[t-1]**2 + beta * h[t-1]
    return h

# -------------------------------
# 5. Acf Helper
# -------------------------------
def acf(data, max_lag=10):
    ac = {}
    for lag in range(1, max_lag+1):
        ac[lag] = np.corrcoef(data[lag:], data[:-lag])[0, 1]
    return ac

# -------------------------------
# 6. Main Program Loop for Multiple Series
# -------------------------------
if __name__ == '__main__':
    # True parameters for simulation
    true_omega = 0.1
    true_alpha = 0.05
    true_beta  = 0.9
    T = 1000
    n_series = 3 # Number of series to simulate and fit

    for sim in range(n_series):
        seed = 123 + sim  # Varying seed for each simulation
        print(f"\n--- Simulation Series {sim+1} ---")
        y, h_true = simulate_garch(T, true_omega, true_alpha, true_beta, seed=seed)
        
        # Initial guess for [omega, alpha, beta]
        initial_theta = [0.2, 0.1, 0.8]
        estimated_theta, loglik_history = bhhh_garch(y, initial_theta)
        
        print("\nTrue parameters:")
        print(f"omega = {true_omega}, alpha = {true_alpha}, beta = {true_beta}")
        
        print("\nEstimated parameters:")
        print(f"omega = {estimated_theta[0]:.4f}, alpha = {estimated_theta[1]:.4f}, beta = {estimated_theta[2]:.4f}")
        
        # --- Statistics for Simulated Returns ---
        sim_mean = np.mean(y)
        sim_std = np.std(y)
        sim_kurt = kurtosis(y, fisher=True)  # Excess kurtosis (kurtosis - 3)
        ac_simulated_sq = acf(y**2, max_lag=10)
        print("\nSimulated Returns Statistics:")
        print(f"Mean: {sim_mean:.4f}, Std: {sim_std:.4f}, Excess Kurtosis: {sim_kurt:.4f}")
        print("Acfs of Squared Returns (lags 1–10):", 
              ", ".join(f"{ac_simulated_sq[lag]:.4f}" for lag in range(1, 11)))
        
        # --- True Conditional Standard Deviation Statistics ---
        true_std = np.sqrt(h_true)
        print("\nTrue Conditional Std Dev Statistics:")
        print(f"Mean: {np.mean(true_std):.4f}, Std: {np.std(true_std):.4f}, Excess Kurtosis: {kurtosis(true_std, fisher=True):.4f}")
        
        # --- Estimated Conditional Standard Deviation Statistics ---
        h_est = compute_garch_variance(y, estimated_theta)
        est_std = np.sqrt(h_est)
        print("\nEstimated Conditional Std Dev Statistics:")
        print(f"Mean: {np.mean(est_std):.4f}, Std: {np.std(est_std):.4f}, Excess Kurtosis: {kurtosis(est_std, fisher=True):.4f}")
        
        # --- Correlation between True and Estimated Conditional Std Dev ---
        corr_std = np.corrcoef(true_std, est_std)[0, 1]
        print(f"\nCorrelation between True and Estimated Conditional Std: {corr_std:.4f}")
        
        # --- Statistics for Normalized Returns (y / estimated_std) ---
        norm_returns = y / est_std
        print("\nNormalized Returns (y/estimated_std) Statistics:")
        print(f"Mean: {np.mean(norm_returns):.4f}, Std: {np.std(norm_returns):.4f}, Excess Kurtosis: {kurtosis(norm_returns, fisher=True):.4f}")

    print("\n#obs, #series:", T, n_series)
