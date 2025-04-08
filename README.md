# GARCH-BHHH
Estimate GARCH models using the BHHH algorithm. Output:

```
--- Simulation Series 1 ---
Iteration 0: logLik = -1761.9998, theta = [0.2 0.1 0.8]
Iteration 20: logLik = -1756.2891, theta = [0.07935555 0.03513497 0.92448429]
Convergence achieved at iteration 22.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.0794, alpha = 0.0351, beta = 0.9245

Simulated Returns Statistics:
Mean: -0.0593, Std: 1.4160, Excess Kurtosis: 0.0502
Acfs of Squared Returns (lags 110): 0.0412, 0.0858, 0.1039, 0.0728, 0.0775, 0.1139, 0.0473, 0.0882, 0.0436, 0.0621

True Conditional Std Dev Statistics:
Mean: 1.4075, Std: 0.1553, Excess Kurtosis: 3.4744

Estimated Conditional Std Dev Statistics:
Mean: 1.4030, Std: 0.1342, Excess Kurtosis: 2.9555

Correlation between True and Estimated Conditional Std: 0.9913

Normalized Returns (y/estimated_std) Statistics:
Mean: -0.0405, Std: 1.0021, Excess Kurtosis: -0.0353

--- Simulation Series 2 ---
Iteration 0: logLik = -1763.4567, theta = [0.2 0.1 0.8]
Iteration 20: logLik = -1762.1122, theta = [0.14240568 0.06641255 0.86368603]
Iteration 40: logLik = -1762.1116, theta = [0.14031383 0.06597169 0.86514988]
Iteration 60: logLik = -1762.1116, theta = [0.1400997  0.06592656 0.86529955]
Iteration 80: logLik = -1762.1116, theta = [0.14007791 0.06592197 0.86531478]
Convergence achieved at iteration 96.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.1401, alpha = 0.0659, beta = 0.8653

Simulated Returns Statistics:
Mean: 0.0140, Std: 1.4278, Excess Kurtosis: -0.0333
Acfs of Squared Returns (lags 110): 0.0834, 0.0209, 0.0862, 0.0743, 0.1080, 0.0765, 0.0903, 0.0495, 0.0103, 0.0414

True Conditional Std Dev Statistics:
Mean: 1.4119, Std: 0.1545, Excess Kurtosis: 0.7458

Estimated Conditional Std Dev Statistics:
Mean: 1.4172, Std: 0.1666, Excess Kurtosis: 0.9333

Correlation between True and Estimated Conditional Std: 0.9890

Normalized Returns (y/estimated_std) Statistics:
Mean: 0.0016, Std: 1.0011, Excess Kurtosis: -0.1476

--- Simulation Series 3 ---
Iteration 0: logLik = -1693.7541, theta = [0.2 0.1 0.8]
Convergence achieved at iteration 15.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.1209, alpha = 0.0000, beta = 0.9284

Simulated Returns Statistics:
Mean: 0.0896, Std: 1.2998, Excess Kurtosis: -0.0521
Acfs of Squared Returns (lags 110): -0.0018, -0.0306, 0.0121, -0.0167, 0.0392, -0.0289, 0.0447, -0.0472, 0.0182, 0.0144

True Conditional Std Dev Statistics:
Mean: 1.3559, Std: 0.0970, Excess Kurtosis: -0.1975

Estimated Conditional Std Dev Statistics:
Mean: 1.2995, Std: 0.0000, Excess Kurtosis: -0.0517

Correlation between True and Estimated Conditional Std: 0.9839

Normalized Returns (y/estimated_std) Statistics:
Mean: 0.0689, Std: 1.0002, Excess Kurtosis: -0.0521

#obs, #series: 1000 3
```
