# GARCH-BHHH
Estimate GARCH models using the BHHH algorithm. Output:

```
--- Simulation Series 1 ---
Iteration 0: logLik = -17534.2088, theta = [0.2 0.1 0.8]
Convergence achieved at iteration 12.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.0886, alpha = 0.0437, beta = 0.9114

Simulated Returns Statistics:
Mean: 0.0132, Std: 1.4092, Excess Kurtosis: 0.1577
Acfs of Squared Returns (lags 110): 0.0702, 0.0541, 0.0752, 0.0726, 0.0477, 0.0585, 0.0734, 0.0558, 0.0545, 0.0708

True Conditional Std Dev Statistics:
Mean: 1.4032, Std: 0.1548, Excess Kurtosis: 4.1896

Estimated Conditional Std Dev Statistics:
Mean: 1.3995, Std: 0.1478, Excess Kurtosis: 4.3529

Correlation between True and Estimated Conditional Std: 0.9983

Normalized Returns (y/estimated_std) Statistics:
Mean: 0.0097, Std: 1.0002, Excess Kurtosis: -0.0308

--- Simulation Series 2 ---
Iteration 0: logLik = -17309.2966, theta = [0.2 0.1 0.8]
Convergence achieved at iteration 8.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.1178, alpha = 0.0492, beta = 0.8885

Simulated Returns Statistics:
Mean: -0.0046, Std: 1.3778, Excess Kurtosis: 0.1713
Acfs of Squared Returns (lags 110): 0.0750, 0.0749, 0.0714, 0.0690, 0.0720, 0.0554, 0.0638, 0.0495, 0.0492, 0.0396

True Conditional Std Dev Statistics:
Mean: 1.3883, Std: 0.1479, Excess Kurtosis: 5.3606

Estimated Conditional Std Dev Statistics:
Mean: 1.3696, Std: 0.1378, Excess Kurtosis: 5.5972

Correlation between True and Estimated Conditional Std: 0.9985

Normalized Returns (y/estimated_std) Statistics:
Mean: -0.0043, Std: 1.0001, Excess Kurtosis: 0.0087

--- Simulation Series 3 ---
Iteration 0: logLik = -17505.4045, theta = [0.2 0.1 0.8]
Convergence achieved at iteration 9.

True parameters:
omega = 0.1, alpha = 0.05, beta = 0.9

Estimated parameters:
omega = 0.0918, alpha = 0.0482, beta = 0.9053

Simulated Returns Statistics:
Mean: 0.0102, Std: 1.4144, Excess Kurtosis: 0.3305
Acfs of Squared Returns (lags 110): 0.1322, 0.1237, 0.1099, 0.1006, 0.1169, 0.0841, 0.0754, 0.0692, 0.0709, 0.0918

True Conditional Std Dev Statistics:
Mean: 1.4040, Std: 0.1717, Excess Kurtosis: 11.5353

Estimated Conditional Std Dev Statistics:
Mean: 1.3994, Std: 0.1725, Excess Kurtosis: 11.4058

Correlation between True and Estimated Conditional Std: 0.9997

Normalized Returns (y/estimated_std) Statistics:
Mean: 0.0075, Std: 1.0001, Excess Kurtosis: -0.0070

#obs, #series: 10000 3
```
