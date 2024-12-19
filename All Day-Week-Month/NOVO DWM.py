import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


novo_nordisk_data = yf.download(
    "NVO", start="2023-01-01", end="2023-12-31", progress=False)


def GBM(S_0, drift, sigma, T, N):
    dt = T / N
    prices = [S_0]
    for _ in range(N):
        Z = np.random.normal()
        S_t = prices[-1] * np.exp((drift - 0.5 * sigma**2)
                                  * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_t)
    return np.array(prices)


def calculate_parameters(data, time_interval):
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    u_bar = log_returns.mean()
    v = np.sqrt(((log_returns - u_bar) ** 2).sum() / (len(log_returns) - 1))
    sigma_star = v / np.sqrt(time_interval)
    drift = u_bar / time_interval + 0.5 * (sigma_star**2)
    return drift, sigma_star


actual_price = novo_nordisk_data['Close'].iloc[-1]

s0_points = {
    '1-day': novo_nordisk_data['Close'].iloc[-2],
    '1-week': novo_nordisk_data['Close'].iloc[-6],
    '1-month': novo_nordisk_data['Close'].iloc[-21]
}


time_horizons = {'1-day': 1/252, '1-week': 1/52, '1-month': 1/12}


results = {}

for horizon, time_interval in time_horizons.items():
    S_0 = s0_points[horizon]
    drift, sigma_star = calculate_parameters(novo_nordisk_data, time_interval)
    simulations = [GBM(S_0, drift, sigma_star, time_interval, 20)
                   for _ in range(1000)]
    final_prices = [sim[-1] for sim in simulations]

    ci_lower = np.percentile(final_prices, 2.5)
    ci_upper = np.percentile(final_prices, 97.5)

    results[horizon] = {
        'S_0': S_0,
        'CI': (ci_lower, ci_upper),
        'Coverage': ci_lower <= actual_price <= ci_upper,
        'MSE': np.mean((np.array(final_prices) - actual_price) ** 2),
        'Mean Predicted Price': np.mean(final_prices),
        'Drift': drift,
        'Sigma': sigma_star
    }


for horizon, res in results.items():
    print(f"\nTime Horizon: {horizon}")
    print(f"  Starting Price (S_0): {res['S_0']:.2f}")
    print(f"  Actual Closing Price: {actual_price:.2f}")
    print(f"  95% Confidence Interval: {res['CI']}")
    print(f"  Coverage (Actual Price in CI): {res['Coverage']}")
    print(f"  Mean Squared Error: {res['MSE']:.4f}")
    print(f"  Mean Predicted Price: {res['Mean Predicted Price']:.2f}")
    print(f"  Drift: {res['Drift']:.4f}, Sigma*: {res['Sigma']:.4f}")


plt.figure(figsize=(10, 6))
for horizon, time_interval in time_horizons.items():
    S_0 = s0_points[horizon]
    drift, sigma_star = calculate_parameters(novo_nordisk_data, time_interval)
    S_t = GBM(S_0, drift, sigma_star, time_interval, 20)
    plt.plot(range(len(S_t)), S_t, label=f"{horizon}")

plt.title("GBM Simulations for Novo Nordisk (NVO)")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
