import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Function to retrieve maximum available stock data
def get_max_stock_history(ticker):
    stock_data = yf.download(ticker, period="max")  # Fetches maximum available history
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    mean_return = stock_data['Daily Return'].mean()
    volatility = stock_data['Daily Return'].std()
    last_price = stock_data['Adj Close'][-1]
    return mean_return, volatility, last_price

# Monte Carlo Simulation with Heston Model
def monte_carlo_heston(ticker, num_simulations=1000, steps=254, dt=0.00198, kappa=1.0, theta=0.04, xi=0.1, rho=-0.7, v0=0.04):
    mean_return, _, S0 = get_max_stock_history(ticker)

    # Arrays to store the results of all simulations
    all_simulations = np.zeros((num_simulations, steps))

    for simulation in range(num_simulations):
        # Arrays for price and variance in this simulation
        price = np.zeros(steps)
        variance = np.zeros(steps)
        price[0] = S0
        variance[0] = v0

        increments_S = np.random.normal(0, 1, steps)
        increments_v = np.random.normal(0, 1, steps)

        for i in range(1, steps):
            # Update variance using the Heston model SDE
            dv = kappa * (theta - variance[i-1]) * dt + xi * np.sqrt(variance[i-1]) * np.sqrt(dt) * increments_v[i]
            variance[i] = max(variance[i-1] + dv, 0)  # Ensuring variance remains non-negative
            
            # Update price using the stock price SDE with stochastic volatility
            dS = mean_return * dt + np.sqrt(variance[i-1]) * np.sqrt(dt) * increments_S[i]
            price[i] = price[i-1] * np.exp(dS)

        all_simulations[simulation] = price

    # Calculate the average price path across all simulations
    average_price = np.mean(all_simulations, axis=0)

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(average_price, lw=2, label="Average Price Path")
    for i in range(min(num_simulations, 10)):  # Plot a few individual simulations for reference
        plt.plot(all_simulations[i], lw=0.5, alpha=0.6)
    plt.title(f'{ticker} Monte Carlo Simulation (Heston Model)')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example usage
monte_carlo_heston('TSLA')

