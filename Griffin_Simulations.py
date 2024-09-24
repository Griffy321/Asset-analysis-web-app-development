# Checklist:
    # Single stock simulation 
    # Portfolio simulation
    # Log returns
    # Heston Volatility model
# To add: 
    # Stock returns corrolation

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# Function to fetch stock data and calculate mean return and volatility
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    
    # Calculate daily log returns
    daily_returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    
    # Mean return and volatility (annualized)
    annual_mean_return = daily_returns.mean() * 252
    annual_volatility = daily_returns.std() * np.sqrt(252)

    # Convert to daily values
    daily_mean_return = annual_mean_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Last closing price as the starting price
    last_price = hist['Close'][-1]
    
    return daily_mean_return, daily_volatility, last_price

# Monte Carlo simulation function
def monte_carlo_simulation(S0, mean_return, volatility, T, N):
    dt = T / N
    price = np.zeros(N)
    price[0] = S0

    # Simulate stock prices using random walk (Monte Carlo method)
    for i in range(1, N):
        z = np.random.normal(0, 1)
        price[i] = price[i-1] * np.exp((mean_return - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
    
    return price

# Streamlit UI
st.title("Welcome to Mr Griffin's Investing and Forecasting Site")

st.write("""
    None of the information or recommendations you receive from this website should be listened to, 
    you would be a fool to take financial advice solely off the predictions or 
    otherwise given by this website.
""")

# Sidebar for stock and portfolio simulations
st.sidebar.title("Menu")
st.sidebar.title("Capabilities")
Operation_option = st.sidebar.selectbox("Select an option:", ["Stock Return Simulation", "Portfolio Returns Simulation", "Stock Returns Correlation"])

# Stock Return Simulation
if Operation_option == "Stock Return Simulation":
    st.title("Stock Return Simulation")

    # Input: stock ticker and number of days
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
    num_days = st.slider("Select number of days to simulate:", 1, 3000, 365)

    if st.button("Run Stock Simulation"):
        try:
            # Fetch historical data
            mean_return, volatility, last_price = fetch_stock_data(ticker)
            st.write(f"Daily Mean Return: {mean_return}, Daily Volatility: {volatility}, Last Price: {last_price}")
            
            # Run 50 stock simulations using Monte Carlo method
            simulations = []
            final_returns = []
            for _ in range(50):
                simulation = monte_carlo_simulation(last_price, mean_return, volatility, num_days, num_days)
                simulations.append(simulation)
                final_return = (simulation[-1] - last_price) / last_price
                final_returns.append(final_return)
            
            # Plot the simulations
            plt.figure(figsize=(10,6))
            for simulation in simulations:
                plt.plot(simulation, lw=0.5)
            plt.title(f"Monte Carlo Simulation for {ticker} over {num_days} days")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.grid(True)
            st.pyplot(plt)
            
            # Calculate and display average return and standard deviation
            avg_return = np.mean(final_returns) * 100
            std_dev_return = np.std(final_returns) * 100
            st.write(f"Average Percentage Return: {avg_return:.2f}%")
            st.write(f"Standard Deviation of Returns: {std_dev_return:.2f}%")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Portfolio Returns Simulation
if Operation_option == "Portfolio Returns Simulation":
    st.header("Portfolio Returns Simulation")
    
    time_horizon = st.slider("Select number of days to simulate:", 1, 1000, 60)
    num_simulations = st.slider("Select number of simulations:", 10, 1000, 100)

    tickers = st.text_area("Enter up to 20 stock tickers separated by commas (e.g., AAPL, MSFT, GOOG):").split(",")
    tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]

    investments = []
    if tickers:
        for i, ticker in enumerate(tickers):
            investment = st.number_input(f"Investment in {ticker}:", min_value=0.0, value=1000.0, step=100.0)
            investments.append(investment)
    
    if st.button("Run Portfolio Simulation") and tickers:
        try:
            stock_data_dict = {}
            stats_dict = {}
            starting_balance = sum(investments)

            for i, ticker in enumerate(tickers):
                mean_return, sigma, last_close = fetch_stock_data(ticker)
                stats_dict[ticker] = {
                    'Mean Return': mean_return,
                    'Sigma': sigma,
                    'Last Close': last_close,
                    'Investment': investments[i]
                }

            plt.figure(figsize=(10,6))
            final_portfolio_values = []
            for sim in range(num_simulations):
                portfolio_values = np.zeros(time_horizon)
                for ticker in tickers:
                    S0 = stats_dict[ticker]['Last Close']
                    mu = stats_dict[ticker]['Mean Return']
                    sigma = stats_dict[ticker]['Sigma']
                    investment = stats_dict[ticker]['Investment']
                    n_shares = investment / S0
                    
                    # Simulate using the Monte Carlo method for individual stock
                    simulated_prices = monte_carlo_simulation(S0, mu, sigma, time_horizon, time_horizon)
                    stock_values = simulated_prices * n_shares
                    portfolio_values += stock_values
                
                final_portfolio_values.append(portfolio_values[-1])
                plt.plot(portfolio_values, lw=0.5, alpha=0.3)

            plt.title(f"Portfolio Value Simulation Over {time_horizon} Days ({num_simulations} Simulations)")
            plt.xlabel('Time Steps')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            st.pyplot(plt)

            avg_final_value = np.mean(final_portfolio_values)
            std_dev_final_value = np.std(final_portfolio_values)

            avg_return = (avg_final_value - starting_balance) / starting_balance * 100
            st.write(f"Average Final Portfolio Value: ${avg_final_value:.2f}")
            st.write(f"Standard Deviation of Final Portfolio Values: ${std_dev_final_value:.2f}")
            st.write(f"Average Portfolio Return: {avg_return:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Heston Model Simulation is omitted here but can be added similarly


elif Operation_option == "Stock Returns Correlation":
    st.header("Stock Returns Correlation")

    # Input for stock tickers and date range
    ticker1 = st.text_input("Enter first stock ticker (e.g., AAPL, MSFT):", "AAPL")
    ticker2 = st.text_input("Enter second stock ticker (e.g., MSFT, GOOGL):", "MSFT")
    start_date = st.date_input("Start date", pd.to_datetime('2017-09-21'))
    end_date = st.date_input("End date", pd.to_datetime('2024-09-21'))

    if st.button("Calculate Correlation"):
        try:
            # Function to fetch stock data
            def fetch_stock_data(tickers, start_date, end_date):
                data = yf.download(tickers, start=start_date, end=end_date)
                adj_close = data['Adj Close']
                returns = adj_close.pct_change().dropna()
                return returns

            # Fetch data and calculate correlation
            tickers = [ticker1, ticker2]
            returns = fetch_stock_data(tickers, start_date, end_date)
            correlation_matrix = returns.corr()
            correlation_coefficient = correlation_matrix.loc[tickers[0], tickers[1]]

            # Plot the correlation graph
            plt.figure(figsize=(10, 6))
            plt.scatter(returns[tickers[0]], returns[tickers[1]], alpha=0.5, color='purple', label=f'{tickers[0]} Returns')
            plt.scatter(returns[tickers[1]], returns[tickers[0]], alpha=0.5, color='orange', label=f'{tickers[1]} Returns')
            plt.title(f'Daily Returns of {tickers[0]} vs {tickers[1]}')
            plt.xlabel(f'{tickers[0]} Daily Returns')
            plt.ylabel(f'{tickers[1]} Daily Returns')
            plt.axhline(0, color='grey', lw=0.8)
            plt.axvline(0, color='grey', lw=0.8)
            plt.grid()
            plt.legend()
            st.pyplot(plt)

            # Function to classify correlation strength
            def correlation_strength(corr):
                if corr < 0:
                    correlation_type = 'Negative. Only buy the one you have more interest in, together they are a weak investment.'
                    if -0.2 <= corr > -0.19:
                        strength = 'Very weak negative. Only buy the one you have more interest in.'
                    elif -0.4 <= corr < -0.2:
                        strength = 'Weak negative. Buying one and selling the other would lead to a low Beta portfolio.'
                    elif -0.6 <= corr < -0.4:
                        strength = 'Moderate negative. Buying one and selling the other would lead to a very low Beta portfolio.'
                    elif -0.8 <= corr < -0.6:
                        strength = 'Strong negative. Consider buying one and shorting the other under the right conditions.'
                    elif -1 <= corr < -0.8:
                        strength = 'Very strong negative. Consider buying one and shorting the other under the right conditions.'
                else:
                    correlation_type = 'Positive'
                    if 0 <= corr < 0.2:
                        strength = 'Very weak. Only buy the one you have more interest in.'
                    elif 0.2 <= corr < 0.4:
                        strength = 'Weak. Buying both could reduce Beta slightly.'
                    elif 0.4 <= corr < 0.6:
                        strength = 'Moderate. Buying both could help diversify without hurting positive returns too much.'
                    elif 0.6 <= corr < 0.8:
                        strength = 'Strong. Buying both could diversify systematic risk.'
                    elif 0.8 <= corr <= 1:
                        strength = 'Very strong. Buying both could diversify systematic risk significantly.'
                    else:
                        strength = 'Undefined'
                
                return correlation_type, strength

            # Display correlation results
            correlation_type, strength = correlation_strength(correlation_coefficient)
            st.write(f'Correlation Coefficient between {tickers[0]} and {tickers[1]}: {correlation_coefficient:.4f}')
            st.write(f'The correlation is {correlation_type} and the strength is: {strength}')

        except Exception as e:
            st.error(f"An error occurred: {e}")


# To run the app:
# 1. Save this file as app.py
# 2. Open a terminal and run: streamlit run Griffin_Simulations.py
