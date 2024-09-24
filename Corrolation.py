import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the stock tickers and the date range
tickers = ['ASTS', 'UNH']
start_date = '2017-09-21'
end_date = '2024-09-21'

# Fetch the stock data
data = yf.download(tickers, start=start_date, end=end_date)

# Get the adjusted close prices
adj_close = data['Adj Close']

# Calculate daily returns
returns = adj_close.pct_change().dropna()

# Calculate the correlation coefficient
correlation_matrix = returns.corr()
correlation_coefficient = correlation_matrix.loc[tickers[0], tickers[1]]
print(f'Correlation Coefficient between {tickers[0]} and {tickers[1]}: {correlation_coefficient:.4f}')

# Function to classify correlation strength
def correlation_strength(corr):
    if corr < 0:
        correlation_type = 'Negative. Only buy the one you have more intrest in, together they are a weak investment.'
        if -0.2 <= corr > -0.19:
            strength = 'Very weak negative. Only buy the one you have more intrest in, together they are a weak investment if gains are your top priority.'
        elif -0.4 <= corr < -0.2:
            strength = 'Weak negative. Buying one and selling the other would lead to a decently low Beta portfolio.'
        elif -0.6 <= corr < -0.4:
            strength = 'Moderate negative. Buying one and selling the other would lead to a very low Beta portfolio.'
        elif -0.8 <= corr < -0.6:
            strength = 'Strong negative. Think about buying one and shorting the other under the right economic conditions'
        elif -1 <= corr < -0.8:
            strength = 'Very strong negative. Think about buying one and shorting the other under the right economic conditions'
    else:
        correlation_type = 'Positive'
        if 0 <= corr < 0.2:
            strength = 'Very  weak. Only buy the one you have more intrest in, together they are a weak investment but give you a lower Beta.'
        elif 0.2 <= corr < 0.4:
            strength = 'Weak. Only buy the one you have more intrest in, together they are a weak investment if gains are your top priority but lower your Beta.'
        elif 0.4 <= corr < 0.6:
            strength = 'Moderate. Buying both could help diversify your portfolio without hurting your Beta too much.'
        elif 0.6 <= corr < 0.8:
            strength = 'Strong. You may want to buy both of these stocks to diversify some systematic risk away.'
        elif 0.8 <= corr <= 1:
            strength = 'Very strong. You may want to buy both of these stocks to diversify some systematic risk away.'
        else:
            strength = 'Undefined'
    
    return correlation_type, strength

# Determine the strength and type of correlation
correlation_type, strength = correlation_strength(correlation_coefficient)
print(f'The correlation is {correlation_type} and the strength is: {strength}')

# Plotting the returns
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
plt.show()
