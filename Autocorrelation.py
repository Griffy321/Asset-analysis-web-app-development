from polygon import RESTClient
from typing import cast
from urllib3 import HTTPResponse
import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# API setup and data retrieval
api_key = 'your polygon api key'
client = RESTClient(api_key)

# Get the current date and subtract one day to get the previous date
end_date = datetime.now().date()  # Current date
start_date = end_date - timedelta(days=1)  # Previous date

# Convert dates to strings in the format 'YYYY-MM-DD'
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Fetch the data
aggs = cast(
    HTTPResponse,
    client.get_aggs(
        'META',  
        1,
        'minute',
        start_date_str,
        end_date_str,
        raw=True
    ),
)

poly_data = json.loads(aggs.data)
poly_data = poly_data['results']

# Check the length of the data to ensure full period coverage
print(f"Number of data points fetched: {len(poly_data)}")

# Prepare the data for DataFrame
dates = []
open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []

for bar in poly_data:
    dates.append(pd.Timestamp(bar['t'], tz='GMT', unit='ms'))
    open_prices.append(bar['o'])
    high_prices.append(bar['h'])
    low_prices.append(bar['l'])
    close_prices.append(bar['c'])
    volumes.append(bar['v'])

data = {
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volumes
}

dataFrame = pd.DataFrame(data, index=dates)

# Calculate returns (percentage change in closing prices)
dataFrame['Returns'] = dataFrame['Close'].pct_change()

# Create a figure and subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# 1. Plot the line chart for the closing prices
ax[0].set_title('NVDA Line Chart - Closing Prices')
ax[0].plot(dataFrame.index, dataFrame['Close'], label='Close Price')
ax[0].set_ylabel('Price')
ax[0].legend()

# 2. Calculate and plot rolling autocorrelation of returns
rolling_autocorr = dataFrame['Returns'].rolling(window=30).apply(lambda x: x.autocorr(), raw=False)
rolling_autocorr = rolling_autocorr.dropna()  # Drop NaN values from autocorrelation
ax[1].plot(rolling_autocorr, label='Rolling Autocorrelation (30-minute window)')
ax[1].set_title('Rolling Autocorrelation of Returns')
ax[1].set_ylabel('Autocorrelation')
ax[1].legend()

# 3. Plot the volume data
ax[2].bar(dataFrame.index, dataFrame['Volume'], width=0.0005, label='Volume', color='orange')
ax[2].set_title('Volume over Time')
ax[2].set_ylabel('Volume')
ax[2].legend()

# Format the x-axis for better readability (time)
plt.tight_layout()
plt.show()
