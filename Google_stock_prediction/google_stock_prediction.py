import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


data = pd.read_csv("googl_data_2020_2025.csv")

print(data.head())

'''plt.style.use("fivethirtyeight")
plt.figure(figsize=(16, 8))
plt.title("Google Stock Price 2020-2025")
plt.plot(data["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()'''

data = data[["Price","Close"]]
data = data.rename(columns={"Price": "ds", "Close": "cs"})
print(data.head())

series = data.set_index("ds")["cs"]

model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

result = model.fit(disp=False)

forecast = result.get_forecast(steps=365)
forecast_df = forecast.summary_frame()

plt.figure(figsize=(16, 8))
plt.plot(series, label="Observed")
plt.plot(forecast_df.index, forecast_df["mean"],label="Forecast")
plt.fill_between(forecast_df.index, 
                 forecast_df["mean_ci_lower"], 
                 forecast_df["mean_ci_upper"], 
                 alpha=0.3)

plt.title("Google Stock Price Forecast")
plt.xlabel("Date")  
plt.ylabel("Close Price USD ($)")
plt.legend()
plt.show()

#https://github.com/ashishpatel26