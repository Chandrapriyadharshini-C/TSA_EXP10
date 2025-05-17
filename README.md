# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 17/05/2025

### AIM:
To implement SARIMA model using python.

### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions

### PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/weather_classification_data.csv')

# Step 1: Data Preprocessing
# We'll assume each entry represents daily data and focus on "Temperature".
# Adding a DateTime index if none is present.
data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
data.set_index('Date', inplace=True)

# Step 2: Check Stationarity
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary. Differencing may be needed.")

check_stationarity(data['Temperature'])

# Step 3: Differencing if needed
data['Temperature_diff'] = data['Temperature'].diff().dropna()
check_stationarity(data['Temperature_diff'].dropna())

# Step 4: Plot ACF and PACF to determine p, d, q parameters
plt.figure(figsize=(12,6))
plt.subplot(121)
plot_acf(data['Temperature_diff'].dropna(), ax=plt.gca(), lags=30)
plt.subplot(122)
plot_pacf(data['Temperature_diff'].dropna(), ax=plt.gca(), lags=30)
plt.show()

# Step 5: Fit SARIMA Model (Example parameters p=1, d=1, q=1; adjust as needed)
model = SARIMAX(data['Temperature'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Step 6: Forecasting
forecast = model_fit.get_forecast(steps=30)
forecast_ci = forecast.conf_int()

# Plotting the results
plt.figure(figsize=(10,5))
plt.plot(data['Temperature'], label='Observed')
plt.plot(forecast.predicted_mean, color='r', label='Forecast')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink')
plt.legend()
plt.show()

# Step 7: Model Evaluation
train_size = int(len(data) * 0.8)
train, test = data['Temperature'][:train_size], data['Temperature'][train_size:]
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)
predictions = model_fit.forecast(len(test))
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/771f9216-4c0a-45d4-a8f4-2f5c59faec66)

![image](https://github.com/user-attachments/assets/245e86ce-48a6-4173-8759-a1e433924eb2)

![image](https://github.com/user-attachments/assets/119ca345-2545-42c5-b162-09ab8feebd07)

### RESULT:
Thus the program run successfully based on the SARIMA model.
