# Ex.No: 6               HOLT WINTERS METHOD
### AIM:To implement the Holt Winters Method Model using Python.

### Date: 
### Developed by: ANUSHARON.S
### Register no.: 212222240010



### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import math
sales_data = pd.read_csv('/content/daily_sales_data.csv', parse_dates=['date'], index_col='date')
print(sales_data.head())
print(sales_data.describe())
monthly_sales_data = sales_data['sales'].resample('MS').sum()  # 'MS' stands for Month Start
print(monthly_sales_data.head())
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales_data, label='Monthly Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
decomposition = seasonal_decompose(monthly_sales_data, model='additive')
decomposition.plot()
plt.show()
def rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))
model = ExponentialSmoothing(monthly_sales_data, trend='additive', seasonal='additive', seasonal_periods=12)
hw_model = model.fit()
predictions = hw_model.forecast(12)
print(predictions)
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales_data, label='Original Sales Data')
plt.plot(predictions, label='Holt-Winters Predictions', color='red')
plt.title('Holt-Winters Predictions vs Original Sales Data')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
fitted_values = hw_model.fittedvalues
error = rmse(monthly_sales_data, fitted_values)
print(f"Root Mean Squared Error (RMSE): {error}")
mean_sales = monthly_sales_data.mean()
std_sales = monthly_sales_data.std()

print(f"Mean Sales: {mean_sales}")
print(f"Standard Deviation of Sales: {std_sales}")
```

### OUTPUT:


TEST_PREDICTION

![Screenshot 2024-10-16 132552](https://github.com/user-attachments/assets/202f2196-ee5a-4602-9280-b17c63b6f44a)


FINAL_PREDICTION
![Screenshot 2024-10-16 132618](https://github.com/user-attachments/assets/432f9b77-da1e-48a6-aff8-0350502d028a)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
