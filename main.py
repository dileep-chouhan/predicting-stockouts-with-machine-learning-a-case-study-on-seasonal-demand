import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_days = 365 * 3 # 3 years of data
dates = pd.date_range(start='2020-01-01', periods=num_days)
sales = 100 + 50 * np.sin(2 * np.pi * np.arange(num_days) / 365) + 20 * np.random.randn(num_days) # Seasonal trend + noise
sales = np.maximum(0, sales) # Ensure sales are non-negative
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# Add some features that might influence sales
df['Holiday'] = np.random.choice([0, 1], size=num_days, p=[0.9, 0.1]) # 10% chance of a holiday
df['Promotion'] = np.random.choice([0, 1], size=num_days, p=[0.8, 0.2]) # 20% chance of a promotion
# --- 2. Data Preparation ---
# Create lagged sales features
df['Sales_Lag1'] = df['Sales'].shift(1)
df['Sales_Lag7'] = df['Sales'].shift(7)
df['Sales_Lag30'] = df['Sales'].shift(30)
df = df.dropna() # Remove rows with NaN values due to lagging
# Define features (X) and target (y)
X = df[['Month', 'Year', 'Holiday', 'Promotion', 'Sales_Lag1', 'Sales_Lag7', 'Sales_Lag30']]
y = df['Sales']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training and Evaluation ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales')
plt.plot(y_pred, label='Predicted Sales')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'actual_vs_predicted_sales.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5. Stockout Prediction ---
# A simple example: predict stockouts based on a threshold
stockout_threshold = 50
predicted_stockouts = (y_pred < stockout_threshold).sum()
print(f"Predicted number of stockouts: {predicted_stockouts}")