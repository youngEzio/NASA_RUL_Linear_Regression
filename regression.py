import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('nasa_rul_data.csv')

# Select the input features
X = df[['sensor1', 'sensor2', 'sensor3']]

# Select the target variable
y = df['RUL']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.mean((y_pred - y_test)**2)
rmse = np.sqrt(mse)

print('RMSE on test set:', rmse)
