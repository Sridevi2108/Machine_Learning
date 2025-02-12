import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv('dogs_dataset.csv')
np.random.seed(42)
f['Price'] = f['Age (Years)'] * f['Weight (kg)'] + np.random.normal(0, 50, size=f.shape[0])
f_encoded = pd.get_dummies(f, columns=['Breed', 'Color', 'Gender'], drop_first=True)

f_encoded = f_encoded.astype(np.float64)
print(f_encoded.dtypes)
X = f_encoded.drop('Price', axis=1).values  # All columns except 'Price' will be features
Y = f_encoded['Price'].values
X = np.column_stack([np.ones(X.shape[0]), X])
X_transpose = X.T
X_transposeX = np.dot(X_transpose, X)
X_transposeY = np.dot(X_transpose, Y)
X_transposeX = X_transposeX.astype(np.float64)
X_transposeY = X_transposeY.astype(np.float64)

weights = np.linalg.inv(X_transposeX).dot(X_transposeY)
b0, *b_values = weights
print("Calculated Coefficients:")
print(f"b0: {b0}, b1, b2, ..., bn: {b_values}")
new_dog = {
    'Age (Years)': 3,
    'Weight (kg)': 15,
    'Breed': 'Labrador Retriever',
    'Color': 'Brown',
    'Gender': 'Male'
}
new_dog_df = pd.DataFrame([new_dog])
new_dog_encoded = pd.get_dummies(new_dog_df, columns=['Breed', 'Color', 'Gender'], drop_first=True)
missing_cols = set(f_encoded.columns) - set(new_dog_encoded.columns)
for col in missing_cols:
    new_dog_encoded[col] = 0
new_dog_encoded = new_dog_encoded[f_encoded.columns.difference(['Price'])]

new_dog_array = np.array([1] + new_dog_encoded.values.flatten().tolist())

predicted_price = new_dog_array.dot(weights)
print("Predicted Price for the dog:", predicted_price)

predictions = X.dot(weights)
residuals = Y - predictions

SS_res = np.sum(residuals**2)
SS_tot = np.sum((Y - np.mean(Y))**2)
R_squared = 1 - (SS_res / SS_tot)
print("R-squared:", R_squared)

# Step 16: Plot Residual Analysis
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

plt.hist(residuals, bins=10, edgecolor='k')
plt.title('Residuals Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
