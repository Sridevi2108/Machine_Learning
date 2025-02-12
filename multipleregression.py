import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f=pd.read_excel('multiple.xlsx',usecols=['Age','Mileage','Engine','NoofDoors','Price'])
print(f.head())
X = f[['Age', 'Mileage', 'Engine', 'NoofDoors']].values
Y=f['Price'].values
#Y=b0+b1X1+b2X2+⋯+bnXn+ϵ
#step1: Adding 1's to x since we have b0
X=np.column_stack([np.ones(X.shape[0]),X])
#step2: b=(xtranspose x)inverse * Xtranspose*Y
X_transpose=X.T
X_transposex=np.dot(X_transpose,X)
X_transposey=np.dot(X_transpose,Y)
weight=np.linalg.inv(X_transposex).dot(X_transposey)

b0,b1,b2,b3,b4=weight
print("Calculated Coefficients:")
print(f"b0: {b0}, b1: {b1}, b2: {b2}, b3: {b3}")

new_car = np.array([1, 4, 55000, 2.2, 4])
predicted_price = new_car.dot(weight)
print("Predicted Price for the car:", predicted_price)

predictions = X.dot(weight)

residuals = Y - predictions

SS_res = np.sum(residuals**2)
SS_tot = np.sum((Y - np.mean(Y))**2)

R_squared = 1 - (SS_res / SS_tot)
print("R-squared:", R_squared)

plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# Histogram of residuals to check normality
plt.hist(residuals, bins=10, edgecolor='k')
plt.title('Residuals Histogram')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


