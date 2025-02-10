import matplotlib.pyplot as plt
import pandas as pd
f=pd.read_excel('linear.xlsx',usecols=['Size','Price'])
print(f.head())
X=f['Size'].values
Y=f['Price'].values
print("X:",X)
print("Y:",Y)
xbar=sum(X)/len(X)
ybar=sum(Y)/len(Y)
n=len(X)
#y=b0+b1x
#b1=(n*∑𝑋𝑌-∑X∑Y)/n*∑X**2-(∑X)**2
#b0=Ybar-b1*xbar
sumofxy=0
sumofsquare=0
count=0
for i in range(len(X)):
    sumofxy+=X[i]*Y[i]
for i in range(len(X)):
    sumofsquare+=X[i]**2

b1=(n*sumofxy-sum(X)*sum(Y))/(n*sumofsquare - (sum(X))**2)
b0=ybar-b1*xbar
size=1400
predicted_price=b0+b1*size
print("Predicted price:",predicted_price)

#Rsquare=1-(SSres/SStot)
ss_total=sum((Y-ybar)**2)
ss_res=sum((Y-(b0+b1*X))**2)
r_square=1-(ss_res/ss_total)
print("R Square",r_square)

#RMSE= (∑(𝑦𝑖−(𝑦𝑖^))**2/n)**1/2
rmse=((sum(Y-(b0+b1*X))**2)/n)**1/2
print("RMSE:",rmse)

#MAE=∑|yi-ycapi|/n
mae=sum(abs(Y-(b0+b1*X)))/n
print("MAE:",mae)

plt.scatter(X,Y,color='blue',label='Given data')
plt.plot(X,b0+b1*X,color='red',label='regression line')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Linear regression')
plt.legend()
plt.show()






