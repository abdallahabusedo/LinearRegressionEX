import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']= (20.0,10.0)

#reading data
data=pd.read_csv("headbrain.csv")
print(data.shape)
data.head()

#Callecting X & Y
X =data["Head Size(cm^3)"].values
Y =data["Brain Weight(grams)"].values

# Mean X & Y
MeanX = np.mean(X)
MeanY = np.mean(Y)

#total number of values
n= len(X)

# using the formula to calculate B1 and B0 
numer = 0 
denom  = 0
for i in range(n):
    numer +=(X[i]-MeanX)*(Y[i]-MeanY)
    denom +=(X[i]-MeanX)**2
B1 = numer /denom
B0 = MeanY - (B1 * MeanX)

#print Cofficients 
print(B1, B0)

#plotting Values and Regression Line 
MaxX= np.max(X)+100
MinX=np.min(X)-100

#Calculating Line values X and Y
x = np.linspace(MinX,MaxX,1000)
y =B0+B1*x

#ploting line 
plt.plot(x,y,color='#58b970',label="Regression")
#ploting scatter points
plt.scatter(X,Y,color="#ef5423", label="Scatter Plot")

plt.xlabel("Head Size in Cm3")
plt.ylabel("Brain Weght in grams")
plt.legend()
plt.show()

ss_t = 0 
ss_r = 0 
for i in range(n):
    y_pred = B0 +B1 * X[i]
    ss_t += (Y[i] - MeanY) **2
    ss_r += (Y[i] - y_pred) **2
r2 = 1- (ss_r/ss_t)
print(r2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Cannot use Rank 1 matrix in scikit learn 
X= X.reshape((n,1))
#creating Model
reg=LinearRegression()
#fitting training data
reg = reg.fit(X,Y)
#Y prediction 
y_pred = reg.predict(X)
#calculating R2 Score
R2_score=reg.score(X,Y)
print(R2_score)
