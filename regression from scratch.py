# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:59:38 2019

@author: Ahmed Khaled
"""

#import libararies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
path = 'C:\\Users\\Ahmed Khaled\\Downloads\\house-prices-advanced-regression-techniques\\regression_data1.txt'
#Read Data
data = pd.read_csv(path,header=None ,names=['population','profit'])
#show data details 
print('data\n',data)
print('******************************************************************')
print('data head(10) = \n',data.head(10))
print('******************************************************************')
#Data describe to get mean ,max , min , avrg, ........
print('data.describe = \n',data.describe())
print('******************************************************************')
#drow data 
data.plot(kind = 'scatter' , x ='population',y = 'profit',figsize=(5,5))
#adding anew column called ones befor the data (x0)
data.insert(0,'ones',1)
print('new data = \n',data.head(10))
print('******************************************************************')
#seperate x (traning data) from y (target data)
cols = data.shape[1]         # no of cols in your data = 3
print('cols = \n' ,cols)
x = data.iloc[:,0:cols-1]      # 0,1
y = data.iloc[:,cols-1:cols]  #2:3
print('******************************************************************')
print('x data =\n ' , x.head(10))
print('y data =\n ' , y.head(10))
print('******************************************************************')
#convert from data frames to numpy matrices 
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))  # make matrix with two variable for theta1 , theta2 and those ara assumptions 
print('theta.shape = \n',theta.shape)
print('x = \n',x)
print('x.shape =\n',x.shape)
print('theta =\n',theta)
print('y = \n',y)
print('y.shape =\n',y.shape)
print('******************************************************************')
# cost function 
def computeCost(x, y, theta):
    z = np.power(((x * theta.T) - y), 2)
    #print('z = \n ',z)
    #print('m  = \n',len(x)) # m is no of rows 
    return np.sum(z) / (2 * len(x))
    
def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape)) # thetas are temparory 
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (x * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(x, y, theta)
        
    return theta, cost
# intialize variable for learning rate and iterations 
alpha = 0.01
iters = 1000
#perform gradient descent to fit the model parameter
g, cost = gradientDescent(x, y, theta, alpha, iters)
print('g= \n',g)
print('cost = \n',cost[0:50])
print('computecost = \n',computeCost(x,y,g))
print('******************************************************************')
# get best fit line for Size vs. Price
x = np.linspace(data.population.min(), data.population.max(), 100)
print('x \n',x)
print('g \n',g)


f = g[0, 0] + (g[0, 1] * x)      
print('f \n',f)               # linear equations , h(x) = theta 0 + theta 1 * x

# draw the line 
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('population vs. profit')
#draw error graph 
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')






