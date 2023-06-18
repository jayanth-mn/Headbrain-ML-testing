#!/usr/bin/env python
# coding: utf-8

# In[1]:


x=100


# In[2]:


print(x)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0,10.0)

#reading data
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()


# In[5]:


#collecting X and y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values


# In[7]:


mean_x = np.mean(X) 
mean_y = np.mean(Y)

#total number of values 
m=len(X)

#using the formula to calculate b1 and b2
numer =0
denom =0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y) 
    denom += (X[i] - mean_x) ** 2
    
b1 = numer/denom 
b0 = mean_y - (b1 * mean_x)

#print coefficients 
print(b1,b0)


# In[11]:


#plotting values and regression line
max_x =np.max(X) + 100
min_x =np.min(X) - 100

#calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

#plotting line
plt.plot(x,y, color='#58b970',label='Regression line')
#plotting scatter points
plt.scatter(X,Y, c ='#ef5423', label='Scatter Plot')

plt.xlabel('Head size in cm')
plt.ylabel('Brain weight in grams')
plt.legend()
plt.show()


# In[12]:


ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 *X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2  = 1-(ss_r/ss_t)
print(r2)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#cannot use rank 1 matrix in scikit learn
X= X.reshape((m,1))
#creating model
reg = LinearRegression()
#fitting training data
reg = reg.fit(X,Y)
# y prediction
Y_pred = reg.predict(X)

# calculatin R2 score

r2_score = reg.score(X,Y)
 
print(r2_score)


# In[ ]:




