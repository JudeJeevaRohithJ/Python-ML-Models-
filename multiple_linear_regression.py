#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# ## Importing the libraries

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[30]:


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# The dataset Startups has been imported from the folder in the local files.

# Interpretation:
# The above code separates the dataset into X,y that means dependent and independant variables.

# In[31]:


print(X)


# ##### This display the training data

# ## Encoding categorical data

# In[15]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# ## **One Hot Encoding**
# One hot encoding can be defined as the essential process of converting the categorical data variables to be provided to machine and deep learning algorithms which in turn improve predictions as well as classification accuracy of a model.

# In[16]:


print(X)


# ## Splitting the dataset into the Training set and Test set

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# The train-test split is a technique for evaluating the performance of a machine learning algorithm. Here sklearn library is used to import the model selection and thus the dataset splitted into train and test data.

# ## Training the Multiple Linear Regression model on the Training set

# In[18]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# **LinearRegression** fits a linear model with coefficients  to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

# ## Predicting the Test set results

# In[19]:


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[20]:


regressor.coef_


# In[21]:


regressor.intercept_


# In[33]:


plt.scatter(X_test[:,5],y_pred, color = 'brown')


# In[23]:


plt.scatter(X_test[:,3],y_pred, color = 'blue')


# In[32]:


plt.scatter(X_test[:,4],y_pred, color = 'red')


# Finally the dataset tested and fit into the model and shown in visualization form.
