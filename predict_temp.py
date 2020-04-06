#!/usr/bin/env python
# coding: utf-8

# In[2]:


###Basic Getting started with pandas and numpy
###pandas for data cleaning and visualization
### Create a Series(Numpy arrays similar to python arrays)

import pandas as pd
import numpy as np

s  = pd.Series(np.random.randn(6),index = ['a','b','c','d','e','f'])
s1 = pd.Series([0,1,2,3,4,5] , index = ['a','b','c','d','e','f'])

print (s)
print ('\n')
print (s1)


# In[3]:


#create a DataFrame(Multiple SERIES in a table)

df = pd.DataFrame(s, columns = ["Column1"])
df


# In[4]:


df['Column1']
df["Column2"] = df["Column1"] * 4
df


# In[5]:


#mean = np.mean(x)
df.apply(lambda x :min(x)+ max(x))


# In[6]:


Mean = df.apply(lambda x :np.mean(x))
print(Mean)

##Describe 

#help(df.describe) 


# In[7]:


###Download data (from kaggle) in CSV format .Create a data frame

###data_frame is a one dimentional matrix

import matplotlib.pyplot as plt  #ec(%matplotlib inline)

DF= pd.read_csv("/home/user/Desktop/Python/data/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")


# In[8]:


# Let's just consider the LandAverageTemperature
# "A primarily label-location based indexer

DF = DF.iloc[:,:2]
print (DF.head())
print (DF.tail())


# In[9]:


DF.describe()


# In[10]:


plt.figure(figsize=(20,5))
plt.plot(DF["LandAverageTemperature"])
plt.title("Average Land Temperature 1750 - 2015")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.show()


# In[11]:


###Complex scatter plot by year


plt.figure(figsize=(20,5))
plt.scatter(x = DF["LandAverageTemperature"].index, y = DF["LandAverageTemperature"])
plt.title("Average Land Temperature 1750 - 2015")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.show()


# In[12]:


###Histogram plot by year This is wrong most likely
print (DF)

plt.figure(figsize=(20,5))
plt.hist(DF["LandAverageTemperature"])
plt.title("Average Land Temperature 1750 - 2015")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.show()


# In[13]:



print (DF.count())
print (DF.head(10))


# In[14]:


times = pd.DatetimeIndex(DF['dt'])

grouped = DF.groupby([times.year]).mean()


# In[15]:


#Plot a New Graph With NewlyFilled data
plt.figure(figsize=(20,5))
plt.plot(grouped["LandAverageTemperature"])
plt.title("Average Land Temperature 1750 - 2015")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.show()


# In[16]:


grouped.head()


# In[17]:


DF[times.year == 1752]


# In[18]:


DF[np.isnan(DF["LandAverageTemperature"])]


# In[19]:


# Use previous valid observation to fill gap(fill Nan)

DF["LandAverageTemperature"]  = DF["LandAverageTemperature"].fillna(method = "ffill")


# In[20]:


grouped = DF.groupby([times.year]).mean()


plt.figure(figsize=(15,5))
plt.plot(grouped["LandAverageTemperature"])
plt.title("Average Land Temperature 1750 - 2015")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.show()


# In[21]:


###Modelling 

from sklearn.linear_model import LinearRegression as Linreg


# In[22]:


x = grouped.index.values.reshape(-1,1)
y = grouped["LandAverageTemperature"].values


# In[23]:


reg = Linreg()
reg.fit(x,y)
y_pred = reg.predict(x)
print ("Accuracy:" + str(reg.score(x,y)))


# In[24]:


###Regression Model  

plt.figure(figsize=(15,5))
plt.title("Average Land Temperature 1750 - 2015 Regression Model")
plt.xlabel("YEAR")
plt.ylabel("Average Land Temp")
plt.scatter(x=x ,y=y_pred)
plt.scatter(x=x ,y=y ,c = "r")
plt.show()


# In[31]:


reg.predict(np.array([2050]).reshape(1, 1))


# In[30]:


### Comunicating Data

