#!/usr/bin/env python
# coding: utf-8

# In[1]:


#tumour detection using random forest algo
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv("C:/Users/Lenovo/Desktop/Tumor_Detection.csv")


# In[8]:


df.info()


# In[9]:


#drop id as no use in tumour detection
df.drop('id', axis  =1, inplace = True)


# In[10]:


l= list(df.columns)


# In[11]:


#creating start point
feature_mean = l[1:11]
feature_se = l[11:21]
feature_worst = l[21:]


# In[12]:


print(feature_mean)


# In[13]:


print(feature_se)


# In[14]:


print(feature_worst )


# In[15]:


df.head()


# In[16]:


df['diagnosis'].unique()


# In[17]:


sns.countplot(df['diagnosis'],label="count")


# In[18]:


df.shape


# In[19]:


df.describe()


# In[20]:


corr  =df.corr() #implace attribute is used to overwrite the existing data set 


# In[21]:


corr


# In[23]:


#drawing the heatmap
plt.figure(figsize=(10,10))
sns.heatmap(corr)


# In[25]:


#segregating the data
df['diagnosis']= df['diagnosis'].map({'M':1,'B':0})#M and B are changed to 1 and 0
df.head


# In[26]:


x = df.drop('diagnosis',axis = 1)#axis stands for column number 
x.head()


# In[27]:


y = df['diagnosis']
y


# In[29]:


#trainign the data set 
#devide the data set in train and test set 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)


# In[30]:


df.shape


# In[31]:


x_train.shape


# In[32]:


x_test.shape


# In[33]:


y_train.shape


# In[34]:


y_test.shape


# In[36]:


ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
x_train# generate the array of x_train test


# In[37]:


#apply random forest classifeier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:




