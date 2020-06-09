#!/usr/bin/env python
# coding: utf-8

# Decision Tree : Decision tree is the most powerful and popular tool for classification and prediction. A Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.
# 
# 

# ## import libraries

# In[29]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image 
import pydot
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np


# ## load data and slicing

# In[30]:


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[31]:


X.head()


# In[32]:


y = pd.get_dummies(y)
y.head()


# Gini index:
# 
# 
# Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified.
# It means an attribute with lower gini index should be preferred.
# Sklearn supports “gini” criteria for Gini Index and by default, it takes “gini” value.
# 
# 
# 

# Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy the more the information content.
# 

# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[34]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[35]:


dot_data = StringIO()


# In[36]:


export_graphviz(dt, out_file=dot_data, feature_names=iris.feature_names)
(graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Accuracy score
# 
# Accuracy score is used to calculate the accuracy of the trained classifier.

# In[37]:


y_pred = dt.predict(X_test)


# Confusion Matrix
# 
# Confusion Matrix is used to understand the trained classifier behavior over the test dataset or validate dataset.

# In[38]:


species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)
confusion_matrix(species, predictions)


# In[ ]:





# In[ ]:




