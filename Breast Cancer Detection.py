#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


dataset = pd.read_csv('detect_bc.csv')


# In[6]:


dataset 
#displays the first five rows


# In[47]:


dataset.isnull().sum()

# the dataset is clean with no null values


# In[7]:


dataset.columns


# In[8]:


dataset['diagnosis'].value_counts()

# since the data is roughly split equally between 'M' and 'B', its valid
#otherwise we would have had to break it down and remove repetetive classifiers


# In[9]:


dataset.describe()
#statistical representation of the coloumn attributes


# In[10]:


# our main objective here is finding the correlation between the different features and breast cancer detection,
# and prediciting M or B on unseen data
# we will be dropping out irrelevant coloumns such as id and unnamed 32


# In[11]:


extra_col = ['id', 'Unnamed: 32']

dataset = dataset.drop(extra_col, axis = 1, inplace = False)
dataset.head()


# In[12]:


dataset.shape


# In[13]:


#it's important in supervised learning to split the training and test/validation data to train a model before its operated on unseen data


data = dataset.sample(frac=0.85, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# NOTE: the 569 rows are divided into 484 and 85 rows with the 95-5 ratio
# By default, the system divides 484 rows into 70-30 ratio which can be changed by train_size parameter


# In[14]:


from pycaret.classification import *

exp_clf101 = setup(data = data, target = 'diagnosis', session_id=123) 

# we setup the pycare module and set diagnosis as the target variable 
# in case if we dont initialise session_id, the system will randomly choose one from the dataset


# In[15]:


dataset.shape


# In[16]:


compare_models()

#this function lists all machine learning models and computes the accuracy and other classification metrics
# the highlighted values represent the max values of a given coloumn
# the programmer can then select a model based on the ideal characterisitics
# usually, a model is selected based on accuracy or AUC parameters.


# In[17]:


# selecting best model
best = automl()
best


# In[18]:


#since xgboost is the best perfoming model based on accuracy and other factors, we proceed with it
xgboost = create_model('xgboost')


# In[19]:


# this function tunes the metrics to improve the accuracy of the attributes
# the values might be less  after tuning however it should validate once the SD is accounted for

tuned_xgboost = tune_model(xgboost)


# In[20]:


plot_model(tuned_xgboost, plot='feature')


# In[46]:


evaluate_model(tuned_xgboost)


# In[22]:


plot_model(tuned_xgboost, plot='feature')


# In[23]:


plot_model(tuned_xgboost, plot='auc')


# In[24]:


plot_model(tuned_xgboost, plot='pr')


# In[25]:


plot_model(tuned_xgboost, plot='confusion_matrix')


# In[26]:


plot_model(tuned_xgboost, plot='threshold')


# In[27]:


plot_model(tuned_xgboost, plot='error')


# In[28]:


plot_model(tuned_xgboost, plot='class_report')


# In[29]:


plot_model(tuned_xgboost, plot='boundary')


# In[30]:


plot_model(tuned_xgboost, plot='learning')


# In[31]:


plot_model(tuned_xgboost, plot='manifold')


# In[32]:


plot_model(tuned_xgboost, plot='calibration')


# In[33]:


plot_model(tuned_xgboost, plot='vc')


# In[36]:


# the above operations were all on the 70% training data
# we now test our 30% test data using the 70% data's newly trained model

predict_model(tuned_xgboost)

# the different between training and set accuracy is not that significant 


# In[37]:


# this function fits the model onto the complete dataset including the test/hold-out sample 30%. 
# The purpose of this function is to train the model on the complete dataset before it is deployed on the 5% dataset.

final_xgboost = finalize_model(tuned_xgboost)
final_xgboost


# In[38]:


# the entire 70-30 dataset is used as a training model which will be used on the unseen dataset
predict_model(final_xgboost)

# we achieve a 1.0 in the metrics as we optimise the hyperparameters once again when both the training-test are iterated upon.
# this however does not mean the model predicts the M or B with a 100% accuracy
# as seen below, the model has a very high accuracy classifying M with ~99% accuracy


# In[39]:


# Label is the prediction and score is the probability of the prediction.

unseen_predictions = predict_model(final_xgboost, data=data_unseen)
unseen_predictions
#unseen_predictions.head()


# In[40]:


save_model(final_xgboost,'Final XGB Model')


# In[41]:


saved_final_xgboost = load_model('Final XGB Model')


# In[42]:


# our model is now ready to be deployed on any unseen data


# In[43]:


new_prediction = predict_model(saved_final_xgboost, data=data_unseen)


# In[44]:


new_prediction

# the label and score can be found on the right end coloumn

