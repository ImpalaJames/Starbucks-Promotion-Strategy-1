#!/usr/bin/env python
# coding: utf-8

# ## Portfolio Exercise: Starbucks
# <br>
# 
# <img src="https://opj.ca/wp-content/uploads/2018/02/New-Starbucks-Logo-1200x969.jpg" width="200" height="200">
# <br>
# <br>
#  
# #### Background Information
# 
# The dataset you will be provided in this portfolio exercise was originally used as a take-home assignment provided by Starbucks for their job candidates. The data for this exercise consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7.
# 
# #### Optimization Strategy
# 
# Your task is to use the training data to understand what patterns in V1-V7 to indicate that a promotion should be provided to a user. Specifically, your goal is to maximize the following metrics:
# 
# * **Incremental Response Rate (IRR)** 
# 
# IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (_treatment_) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (_control_).
# 
# $$ IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}} $$
# 
# 
# * **Net Incremental Revenue (NIR)**
# 
# NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.
# 
# $$ NIR = (10\cdot purch_{treat} - 0.15 \cdot cust_{treat}) - 10 \cdot purch_{ctrl}$$
# 
# For a full description of what Starbucks provides to candidates see the [instructions available here](https://drive.google.com/open?id=18klca9Sef1Rs6q8DW4l7o349r8B70qXM).
# 
# Below you can find the training data provided.  Explore the data and different optimization strategies.
# 
# #### How To Test Your Strategy?
# 
# When you feel like you have an optimization strategy, complete the `promotion_strategy` function to pass to the `test_results` function.  
# From past data, we know there are four possible outomes:
# 
# Table of actual promotion vs. predicted promotion customers:  
# 
# <table>
# <tr><th></th><th colspan = '2'>Actual</th></tr>
# <tr><th>Predicted</th><th>Yes</th><th>No</th></tr>
# <tr><th>Yes</th><td>I</td><td>II</td></tr>
# <tr><th>No</th><td>III</td><td>IV</td></tr>
# </table>
# 
# The metrics are only being compared for the individuals we predict should obtain the promotion – that is, quadrants I and II.  Since the first set of individuals that receive the promotion (in the training set) receive it randomly, we can expect that quadrants I and II will have approximately equivalent participants.  
# 
# Comparing quadrant I to II then gives an idea of how well your promotion strategy will work in the future. 
# 
# Get started by reading in the data below.  See how each variable or combination of variables along with a promotion influences the chance of purchasing.  When you feel like you have a strategy for who should receive a promotion, test your strategy against the test dataset used in the final `test_results` function.

# In[1]:


# load in packages
from itertools import combinations

import statsmodels.api as sm
from test_results import test_results, score
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')

# load in the data
train_data = pd.read_csv('./training.csv')
train_data.head()


# In[ ]:





# ## Exploratory Data Analysis

# In[2]:


train_data.info()


# In[3]:


print('count of duplicate data: ', train_data.duplicated(subset='ID').sum())
print('count of null data: ', train_data.isnull().sum().sum())


# In[4]:


for column in train_data.columns:
    print(column)
    print('---')
    print(train_data[column].unique())


# This indicates that V2 and V3 are continuous variables, while the reset could be categorical variables or discrete variables.

# ## Significance test
# ### Analyzing Invariant Metric

# The first step will be to analyze our invariant metric of number of participants in our control and experiment (i.e those who recieved a promotional offer vs those who did not).
# - H0：p_treat = p_control
# - H1：p_treat <> p_control

# In[5]:


train_data.Promotion.value_counts()
n_total = train_data.shape[0]
n_treat = train_data.Promotion.value_counts()['Yes']
n_control = train_data.Promotion.value_counts()['No']
actual_p_treat = n_treat/n_total
print("Percentage of treat group population in the sample population: ", "%.2f"%actual_p_treat)


# In[6]:


import scipy.stats as stats

p_treat_null = 0.5
std_treat_null = np.sqrt(p_treat_null*(1-p_treat_null)/n_total)
null_dist = stats.norm(loc=p_treat_null, scale = std_treat_null)
p_val = (1-null_dist.cdf(actual_p_treat))* 2
p_val


# #### Conclusion
# Our P value is larger than alpha=0.05 and therefore we fail to reject the null hypothesis. This implies that there is no statistical signifigance in the difference of our sampling populations.

# ### Analyzing Effect Metric

# In[7]:


# Calculate actual IRR
n_treat_purch = train_data.query("Promotion =='Yes' and purchase == 1").shape[0]
n_ctrl_purch = train_data.query("Promotion =='No' and purchase == 1").shape[0]

irr, nir = score(train_data)
irr,nir


# #### IRR
# We shall determine if the experiment had a positive effect on the IRR metric.
# - H0: Incremental Response Rate = 0
# - H1: Incremental Response Rate > 0
# 
# alpha = 0.05
# 
# Since the sample size is larger than 30, the sampling is considered to be normal distributed.

# In[8]:


# significance testing - IRR
zstat, p_val = sm.stats.proportions_ztest([n_treat_purch, n_ctrl_purch],[n_treat, n_control], alternative='larger')
print("p_val_irr: ", "%.2f"%p_val)


# Conclusion:
# p_val = 0.0 < alpha=0.05, so we reject null hypothesis. This implies that there is a statistical increase in IRR between our control and experiemental group. So, we can see that our campaign does have a positive effect on the number of customers who purchased the product with the promotion, as compared to if they didn't receive the promotion.

# #### NIR
# We shall determine if the experiment had a positive effect on the NIR metric.
# - H0: NIR <= 0
# - H1: NIR > 0
# 
# equaled to:
# - H0: IRP <= 1.5%
# - H1: IRP > 1.5%
# 
# 1-beta = 0.80; alpha = 0.05

# In[9]:


sample_purch_p = (n_ctrl_purch + n_treat_purch)/n_total
sample_purch_p


# In[10]:


# Calculate sample size required to get the statistic power of 0.80
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
irr_null = 0
irr_alt = 0.015

# leave out the "nobs1" parameter to solve for it
needed_sample_size = NormalIndPower().solve_power(effect_size = proportion_effectsize(.01123+0.015, .01123), nobs1=None,alpha = .05, power = 0.8,
                             alternative = 'larger')
needed_sample_size


# In[11]:


irr_mean_null = 0.015
zstat, p_val=sm.stats.proportions_ztest([n_treat_purch,n_ctrl_purch],[n_treat, n_control], value = irr_mean_null, alternative='larger')
p_val


# p_val = 1 >alpha = 0.05 and therefore we fail to reject the null hypothesis, which means the probability of reaching IRR is close to 1 under the assumption that NIR = 0. This implies that the promotion strategy itself, though increases revenue, cannot bring actual profit due to the amount of increased revenue is less than the cost of the promotion.

# ## Exploration Analysis II
# From previous conclusions of the testings, promotion strategy should be optimized, otherwise the loss will increase.

# ### Single Variable Analysis
# Look at the distribution of the categorical variables and continuous variables respectively

# In[12]:


train_data.head()


# In[13]:


## Analysis of Discrete variables
train_data[['V1','V4','V5','V6','V7']]=train_data[['V1','V4','V5','V6','V7']].applymap(lambda x:str(x))
train_data.info()


# In[14]:


discret_val = train_data[['purchase','V1','V4','V5','V6','V7']]


# In[15]:


discret_features = np.array(['V1','V4','V5','V6','V7'])
fig,axes = plt.subplots(1,5,figsize=(16,4))
for i,feature in enumerate(discret_features):
    data = (discret_val         .query("purchase == 1").groupby([feature]).size()
        /discret_val.groupby([feature]).size()).reset_index(name="purchase_rate")
    axes[i].bar(feature, 'purchase_rate', color='green', alpha=.5, data=data)
    axes[i].set_xticks(data[feature])
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Purchase Rate')
plt.subplots_adjust(wspace=0.5);


# From the plots above:
# - V1: users with value of 3 perform lower purchase rate than other values
# - V4: users with value of 2 perform significantly higher purchase rate than users with value 1
# - V5: users with value of 2 perform lower purchase rate than other values
# - V6 & V7: no differential distributions among all the values

# In[16]:


## Analysis of Continuous variables
train_data.describe()


# In[ ]:





# In[17]:


# histogram of the distribution of the continuous variables
constant_df = train_data[['purchase','V2','V3']]
constant_purch_df = train_data[['purchase','V2','V3']].query("purchase == 1")
fig,axes = plt.subplots(2,1, figsize=(12,10),sharex=False,sharey=False)

# Distribution of the V2 variable
sb.histplot(constant_purch_df, x="V2",element="step", ax=axes[0],color='green',fill=False)
axes1 = axes[0].twinx()
sb.histplot(constant_df, x="V2",element="step", ax=axes1,color='purple',fill=False)
axes[0].set(xticks = np.arange(10,50,5),
            xlabel='V2',
            ylabel='Purchase Num')
axes1.set(xticks = np.arange(10,50,5),
            xlabel='V2',
            ylabel='Total Num')

# Distribution of the V3 variable
sb.histplot(constant_purch_df, x="V3",element="step", ax=axes[1],color='green',fill=False,);
axes2 = axes[1].twinx()
sb.histplot(constant_df, x="V3",element="step", ax=axes2,color='purple',fill=False)
axes[1].set(xticks = np.arange(-2,2,0.2),
            xlabel='V3',
            ylabel='Purchase Num')
axes2.set(xticks = np.arange(-2,2,0.2),
          xlabel='V3',
          ylabel='Total Num')
axes[0].legend(labels=['purchase num','total num'])
plt.subplots_adjust(hspace=0.2);


# ## Modeling 
# ere we will create a model that can accurately predict if a customer will be responsive to the campaign.

# In[20]:


# Only considering the experiment group for our model training and dropping the irrelavant columns
data_exp = train_data[train_data['Promotion']=="Yes"].drop(['ID','Promotion'], axis = 1)


# In[ ]:


print('Shape: ', data_exp.shape)
data_exp.head()


# In[ ]:


# Splitting data into predictors and target variables
X = data_exp.drop(['purchase'], axis=1)
y = data_exp.purchase


# In[ ]:


# Scaling inputs 
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# In[ ]:


# Training Model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X,y);


# ### Loading Data to the Model

# In[19]:


# Loading in our test data 
test_data = pd.read_csv('./test.csv')

data_exp_test = test_data[test_data['Promotion']=="Yes"].drop(['ID','Promotion'], axis = 1)


# In[ ]:


print('Shape: ', data_exp_test.shape)
data_exp_test.head()


# In[18]:


# Splitting data into predictors and target variables
X_test = data_exp_test.drop(['purchase'],axis=1)
y_test = data_exp_test.purchase


# In[ ]:


# Scaling inputs 
X_test = min_max_scaler.fit_transform(X_test)


# In[ ]:


# Predicting our target values
y_pred = clf.predict(X_test)


# In[ ]:


# Checking our accuracy for the model 
accuracy = (y_pred == y_test).mean()
print("The accuracy of the model is {0:.5f}%".format(accuracy))


# In[ ]:


# Confusion Matrix
cf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)

# Plotting
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

group_counts = cf_matrix.flatten()

group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues');


# In[ ]:





# In[ ]:


print('Shape: ', data_exp.shape)
data_exp.head()


# In[ ]:


# Splitting data into predictors and target variables
X = data_exp.drop(['purchase'], axis=1)
y = data_exp.purchase


# In[ ]:


# Scaling inputs 
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# In[ ]:





# In[ ]:





# In[ ]:


def promotion_strategy(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    df = min_max_scaler.fit_transform(df)
    
    y_pred = clf.predict(df)
    
    pred_yes_no = []
    for value in y_pred:
        if value == 0:
            pred_yes_no.append("No")
        if value == 1:
            pred_yes_no.append("Yes")
            
    promotion = np.asarray(pred_yes_no)
    
    return promotion


# In[ ]:


# This will test your results, and provide you back some information 
# on how well your promotion_strategy will work in practice

test_results(promotion_strategy)


# ## Conclusion
# We managed to get a better IRR but a signifigantly worse NIR. Regardless, we still managed a signifigantly better approach than what we had observed with the experiment.
# 
# Our confusion matrix indidcated that our accuracy is only hindered by a small number of false negatives. Prehaps if we had a less conservative model we may see a higher NIR rate. Allowing for a higher number of false positives may actually improve our NIR metric at the cost of our IRR.
