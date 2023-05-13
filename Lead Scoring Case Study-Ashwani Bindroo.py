#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Pandas and NumPy
import pandas as pd, numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Importing lead dataset
lead_data = pd.read_csv("Leads.csv")
lead_data.head()


# In[4]:


# checking the shape of the data 
lead_data.shape
# # We have 9240 rows and 37 columns in our leads dataset


# In[5]:


# checking non null count and datatype of the variables
lead_data.info()


# In[ ]:


# All the dataypes of the variables are in correct format.


# In[6]:


# Describing data
lead_data.describe()


# In[ ]:


# From above description about counts, we can see that there are missing values present in our data.


# In[ ]:


# Data Cleaning
# 1)Handling the 'Select' level that is present in many of the categorical variables.
# We observe that there are 'Select' values in many columns.
# It may be because the customer did not select any option from the list, hence it shows 'Select'.'Select' values are as good as NULL. So we can convert these values to null values.


# In[7]:


# Converting 'Select' values to NaN.
lead_data = lead_data.replace('Select', np.nan)


# In[8]:


# checking the columns for null values
lead_data.isnull().sum()


# In[9]:


# Finding the null percentages across columns
round(lead_data.isnull().sum()/len(lead_data.index),2)*100


# In[ ]:


# We see that for some columns we have high percentage of missing values. We can drop the columns with missing values greater than 40% .


# In[10]:


# dropping the columns with missing values greater than or equal to 40% .
lead_data=lead_data.drop(columns=['How did you hear about X Education','Lead Quality','Lead Profile',
                                  'Asymmetrique Activity Index','Asymmetrique Profile Index','Asymmetrique Activity Score',
                                 'Asymmetrique Profile Score'])


# In[11]:


# Finding the null percentages across columns after removing the above columns
round(lead_data.isnull().sum()/len(lead_data.index),2)*100


# In[ ]:


# This column has 37% missing values


# In[12]:


plt.figure(figsize=(17,5))
sns.countplot(lead_data['Specialization'])
plt.xticks(rotation=90)


# In[ ]:


# There is 37% missing values present in the Specialization column .It may be possible that the lead may leave this column blank if he may be a student or not having any specialization or his specialization is not there in the options given. So we can create a another category 'Others' for this.


# In[13]:


# Creating a separate category called 'Others' for this 
lead_data['Specialization'] = lead_data['Specialization'].replace(np.nan, 'Others')


# In[ ]:


# 2) Tags column
#'Tags' column has 36% missing values


# In[14]:


# Visualizing Tags column
plt.figure(figsize=(10,7))
sns.countplot(lead_data['Tags'])
plt.xticks(rotation=90)


# In[ ]:


# Since most values are 'Will revert after reading the email' , we can impute missing values in this column with this value.


# In[15]:


# Imputing the missing data in the tags column with 'Will revert after reading the email'
lead_data['Tags']=lead_data['Tags'].replace(np.nan,'Will revert after reading the email')


# In[ ]:


# 3) Column: 'What matters most to you in choosing a course'
# this column has 29% missing values


# In[16]:


# Visualizing this column
sns.countplot(lead_data['What matters most to you in choosing a course'])
plt.xticks(rotation=45)


# In[17]:


# Finding the percentage of the different categories of this column:
round(lead_data['What matters most to you in choosing a course'].value_counts(normalize=True),2)*100


# In[ ]:


# We can see that this is highly skewed column so we can remove this column.


# In[18]:


# Dropping this column 
lead_data=lead_data.drop('What matters most to you in choosing a course',axis=1)


# In[19]:


sns.countplot(lead_data['What is your current occupation'])
plt.xticks(rotation=45)


# In[20]:


# Finding the percentage of the different categories of this column:
round(lead_data['What is your current occupation'].value_counts(normalize=True),2)*100


# In[ ]:


# Since the most values are 'Unemployed' , we can impute missing values in this column with this value.


# In[21]:


# Imputing the missing data in the 'What is your current occupation' column with 'Unemployed'
lead_data['What is your current occupation']=lead_data['What is your current occupation'].replace(np.nan,'Unemployed')


# In[ ]:


# 5) Column: 'Country'
# This column has 27% missing values


# In[22]:


plt.figure(figsize=(17,5))
sns.countplot(lead_data['Country'])
plt.xticks(rotation=90)


# In[ ]:


# We can see that this is highly skewed column but it is an important information w.r.t. to the lead. Since most values are 'India' , we can impute missing values in this column with this value.


# In[ ]:


# 6) Column: 'City'
# This column has 40% missing values


# In[23]:


plt.figure(figsize=(10,5))
sns.countplot(lead_data['City'])
plt.xticks(rotation=90)


# In[24]:


# Finding the percentage of the different categories of this column:
round(lead_data['City'].value_counts(normalize=True),2)*100


# In[ ]:


# Since most values are 'Mumbai' , we can impute missing values in this column with this value.


# In[25]:


# Imputing the missing data in the 'City' column with 'Mumbai'
lead_data['City']=lead_data['City'].replace(np.nan,'Mumbai')


# In[26]:


# Finding the null percentages across columns after removing the above columns
round(lead_data.isnull().sum()/len(lead_data.index),2)*100


# In[ ]:


# Rest missing values are under 2% so we can drop these rows


# In[27]:


# Dropping the rows with null values
lead_data.dropna(inplace = True)


# In[28]:


# Finding the null percentages across columns after removing the above columns
round(lead_data.isnull().sum()/len(lead_data.index),2)*100


# In[ ]:


# Now we don't have any missing value in the dataset.


# In[ ]:


# We can find the percentage of rows retained


# In[29]:


# Percentage of rows retained 
(len(lead_data.index)/9240)*100


# In[ ]:


# We have retained 73% of the rows after cleaning the data


# In[ ]:


# Exploratory Data Analysis


# In[ ]:


#Checking for duplicates


# In[30]:


lead_data[lead_data.duplicated()]


# In[ ]:


# We see there are no duplicate records in our lead dataset.


# In[ ]:


# Univariate Analysis and Bivariate Analysis


# In[ ]:


# 1) Converted
# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0)


# In[31]:


Converted = (sum(lead_data['Converted'])/len(lead_data['Converted'].index))*100
Converted


# In[ ]:


# The lead conversion rate is 37%


# In[ ]:


# 2) Lead Origin`


# In[32]:


plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Origin", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 45)


# In[ ]:


# Inference :
# API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# Lead Import are very less in coun
# To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.


# In[ ]:


# 3) Lead Source


# In[33]:


plt.figure(figsize=(13,5))
sns.countplot(x = "Lead Source", hue = "Converted", data = lead_data, palette='Set1')
plt.xticks(rotation = 90)


# In[34]:


# Need to replace 'google' with 'Google'
lead_data['Lead Source'] = lead_data['Lead Source'].replace(['google'], 'Google')


# In[35]:


# Creating a new category 'Others' for some of the Lead Sources which do not have much values.
lead_data['Lead Source'] = lead_data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


# In[36]:


# Visualizing again
plt.figure(figsize=(10,5))
sns.countplot(x = "Lead Source", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Google and Direct traffic generates maximum number of leads.
# Conversion Rate of reference leads and leads through welingak website is high.
# To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.


# In[ ]:


# 4) Do not Email


# In[37]:


sns.countplot(x = "Do Not Email", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 5) Do not call


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 6) TotalVisits


# In[38]:


lead_data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[39]:


sns.boxplot(lead_data['TotalVisits'],orient='vert',palette='Set1')


# In[ ]:


# As we can see there are a number of outliers in the data. We will cap the outliers to 95% value for analysis.


# In[40]:


percentiles = lead_data['TotalVisits'].quantile([0.05,0.95]).values
lead_data['TotalVisits'][lead_data['TotalVisits'] <= percentiles[0]] = percentiles[0]
lead_data['TotalVisits'][lead_data['TotalVisits'] >= percentiles[1]] = percentiles[1]


# In[41]:


# Visualizing again
sns.boxplot(lead_data['TotalVisits'],orient='vert',palette='Set1')


# In[42]:


sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_data,palette='Set1')


# In[ ]:


# Inference
# Median for converted and not converted leads are the same.
# Nothing can be concluded on the basis of Total Visits.


# In[ ]:


# 7) Total Time Spent on Website


# In[43]:


lead_data['Total Time Spent on Website'].describe()


# In[44]:


sns.boxplot(lead_data['Total Time Spent on Website'],orient='vert',palette='Set1')


# In[45]:


sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead_data,palette='Set1')


# In[ ]:


# Inference
# Leads spending more time on the weblise are more likely to be converted.
# Website should be made more engaging to make leads spend more time.


# In[ ]:


# 8) Page Views Per Visit


# In[46]:


lead_data['Page Views Per Visit'].describe()


# In[47]:


sns.boxplot(lead_data['Page Views Per Visit'],orient='vert',palette='Set1')


# In[ ]:


# As we can see there are a number of outliers in the data. We will cap the outliers to 95% value for analysis.


# In[48]:


percentiles = lead_data['Page Views Per Visit'].quantile([0.05,0.95]).values
lead_data['Page Views Per Visit'][lead_data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
lead_data['Page Views Per Visit'][lead_data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[49]:


# Visualizing again
sns.boxplot(lead_data['Page Views Per Visit'],palette='Set1',orient='vert')


# In[50]:


sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data =lead_data,palette='Set1')


# In[ ]:


# Inference
# Median for converted and unconverted leads is the same.
# Nothing can be said specifically for lead conversion from Page Views Per Visit


# In[ ]:


# 9) Last Activity


# In[51]:


lead_data['Last Activity'].describe()


# In[52]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[53]:


# We can club the last activities to "Other_Activity" which are having less data.
lead_data['Last Activity'] = lead_data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


# In[54]:


# Visualizing again
plt.figure(figsize=(15,6))
sns.countplot(x = "Last Activity", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most of the lead have their Email opened as their last activity.
# Conversion rate for leads with last activity as SMS Sent is almost 60%.


# In[ ]:


# 10) Country


# In[55]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Country", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most values are 'India' no such inference can be drawn


# In[ ]:


# 11) Specialization


# In[56]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Specialization", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Focus should be more on the Specialization with high conversion rate.


# In[ ]:


# 12) What is your current occupationc


# In[57]:


plt.figure(figsize=(15,6))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Working Professionals going for the course have high chances of joining it.
# Unemployed leads are the most in numbers but has around 30-35% conversion rate.


# In[ ]:


# 13) Search


# In[58]:


sns.countplot(x = "Search", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 14) Magazine


# In[59]:


sns.countplot(x = "Magazine", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 15) Newspaper Article


# In[60]:


sns.countplot(x = "Newspaper Article", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 16) X Education Forums


# In[61]:


sns.countplot(x = "X Education Forums", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 17) Newspaper


# In[62]:


sns.countplot(x = "Newspaper", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter


# In[ ]:


# 18) Digital Advertisement


# In[63]:


sns.countplot(x = "Digital Advertisement", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 19) Through Recommendations


# In[64]:


sns.countplot(x = "Through Recommendations", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter


# In[ ]:


# 20) Receive More Updates About Our Courses


# In[65]:


sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter


# In[ ]:


# 21) Tags


# In[66]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Tags", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Since this is a column which is generated by the sales team for their analysis , so this is not available for model building . So we will need to remove this column before building the model.


# In[ ]:


# 22) Update me on Supply Chain Content


# In[67]:


sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 23) Get updates on DM Content


# In[68]:


sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 24) City


# In[69]:


plt.figure(figsize=(15,5))
sns.countplot(x = "City", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most leads are from mumbai with around 50% conversion rate.


# In[ ]:


# 25) I agree to pay the amount through cheque


# In[70]:


sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 26) A free copy of Mastering The Interview


# In[71]:


sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.


# In[ ]:


# 27) Last Notable Activity


# In[72]:


plt.figure(figsize=(15,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = lead_data,palette='Set1')
plt.xticks(rotation = 90)


# In[ ]:


# Results
# Based on the univariate analysis we have seen that many columns are not adding any information to the model, hence we can drop them for further analysis


# In[73]:



lead_data = lead_data.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article','X Education Forums',
                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                            'A free copy of Mastering The Interview'],1)


# In[74]:


lead_data.shape


# In[75]:


lead_data.info()


# In[ ]:


# Data Preparation


# In[ ]:


# 1) Converting some binary variables (Yes/No) to 1/0


# In[76]:


vars =  ['Do Not Email', 'Do Not Call']

def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

lead_data[vars] = lead_data[vars].apply(binary_map)


# In[ ]:


# 2) Creating Dummy variables for the categorical features:
'Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City','Last Notable Activity'


# In[77]:


# Creating a dummy variable for the categorical variables and dropping the first one.
dummy_data = pd.get_dummies(lead_data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity']], drop_first=True)
dummy_data.head()


# In[78]:


# Concatenating the dummy_data to the lead_data dataframe
lead_data = pd.concat([lead_data, dummy_data], axis=1)
lead_data.head()


# In[ ]:


# Dropping the columns for which dummies were created


# In[79]:


lead_data = lead_data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity'], axis = 1)


# In[80]:


lead_data.head()


# In[ ]:


# 3) Splitting the data into train and test set.


# In[81]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = lead_data.drop(['Prospect ID','Converted'], axis=1)
X.head()


# In[82]:


# Putting target variable to y
y = lead_data['Converted']

y.head()


# In[83]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[ ]:


# 4) Scaling the features


# In[84]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[85]:


# Checking the Lead Conversion rate
Converted = (sum(lead_data['Converted'])/len(lead_data['Converted'].index))*100
Converted


# In[ ]:


# We have almost 37% lead conversion rate


# In[ ]:


# Feature Selection Using RFE


# In[89]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg)             
# running RFE with 20 variables as output
rfe = rfe.fit(X_train, y_train)


# In[90]:


rfe.support_


# In[91]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[92]:


# Viewing columns selected by RFE
cols = X_train.columns[rfe.support_]
cols


# In[ ]:


# Model Building


# In[93]:


import statsmodels.api as sm


# In[94]:


X_train_sm = sm.add_constant(X_train[cols])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
result = logm1.fit()
result.summary()


# In[ ]:


# Since Pvalue of 'What is your current occupation_Housewife' is very high, we can drop this column.


# In[95]:


# Dropping the column 'What is your current occupation_Housewife'
col1 = cols.drop('What is your current occupation_Housewife')


# In[ ]:


# Model-2


# In[96]:


X_train_sm = sm.add_constant(X_train[col1])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


# Since Pvalue of 'Last Notable Activity_Had a Phone Conversation' is very high, we can drop this column.


# In[97]:


col1 = col1.drop('Last Notable Activity_Had a Phone Conversation')


# In[ ]:


# Model-3


# In[98]:


X_train_sm = sm.add_constant(X_train[col1])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[ ]:


# Since Pvalue of 'What is your current occupation_Student' is very high, we can drop this column.


# In[99]:


col1 = col1.drop('What is your current occupation_Student')


# In[ ]:


# Model-4


# In[100]:


X_train_sm = sm.add_constant(X_train[col1])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


# Since Pvalue of 'Lead Origin_Lead Add Form' is very high, we can drop this column.


# In[101]:


col1 = col1.drop('Lead Origin_Lead Add Form')


# In[ ]:


# Model-5


# In[102]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[ ]:


# Checking for VIF values:


# In[103]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[104]:


# Dropping the column  'What is your current occupation_Unemployed' because it has high VIF
col1 = col1.drop('What is your current occupation_Unemployed')


# In[ ]:


# Model-6


# In[105]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[107]:


# Model-7


# In[108]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[ ]:


# Checking for VIF values:


# In[118]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[119]:


# Dropping the column  'Last Activity_Unsubscribed' to reduce the variables
col1 = col1.drop('Last Activity_Unsubscribed')


# In[120]:


# Model-8


# In[121]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[122]:


# Checking for VIF values:


# In[123]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vifX_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[124]:


# Dropping the column  'Last Notable Activity_Unreachable' to reduce the variables
col1 = col1.drop('Last Notable Activity_Unreachable')


# In[125]:


# Model-9


# In[126]:


X_train_sm = sm.add_constant(X_train[col1])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# In[128]:


# Checking for VIF values:


# In[129]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col1].columns
vif['VIF'] = [variance_inflation_factor(X_train[col1].values, i) for i in range(X_train[col1].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[130]:


# Since the Pvalues of all variables is 0 and VIF values are low for all the variables, model-9 is our final model. We have 12 variables in our final model.


# In[131]:


# Making Prediction on the Train set¶


# In[132]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[133]:


# Reshaping into an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[134]:


# Creating a dataframe with the actual Converted flag and the predicted probabilities


# In[135]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[136]:


# Choosing an arbitrary cut-off probability point of 0.5 to find the predicted labels¶


# In[137]:


# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0


# In[138]:


y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[139]:


# Making the Confusion matrix¶


# In[140]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[141]:


# The confusion matrix indicates as below
# Predicted     not_converted    converted
# Actual
# not_converted        2675      346
# converted            580       1143  


# In[142]:


# Let's check the overall accuracy.
print('Accuracy :',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[146]:


# Metrics beyond simply accuracy


# In[147]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[148]:


# Sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[149]:


# Let us calculate specificity
print("Specificity : ",TN / float(TN+FP))


# In[150]:


# Calculate false postive rate - predicting converted lead when the lead actually was not converted
print("False Positive Rate :",FP/ float(TN+FP))


# In[151]:


# positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[152]:


# Negative predictive value
print ("Negative predictive value :",TN / float(TN+ FN))


# In[ ]:


# We found out that our specificity was good (~88%) but our sensitivity was only 70%. Hence, this needed to be taken care of.
# We have got sensitivity of 70% and this was mainly because of the cut-off point of 0.5 that we had arbitrarily chosen. Now, this cut-off point had to be optimised in order to get a decent value of sensitivity and for this we will use the ROC curve.


# In[ ]:


# Plotting the ROC Curve
# An ROC curve demonstrates several things:

# It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.


# In[155]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[156]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[157]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[ ]:


# Since we have higher (0.89) area under the ROC curve , therefore our model is a good one.

# Finding Optimal Cutoff Point
# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that. Optimal cutoff probability is that prob where we get balanced sensitivity and specificity


# In[158]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[159]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[160]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:


# From the curve above, 0.34 is the optimum point to take it as a cutoff probability.¶


# In[161]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.34 else 0)

y_train_pred_final.head()


# In[ ]:


# Assigning Lead Score to the Training data


# In[162]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final.head()


# In[163]:


# Model Evaluation


# In[164]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))


# In[165]:


# Confusion matrix
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[166]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[167]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[168]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[169]:


# Calculate false postive rate - predicting converted lead when the lead was actually not have converted
print("False Positive rate : ",FP/ float(TN+FP))


# In[170]:


# Positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[171]:


# Negative predictive value
print("Negative Predictive Value : ",TN / float(TN+ FN))


# In[ ]:


# Precision and Recall
# Precision = Also known as Positive Predictive Value, it refers to the percentage of the results which are relevant.
# Recall = Also known as Sensitivity , it refers to the percentage of total relevant results correctly classified by the algorithm.


# In[172]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion


# In[173]:


# Precision
TP / TP + FP

print("Precision : ",confusion[1,1]/(confusion[0,1]+confusion[1,1]))


# In[178]:


# Recall
TP / TP + FN

print("Recall :",confusion[1,1]/(confusion[1,0]+confusion[1,1]))


# In[175]:


# Using sklearn utilities for the same


# In[176]:


from sklearn.metrics import precision_score, recall_score


# In[177]:


print("Precision :",precision_score(y_train_pred_final.Converted , y_train_pred_final.predicted))


# In[179]:


print("Recall :",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[180]:


# Precision and recall tradeoff


# In[181]:


from sklearn.metrics import precision_recall_curve

y_train_pred_final.Converted, y_train_pred_final.predicted


# In[182]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[183]:


# plotting a trade-off curve between precision and recall
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[184]:


# The above graph shows the trade-off between the Precision and Recall 


# In[185]:


# Making predictions on the test set


# In[186]:


# Scaling the test data


# In[187]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',
                                                                                                        'Total Time Spent on Website',
                                                                                                        'Page Views Per Visit']])


# In[189]:


# Assigning the columns selected by the final model to the X_test 
X_test = X_test[col1]
X_test.head()


# In[190]:


# Adding a const
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]


# In[191]:


# Converting y_test_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[192]:


# Let's see the head
y_pred_1.head()


# In[193]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[194]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[195]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[196]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(columns=['Prospect ID','Converted','Converted_prob'])


# In[197]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[198]:


y_pred_final['final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()


# In[199]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))


# In[200]:


# Making the confusion matrix
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2


# In[201]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[202]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))
Sensitivity : 0.8048533872598584
# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[204]:


# Assigning Lead Score to the Testing data


# In[205]:


y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))

y_pred_final.head()


# In[206]:


# # Results :
# # 1) Comparing the values obtained for Train & Test:
# # Train Data:
# Accuracy : 81.0 %
# Sensitivity : 81.7 %
# Specificity : 80.6 %
# Test Data:
# Accuracy : 80.4 %
# Sensitivity : 80.4 %
# Specificity : 80.5 %
# Thus we have achieved our goal of getting a ballpark of the target lead conversion rate to be around 80% . The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model to get a higher lead conversion rate of 80%.

# 2) Finding out the leads which should be contacted:
# The customers which should be contacted are the customers whose "Lead Score" is equal to or greater than 85. They can be termed as 'Hot Leads'.


# In[207]:


hot_leads=y_pred_final.loc[y_pred_final["Lead_Score"]>=85]
hot_leads


# In[208]:


# So there are 368 leads which can be contacted and have a high chance of getting converted. The Prospect ID of the customers to be contacted are :


# In[209]:


print("The Prospect ID of the customers which should be contacted are :")

hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)
hot_leads_ids


# In[210]:


# 3) Finding out the Important Features from our final model:


# In[211]:


res.params.sort_values(ascending=False)


# In[212]:


# # Recommendations:
# The company should make calls to the leads coming from the lead sources "Welingak Websites" and "Reference" as these are more likely to get converted.
# The company should make calls to the leads who are the "working professionals" as they are more likely to get converted.
# The company should make calls to the leads who spent "more time on the websites" as these are more likely to get converted.
# The company should make calls to the leads coming from the lead sources "Olark Chat" as these are more likely to get converted.
# The company should make calls to the leads whose last activity was SMS Sent as they are more likely to get converted.

# The company should not make calls to the leads whose last activity was "Olark Chat Conversation" as they are not likely to get converted.

# The company should not make calls to the leads whose lead origin is "Landing Page Submission" as they are not likely to get converted.
# The company should not make calls to the leads whose Specialization was "Others" as they are not likely to get converted.
# The company should not make calls to the leads who chose the option of "Do not Email" as "yes" as they are not likely to get converted.

