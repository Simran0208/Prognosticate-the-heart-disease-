#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for visualization
sns.set()
get_ipython().magic('matplotlib inline')
import sklearn # for scientific calculations
from sklearn import preprocessing
from matplotlib import rcParams
from seaborn import distplot
import warnings
warnings.filterwarnings("ignore")
import pickle
from flask import Flask, request, render_template

# preprocessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC


# In[2]:


data = pd.read_csv("C:\\Users\\SIMRAN\\Desktop\cardiovascular.csv", sep=";")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


# Rename columns to make features more clearly understood
data.rename(columns={'ap_hi': 'systolic', 'ap_lo': 'diastolic', 'gluc': 'glucose', 'alco': 'alcohol', 'cardio': 'target'}, inplace=True)


# In[8]:


data.head()


# In[9]:


data = data.drop('id', axis=1)


# In[10]:


data.head()


# In[11]:


# 24 Duplicated entries
print(f'{data.duplicated().sum()} duplicates recorded.')


# In[12]:


data[data.duplicated()]


# In[13]:


data.drop_duplicates(inplace=True)


# In[14]:


data.count()


# In[15]:


print(f'{data.dtypes.value_counts()}')


# In[16]:


data.shape


# In[17]:


print('Let us now get a quick summary of features available.')
data.describe().T.round(2)


# In[18]:


# Convert the ages from number of days to categorical values
def calculate_age(days):
  days_year = 365.2425
  age = int(days // days_year)
  return age


# In[19]:


def categorize_age(age):
  if 0 < age <= 2:
    return "Infants"
  elif 2 < age <= 5:
    return "Pre School Child"
  elif 5 < age <= 12:
    return "Child"
  elif 12 < age <= 19:
    return "Adolescent"
  elif 19 < age <= 24:
    return "Young Adult"
  elif 24 < age <= 44:
    return "Adult"
  elif 44 < age <= 65:
    return "Middle Aged"
  elif 65 < age:
    return "Aged"


# In[20]:


def categorize_age_tees(age):
  if 0 < age <= 10:
    return "10s"
  elif 10 < age <= 30:
    return "20s"
  elif 20 < age <= 30:
    return "30s"
  elif 30 < age <= 40:
    return "40s"
  elif 40 < age <= 50:
    return "50s"
  elif 50 < age <= 60:
    return "60s"
  elif 60 < age <= 70:
    return "70+"


# In[21]:


# age transformations
data['age'] = data['age'].apply(lambda x: calculate_age(x))
data['age_cat'] = data['age'].apply(lambda x: categorize_age(x))
data['age_tees'] = data['age'].apply(lambda x: categorize_age_tees(x))
data.head()


# In[22]:


rcParams['figure.figsize'] = 7, 7
sns.countplot(x ='age_cat', data = data) 


# In[23]:


# Visulization of age groups with cvd info
rcParams['figure.figsize'] = 7, 7
sns.countplot(x ='age_cat', hue = 'target', data = data) 


# In[24]:


plt.rcParams['figure.figsize'] = (10, 5)
sns.distplot(data['age'], color = 'red')
plt.title('Distribution of Age', fontsize = 20)
plt.show()


# In[25]:


rcParams['figure.figsize'] = 11, 8
sns.countplot(x='age', hue='target', data = data, palette="Set2");


# In[26]:


rcParams['figure.figsize'] = 10, 8
sns.countplot(x='age_tees', hue='target', data = data, palette="Set2", order = ['10s','20s','30s','40s','50s','60s','70+']);


# In[27]:


# Height comparison 
data.groupby('gender')['height'].mean()


# In[28]:


# Alcohol consumption 
data.groupby('gender')['alcohol'].sum()


# In[29]:


# Gender Ratio
data['gender'].value_counts()


# In[30]:


# Calcualte the CVD distribution based on Gender
data['target'].value_counts(normalize=True)


# In[31]:


sns.lmplot(x="weight",y="height",hue="gender", data=data, fit_reg=False)
plt.show()


# In[32]:


rcParams['figure.figsize'] = 7, 7
sns.countplot(x='gender', hue='target', data = data, palette="Set2");


# In[33]:


for col in ["height", "weight"]:
    sns.kdeplot(data[col], shade=True)
    


# In[34]:


# Height Distribution
data_melt = pd.melt(frame=data, value_vars=['height'], id_vars=['gender'])
plt.figure(figsize=(7, 7))
ax = sns.violinplot(
    x='variable', 
    y='value', 
    hue='gender', 
    split=True, 
    data=data_melt, 
    scale='count',
    scale_hue=False,
    palette="Set1");


# In[35]:


# Weight Distribution
data_melt = pd.melt(frame=data, value_vars=['weight'], id_vars=['gender'])
plt.figure(figsize=(7, 7))
ax = sns.violinplot(
    x='variable', 
    y='value', 
    hue='gender', 
    split=True, 
    data=data_melt, 
    scale='count',
    scale_hue=False,
    palette="Set1");


# In[36]:


plt.figure(figsize=(6,4))
sns.countplot(x='cholesterol', hue='target', data=data,palette="BrBG")
plt.show()
# There appears to be a correlation between higher cholesterol levels and cardiovascular disease
# chloesterol levels: 1 = normal, 2 = above normal, 3 = well above normal


# In[37]:


plt.figure(figsize=(6,4))
sns.countplot(x='glucose', hue='target', data=data)
plt.show()
# There appears to be another correlation between higher glucose levels and cardiovascular disease
# glucose levels: 1 = normal, 2 = above normal, 3 = well above normal


# In[38]:


plt.figure(figsize=(8,6))
sns.countplot(x='active', hue='target', data=data)
plt.show()


# In[39]:


plt.figure(figsize=(8,6))
sns.countplot(x='smoke', hue='target', data=data)
plt.show()


# In[40]:


plt.figure(figsize=(8,6))
sns.countplot(x='alcohol', hue='target', data=data)
plt.show()


# In[41]:


plt.figure(figsize=(10,4))
sns.distplot(data['weight'], kde=False)
plt.show()


# In[42]:


data['weight'].sort_values().head()


# In[43]:


plt.figure(figsize=(10,4))
sns.distplot(data['height'], kde=False)
plt.show()


# In[44]:


data['height'].max()
#This maximum height of 250 cm/8.2 ft seems unlikely


# In[45]:


data['height'].sort_values().head()
#The minimum height of 55 cm/1.8 ft also seems unlikely and unrealistic.
#This dataset may not be legitimate; however, we will continue on with the data analysis and model selection.


# In[46]:


# calculate the BMI score 
data['BMI'] = data['weight']/((data['height']/100)**2)
data['pulse pressure'] = data['systolic'] - data['diastolic']


# categorize normal & abnormal
def bmi_categorize(bmi_score):
  if 18.5 <= bmi_score <= 25:
    return "Normal"
  else:
    return "Abnormal"

data["BMI_State"] = data["BMI"].apply(lambda x: bmi_categorize(x))
data["BMI_State"].value_counts().plot(kind='pie')


# In[47]:


data.head()
# Quick look at the dataframe to make sure these new features have been added


# In[48]:


data[data['BMI'] > 100].head(10)


# In[49]:


data[(data['pulse pressure'] >= 60 ) & (data['cholesterol'] == 3)].head(15)


# In[50]:


rcParams['figure.figsize'] = 7, 7
sns.countplot(x='BMI_State', hue='target', data = data, palette="Set2");


# In[51]:


# Alcohol consumption 
data["alcohol"].value_counts().plot(kind='pie')


# In[52]:


with sns.axes_style('white'):
    g = sns.factorplot("alcohol", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set2")
    g.set_ylabels('Number of Patients')


# In[53]:


data["smoke"].value_counts().plot(kind='pie')


# In[54]:


with sns.axes_style('white'):
    g = sns.factorplot("smoke", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set2")
    g.set_ylabels('Number of Patients')


# In[55]:


data["active"].value_counts().plot(kind='pie')


# In[56]:


with sns.axes_style('white'):
    g = sns.factorplot("active", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set2")
    g.set_ylabels('Number of Patients')


# In[57]:


# Let us first have a look at our target variable.
plt.figure(figsize=(4,3))
fig, ax = plt.subplots(1,1)
sns.countplot(data['target'], ax = ax)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2,height,'{:.2f}'.format((i.get_height()/len(data['target']))*100,'%'))
plt.show()


# In[58]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.distplot(data['age'][data['target']==0], ax = ax1, color='green')
sns.distplot(data['age'][data['target']==1], ax = ax1,color='coral')
ax1.set_title('Age Distribution')
ax1.legend()

sns.distplot(data['age'][(data['gender']==1) & (data['target']==1)],ax = ax2,color='pink')
sns.distplot(data['age'][(data['gender']==2) & (data['target']==1)],ax = ax2,color='blue')
ax2.set_title('Disease count distribution by  gender, aged below 54.')
plt.show()


# In[59]:


fig, (ax1) = plt.subplots(1,1, figsize=(10,10))
sns.boxenplot(data['target'],(data['height']*0.0328084),ax=ax1)
ax1.set_title('Height / Diseased')
plt.show()


# In[60]:


fig, (ax1) = plt.subplots(1,1, figsize=(10,10))
sns.boxenplot(data['target'],(data['height']),ax=ax1)
ax1.set_title('Height / Diseased')
plt.show()


# In[61]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.scatterplot(data['age'],data['height'][(data['height']*0.0328084)<4]*0.0328084,hue=data['target'],ax=ax1)
ax1.set_title('Height vs Age')
sns.scatterplot(data['weight'],data['height'][(data['height']*0.0328084)<4]*0.0328084,hue=data['target'],ax=ax2)
ax2.set_title('Height vs Weight')
plt.show()


# In[62]:


#Converting height in cms to foot.
data['height'] = data['height']*0.0328084 
filt =(data['height']>8) | (data['height']<3) 

data.drop(index = list(data[filt].index),inplace=True)
print(f'Dataset: {data.shape}')


# In[63]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,5))
sns.boxenplot(data['target'],(data['weight']),ax=ax1)
ax1.set_title('Weight / Diseased')
sns.scatterplot(data['weight'],data['height'],ax=ax2,hue=data['target'])
ax2.set_title('height vs weight')
plt.show()


# In[64]:


# 1. Weight < 25 kg
filt1 = data['weight']<25
data.drop(index=list(data[filt1].index),inplace=True)

# 2. Weight > 175 kg
filt2 = data['weight']>175
data.drop(index=list(data[filt2].index),inplace=True)

# 3. Height < 4.5 & Weight > 150 kg
filt3 = (data['height']<4.5) & (data['weight']>150)
data.drop(index=list(data[filt3].index),inplace=True)


# In[65]:


# Gender
fig,(ax) = plt.subplots(1,1, figsize=(8,6))
tmp = pd.crosstab(data['gender'],data['target'],normalize='index').round(4)*100
tmp.columns = ['Not Diseased','Diseased']
ax1 = sns.countplot(data['gender'])
ax2 = ax1.twinx()
sns.pointplot(tmp.index,tmp['Diseased'],ax=ax2, color='red')
for x in ax1.patches:
    height = x.get_height()
    ax1.text(x.get_x()+x.get_width()/2,height,'{:.2f}{}'.format((height/len(data))*100,'%'))
plt.show()


# In[66]:


size = data['gender'].value_counts()
colors = ['lightblue', 'lightgreen']
labels = "Male", "Female"


my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.rcParams['figure.figsize'] = (5, 5)
plt.pie(size,colors = colors, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Distribution of Gender', fontsize = 20)
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.legend()
plt.show()


# In[67]:


# systolic
filt = (data['systolic']<90) | (data['systolic']>140)
print(f'Normal systolic blood pressure range is between 90 and 120. However, from our dataset we can see that we have {len(data[filt])} records that are not falling within the normal range. We can replace them with their median values.')


# In[68]:


data['systolic'].replace(data[filt]['systolic'].values,data['systolic'].median(),inplace=True)


# In[69]:


# filt =  (data['ap_lo']>90) | (data['ap_lo']<60)
fig, ax = plt.subplots(1,1, figsize = (20,10))
sns.distplot(data['diastolic'][data['diastolic']<200],bins = 25, kde = True, ax = ax)
xticks = [i*10 for i in range(-5,20)]
ax.set_xticks(xticks)
ax.tick_params(axis='x',labelsize = 16, pad = 12,  
               colors ='r')
plt.show()
print(f'Similar to Systolic Blood Pressure Range the diastolic bp range should be between 60-90 for a healthy individual. However, in this case we have median values for diastolic as {data.diastolic.median()} which does not look correct to me. Considering this in mind we would have to do some further analysis if the data source is correct or not.')


# In[70]:


plt.figure(figsize=(10,10))
sns.boxenplot(data['target'],data['diastolic'][data['diastolic']<150])
plt.show()


# In[71]:


data.tail(5)


# In[72]:


# cholesterol
tmp = pd.crosstab(data['cholesterol'],data['target'],normalize='index')
tmp.columns = ['not diseased','diseased']
fig, ax = plt.subplots(1,1, figsize=(6,6))
sns.countplot(data['cholesterol'], ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['cholesterol'])*100,'%'))
plt.show()


# In[73]:


# Glucose
tmp = pd.crosstab(data['glucose'],data['target'],normalize='index')
tmp.columns = ['not diseased','diseased']
fig, ax = plt.subplots(1,1)
sns.countplot(data['glucose'], ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['glucose'])*100,'%'))
plt.show()


# In[74]:


data['smoke/alcohol'] = data['smoke'].apply(str)+'|'+data['alcohol'].apply(str)

tmp = pd.crosstab(data['smoke/alcohol'],data['target'],normalize='index')
tmp.columns = ['Not diseased','diseased']

fig, ax = plt.subplots(1,1)
sns.countplot(data['smoke/alcohol'], ax=ax)
plot2 = ax.twinx()
sns.pointplot(tmp.index,tmp['diseased'],ax=plot2)
for patch in ax.patches:
    height = patch.get_height()
    ax.text(patch.get_x()+patch.get_width()/2,height,'{:.2f}{}'.format(height/len(data['smoke/alcohol'])*100,'%'))
plt.show()


# In[75]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
sns.boxenplot(data['target'],data['BMI'],ax=ax1)
sns.distplot(data[data['target']==0]['BMI'],color='g',ax=ax2)
sns.distplot(data[data['target']==1]['BMI'],color='b',ax=ax2)
plt.show()


# In[76]:


def categorize_blood_pressure(x):
  if x['systolic'] < 120 and x['diastolic'] < 80:
    return "Normal"
  elif 120 <= x['systolic'] <= 129 and x['diastolic'] < 80:
    return "Elevated"
  elif 130 <= x['systolic'] <= 139 or 80 <= x['diastolic'] <= 89:
    return "High Blood Pressure(Stage 1)"
  elif  140 <= x['systolic'] <= 180 or 90 <= x['diastolic'] <= 120:
    return "High Blood Pressure(Stage 2)"
  elif  (x['systolic'] > 180 and  x['diastolic'] > 120) or (x['systolic'] > 180 or x['diastolic'] > 120):
    return "Hypertensive Crisis"


# In[77]:


# categorizing blood pressure
data['blood_category'] = data.apply(categorize_blood_pressure, axis=1)
 
data.head()


# In[78]:


# Visulization of blood pressure category
data["blood_category"].value_counts().plot(kind='pie')


# In[79]:


with sns.axes_style('white'):
    g = sns.factorplot("blood_category", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set1", order=["Normal", "Elevated", "High Blood Pressure(Stage 1)", "High Blood Pressure(Stage 2)", "Hypertensive Crisis"])
    g.set_ylabels('Number of Patients')


# In[80]:


with sns.axes_style('white'):
    g = sns.factorplot("cholesterol", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set2")
    g.set_ylabels('Number of Patients')


# In[81]:


with sns.axes_style('white'):
    g = sns.factorplot("glucose", data=data, aspect=4.0, kind='count',
                       hue='target', palette="Set2")
    g.set_ylabels('Number of Patients')
    


# In[82]:


fig, ax = plt.subplots(figsize = (10, 5))

plt.subplot(2,1,1)
sns.countplot('cholesterol', data= data)
plt.title('Cholesterol')

plt.subplot(2,1,2)
sns.countplot('glucose', data= data)
plt.title('Glucose')

plt.subplots_adjust(hspace= 0.6)
plt.show()


# In[83]:


fig, ax = plt.subplots(figsize = (10, 5))

plt.subplot(2,2,1)
sns.countplot('smoke', data = data)
plt.title('Smoke')

plt.subplot(2,2,2)
sns.countplot('alcohol', data = data)
plt.title('Alcohol')

plt.subplot(2,2,3)
sns.countplot('active', data = data)
plt.title('Activity')

plt.subplot(2,2,4)
sns.countplot('target', data = data)
plt.title('Cardio disease')

plt.subplots_adjust(hspace= 0.6)
plt.show()


# In[84]:


# Filtering out the required features
new_data = data[["gender","age","age_tees","height","BMI","BMI_State","cholesterol","glucose","smoke","alcohol","active","blood_category","pulse pressure","target"]].copy()
new_data.head()


# In[85]:


# Checking any missing values
new_data.isnull().sum()


# In[86]:


categorical_val = []
continous_val = []
for column in new_data.columns:
    print('==============================')
    print(f"{column} : {new_data[column].unique()}")
    if len(new_data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[87]:


categorical_val


# In[88]:


# Label encode the categorical columns BMI_State & blood category
le = preprocessing.LabelEncoder()

# BMI_State
le.fit(new_data['BMI_State'])
new_data['BMI_State'] = le.transform(new_data['BMI_State'])

# blood category
le.fit(new_data['blood_category'])
new_data['blood_category'] = le.transform(new_data['blood_category'])

# age tees
le.fit(new_data['age_tees'])
new_data['age_tees'] = le.transform(new_data['age_tees'])

new_data.head()


# In[89]:


# plotting correlation map
corr = new_data.corr()
f, ax = plt.subplots(figsize = (15,15))
sns.heatmap(corr, annot=True, fmt=".3f", linewidths=0.4, ax=ax)


# In[90]:


new_data.head(5)


# In[91]:


new_data.shape


# In[92]:


categorical_val.remove('target')
new_data = pd.get_dummies(new_data, columns = categorical_val)


# In[93]:


new_data.head()


# In[94]:


print(data.columns)
print(new_data.columns)


# In[95]:


from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_to_scale = ['age', 'pulse pressure', 'height', 'BMI']
new_data[col_to_scale] = s_sc.fit_transform(new_data[col_to_scale])


# In[96]:


new_data.head()


# In[97]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,auc,roc_auc_score,roc_curve

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        


# In[98]:


from sklearn.model_selection import train_test_split

X = new_data.drop('target', axis=1)
y = new_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                                                    


# In[99]:


X_train.head(10)


# In[100]:


X_test.head(10)


# In[101]:


X_train.info()


# In[102]:


X_test.info()


# In[103]:


#%% split training set to validation set
#Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.3)
                                              


# In[104]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)


# In[105]:


train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score]], columns=['Algorithm', 'Training Accuracy %'])
results_df


# In[106]:


from sklearn.svm import SVC


svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm_clf.fit(X_train, y_train)

print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)


# In[107]:


train_score = accuracy_score(y_train, svm_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Support Vector Machine", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[108]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)


# In[109]:


train_score = accuracy_score(y_train, rf_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Random Forest Classifier", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[110]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)


# In[111]:


train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["K-nearest neighbors", train_score]], columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[112]:


# Gaussian Naive Bayes

gaussian_clf = GaussianNB()
gaussian_clf.fit(X_train, y_train)
print_score(gaussian_clf, X_train, y_train, X_test, y_test, train=True)


# In[113]:


train_score = accuracy_score(y_train, gaussian_clf.predict(X_train)) * 100
results_df_2 = pd.DataFrame(data=[["Gaussian naive bayes", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[114]:


percept_clf = Perceptron()
percept_clf.fit(X_train, y_train)
print_score(percept_clf, X_train, y_train, X_test, y_test, train=True)


# In[115]:


train_score = accuracy_score(y_train, percept_clf.predict(X_train)) * 100
results_df_2 = pd.DataFrame(data=[["Perceptron", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[116]:


from sklearn.tree import DecisionTreeClassifier


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)


# In[117]:


train_score = accuracy_score(y_train, tree_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Decision Tree Classifier", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[118]:


#stochastic gradient descent

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)


print_score(sgd_clf, X_train, y_train, X_test, y_test, train=True)


# In[119]:


train_score = accuracy_score(y_train, sgd_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Stochastic gradient descent", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
results_df = results_df.append(results_df_2, ignore_index=True)
results_df


# In[120]:


# Predicted values
y_head_lr = lr_clf.predict(X_train)
y_head_svm = svm_clf.predict(X_train)
y_head_rf = rf_clf.predict(X_train)
y_head_knn = knn_clf.predict(X_train)
y_head_gaussian = gaussian_clf.predict(X_train)
y_head_percept = percept_clf.predict(X_train)
y_head_dtc = tree_clf.predict(X_train)
y_head_sgd = sgd_clf.predict(X_train)


# In[121]:


cm_lr = confusion_matrix(y_train,y_head_lr)
cm_svm = confusion_matrix(y_train,y_head_svm)
cm_rf = confusion_matrix(y_train,y_head_rf)
cm_knn = confusion_matrix(y_train,y_head_knn)
cm_gaussian = confusion_matrix(y_train,y_head_gaussian)
cm_percept = confusion_matrix(y_train,y_head_percept)
cm_dtc = confusion_matrix(y_train,y_head_dtc)
cm_sgd = confusion_matrix(y_train,y_head_sgd)


# In[122]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(3,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,2)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,3)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(3,3,4)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,5)
plt.title("Gaussian naive bayes Confusion Matrix")
sns.heatmap(cm_gaussian,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,6)
plt.title("Perceptron Confusion Matrix")
sns.heatmap(cm_percept,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(3,3,7)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(3,3,8)
plt.title("Stochastic gradient descent Confusion Matrix")
sns.heatmap(cm_sgd,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.show()


# In[123]:


from sklearn.model_selection import GridSearchCV


# In[124]:


params = {"C": np.logspace(-4, 4, 20),
          "solver": ["liblinear"]}

lr_clf = LogisticRegression()

lr_cv = GridSearchCV(lr_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5, iid=True)
lr_cv.fit(X_train, y_train)
best_params = lr_cv.best_params_
print(f"Best parameters: {best_params}")
lr_clf = LogisticRegression(**best_params)

lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
#print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# In[125]:


train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

tuning_results_df = pd.DataFrame(data=[["Tuned Logistic Regression", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
tuning_results_df


# In[126]:


knn_clf = KNeighborsClassifier(n_neighbors=27)
knn_clf.fit(X_train, y_train)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)


# In[127]:


train_score = accuracy_score(y_train, knn_clf.predict(X_train)) * 100

results_df_2 = pd.DataFrame(data=[["Tuned K-nearest neighbors", train_score]], 
                          columns=['Algorithm', 'Training Accuracy %'])
tuning_results_df = tuning_results_df.append(results_df_2, ignore_index=True)
tuning_results_df


# In[128]:


from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = rf_clf, X = X_train, y = y_train, cv=10)
print("Cross validation accuracy of Random forest model = ", cross_validation)
print("\nCross validaion mean accuracy of Random forest model = ", cross_validation.mean())


# In[ ]:




