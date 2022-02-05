#==============================================================================
# Initiating
#==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import PyALE
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from scipy import stats
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
shap.initjs()
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#==============================================================================
# Load and Preprocess Data + Select 4 Key Features
#==============================================================================


# Add dashboard title and description
st.title("Communication Skills Churn Analysis Dashboard")
st.write("Project Members: Fangda Fan, Mario Cortez, Sofie Ghysels")


# load train churn data as:
train_data = pd.read_csv("C:/Users/ffan1/OneDrive - IESEG (1)/Communications Skills/2021/GroupProject/OneDrive_1_11-2-2021/Data-20211102/churn_train.csv")

# preprocessing, encoding the categorical values

# variables to encode: state, area_code, international_plan, voice_mail_plan, churn variables
l_state = LabelEncoder()
train_data["state"] = l_state.fit_transform(train_data["state"])

l_international_plan = LabelEncoder()
train_data["international_plan"] = l_international_plan.fit_transform(train_data["international_plan"])

l_voice_mail_plan = LabelEncoder()
train_data["voice_mail_plan"] = l_voice_mail_plan.fit_transform(train_data["voice_mail_plan"])

l_area_code = LabelEncoder()
train_data["area_code"] = l_area_code.fit_transform(train_data["area_code"])

l_churn = LabelEncoder()
train_data["churn"] = l_churn.fit_transform(train_data["churn"])

# variable selection with forward stepwise variable selection procedure and AUC function
# functions imported from DataCamp 'Introduction to Predictive Analytics in Python'

def auc(variables, target, basetable):    
    X = basetable[variables]    
    y = basetable[target]    
    logreg = linear_model.LogisticRegression()    
    logreg.fit(X, y)    
    predictions = logreg.predict_proba(X)[:,1]    
    auc = roc_auc_score(y, predictions)
    return(auc)

def next_best(current_variables,candidate_variables, target, basetable):    
    best_auc = -1    
    best_variable = None
    for v in candidate_variables:
        auc_v = auc(current_variables + [v], target, basetable)
        if auc_v >= best_auc:
            best_auc = auc_v
            best_variable = v
    return best_variable

def auc_train_test(variables, target, train, test):
    X_train = train[variables]
    X_test = test[variables]
    Y_train = train[target]
    Y_test = test[target]
    logreg = linear_model.LogisticRegression()
    
    # Fit the model on train data
    logreg.fit(X_train, Y_train)
    
    # Calculate the predictions both on train and test data
    predictions_train = logreg.predict_proba(X_train)[:,1]
    predictions_test = logreg.predict_proba(X_test)[:,1]
    
    # Calculate the AUC both on train and test data
    auc_train = roc_auc_score(Y_train, predictions_train)
    auc_test = roc_auc_score(Y_test,predictions_test)
    return(auc_train, auc_test)

# we will try every variable in the dataframe until we get the combination of features that results in the highest value of AUC

candidate_variable = list(train_data.columns.values)
candidate_variable.remove("churn")


# What would the AUC be if we select 6 features 
# It is important to run the cell above (Cell 273) first, then run this cell 

current_variables = []
target = ['churn']
max_number_variables = 6
number_iterations = min(max_number_variables, len(candidate_variable))
basetable = train_data

for i in range(0,number_iterations):    
    next_var = next_best(current_variables,candidate_variable,target,basetable)    
    current_variables = current_variables + [next_var]    
    candidate_variable.remove(next_var)
print(current_variables)

# With 6 features, we are getting an AUC of 0.83

testauc = auc(current_variables, ['churn'], train_data)
print(round(testauc,2))

# we will try every variable in the dataframe until we get the combination of features that results in the highest value of AUC


candidate_variable = list(train_data.columns.values)
candidate_variable.remove("churn")
print(candidate_variable)
print(len(candidate_variable))

# What would the AUC be if we select 4 features  

current_variables = []
target = ['churn']
max_number_variables = 4
number_iterations = min(max_number_variables, len(candidate_variable))
basetable = train_data

for i in range(0,number_iterations):    
    next_var = next_best(current_variables,candidate_variable,target,basetable)    
    current_variables = current_variables + [next_var]    
    candidate_variable.remove(next_var)
print(current_variables)

# we test the AUC for this set of variables, if we keep adding more variables to the model the AUC won't increase, instead at variable number 11 the AUC starts to decrease:
# With 4 features, we are getting an AUC of 0.82

testauc = auc(current_variables, ['churn'], train_data)
print(round(testauc,2))
# should we stay with 4 variables and AUC of 0.82 or can we get the six variables and increase the AUC to 0.83 
# Is it worth it to manage 2 more variables for this little increase?
# After giving this some thought, we decided to check for correlation between variables using the pearson test to calculate their respective p-values

# we evaluate pearson correlation and it's p value 
# we want to see which features are correlated with each other

coef_correlation_list = []
p_correlation_list = []
name_column = []

for (index,column) in enumerate(train_data.columns): 
    a = train_data[column]
    b = train_data['churn']
    coef, p =  stats.pearsonr(a, b)
    
    #we define a small p value to validate what candidates are the ones we're keeping
    if p < 0.0001:
        name_column.append(column)
        coef_correlation_list.append(coef)
        p_correlation_list.append(p)
    
print(name_column)
print(coef_correlation_list)
print(p_correlation_list)  

# Variables 'total_day_minutes' and 'total_day_charge' have extremely high correlation (9.6e**-46)
# Variables 'total_eve_minutes' and 'total_eve_charge' have extremely high correlation (2.6e**-07)
# Therefore, only selecting 1 variable out of these 4 is enough
# So after taking out these extremely correlated variables aboe, our selection of 4 current_variables makes sense

#==============================================================================
# Perform Initial Insights in the Dataset
#==============================================================================
# total number of churners in international plans
train_data[train_data['international_plan']==1]['churn'].value_counts()
# it seems that this feature is one of the most important, as 42% of the clients that have an international plan have churned

# total number of churners 
train_data['churn'].value_counts()

# Total number of customers 
train_data.value_counts()

# 7% of the clientes that have a voice mail plan have churned
train_data[train_data['voice_mail_plan']==1]['churn'].value_counts()

# We did some Preliminary exploratory analysis of the impact on churn behavior by total_day_charge
# There is a positive relationship between churn behavior and total_day_charge
# When total day charge increase, customers tend to churn 

catplot_total_day_charge = sns.catplot(x="churn", y="total_day_charge", kind="point", data=train_data)
# We did some Preliminary exploratory analysis of the impact on churn behavior by number_customer_service_calls
# There is a positive relationship between churn behavior and number_customer_service_calls
# when number of customer service calls increase, customers tend to churn
catplot_nservicecalls = sns.catplot(x="churn", y='number_customer_service_calls', kind="point", data=train_data)
# We did some Preliminary exploratory analysis of the impact on churn behavior by international_plan
# There is a positive relationship between churn behavior and international_plan
# when customers have an international plan, they tend to churn
catplot_internationalplan = sns.catplot(x="churn", y='international_plan', kind="point", data=train_data)
# We did some Preliminary exploratory analysis of the impact on churn behavior by voice_mail_plan
# There is a negative relationship between churn behavior and voice_mail_plan
# when customers have a voicemail plan, they tend to not churn
catplot_voicemailplan = sns.catplot(x="churn", y='voice_mail_plan', kind="point", data=train_data)

# We have selected the Logistic Regression Model to be our linear model
# Logistic Regression Model

# split data in train and test (stratify y)
X = train_data[['total_day_charge','number_customer_service_calls','international_plan','voice_mail_plan']] 
y = train_data[['churn']]

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.3, random_state=42)

# fit model
logreg = LogisticRegression(fit_intercept=True)
logreg = logreg.fit(X_train, y_train)

# predict
pred_train1 = logreg.predict_proba(X_train)[:,1]
pred_test1 = logreg.predict_proba(X_test)[:,1]


auc_log_1 = roc_auc_score(y_train, pred_train1)
auc_log_2 = roc_auc_score(y_test, pred_test1)

# In general, an AUC of 0.5 suggests no discrimination / random / average value (i.e., ability to diagnose patients with 
# and without the disease or condition based on the test), 0.7 to 0.8 is considered acceptable, 
# 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding.

# We have selected Neural Network to be our first black box model
# Neural Network
mlp_cla = MLPClassifier(hidden_layer_sizes=(32, 32, 16, 16), batch_size=16, early_stopping=True, random_state=42)
mlp_cla = mlp_cla.fit(X_train.values, y_train) # .values transforms df in np.array (avoid sklearn warning)


# predict
pred_train2 = mlp_cla.predict_proba(X_train)[:,1]
pred_test2 = mlp_cla.predict_proba(X_test)[:,1]

auc_neural_1 = roc_auc_score(y_train, pred_train2)
auc_neural_2 = roc_auc_score(y_test, pred_test2)


# We have selected Random Forest to be our 2nd black box model
# Random Forest

# fit random forest
rf = RandomForestClassifier(max_depth= 6, random_state=42).fit(X_train,y_train)

# predict
pred_train3 = rf.predict_proba(X_train)[:,1]
pred_test3 = rf.predict_proba(X_test)[:,1]

auc_rf_1 = roc_auc_score(y_train, pred_train3)
auc_rf_2 = roc_auc_score(y_test, pred_test3)


# The depth of the tree should be enough to split each node to your desired number of observations. There has been some 
# work that says best depth is 5-8 splits. We decide to go with 6 splits.

# Extract feature importance
pd.DataFrame(rf.feature_importances_, index = X.columns, columns = ['Feature importance'])

# We evaluate the 3 AUC scores for each of our 3 model:
eval_dict = {
    
     "Logistic Regression":{
        "AUC_train":auc_log_1,
        "AUC_test":auc_log_2
    },   
    "Neural Network":{
        "AUC_train":auc_neural_1,
        "AUC_test":auc_neural_2
    },
    "Random Forest":{
        "AUC_train":auc_rf_1,
        "AUC_test":auc_rf_2
    }
}

eval_df = pd.DataFrame(eval_dict)


# Conclusion about our best performing model: Our best performing model is the one with the highest AUC for the train 
# partition, therefore Random Forest performs the best. We will use this model for the next steps.



    
def run():
    st.title("Initial Exploratory Analysis with 4 Selected Features")

    c1,c2,c3,c4 = st.columns((4))

    c1.pyplot(catplot_total_day_charge)
    c2.pyplot(catplot_nservicecalls)
    c3.pyplot(catplot_internationalplan)
    c4.pyplot(catplot_voicemailplan)
    
    st.dataframe(eval_df)