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
import streamlit.components.v1 as components

#==============================================================================
# Load and Preprocess Data + Select 4 Key Features
#==============================================================================


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

# Conclusion about our best performing model: Our best performing model is the one with the highest AUC for the train 
# partition, therefore Random Forest performs the best. We will use this model for the next steps.

# Train and evaluate the model on churn_train.csv and interpret the model predictions on churn_test.csv.
# load churn_test data as:
test_data = pd.read_csv("C:/Users/ffan1/OneDrive - IESEG (1)/Communications Skills/2021/GroupProject/OneDrive_1_11-2-2021/Data-20211102/churn_test.csv")


# Preprocessing the Test Data, encoding the categorical values


l_state = LabelEncoder()
test_data["state"] = l_state.fit_transform(test_data["state"])

l_international_plan = LabelEncoder()
test_data["international_plan"] = l_international_plan.fit_transform(test_data["international_plan"])

l_voice_mail_plan = LabelEncoder()
test_data["voice_mail_plan"] = l_voice_mail_plan.fit_transform(test_data["voice_mail_plan"])

l_area_code = LabelEncoder()
test_data["area_code"] = l_area_code.fit_transform(test_data["area_code"])

X_test_data = test_data[['total_day_charge','number_customer_service_calls','international_plan','voice_mail_plan']]

#Interpretability Technique 3: Shapley Values

# Exact Explainer (.predict)
def run_exact():
    global explainer, shap_values
    explainer = explainer = shap.Explainer(rf.predict, X_test_data)
    shap_values = explainer(X_test_data)

# Kernel Explainer (.predict_proba)
def run_kernel():
    global explainer, shap_values
    explainer = shap.Explainer(rf)
    shap_values = explainer.shap_values(X_test_data)

run_exact()

# helper to plot js plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 
def plot_bar():
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    return fig

def plot_beeswarm():
    fig = plt.figure()
    shap.plots.beeswarm(shap_values)
    return fig

def plot_summary_plot1():
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test_data, class_names=["Did not churn", "Churned"], show=False)
    return fig

def plot_summary_plot2():
    fig = plt.figure()
    shap.summary_plot(shap_values[1], X_test_data, plot_type="bar", show=False)
    return fig

def plot_summary_plot3():
    fig = plt.figure()
    shap.summary_plot(shap_values[1], X_test_data, plot_type="dot", show=False)
    return fig

def plot_decision_plot():
    fig = plt.figure()
    shap.decision_plot(explainer.expected_value[1], shap_values[1], X_test_data, show=False)
    return fig

def plot_force():
    fig = shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_data)
    return fig

def run():
    st.title("SHAP")

    selectbox = st.sidebar.selectbox(
        "Select SHAP Explainer type:",
        ("ExactExplainer", "KernelExplainer"))

    if selectbox =="ExactExplainer":
        run_exact()
        c1,c2 = st.columns(2)
        c1.pyplot(plot_bar())
        c2.pyplot(plot_beeswarm())

        
    elif selectbox =="KernelExplainer":
        run_kernel()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.pyplot(plot_summary_plot1())
        c2.pyplot(plot_summary_plot2())
        c3.pyplot(plot_summary_plot3())
        c4.plotly_chart(plot_decision_plot())
        c5.plotly_chart(st_shap(plot_force(), 400))


















    
