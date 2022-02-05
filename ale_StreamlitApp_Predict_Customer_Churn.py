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
import PyALE
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

# We have selected Neural Network to be our first black box model
# Neural Network
mlp_cla = MLPClassifier(hidden_layer_sizes=(32, 32, 16, 16), batch_size=16, early_stopping=True, random_state=42)
mlp_cla = mlp_cla.fit(X_train.values, y_train) # .values transforms df in np.array (avoid sklearn warning)


# predict
pred_train2 = mlp_cla.predict_proba(X_train)[:,1]
pred_test2 = mlp_cla.predict_proba(X_test)[:,1]

auc_neural_1 = roc_auc_score(y_train, pred_train2)
auc_neural_2 = roc_auc_score(y_test, pred_test2)


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

#Interpretability Technique 4: ALE

# adapt PyALE.ale function to incorporate classification models 
# reference: https://htmlpreview.github.io/?https://github.com/DanaJomar/PyALE/blob/master/examples/ALE%20plots%20for%20classification%20models.html
def ale(target=None, print_meanres=False, **kwargs):
    if target is not None:
        class clf():
            def __init__(self, classifier):
                self.classifier = classifier
            def predict(self, X):
                return(self.classifier.predict_proba(X)[:, target])
        clf_dummy = clf(kwargs["model"])
        kwargs["model"] = clf_dummy
    if (print_meanres & len(kwargs["feature"])==1):
        mean_response = np.mean(kwargs["model"].predict(kwargs["X"]), axis=0)
        print(f"Mean response: {mean_response:.5f}")
    return PyALE.ale(**kwargs)




def plot_ale1d():
    fig, axs = plt.subplots(1,4,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)

    ale_total_day_charge = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["total_day_charge"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[0], print_meanres=True)

    ale_number_customer_service_calls = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["number_customer_service_calls"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[1], print_meanres=True)
    
    ale_international_plan = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["international_plan"],
        include_CI=True,
        target=1,
        print_meanres=True,
        fig=fig,
        ax=axs[2])
    
    ale_voice_mail_plan = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["voice_mail_plan"],
        include_CI=True,
        target=1,
        print_meanres=True,
        fig=fig,
        ax=axs[3]
    )

    return fig

def plot_ale2d():
    fig, axs = plt.subplots(1,3,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)
    
    #fig, ax = plt.subplots(figsize=(2, 2))
    ale_2d_1 = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["total_day_charge", "number_customer_service_calls"],
        include_CI=True,
        target=1, 
        fig=fig,
        ax=axs[0])
    
    ale_2d_2 = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["total_day_charge", "international_plan"],
        include_CI=True,
        target=1, 
        fig=fig,
        ax=axs[1])
    
    ale_2d_3 = ale(
        X=X_test_data,
        model=mlp_cla,
        feature=["number_customer_service_calls", "international_plan"],
        include_CI=True,
        target=1, 
        fig=fig,
        ax=axs[2])   
    
    return fig


def run():
    st.title("ALE")
    st.pyplot(plot_ale1d())

    st.pyplot(plot_ale2d())



