import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import fnmatch
import sqlite3
import sys
from scipy.stats import ttest_ind
from imblearn.combine import SMOTETomek
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost.sklearn import XGBClassifier

sys.path.append("src")
import Feature_selection
import Model_training

con = sqlite3.connect("data/failure.db")
cur = con.cursor()
cur.execute("SELECT * FROM failure")
failure_df = pd.read_sql_query("SELECT * FROM failure", con)
con.close()

failure_df = Feature_selection.split_column(full_df = failure_df, split_col = "Model", new_cols = ["Model_no", "Year"], split_by = ',')

#filling nan values
failure_df["Membership"] = failure_df["Membership"].replace(np.nan, "None")

#drop duplicates
failure_df = failure_df.drop_duplicates(subset = "Car ID").reset_index(drop=True)

#convert degrees F to C
failure_df["Temperature"].astype(str)
failure_df = Feature_selection.split_column(full_df = failure_df, split_col = "Temperature", new_cols = ["Temp", "Unit"], split_by = " °")
failure_df["Temp"] = failure_df["Temp"].astype(float)
failure_df["Unit"] = failure_df["Unit"].astype(str)
failure_df["Unit"] = failure_df["Unit"].map({"C": 0, "F": 1})

new_temp = []
failure_df["Year"] = failure_df["Year"].astype(int)

for index, row in failure_df.iterrows():
    if row["Unit"] == 1:
        # convert Fahrenheit to Celcius
        new_temp.append(((row["Temp"]-32)*5)/9)
    elif row["Unit"] == 0:
        new_temp.append(row["Temp"])
        
failure_df["Temp"] = new_temp

#drop non-exsistent cities
failure_df = failure_df[failure_df.Factory != "Seng Kang, China"]
failure_df = failure_df[failure_df.Factory != "Newton, China"]
failure_df = failure_df[failure_df.Factory != "Bedok, Germany"]


#remove extreme outliers
failure_df = failure_df[failure_df.Temp != 230.7]
failure_df = failure_df[failure_df["RPM"]>0]

#create dummies
model_dummy = pd.get_dummies(failure_df["Model_no"], prefix= "Mod_")
factory_dummy = pd.get_dummies(failure_df["Factory"], prefix= "Fac_")
usage_dummy = pd.get_dummies(failure_df["Usage"], prefix= "Use_")
membership_dummy = pd.get_dummies(failure_df["Membership"], prefix= "Mem_")

clean_df = failure_df.drop(["Car ID", "Model_no", "Color", "Unit", "Factory", "Usage", "Membership"], axis = 1)

ml_df = pd.concat([model_dummy, factory_dummy, usage_dummy, membership_dummy, clean_df], axis = 1)

#making X and y axis
X = ml_df.drop(["Failure A", "Failure B","Failure C","Failure D","Failure E"], axis = 1)
y = clean_df.drop(["Year", "Temp","RPM","Fuel consumption"], axis = 1)
upscaler = SMOTETomek(sampling_strategy = "auto", random_state = 11)

#splitting y axis
ya = y["Failure A"]
yb = y["Failure B"]
yc = y["Failure C"]
yd = y["Failure D"]
ye = y["Failure E"]

#ML pipeline
y_list = [ya, yb, yc, yd, ye]
f1_list = []
recall_list = []
faults_list = []

for y_type in y_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=11)
    X_train, y_train = upscaler.fit_resample(X_train, y_train)
    pred = Model_training.pipeline_grid(X_train, X_test, y_train, y_test, model = "xgb", scaler = "mm")
    f1, recall, accuracy = Model_training.assess_model_simple(y_test , pred)
    faults = pred.sum()
    faults_list.append(faults)


summary = pd.DataFrame(index = ["Failure A","Failure B", "Failure C", "Failure D", "Failure E"], columns = ["Predicted faults"])
summary["Predicted faults"] = faults_list

#output results

print("""\

  _____ _            _ _                          _               _   
 |  __ (_)          | (_)                        | |             | |  
 | |__) | _ __   ___| |_ _ __   ___    ___  _   _| |_ _ __  _   _| |_ 
 |  ___/ | '_ \ / _ \ | | '_ \ / _ \  / _ \| | | | __| '_ \| | | | __|
 | |   | | |_) |  __/ | | | | |  __/ | (_) | |_| | |_| |_) | |_| | |_ 
 |_|   |_| .__/ \___|_|_|_| |_|\___|  \___/ \__,_|\__| .__/ \__,_|\__|
         | |                                         | |              
         |_|                                         |_|           

                    """)
print("Total cars predicted:")

print(len(y_test))
print("\n")

print(summary)
print("\n")
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"Recall: {recall}")




