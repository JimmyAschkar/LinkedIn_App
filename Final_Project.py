

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

# [theme]
# backgroundColor="#fafdff"
# secondaryBackgroundColor="#edf1f3"
# textColor="#080808"

# col1, col2 = st.columns([70, 150])
# with col1:
#     st.image("linkedin-logo.png", width = 350)
# with col2:
#     st.markdown("# User Prediction App")

st.image("linkedin-logo.png")
st.markdown("# User Prediction App")

st.markdown("***")

st.subheader("Instructions:\n Please complete the attribute form below to determine the likelyhood of being a LinkedIn user")
#st.markdown("###### Please complete the attribute form below to determine the likelyhood of being a LinkedIn user")

st.subheader("Attributes:")
##EDUCATION## 
educ = st.selectbox("Education level", 
             options = ["Less Than High School",
                        "High School Incomplete",
                        "High School Graduate",
                        "Some College, No Degree",
                        "Two-Year Associate Degree", 
                        "College Degree",
                        "Some Postgraduate",
                        "Postgraduate Degree"])

if educ == "Less Than High School":
    educ = 1
elif educ == "High School Incomplete":
    educ = 2
elif educ == "High School Graduate":
    educ = 3
elif educ == "Some College, No Degree":
    educ = 4
elif educ == "Two-Year Associate Degree":
    educ = 5
elif educ == "College Degree":
    educ = 6
elif educ == "Some Postgraduate":
    educ = 7
else:
    educ = 8

##INCOME##
inc = st.selectbox("Household Income", 
             options = ["Less than $10,000",
                        "10 to under $20,000",
                        "20 to under $30,000",
                        "30 to under $40,000",
                        "40 to under $50,000", 
                        "50 to under $75,000",
                        "75 to under $100,000",
                        "100 to under $150,000",
                        "$150,000 +"])

if inc == "Less than $10,000":
    inc = 1
elif inc == "10 to under $20,000":
    inc = 2
elif inc == "20 to under $30,000":
    inc = 3
elif inc == "30 to under $40,000":
    inc = 4
elif inc == "40 to under $50,000":
    inc = 5
elif inc == "50 to under $75,000":
    inc = 6
elif inc == "75 to under $100,000":
    inc = 7
elif inc == "100 to under $150,000":
    inc = 8
else:
    inc = 9

##PARENT##
par = st.selectbox("Parent", 
             options = ["Yes",
             "No"])

if par == "Yes":
    par = 1
else:
    par = 0

##MARIED##
mar = st.selectbox("Marital Status", 
             options = ["Married",
             "Not Married"])

if mar == "Married":
    mar = 1
else:
    mar = 0

##GENDER##
gend = st.selectbox("Gender", 
             options = ["Female",
             "Male"])

if gend == "Married":
    gend = 1
else:
    gend = 0

Ag = st.slider(label="Age - Drag Slider", 
          min_value=1,
          max_value= 97,
          value= 25)

##PREDICTION MODEL##

s = pd.read_csv("social_media_usage.csv")

def clean_sm (x):
    x = np.where(x == 1, 1, 0)
    return(x)

s["sm_li"] = clean_sm(s["web1h"])

ss = pd.DataFrame({
    "LinkedIn User": s["sm_li"], 
    "Income":np.where(s["income"] <= 9, s["income"], np.nan),
    "Education":np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "Parent":np.where(s["par"] > 2, np.nan, 
                    np.where(s["par"] == 1, 1, 0)),
    "Marital Status":np.where(s["marital"] == 1, 1, 0),
    "Gender":np.where(s["gender"] > 2, np.nan,
                     np.where(s["gender"] == 2, 1, 0)),
    "Age":np.where(s["age"] > 97, np.nan, s["age"])
})

ss = ss.dropna()

y = ss["LinkedIn User"]
X = ss[["Household Income", "Education", "Parent", "Marital Status", "Gender", "Age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                   stratify = y,
                                                   test_size = 0.2,
                                                   random_state = 987)

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X_train, y_train)

# newdf = pd.DataFrame({
#     "Income": [inc], 
#     "Education": [educ],
#     "Parent": [par], 
#     "Marital Status": [mar], 
#     "Gender": [gend], 
#     "Age": [Ag]
# })

# newdf["prediction_LinkedIn_user"] = lr.predict(newdf)

person = [inc, educ, par, mar, gend, Ag]

predicted_class = lr.predict([person])
probs = lr.predict_proba([person])

st.markdown("***")

st.write(probs)


