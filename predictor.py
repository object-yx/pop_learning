#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib as plt


# In[5]:


model = joblib.load("xgb2.pkl")


# In[6]:


x_test = pd.read_csv("x_test_scaler.csv")


# In[7]:


feature_names = ["Age","BMI","Parity","VD","C/S","MISC","HTN","DM","UFs","OPL","SUI","St","VL","CHD"]


# In[8]:


import warnings


# In[9]:


st.title("Pelvic Organ Prolapse Predictor")


# In[10]:


age = st.number_input("Age:",min_value=0,max_value=100,value=55)
BMI = st.number_input("BMI:",min_value=0,max_value=100,value=20)
parity = st.number_input("Parity:",min_value=0,max_value=100,value=2)
vd = st.number_input("VD:",min_value=0,max_value=100,value=1)
cs = st.number_input("C/S:",min_value=0,max_value=100,value=1)
mi = st.number_input("MISC:",min_value=0,max_value=100,value=55)


# In[11]:


htn = st.selectbox("HTN:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
dm = st.selectbox("DM:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
zgjl = st.selectbox("UFs:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
opl = st.selectbox("OPL:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
sui = st.selectbox("SUI:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
ST = st.selectbox("St:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
vl = st.selectbox("VL:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")
gxb = st.selectbox("CHD:",options=[0,1],format_func=lambda x:"是" if x==1 else "否")


# In[12]:


feature_values = [age, BMI, parity, vd, cs, mi, htn, dm, zgjl, opl, sui, ST, vl, gxb]


# In[13]:


features = np.array([feature_values])


# In[14]:


tab1, tab2 = st.tabs(["Prediction Result","SHAP"])
if st.button("Predict"):
  with tab1:
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    st.write(f"The prediction result for pelvic organ prolapse is:{predicted_class}(1:Prolapse,0:Normal)")
    st.write(f"The likelihood of pelvic organ prolapse is:{predicted_proba}")
    probability = predicted_proba[predicted_class]*100
    if predicted_class == 1:
        advice = (f"High risk"
                  f"The likelihood of pelvic organ prolapse is{probability:1f}%")
    else:
        advice = (f"Low risk"
                  f"The likelihood of pelvic organ prolapse is{probability:1f}%")
    st.write(advice)
  with tab2:
    st.subheader("SHAP")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns = feature_names))
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value, shap_values[1],pd.DataFrame([feature_names], columns = feature_names),matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value, shap_values[0],pd.DataFrame([feature_names], columns = feature_names),matplotlib=True)
    plt.savefig("shap_force_plot.png",bbox_inches="tight",dpi=1200)
    st.image("shap_force_plot.png",caption="SHAP Force Plot Explanation")


# In[ ]:




