#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle

# Load the saved model
with open("nb_model.pkl", "rb") as file:
    model = pickle.load(file)

#Streamlit app code
st.title("Chatgpt Tweets Sentiment Analysis App")

st.header("Enter the tweet here")

#Input text from the user
user_input = st.text_area("", height=100)

#creat predict button
if st.button("Predict"):
    #Preprocess the input text using the loded COuntorlizer
    text_dtm = model['vect'].transform([user_input])
    
    #Make Prediction
    prediction = model['nb'].predict(text_dtm)
    
    st.header("Prediction")
    #Display the pridiction sentiment
    if prediction == 0:
        st.subheader("Negative Sentiment")
    elif prediction == 1:
        st.subheader("Netural Sentiment")
    elif prediction == 2:
        st.subheader("Positive Sentiment")
   


# In[ ]:




