#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,render_template,request,jsonify
import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__)
#sentiment_model = pickle.load(open('sentiment_model.pkl','rb'))
model_pipe = load('text_classification.joblib')

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    message = request.form['message']
    my_pred = model_pipe.predict(["message"])
        
    if my_pred == 1:
        pred = 'positive'
    
    elif my_pred == 0:
        pred = 'neutral'
    
    else:
        pred = 'negative'
                
    return render_template('home.html',prediction = pred)


if __name__ == "__main__":
    
    app.run(port='8088',threaded=True, debug=True,use_reloader=False)


# In[ ]:




