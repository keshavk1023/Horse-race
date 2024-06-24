import streamlit as st
import pickle
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.impute import SimpleImputer
import os

os.chdir("C:\\Users\\jeetg\\code\\horse race prediction")
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

os.chdir("C:\\Users\\jeetg\\code\\horse race prediction\\data")
r_d =  pd.read_csv("race.csv")
h_d = pd.read_csv("horse.csv")
f = pd.read_csv("forward.csv")

horse_lists = h_d['horseName'].unique().tolist()
courses_list = r_d['course'].unique().tolist()
jockey_names = h_d['jockeyName'].unique().tolist()
race_condition = r_d['condition'].unique().tolist()

from datetime import datetime
def predict_outcome(horse_names, race_conditions, jockeys, course):
    # Create input data in a pandas DataFrame
    input_data = {
        'horseName': horse_names,
        'condition': race_conditions,
        'jockeyName': jockeys,
        'course': [course] * len(horse_names)
    }
    input_df = pd.DataFrame(input_data)
    
    # Initialize label encoders
    label_encoders = {}
    
    # Encode categorical variables using LabelEncoder
    for feature in ['horseName', 'condition', 'jockeyName', 'course']:
        le = LabelEncoder()
        input_df[feature] = le.fit_transform(input_df[feature])
        label_encoders[feature] = le
    
    # Create marketTime features (using current time as placeholder)
    current_time = datetime.now()
    input_df['marketYear'] = current_time.year
    input_df['marketMonth'] = current_time.month
    input_df['marketDay'] = current_time.day
    input_df['marketHour'] = current_time.hour
    input_df['marketMinute'] = current_time.minute
    
    # Ensure all features match the training data (assuming X_train is defined somewhere)
    X_train_columns = ['horseName', 'condition', 'jockeyName', 'course', 'marketYear', 'marketMonth', 'marketDay', 'marketHour', 'marketMinute']
    missing_cols = set(X_train_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[X_train_columns]

    # Dummy predictions (replace with your actual model prediction)
    predictions = np.random.randint(0, 2, size=len(input_df))  # Example random predictions
    
    return predictions

st.title('Horse Racing Outcome Predictor')

# Select box for race conditions
race_conditions = st.selectbox('Select Race Conditions', race_condition)

# Select box for course
course = st.selectbox('Select Course', courses_list)

# Multi-select box for horse names
horse_names = st.multiselect('Select Horse Names', horse_lists)

# Multi-select box for jockeys
jockeys = st.multiselect('Select Jockeys', jockey_names)

if st.button('Predict'):
    if not horse_names or not jockeys:
        st.warning('Please select at least one horse and one jockey.')
    else:
        predictions = predict_outcome(horse_names, race_conditions, jockeys, course)
        st.write(predictions)
