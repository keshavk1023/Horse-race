import streamlit as st  # Streamlit is used to create web applications
import pickle  # Pickle is used for serializing and deserializing Python object structures
import numpy as np  # Numpy is used for numerical operations
import pandas as pd  # Pandas is used for data manipulation and analysis
import polars as pl  # Polars is an alternative DataFrame library for data manipulation (similar to pandas)
from sklearn.preprocessing import LabelEncoder  # LabelEncoder is used for encoding categorical variables
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier is a machine learning model
from sklearn.metrics import accuracy_score, classification_report  # For evaluating the model
import os  # Os module provides a way of using operating system dependent functionality

# Change the working directory to where the model file is located
os.chdir("C:\\Users\\keshavk\\code\\horse race prediction")

# Load the pre-trained model from a file
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Change the working directory to where the data files are located
os.chdir("C:\\Users\\keshavk\\code\\horse race prediction\\data")

# Read CSV files into DataFrames
r_d =  pd.read_csv("race.csv")
h_d = pd.read_csv("horse.csv")
f = pd.read_csv("forward.csv")

# Extract unique values for dropdowns in the Streamlit app
horse_lists = h_d['horseName'].unique().tolist()
courses_list = r_d['course'].unique().tolist()
jockey_names = h_d['jockeyName'].unique().tolist()
race_condition = r_d['condition'].unique().tolist()

# Define a function to predict race outcomes
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

# Create the Streamlit app interface
st.title('Horse Racing Outcome Predictor')

# Select box for race conditions
race_conditions = st.selectbox('Select Race Conditions', race_condition)

# Select box for course
course = st.selectbox('Select Course', courses_list)

# Multi-select box for horse names
horse_names = st.multiselect('Select Horse Names', horse_lists)

# Multi-select box for jockeys
jockeys = st.multiselect('Select Jockeys', jockey_names)

# Predict button to trigger prediction
if st.button('Predict'):
    if not horse_names or not jockeys:
        st.warning('Please select at least one horse and one jockey.')
    else:
        predictions = predict_outcome(horse_names, race_conditions, jockeys, course)
        st.write(predictions)
