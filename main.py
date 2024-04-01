import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

red_wine_data = pd.read_csv('data/red_wine.csv')
white_wine_data = pd.read_csv('data/white_wine.csv')

def preprocess_data(data):
    X = data.drop('quality', axis=1)
    y = data['quality']
    return X, y

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', acc)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    st.write('Confusion Matrix:')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    
    plt.xticks(range(len(unique_labels)), labels=unique_labels)
    plt.yticks(range(len(unique_labels)), labels=unique_labels)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    st.pyplot()
    
wine_type = st.sidebar.selectbox('Select Wine Type', ('Red', 'White'))

if wine_type == 'Red':
    data = red_wine_data.copy()
else:
    data = white_wine_data.copy()
    
# Preprocess the data
X, y = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier selection
classifier = st.sidebar.selectbox('Select Classifier', ('Decision Tree', 'Random Forest', 'Extreme Tree'))

if classifier == 'Decision Tree':
    model = DecisionTreeClassifier()
elif classifier == 'Random Forest':
    model = RandomForestClassifier()
else:
    model = ExtraTreesClassifier()

# Train and evaluate the model
st.header('Model Performance')
train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

# Button to input values and get prediction
st.sidebar.header('Input Values for Prediction')

fixed_acidity = st.sidebar.number_input('Fixed Acidity')
volatile_acidity = st.sidebar.number_input('Volatile Acidity')
citric_acid = st.sidebar.number_input('Citric Acid')
residual_sugar = st.sidebar.number_input('Residual Sugar')
chlorides = st.sidebar.number_input('Chlorides')
free_sulfur_dioxide = st.sidebar.number_input('Free Sulfur Dioxide')
total_sulfur_dioxide = st.sidebar.number_input('Total Sulfur Dioxide')
density = st.sidebar.number_input('Density')
pH = st.sidebar.number_input('pH')
sulphates = st.sidebar.number_input('Sulphates')
alcohol = st.sidebar.number_input('Alcohol')

# Button to trigger prediction
if st.sidebar.button('Get Prediction'):
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
    prediction = model.predict(input_data)[0]
    st.sidebar.write('Predicted Quality:', prediction)  