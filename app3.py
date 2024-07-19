#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples (customers)
n = 1000

# Generate synthetic data
customer_ids = np.arange(1, n+1)
ages = np.random.randint(18, 70, size=n)
genders = np.random.choice(['Male', 'Female'], size=n)
items_purchased = np.random.randint(1, 10, size=n)
total_spent = np.random.uniform(10, 500, size=n)

# Create a DataFrame
data = {
    'CustomerID': customer_ids,
    'Age': ages,
    'Gender': genders,
    'ItemsPurchased': items_purchased,
    'TotalSpent': total_spent
}

df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df)


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.metrics import classification_report

X = df[['TotalSpent','Age', 'ItemsPurchased']]  # Features
y = df['Gender']  # Target

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Optional: Print classification report for more detailed evaluation
print(classification_report(y_test, y_pred))


# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

# Load the pre-trained Logistic Regression model
# model = joblib.load('model.pkl') 

# Since the model was not saved, let's re-initialize and re-train it. 
# Assuming 'df', 'X' and 'y' are available from your previous code
model = LogisticRegression()
model.fit(X, y) # Fit on the entire dataset for this example

# Function to predict gender based on age and items purchased
def predict_gender(total_spent,age, items_purchased):
    prediction = model.predict([[total_spent,age, items_purchased]])[0]
    if prediction == 0:
        return 'Female'
    elif prediction == 1:
        return 'Male'
    else:
        return 'Unknown'

# Streamlit App
def main():
    # Page title and description
    st.title('Gender Prediction App')
    st.write('This app predicts gender based on age and items purchased.')

    # Sidebar with input fields
    with st.sidebar:
        st.subheader('Input Parameters')
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        items_purchased = st.number_input('Items Purchased', min_value=1, max_value=20, value=5)
        total_spent=st.number_input('Total Spent', min_value=10, max_value=500, value=50)
    # Predict button
    if st.button('Predict Gender'):
        result = predict_gender(age, items_purchased)
        st.success(f'Predicted Gender: {result}')

# Run the app
if __name__ == '__main__':
    main()

# After running this code successfully, you can save the model for future use.
joblib.dump(model, 'model3.pkl')


# In[ ]:




