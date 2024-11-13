import streamlit as st
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

# Define a custom clipping layer to replace the Lambda layer
class ClippingLayer(Layer):
    def __init__(self, min_value=1, max_value=5, **kwargs):
        super(ClippingLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super(ClippingLayer, self).get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config

# Load pre-trained model and data (replace 'model_path' and 'data_path' with actual file paths)
model = load_model("my_model2.keras", custom_objects={"ClippingLayer": ClippingLayer})  # Use custom layer here
df = pd.read_csv('PREDICTIVE2.csv',encoding='ISO-8859-1')  # Replace with your dataset path
# Define mappings for Customer ID, Product ID, Product Category, and Domain
Cust_ID_Map = {Customer_ID: idx for idx, Customer_ID in enumerate(df['Customer_ID'].unique())}
Prod_ID_Map = {prod: idx for idx, prod in enumerate(df['Shortcut_Dimension_Code'].unique())}
Prod_Cat_ID_Map = {category: idx for idx, category in enumerate(df['Description'].unique())}  # Product category mapping
Dom_ID_Map = {domain: idx for idx, domain in enumerate(df['DOMAIN'].unique())}  # Domain mapping
Prod_ID = df['Shortcut_Dimension_Code'].unique().tolist()  # List of unique product IDs

# Utility function to recommend products
def recommend_prods(customer_id, model, data, num_recommendations=3):
    if customer_id not in Cust_ID_Map:
        st.write(f"Customer ID {customer_id} not found in Cust_ID_Map.")
        return []

    # Map customer ID to internal ID and get unseen products
    cust_internal_id = Cust_ID_Map[customer_id]
    seen_prods = data[data['Customer_ID'] == customer_id]['Shortcut_Dimension_Code'].tolist()
    unseen_prods = [prod for prod in Prod_ID if prod not in seen_prods]

    if not unseen_prods:
        st.write(f"No unseen products for Customer ID {customer_id}.")
        return []

    # Map unseen products and additional fields
    prod_internal_ids = [Prod_ID_Map[prod] for prod in unseen_prods]
    prod_category_ids = [Prod_Cat_ID_Map[data[data['Shortcut_Dimension_Code'] == prod]['Description'].values[0]] for
                         prod in unseen_prods]
    domain_ids = [Dom_ID_Map[data[data['Shortcut_Dimension_Code'] == prod]['DOMAIN'].values[0]] for prod in
                  unseen_prods]

    # Repeat customer internal ID for each product (to match batch size)
    cust_internal_ids = np.array([cust_internal_id] * len(prod_internal_ids))

    # Predict ratings for unseen products using all 4 inputs
    predictions = model.predict(
        [cust_internal_ids, np.array(prod_internal_ids), np.array(prod_category_ids), np.array(domain_ids)])

    # Sort predictions and get the top N recommendations
    recommended_prods = sorted(list(zip(unseen_prods, predictions.flatten())), key=lambda x: x[1], reverse=True)[
                        :num_recommendations]

    return recommended_prods


# Define Prod_ID from the DataFrame
Prod_ID = df['Shortcut_Dimension_Code'].unique().tolist()
# Streamlit app interface
st.title("Product Recommendation System")
st.write("Enter a Customer ID to receive personalized product recommendations.")

# Map customer and product IDs
Cust_ID_Map = {Customer_ID: idx for idx, Customer_ID in enumerate(df['Customer_ID'].unique())}
Prod_ID_Map = {prod: idx for idx, prod in enumerate(df['Shortcut_Dimension_Code'].unique())}

customer_id = st.text_input("Customer ID")

if customer_id:
    customer_id = int(customer_id)
    recommendations = recommend_prods(customer_id, model, df)

    if recommendations:
        st.write(f"Top Recommendations for Customer {customer_id}:")
        for prod, score in recommendations:
            st.write(f"Product: {prod}, Score: {score:.2f}")

