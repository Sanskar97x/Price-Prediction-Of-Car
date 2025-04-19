import streamlit as st
st.set_page_config(page_title="Car Price Predictor", layout="wide")  # <--- FIRST Streamlit command

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache data loading for performance
@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car_data.csv')

# Cache model loading
@st.cache_resource
def load_model():
    return pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Load model and data
try:
    car = load_data()
    model = load_model()
except FileNotFoundError as e:
    logger.error(f"Error loading files: {str(e)}")
    st.error("Error: Required files not found. Please ensure 'LinearRegressionModel.pkl' and 'Cleaned_Car_data.csv' are in the same directory.")
    st.stop()
except Exception as e:
    logger.error(f"Unexpected error loading files: {str(e)}")
    st.error("Error: An unexpected error occurred while loading files.")
    st.stop()

# Prepare dropdown data
companies = sorted(car['company'].unique())
car_models_by_company = {company: sorted(car[car['company'] == company]['name'].unique()) for company in companies}
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

# Custom CSS for minor styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .error {
        color: #dc3545;
    }
    .success {
        color: #28a745;
    }
    .sidebar .stSelectbox, .sidebar .stNumberInput {
        margin-bottom: 15px;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-title">Car Price Predictor</div>', unsafe_allow_html=True)
st.write("Predict the selling price of a used car and explore the dataset insights.")

# Sidebar for inputs
st.sidebar.header("Car Details")
company = st.sidebar.selectbox("Select Company", ["Select Company"] + companies, index=0, key="company")

# Dynamic car model selection
if company and company != "Select Company":
    car_models = car_models_by_company.get(company, [])
    car_model = st.sidebar.selectbox("Select Model", ["Select Model"] + car_models, index=0, key="car_model")
else:
    car_model = st.sidebar.selectbox("Select Model", ["Select a company first"], index=0, key="car_model", disabled=True)

year = st.sidebar.selectbox("Select Year of Purchase", years, index=0, key="year")
fuel_type = st.sidebar.selectbox("Select Fuel Type", fuel_types, index=0, key="fuel_type")
kms_driven = st.sidebar.number_input("Enter Kilometers Driven", min_value=0, step=1, value=0, key="kms_driven")
predict_button = st.sidebar.button("Predict Price")

# Main content
if predict_button:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    try:
        # Validate inputs
        if company == "Select Company" or not company:
            st.markdown('<div class="prediction error">Error: Please select a company</div>', unsafe_allow_html=True)
        elif car_model == "Select Model" or not car_model or car_model not in car_models_by_company.get(company, []):
            st.markdown('<div class="prediction error">Error: Please select a valid car model</div>', unsafe_allow_html=True)
        elif not year or year < 1900 or year > 2025:
            st.markdown('<div class="prediction error">Error: Please select a valid year</div>', unsafe_allow_html=True)
        elif not fuel_type or fuel_type not in fuel_types:
            st.markdown('<div class="prediction error">Error: Please select a valid fuel type</div>', unsafe_allow_html=True)
        elif kms_driven < 0:
            st.markdown('<div class="prediction error">Error: Kilometers driven cannot be negative</div>', unsafe_allow_html=True)
        else:
            # Prepare data for prediction
            input_data = pd.DataFrame(
                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)
            )
            
            # Predict
            prediction = model.predict(input_data)
            price = np.round(prediction[0], 2)
            if price < 0:
                price = 0  # Ensure non-negative price
            
            # Display prediction
            st.markdown(f'<div class="prediction success">Predicted Price: ₹{price:,.2f}</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error predicting price: {str(e)}")
        st.markdown(f'<div class="prediction error">Error: An unexpected error occurred during prediction</div>', unsafe_allow_html=True)

# Data Visualizations
st.markdown('<div class="section-title">Dataset Insights</div>', unsafe_allow_html=True)
st.write("Explore key trends in the car price dataset.")

# Price Distribution
st.markdown("### Price Distribution")
fig, ax = plt.subplots()
sns.histplot(car['Price'], bins=30, kde=True, ax=ax)
ax.set_xlabel("Price (₹)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Car Prices")
st.pyplot(fig)

# Price vs. Year
st.markdown("### Price vs. Year")
fig, ax = plt.subplots()
sns.scatterplot(data=car, x='year', y='Price', hue='fuel_type', ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Price (₹)")
ax.set_title("Car Price vs. Year by Fuel Type")
st.pyplot(fig)

# Price vs. Kilometers Driven
st.markdown("### Price vs. Kilometers Driven")
fig, ax = plt.subplots()
sns.scatterplot(data=car, x='kms_driven', y='Price', hue='fuel_type', ax=ax)
ax.set_xlabel("Kilometers Driven")
ax.set_ylabel("Price (₹)")
ax.set_title("Car Price vs. Kilometers Driven by Fuel Type")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center;">Built with Streamlit | Data Source: Cleaned Car Data</div>', unsafe_allow_html=True)