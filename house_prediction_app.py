import streamlit as st
import class_modules
from class_modules import ClusterSimilarity
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib

st.image('header.jpg')
st.title(':blue[California House Value Prediction App]')
st.write("""-- This app predicts a Prices of houses in California --
""")
st.write(':point_left: (click arrow sign for hide and unhide form) :green[Please Fillup the input field of left side for Prediction.]')
st.sidebar.header('Please Input Features Value')

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def user_input_features():
    longitude = st.sidebar.number_input('Longitude of the region: ')
    latitude = st.sidebar.number_input('Latitude of the region: ')
    housing_median_age = st.sidebar.number_input('Housing median age of the region: ')
    total_rooms = st.sidebar.number_input('Total rooms in that region: ')
    total_bedrooms = st.sidebar.number_input('Total bedrooms in that region: ')
    population = st.sidebar.number_input('Population of that region: ')
    households = st.sidebar.number_input('Number of Households in that region: ')
    median_income = st.sidebar.number_input('Median Income of that region: ')
    ocean_proximity = st.sidebar.selectbox('Ocean Proximity of the region: ',( 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))

    data = {'longitude':longitude, 'latitude':latitude, 'housing_median_age':housing_median_age, 'total_rooms':total_rooms, 'total_bedrooms':total_bedrooms, 'population':population,
            'households':households, 'median_income':median_income, 'ocean_proximity':ocean_proximity}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.write(input_df)

if input_df.isnull().any().any() or (input_df.iloc[0] == 0).any():
    st.warning('Please fill in all the input fields and ensure no values are zero before estimating the price.')
else:
    if st.button("Estimate Price"):
        try:
            # Load the model and make the prediction
            model = joblib.load('my_california_housing_model.pkl')
            prediction = model.predict(input_df)
            st.subheader(f'The estimated house value is: {prediction[0]}')
        except Exception as e:
            st.error(f'An error occurred: {e}')







