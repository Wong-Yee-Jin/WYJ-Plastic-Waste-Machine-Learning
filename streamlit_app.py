import streamlit as st
import pandas as pd

independent_variables = ['Total SG Population', 'SG GDP Per Capita']
dependent_variable = ['Plastic Waste Disposed (Tonnes)']

st.set_page_config(layout="wide")
st.title('♻ Singapore Plastic Waste Disposed Machine Learning')
st.info('This is a machine learning model.')

st.write('Describe how the data is obtained...')

with st.expander('Raw Data'):
  df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
  df

st.write('Describe the independent and dependent variables...')

with st.expander('Independent Variables'):
  X = df.loc(:, independent_variables)
  # X = df.drop(dependent_variable, axis=1)
  X

with st.expander('Dependent Variable'):
  y = df.loc(:, dependent_variable)
  # y = df.drop(independent_variables, axis=1)
  y
