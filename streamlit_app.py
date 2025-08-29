import streamlit as st
import pandas as pd

independent_variables = ['Total SG Population', 'SG GDP Per Capita']
dependent_variable = ['Plastic Waste Disposed (Tonnes)']


st.set_page_config(layout="wide")
st.title('♻ Singapore Plastic Waste Disposed Machine Learning')
st.info("Problem Statement: using a **supervised learning model** (Multiple Linear Regression), how might we **predict the annual volume of plastic waste disposed in Singapore** based on population size and GDP per capita to aid the Singapore Government's decision-making to promote sustainable growth and effective waste management?")


# DATA EXTRACTION
st.write('Describe how the data is obtained...')
with st.expander('Raw Data'):
  df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
  df


# DATA VISUALIZATION
st.write('Describe the independent and dependent variables...')
with st.expander('Independent Variables'):
  X = df[independent_variables]
  X
with st.expander('Dependent Variable'):
  y = df[dependent_variable]
  y
with st.expander('Data Visualization'):
  st.write('**Total SG Population**')
  st.bar_chart(data=df, x='Year', y=independent_variables[0], color=dependent_variable[0])
  st.write('**SD GDP Per Capita**')
  st.bar_chart(data=df, x='Year', y=independent_variables[1], color=dependent_variable[0])


