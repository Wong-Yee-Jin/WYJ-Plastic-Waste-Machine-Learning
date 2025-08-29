import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title('♻ Singapore Plastic Waste Disposed Machine Learning')

st.info('This is a machine learning model.')

with st.expander('Raw Data'):
  st.write('**Description of raw data...**')
  df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
  df
