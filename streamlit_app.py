import streamlit as st
import pandas as pd

st.title('♻ Singapore Plastic Waste Disposed Machine Learning')

st.info('This is a machine learning model.')

df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
df
