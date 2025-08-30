import streamlit as st
import pandas as pd
import numpy as np
from linear_regression_functions import get_features_targets, split_data, normalize_z, prepare_feature, calc_linreg, compute_cost_linreg, gradient_descent_linreg, predict_linreg, build_model_linreg, r2_score, mean_squared_error


independent_variables = ['Total SG Population', 'SG GDP Per Capita']
dependent_variable = ['Plastic Waste Disposed (Tonnes)']
citation_links = ['Singapore Department of Statistics. (2024). Waste Management And Overall Recycling Rates, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_daf568968ab40dc81e7b08887a83c8fa/view',
                 'Singapore Department of Statistics. (2024). Per Capita GDP In Chained (2015) Dollars, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_c43f61819c32009f2e86c29b0550e7fc/view',
                 'Singapore Department of Statistics. (2023). Indicators On Population, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_3d227e5d9fdec73f3bcadce671c333a6/view',
                 'Envcares. (2025). Plastics wastes in Singapore. https://envcares.com.sg/plastics-wastes-in-singapore/']


st.set_page_config(layout="wide")
st.title('♻ Singapore Plastic Waste Disposed Machine Learning')
st.info("Problem Statement: using **supervised learning** (Multiple Linear Regression), how might we **predict the volume of plastic waste disposed in Singapore** based on population size and GDP per capita to aid the Singapore Government's decision-making to promote sustainable growth and effective waste management?")


# DATA EXTRACTION
st.header('Data Extraction')
st.write('TODO: Describe how the data is obtained...')
df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
df


# DATA VISUALIZATION
st.header('Independent and Dependent Variables')
st.write('TODO: Describe the independent and dependent variables used...')
col1, col2 = st.columns(2)
with col1:
  # st.header('**Total SG Population**')
  st.write('**Total SG Population against Year**')
  st.bar_chart(data=df, x='Year', y=independent_variables[0], color=dependent_variable[0])
with col2:
  # st.header('**SG GDP Per Capita**')
  st.write('**SG GDP Per Capita against Year**')
  st.bar_chart(data=df, x='Year', y=independent_variables[1], color=dependent_variable[0])


# DATA PREPARATION
with st.sidebar:
  st.header('Input Features')
  sg_population = st.slider('Singapore Population', 3000000, 1000000000, 5000000)
  sg_gdp = st.slider('Singapore GDP Per Capita', 30000, 1000000000, 500000)
  input_data = {'Total SG Population': sg_population, 'SG GDP Per Capita': sg_gdp}
  input_df = pd.DataFrame(input_data, index=[0])
  input_display = pd.DataFrame({'Total SG Population': f'{sg_population:,}', 'SG GDP Per Capita': f'{sg_gdp:,}'}, index=[0])


# st.header('Selected Input Features')
# input_display


# PREPARE FEATURES & TARGET SETS
df_features, df_target = get_features_targets(df, independent_variables, dependent_variable)
df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, 100, 0.2)
array_features,_,_ = normalize_z(df_features.to_numpy())
X: np.ndarray = prepare_feature(array_features)


# BUILD & TEST LINEAR REGRESSION MODEL
target: np.ndarray = df_target.to_numpy()
iterations: int = 1500
alpha: float = 0.01
beta: np.ndarray = np.zeros((X.shape[1], 1))
beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)
pred: np.ndarray = predict_linreg(df_features, beta)
target: np.ndarray = df_target.to_numpy()
model, J_storage = build_model_linreg(df_features_train, df_target_train, beta, alpha, iterations)
pred: np.ndarray = predict_linreg(df_features_test, model['beta'], model['means'], model['stds'])


# EVALUATE LINEAR REGRESSION MODEL
target: np.ndarray = df_target_test.to_numpy()
mse: float = mean_squared_error(target, pred)
rmse = np.sqrt(mse)
abs_errors = np.abs(target - pred)
mean_abs_error = np.mean(abs_errors)
output_data = {'Mean Squared Error (MSE)': f'{mse:,.4f}', 'Root Mean Squared Error (RMSE)': f'{rmse:,.4f}', 'Mean Absolute Error (MAE)': f'{mean_abs_error:,.4f}'}
output_df = pd.DataFrame(output_data, index=[0])


# PREDICT USING INPUT FEATURES FROM USER
output_predict: np.ndarray = predict_linreg(input_df, model['beta'], model['means'], model['stds'])


st.header('Predict the Volume of Plastic Waste Disposed in Singapore')
st.write('You have selected the following population and GDP per capita in Singapore:')
input_display
st.write('TODO: Describe the MSE, RMSE, and MAE, as well as why we chose these to evaluate the model...')
st.table(output_df)
st.success(f'Prediction: **{round(output_predict[0][0]):,} tonnes**')


st.header('Dataset Reference List')
st.code(citation_links[0], language="None", wrap_lines=True)
st.code(citation_links[1], language="None", wrap_lines=True)
st.code(citation_links[2], language="None", wrap_lines=True)
st.code(citation_links[3], language="None", wrap_lines=True)
