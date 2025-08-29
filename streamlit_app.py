import streamlit as st
import pandas as pd
import numpy as np
from linear_regression_functions import get_features_targets, split_data, normalize_z, prepare_feature, calc_linreg, compute_cost_linreg, gradient_descent_linreg, predict_linreg, build_model_linreg, r2_score, mean_squared_error


independent_variables = ['Total SG Population', 'SG GDP Per Capita']
dependent_variable = ['Plastic Waste Disposed (Tonnes)']


st.set_page_config(layout="wide")
st.title('♻ Singapore Plastic Waste Disposed Machine Learning')
st.info("Problem Statement: using a **supervised learning model** (Multiple Linear Regression), how might we **predict the volume of plastic waste disposed in Singapore** based on population size and GDP per capita to aid the Singapore Government's decision-making to promote sustainable growth and effective waste management?")


# DATA EXTRACTION
st.write('Describe how the data is obtained...')
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


# DATA PREPARATION
with st.sidebar:
  st.header('Input Features')
  sg_population = st.slider('Singapore Population', 3000000, 1000000000, 5000000)
  sg_gdp = st.slider('Singapore GDP Per Capita', 30000, 1000000000, 500000)
  input_data = {'Total SG Population': sg_population, 'SG GDP Per Capita': sg_gdp}
  input_df = pd.DataFrame(input_data, index=[0])


st.write('You have selected the following input features')
input_df


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
# r2: float = r2_score(target, pred)
mse: float = mean_squared_error(target, pred)
rmse = np.sqrt(mse)
abs_errors = np.abs(target - pred)
mean_abs_error = np.mean(abs_errors)
output_df = {'Mean Squared Error': f'{mse:.4f}', 'Root Mean Squared Error': f'{rmse:.4f}', 'Mean Absolute Error': f'{mean_abs_error:.2f}'}
# print(f"R² Score: {r2:.4f}")
# print(f"MSE: {mse:.4f} tonnes²")
# print(f"RMSE: {rmse:.4f} tonnes")
# print(f"MAE: {mean_abs_error:.2f} tonnes")


# PREDICT USING INPUT FEATURES FROM USER
user_array_features,_,_ = normalize_z(input_df.to_numpy())
user_X: np.ndarray = prepare_feature(user_array_features)
output_predict: np.ndarray = predict_linreg(user_X, model['beta'], model['means'], model['stds'])


st.header('Predicted Volume of Plastic Waste Disposed in Singapore')
output_df
st.success(f'Predicted Plastic Waste Disposed in Singapore: {output_predict}')
