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
st.write('Three datasets have been considered for this machine learning project.')
with st.container(border=True):
  st.write('**1. Waste Management and Overall Recycling Rates Annual**')
  st.write('This dataset details the annual volume of waste generated, disposed and recycled in tonnes, with the waste generated being the sum of waste disposed and waste recycled.')
  st.write('From 1996 to 1999, the data was obtained from Envcares which includes a screenshot of the plastic waste statistics from the National Environment Agency (NEA) in Singapore. From 2000 to 2024, the data was obtained from the Singapore Department of Statistics, specifically the material "Plastic" under the category "Total Disposed".')
with st.container(border=True):
  st.write('**2. Per Capita GDP In Chained 2015 Dollars Annual**')
  st.write('This dataset details the annual GDP per capita and manufacturing GDP per capita in chained dollars as well as the year on year growth rates, from 1960 to 2024.')
  st.write('Aligning with the years found in the first dataset, the data in the column "Per Capita GDP in Chained (2015) Dollars" from 1996 to 2024 was selected.')
with st.container(border=True):
  st.write('**3. Indicators On Population Annual**')
  st.write("This dataset details the various characteristics of the country's population such as total population size and its breakdown by type of citizenship as well as population growth rate and density and gender ratio, from 1950 to 2024.")
  st.write('Aligning with the years found in the first dataset, the data in the column "Total Population" from 1996 to 2024 was selected.')
df = pd.read_csv('SG_Plastic_Waste_GDP_Population_Dataset.csv')
df


# DATA VISUALIZATION
st.header('Independent and Dependent Variables')
st.write('Inferring from the problem statement, the dependent variable is "Plastic Waste Disposed in Singapore". As for the independent variables,')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
  st.write('**Total SG Population**')
  st.write('It shows how the total population size changes from 1950 to 2024. We expect that when the population size increases, the total amount of consumer goods consumed in Singapore increases. Given that, the volume of plastic consumption and consequently plastic waste disposed should increase as well.')
with col2:
  st.write('**SG GDP Per Capita**')
  st.write("It shows Singapore's average economic output per person, adjusted for inflation to 2015 prices in SGD, as an indicator of yearly economic performance from 1960 to 2024. As GDP per capita rises, economic growth is expected, boosting consumer wealth and spending on goods. Hence, we expect the volume of plastic consumption and consequently plastic waste disposed to increase.")
with col3:
  st.bar_chart(data=df, x='Year', y=independent_variables[0], color=dependent_variable[0])
with col4:
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
df[dependent_variable[0]] = np.log(df[dependent_variable[0]]) # logarithm y variable
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


# BUILD + TEST + EVALUATE LINEAR REGRESSION MODEL (LOGARITHM)
# pred_log: np.ndarray = predict_linreg(df_features_test, model['beta'], model['means'], model['stds'])
# pred: np.ndarray = np.exp(pred_log)
# target_log: np.ndarray = df_target_test.to_numpy()
# target: np.ndarray = np.exp(target_log)
# mse: float = mean_squared_error(target, pred)
# rmse = np.sqrt(mse)
# abs_errors = np.abs(target - pred)
# mean_abs_error = np.mean(abs_errors)
# output_data = {'Mean Squared Error (MSE)': f'{mse:,.4f}', 'Root Mean Squared Error (RMSE)': f'{rmse:,.4f}', 'Mean Absolute Error (MAE)': f'{mean_abs_error:,.4f}'}
# output_df = pd.DataFrame(output_data, index=[0])


# PREDICT USING INPUT FEATURES FROM USER
output_predict: np.ndarray = predict_linreg(input_df, model['beta'], model['means'], model['stds'])


st.header('Predict the Volume of Plastic Waste Disposed in Singapore')
st.write('You have selected the following population and GDP per capita in Singapore:')
input_display
st.write('Root Mean Squared Error (RMSE) represents the average magnitude of the error between the predicted and actual values, with greater weight given to larger errors. In the context of national-level policy, such a margin of error is considered acceptable due to the scale of the data involved.')
st.write('Mean Absolute Error (MAE) reflects the average absolute difference between predicted and actual values. An error of around 40,000 tonnes indicates that the model produces reasonably close estimates across the dataset.')
st.table(output_df)
st.success(f'Prediction: **{round(output_predict[0][0]):,} tonnes**')


st.header('Discussion and Analysis of Results')
st.write('Our objective was to predict Singapore’s future plastic waste generation, addressing critical issues such as land scarcity and rising consumption. Using machine learning allowed us to:')
# st.html('<ul>  \n  \n<li>Quantify the relationship between macroeconomic indicators (GDP and population) and waste generation trends.</li><li>Forecast future plastic waste levels, even with incomplete historical data, by training a regression model.</li><li>Support policy with evidence-based projections, aiding long-term planning and infrastructure decisions related to waste management</li>')
# st.write('  • Quantify the relationship between macroeconomic indicators (GDP and population) and waste generation trends.  \n  • Forecast future plastic waste levels, even with incomplete historical data, by training a regression model.')
st.markdown("""
- Quantify the relationship between macroeconomic indicators (GDP and population) and waste generation trends.
- Forecast future plastic waste levels by training a regression model.
- Support policy with evidence-based projections, aiding long-term planning and infrastructure decisions related to waste management.
""")
st.write('By evaluating the model using RMSE and MAE, we demonstrate that its predictions are sufficiently accurate to support key decisions on infrastructure, recycling capacity, and sustainability policy. While there is room for refinement, such as incorporating more variables or higher-resolution data—the current model already provides a meaningful, data-driven foundation for sustainable waste management planning.')


st.header('Dataset Reference List')
st.code(citation_links[0], language="None", wrap_lines=True)
st.code(citation_links[1], language="None", wrap_lines=True)
st.code(citation_links[2], language="None", wrap_lines=True)
st.code(citation_links[3], language="None", wrap_lines=True)
