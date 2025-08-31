# ♻ Singapore Plastic Waste Disposed Machine Learning

This is a machine learning app using supervised learning (Multiple Linear Regression model).

Problem Statement: How might we predict the annual volume of plastic waste disposed in Singapore based on population size and GDP per capita to aid the Singapore Government's decision-making to promote sustainable growth and effective waste management?

# 🖐 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://WYJ-Plastic-Waste-Machine-Learning.streamlit.app/)

# 👩‍💻 GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

-------------------------------------------

## 🔎 Data Extraction

Three datasets have been considered for this machine learning project.

**Waste Management and Overall Recycling Rates Annual**
- This dataset details the annual volume of waste generated, disposed and recycled in tonnes, with the waste generated being the sum of waste disposed and waste recycled.
- From 1996 to 1999, the data was obtained from Envcares which includes a screenshot of the plastic waste statistics from the National Environment Agency (NEA) in Singapore. From 2000 to 2024, the data was obtained from the Singapore Department of Statistics, specifically the material "Plastic" under the category "Total Disposed".

**Per Capita GDP In Chained 2015 Dollars Annual**
- This dataset details the annual GDP per capita and manufacturing GDP per capita in chained dollars as well as the year on year growth rates, from 1960 to 2024.
- Aligning with the years found in the first dataset, the data in the column "Per Capita GDP in Chained (2015) Dollars" from 1996 to 2024 was selected.

**Indicators On Population Annual**
- This dataset details the various characteristics of the country's population such as total population size and its breakdown by type of citizenship as well as population growth rate and density and gender ratio, from 1950 to 2024.
- Aligning with the years found in the first dataset, the data in the column "Total Population" from 1996 to 2024 was selected.

## ✌ Independent and Dependent Variables
Inferring from the problem statement, the dependent variable is "Plastic Waste Disposed in Singapore". As for the independent variables,

**Total SG Population**
- It shows how the total population size changes from 1950 to 2024. When the population size increases, the total amount of consumer goods consumed in Singapore is expected to increases. Given that, the volume of plastic consumption and consequently plastic waste disposed should increase as well.

**SG GDP Per Capita**
- It shows Singapore's average economic output per person, adjusted for inflation to 2015 prices in SGD, as an indicator of yearly economic performance from 1960 to 2024. As GDP per capita rises, economic growth is expected, boosting consumer wealth and spending on goods. Hence, the volume of plastic consumption and consequently plastic waste disposed is expected to increase.

## ❓ Predict the Volume of Plastic Waste Disposed in Singapore
**Root Mean Squared Error (RMSE)**
- This represents the average magnitude of the error between the predicted and actual values, with greater weight given to larger errors. In the context of national-level policy, such a margin of error is considered acceptable due to the scale of the data involved.

**Mean Absolute Error (MAE)**
- This reflects the average absolute difference between predicted and actual values. An error of around 70,000 tonnes indicates that the model produces reasonably close estimates across the dataset.

## 🎯 Discussion and Analysis of Results
Our objective was to predict Singapore’s future plastic waste generation, addressing critical issues such as land scarcity and rising consumption. Using machine learning allowed us to:
- Quantify the relationship between macroeconomic indicators (GDP and population) and waste generation trends.
- Forecast future plastic waste levels by training a regression model.
- Support policy with evidence-based projections, aiding long-term planning and infrastructure decisions related to waste management.
By evaluating the model using RMSE and MAE, its predictions demonstrate that they are sufficiently accurate to support key decisions on infrastructure, recycling capacity, and sustainability policy. While there is room for refinement such as incorporating more variables or higher-resolution data, the current model already provides a meaningful, data-driven foundation for sustainable waste management planning.

## 💻 Dataset Reference List
Singapore Department of Statistics. (2024). Waste Management And Overall Recycling Rates, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_daf568968ab40dc81e7b08887a83c8fa/view

Singapore Department of Statistics. (2024). Per Capita GDP In Chained (2015) Dollars, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_c43f61819c32009f2e86c29b0550e7fc/view

Singapore Department of Statistics. (2023). Indicators On Population, Annual (2025) [Dataset]. data.gov.sg. Retrieved from https://data.gov.sg/datasets/d_3d227e5d9fdec73f3bcadce671c333a6/view

Envcares. (2025). Plastics wastes in Singapore. https://envcares.com.sg/plastics-wastes-in-singapore/
