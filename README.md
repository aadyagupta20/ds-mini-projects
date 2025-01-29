# ds-mini-projects
A collection of simple data science projects covering regression, classification, clustering, and data visualization. Each project includes code, explanations, and insights.

# _**File 1:**_
## **Linear Regression on Boston Housing Dataset**
This repository contains the implementation of Linear Regression to predict housing prices (MEDV) from the Boston Housing Dataset. The project demonstrates how model performance improves as more features are added to the dataset, and it evaluates the model using various metrics such as R², MSE, MAE, RMSE, and Adjusted R².

## **Project Overview**
This project applies Linear Regression to predict housing prices in Boston. We explore the impact of using different subsets of features from the dataset on model performance. The analysis is done step by step:
- Starting with a model using just two features.
- Adding more variables in subsequent models.
- Evaluating each model's performance using various metrics.

## **Dataset Description**
The Boston Housing Dataset consists of various attributes related to housing in Boston. The target variable (MEDV) is the median value of owner-occupied homes in $1000s. The features (independent variables) include:

CRIM: Crime rate per capita
ZN: Proportion of residential land zoned for large properties
INDUS: Proportion of industrial land
CHAS: Whether the property is near the Charles River
NOX: Nitrogen oxides concentration
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built before 1940
DIS: Weighted distance to employment centers
RAD: Index of accessibility to radial highways
TAX: Property tax rate
PTRATIO: Pupil-teacher ratio
B: Proportion of residents of African American descent
LSTAT: Percentage of lower status population

## **Installation**
**Prerequisites**
To run the project, you'll need **Jupyter Notebook or Google Colab** for executing the code.

**How to Run the Notebook:**
Clone the repository or directly open the .ipynb file in Jupyter Notebook or Google Colab.

**Download the Boston Housing Dataset** 
Download the dataset: https://docs.google.com/spreadsheets/d/1ev4X69r61KyxWeM7law4BoTqCrhNkCINIxBGeDddgEQ/edit?gid=2058781777#gid=2058781777
and upload the dataset directly to Google Drive.

If you're using Google Colab, you need to mount your Google Drive and load the dataset. Use the following code to mount your Google Drive and set the path for the dataset:

'''from google.colab import drive

drive.mount('/content/drive')

#Change directory to where your dataset is located in Google Drive

import pandas as pd

df = pd.read_csv('/content/drive/My Drive/path_to_your_folder/BostonHousing.csv')  # Adjust path accordingly'''

Once the dataset is loaded, you can run the cells in the notebook to perform the Linear Regression analysis.

## **Models & Analysis**

**A) Model 1: Using CRIM and ZN Only**
- The initial model uses just CRIM (crime rate) and ZN (proportion of residential land) to predict housing prices.
- This model gives a relatively low R² value of 0.422, showing that the model only captures about 42% of the variance in housing prices.

**B) Model 2: Adding More Features (INDUS, CHAS, etc.)**
- Additional features such as INDUS, CHAS, and others are added to the model.
- This improves the model's R² to 0.686, reflecting a better fit to the data.

**C) Model 3: Using All 13 Features**
- The final model includes all 13 variables from the dataset.
- The R² increases significantly to 0.771, indicating a much better fit to the data. The Adjusted R² further confirms that the model is highly predictive.

## **Metrics & Results**
**Model Performance**

**Model 1: Using CRIM and ZN**
R²: 0.422
MSE: 40.72
MAE: 5.76
RMSE: 6.38

**Model 2: Adding More Variables**
R²: 0.686
MSE: 23.84
MAE: 4.24
RMSE: 4.88

**Model 3: Using All Features**
R²: 0.771
MSE: 18.30
MAE: 3.12
RMSE: 1.79
Adjusted R²: 0.726

The incremental improvements in model performance as more features are included demonstrate the power of using comprehensive data for predictive modeling.

## **Key Insights**
1. **Effect of Feature Selection:** As more variables were added, the model performance improved significantly:
   - Model 1 had an R² of 42%, which increased to 77.1% in Model 3.
   - The error metrics (MSE, MAE, RMSE) decreased as we incorporated more features.

2. **Impact of Individual Features**: Features such as CRIM (crime rate), RM (number of rooms), and NOX (nitrogen oxides) showed significant coefficients, indicating their strong influence on housing prices.

3. **Adjusted R²**: The increase in Adjusted R² from 0.726 suggests that the full model is well-optimized, balancing the number of predictors with the variance explained.

4. **Model Robustness**: The model with all features can predict housing prices more accurately, demonstrating that more data can lead to a better model, assuming the features are relevant.

**Contributions**
Feel free to fork this repository, contribute to the analysis, or suggest improvements. Contributions to the documentation or extending the models with new algorithms are welcome.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# _**File 2:**_
## **Life Expectancy Prediction and Analysis**
This project performs a detailed analysis and prediction of life expectancy using various factors such as schooling, income, and adult mortality. The analysis leverages multiple statistical techniques, including correlation analysis, linear regression, and bias-variance decomposition, to explore the relationships between these factors and life expectancy. It also implements model evaluation techniques to assess the performance of the predictive model.

## **Overview**
This repository contains a Python implementation for predicting life expectancy using various factors from a dataset containing life expectancy information across countries. The steps followed include:
- Data cleaning and preprocessing
- Correlation analysis
- Regression model fitting
- Model evaluation (including MSE, RMSE, MAE, and R-squared)
- Bias-variance decomposition to evaluate model performance

## **Dataset**
The dataset used in this analysis is the life expectancy dataset, which contains various features like:

Country: Country name
Year: Year of the data point
Status: Development status of the country
Life expectancy: Life expectancy at birth
Adult Mortality: Mortality rate for adults
Infant deaths: Number of infant deaths
Alcohol: Alcohol consumption per capita
GDP: Gross Domestic Product
Schooling: Average years of schooling

The dataset contains 2938 rows and 22 columns. Missing values are dropped, leaving 1649 rows for analysis.

## **Files Included**
- Life Expectancy Analysis and Prediction (Jupyter Notebook): This file contains all the code for the data analysis and model evaluation.
- Life Expectancy Dataset (CSV): The dataset used for analysis.

## **Installation**
**Prerequisites**
To run the project, you'll need **Jupyter Notebook or Google Colab** for executing the code.

**How to Run the Notebook:**
Clone the repository or directly open the .ipynb file in Jupyter Notebook or Google Colab.

**Download the Boston Housing Dataset** 
Download the dataset: https://docs.google.com/spreadsheets/d/1hE5AKTByaaOeHvZx6Gu2cRZA-4kAVPO3ZlaatwvzyLs/edit?gid=328773461#gid=328773461
and upload the dataset directly to Google Drive.

If you're using Google Colab, you need to mount your Google Drive and load the dataset. Use the following code to mount your Google Drive and set the path for the dataset:

'''from google.colab import drive

drive.mount('/content/drive')

#Change directory to where your dataset is located in Google Drive

import pandas as pd

df = pd.read_csv('/content/drive/My Drive/path_to_your_folder/BostonHousing.csv')  # Adjust path accordingly'''

Once the dataset is loaded, you can run the cells in the notebook to perform the Linear Regression analysis.

## **Models & Analysis**

**A) Model 1: Using Schooling, Income, and Adult Mortality**
- Independent Variables: Schooling, Income, Adult Mortality
- Target Variable: Life Expectancy
- Model Type: Multiple Linear Regression
- R²: 0.7299
- **Interpretation**: This model uses three predictors—Schooling, Income, and Adult Mortality—to predict Life Expectancy. The R² value of 0.73 suggests that around 73% of the variance in life expectancy can be explained by these three variables. The model's performance indicates a good fit.
  
**B) Model 2: Using Only Schooling**
- Independent Variable: Schooling
- Target Variable: Life Expectancy
- Model Type: Simple Linear Regression
- R²: 0.426
- **Interpretation**: When only Schooling is used as the predictor, the model explains just 42.6% of the variance in life expectancy, reflecting a weaker relationship. This shows that the other factors (Income and Adult Mortality) might play a significant role in predicting life expectancy.

## **Metrics & Results**

**Model 1: Using Schooling, Income, and Adult Mortality**
R² (on test data): 0.7299
MSE: 19.18
MAE: 3.11
RMSE: 4.38
Adjusted R²: 0.7275
**Interpretation**: The model explains 73% of the variance in life expectancy. With a lower MSE (19.18) and RMSE (4.38), it performs better than Model 2, indicating a strong 

predictive relationship between the features and life expectancy.
**Model 2: Using Only Schooling**
R² (on test data): 0.426
MSE: 36.71
MAE: 4.38
RMSE: 6.06
Adjusted R²: 0.418
**Interpretation**: Using only Schooling as a predictor leads to a much weaker fit. The MSE and RMSE are considerably higher, which indicates a higher error in predictions. The R² of 0.426 shows that Schooling alone doesn’t explain much of the variance in life expectancy.

## **Model Performance**

**Model 1 (Multiple Independent Variables)**
R²: 0.7299
MSE: 19.18
MAE: 3.11
RMSE: 4.38
Adjusted R²: 0.7275
**Interpretation**: The multiple linear regression model is well-tuned with an R² of 0.73, indicating that it successfully captures most of the variance in life expectancy. The error metrics (MSE, MAE, RMSE) are reasonably low, reflecting a good predictive performance.

**Model 2 (Single Independent Variable)**
R²: 0.426
MSE: 36.71
MAE: 4.38
RMSE: 6.06
Adjusted R²: 0.418
**Interpretation**: With only Schooling as the predictor, the model’s performance significantly declines, as seen with an R² of 0.426. The higher MSE and RMSE indicate greater prediction errors.

## **Key Insights**
1. Effect of Multiple Variables: Adding more relevant features (Income and Adult Mortality) significantly improves the model's performance. This is evident from the substantial increase in R² from 0.426 (Model 2) to 0.7299 (Model 1).
- Model 1: R² = 0.7299, Adjusted R² = 0.7275
- Model 2: R² = 0.426, Adjusted R² = 0.418

2. Bias-Variance Tradeoff: In the bias-variance decomposition for Model 2, the majority of the error comes from bias, indicating that a simple model (with just Schooling) is too simplistic and underfits the data. In contrast, Model 1, which includes additional variables, can better capture the complexity of the relationship.

3. Model Flexibility: The multiple regression model is more flexible, as it incorporates more predictors and thus has a better chance of fitting the data more accurately. This is reflected in the lower MSE, MAE, and RMSE compared to Model 2.

4. Room for Improvement: Even though Model 1 is a significant improvement, the model could potentially benefit from adding even more features or exploring more advanced techniques to better predict life expectancy. For example, including additional socio-economic or health-related factors might improve the model further.

## **Conclusion**
The comparison between the two models shows that including more features significantly improves model performance.

- Model 1 (with Schooling, Income, and Adult Mortality) achieves a higher R² (0.7299) compared to Model 2 (with only Schooling, R² = 0.426), demonstrating that more features help explain more variance in life expectancy.
- The bias-variance decomposition shows that Model 1 has a better balance of bias and variance, while Model 2 suffers from high bias, indicating underfitting.

In summary, using a comprehensive set of relevant features leads to more accurate and reliable predictions.

## **Contributions**
Feel free to fork this repository, contribute to the analysis, or suggest improvements. Contributions to the documentation or extending the models with new algorithms are welcome.
