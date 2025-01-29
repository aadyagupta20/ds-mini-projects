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
