# Purchase Prediction Project

This project aims to predict whether a customer will make a purchase based on various features like Age, Gender, Location, Product Category, Price, and more. The dataset was generated and processed through various steps, including data visualization, preprocessing, and the training of machine learning models like Random Forest and XGBoost.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Generation](#dataset-generation)
- [Data Visualization](#data-visualization)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Authors and Contact](#authors-and-contact)

## Introduction

In this project, I developed a model to predict customer purchase behavior. The entire process is documented step-by-step, from generating synthetic data to evaluating the model's performance.

## Dataset Generation

In this project, I created synthetic data to simulate a real-world scenario. I utilized the Faker library in Python to generate a variety of customer and purchasing data. The dataset was generated within a Jupyter Notebook named `Generate_the_datasets.ipynb`, located in the `FakeData` folder. This notebook includes the process of data creation, ensuring that the generated data is diverse and representative of potential real-world cases.

## Data Visualization

In this section, I will describe the data visualization process used to gain insights from the dataset.

### Introduction to Visualization

Data visualization is crucial for understanding and analyzing the dataset. By visualizing data, I can identify patterns, trends, and anomalies that might not be obvious from raw data alone.

### Libraries Used

For visualization, I utilized libraries such as Matplotlib and Seaborn. These libraries provide powerful tools for creating a wide range of charts and plots.

### Types of Visualizations

I created several types of visualizations to explore different aspects of the data:
- **Histograms**: To show the distribution of numerical features such as price and rating.
- **Scatter Plots**: To examine relationships between features, for example, the relationship between price and average rating.
- **Box Plots**: To visualize the distribution and detect outliers in features like price and rating.

Each visualization was chosen to highlight specific features of the data and provide insights into its structure and distribution.

### Specific Examples

In the `data_visualization` Jupyter Notebook located in the `Model` folder, I have included all the visualizations created. For instance:
- **Price Distribution**: Shows how prices are spread across different products.
- **Rating Distribution**: Illustrates the spread of average ratings given by users.
- **Relationship Between Price and Rating**: Displays how price correlates with average rating.

### Insights

The visualizations revealed several key insights, such as the distribution of product prices and the relationship between price and rating. These insights helped in understanding the dataset better and informed further data preprocessing and modeling steps.

## Preprocessing

In this step, I handled various data preprocessing tasks to prepare the dataset for model training.

### Loading Data

I loaded the data files `User`, `ProductData`, `UserBehaviourData`, and `UserRating` into a Jupyter Notebook from the `Model` folder.

### Handling Missing Values

- For numerical values, I filled missing values with the mean of the respective columns.
- For categorical values, I used the mode (most frequent value) to fill in missing values.

### Feature Engineering

I created the `purchase_made` column to indicate whether a purchase was made (1) or not (0). This column will be used as the target variable for prediction.

### Saving Processed Data

The processed data was saved into a `final_data.csv` file within the `Data/processed/` sub-folder. This file contains all the modifications and updates made during preprocessing.

### Additional Preprocessing Details

I also handled One-Hot Encoding and feature scaling, and ensured that all data was appropriately transformed for model training. Missing values were addressed by filling numerical values with the mean and categorical values with the mode.

## Model Training

In this section, I focused on training machine learning models to predict whether a purchase was made or not.

### Data Preparation

- I applied One-Hot Encoding to the categorical features, such as Gender, Location, Category, and Brand, to convert them into a format suitable for machine learning algorithms.
- The dataset was split into training and testing sets using the `train_test_split` function from scikit-learn.

### Random Forest Classification

- I trained a Random Forest Classifier using the preprocessed training data.
- Hyperparameters were tuned to optimize the model's performance. Some key parameters adjusted include the number of estimators and the maximum depth of the trees.

### XGBoost Model

- I also trained an XGBoost Classifier with the training data.
- For XGBoost, I tuned hyperparameters such as `learning_rate`, `n_estimators`, and `max_depth`. Regularization parameters like `lambda` and `alpha` were also considered to prevent overfitting.

## Evaluation

In this section, I evaluated the performance of the machine learning models and their effectiveness in predicting the target variable.

### Model Performance

- I assessed the performance of the Random Forest and XGBoost models using various metrics, including accuracy, precision, recall, and F1-score.
- The models were evaluated on both the training and test datasets to understand their generalization capability.

### Cross-Validation

- I used k-fold cross-validation for both models to ensure that the evaluations were robust and not dependent on a particular train-test split.
- This technique helps in providing a more reliable estimate of the model's performance.

### Hyperparameter Tuning

- Grid search was employed to find the best hyperparameters for the Random Forest and XGBoost models.
- This process involved testing various combinations of hyperparameters to optimize model performance.

### Results Analysis

- For XGBoost, the accuracy achieved was 77.14% with a standard deviation of 0.73%.
- For Random Forest, the accuracy achieved was 78.12% with a standard deviation of 0.01%.
- The impact of features, including the `purchase_made` column, was analyzed to see how they influenced model accuracy.

### Visualization

- Performance metrics were visualized using plots and graphs to better understand the results and differences between models.
- These visualizations included confusion matrices, ROC curves, and feature importance charts.

## Conclusion

In this project, I embarked on a comprehensive journey to build a machine learning model for predicting customer purchasing behavior. Here’s a summary of what I accomplished and learned:

### Project Overview

- I generated synthetic data using the Faker library to create a realistic dataset for modeling.
- The dataset was preprocessed, including tasks like One-Hot Encoding, missing value imputation, and feature scaling, to prepare it for machine learning algorithms.

### Model Training and Evaluation

- I trained and evaluated two machine learning models: Random Forest and XGBoost.
- Hyperparameter tuning was performed using Grid Search and k-fold cross-validation to optimize model performance.
- The final models showed accuracies of 78.12% for Random Forest and 77.14% for XGBoost.

### Insights and Improvements

- The Random Forest model slightly outperformed the XGBoost model in terms of accuracy.
- Feature engineering, such as creating the Poor Case Mates column, proved valuable in enhancing model performance.
- Further improvements could involve experimenting with additional features, advanced hyperparameter tuning, or trying other machine learning techniques.

### Future Work

- Future steps may include exploring more sophisticated models or techniques such as ensemble methods or deep learning.
- Continuous refinement of the data preprocessing steps and model parameters could lead to better accuracy and robustness.

This project provided valuable experience in data generation, preprocessing, model training, and evaluation. The lessons learned here will be instrumental in tackling future machine learning challenges.

## Authors and Contact

This project was developed by Bahna Darius.

For more information or to connect, please visit my profiles:

- [LinkedIn](https://www.linkedin.com/in/darius-bahn%C4%83-2224b7264/)
- [GitHub](https://github.com/Bahna-Darius)
