# Lab 1

In our first lab session, we implemented a serverless machine-learning system, tackling two distinct machine-learning problems. Every notebook presents comments and data visualization for clarity and depth.

# Iris Dataset Prediction 🌸

The first machine learning problem addressed is the classical Iris dataset prediction.
- We stored the features in Hopsworks FeatureStore.
- We trained the serverless K-nearest neighbors algorithm using Modal
- Finally, we hosted a Gradio application on Huggingface Spaces to show the User Interface.

# Wine Quality Prediction System

## Overview 🍇🍷
This Wine Quality Prediction System leverages a comprehensive data analysis and machine learning approach to predict the quality of wine. The system uses the 'winequalityN.csv' dataset and applies various Python libraries and machine learning techniques to predict the 'quality' target variable. The README provides a detailed walkthrough of each step in the pipeline, from data preparation to model deployment and monitoring.
## 1. Exploratory Data Analysis (EDA) 🔍
In `wine-eda-and-backfill-feature-group.py`, our goal was to understand dataset characteristics.
Techniques Used: Seaborn, Pandas, and NumPy.
Key Steps:
- Data Loading: Connection established with Hopsworks.
- Column Renaming: Syntax conflict avoidance.
- Data Inspection: Identifying data types and missing values.
- Data Visualization: Distribution of features and quality scores.
## 2. Data Preprocessing and Feature Engineering ⚙️
After the first explorative phase, we prepared the dataset for modeling.
Processes:
- One-Hot Encoding: For categorical variables.
- Handling Missing Values: Random replacement with sampling from the same column.
- Feature Selection: Dropping less predictive features.
- Handling Duplicates: Removing duplicate rows.
## 3. Feature Storage 📦
- After preprocessing we created a Feature View of the Group and stored it in Hopsworks
- The feature group was later accessed as a feature view to perform train-test splitting and model predictions
## 4. Model Training and Evaluation 🤖📊
In `wine-training-pipeline-regression.py` and `wine-training-pipeline-classification.py`, we created and improved different ML models, which were later exported on Hopsworks.
Models Used: KNN, Random Forest, XGBoost, for different approaches to the problem. Classification was later preferred due to the unbalanced nature of the dataset.
Techniques:
- Grid Search: For hyperparameter tuning.
- Model Evaluation: Using RMSE (regression), accuracy, precision, recall, and F1 score (classification).
- Confusion Matrix: To visualize model performance.
- Class weights + ADASYN for dataset oversampling, to generate more samples for the minority classes.

Algorithm	Accuracy
KNN	41.2%
Random Forest	60.0%
Oversampled RF	84.4%

## 5. Daily Sample Generation 🧰
- In `wine-feature-pipeline-daily.py`, we developed a function generating synthetic wines based on EDA insights, incorporating historical knowledge.
- The function takes into consideration the distribution of each variable according to the sample quality, to create relevant samples.

## 6. Deployment & Inference 🔮
- The prediction of the quality for synthetic wines was carried out in `wine-batch-inference-pipeline.py`.


### Huggingface Spaces 🤗
Additionally, there are two Huggingface Spaces for each task:  
Iris classification task:
- The first one is called 'iris'. It allows the user to input iris features and then predicts the type based on the user input. Access it [here](https://laura000-iris.hf.space).
- The second one is called iris-monitor. It displays the latest predicted and actual flower, the last 4 predictions in a dataframe, and a confusion matrix. Access it [here](https://laura000-iris-monitor.hf.space).

Wine classification task:
- The first one is called 'wine'. It allows the user to input wine features and then predicts the wine quality based on the user input. Access it [here](https://laura000-wine.hf.space/).
- The second one is called wine-monitor. It displays the latest predicted and actual quality, the last 4 predictions in a dataframe, and a confusion matrix. Access it [here](https://laura000-wine-monitor.hf.space).
