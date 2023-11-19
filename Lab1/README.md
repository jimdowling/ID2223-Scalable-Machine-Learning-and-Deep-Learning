# Lab 1

In my first lab session, we implemented a serverless machine learning system, tackling two distinct machine learning problems.

## Iris Dataset Prediction

The first machine learning problem addressed is the classical Iris dataset prediction.
- We stored the features in Hopsworks FeatureStore.
- We trained the serverless K-nearest neighbours algorithm using Modal
- Finally we hosted a Gradio application on Huggingface Spaces to show the User Iterface.


## Wine Dataset Prediction

For the wine dataset:
- A Jupyter Notebook was dedicated to Exploratory Data Analysis (EDA).
- We dropped the non-relevant features and the duplicate rows, filled the missing values and encoded the 'type' feature.
- `wine-training-pipeline_regression.ipynb` involved creating a Feature View of the Group and training an XGBoost algorithm with default features, saving the model in Hopsworks.
- Given that the regression algorithm wasn't accurate enough, we did the same procedure using Random Forest Classification algorithm in `wine-training-pipeline_classification.ipynb`, which resulted in higher performance.
- In `wine-feature-pipeline-daily.py`, we developed a function generating synthetic wines based on EDA insights, incorporating historical knowledge.
- The prediction of the quality for synthetic wines was carried out in `wine-batch-inference-pipeline.py`.

### Huggingface Spaces
Additionally, there are two Huggingface Spaces:
- The first one is called 'wine'. It allows the user to input wine features and then predicts the wine quality based on the user input. Access it [here](https://07a2e891-65d9-49c8.gradio.live).
- The second one is called wine-monitor. It displays the latest predicted and actual quality, the last 4 predictions in a dataframe and a confusion matrix. Access it [here](https://0a17038e-2340-4566.gradio.live).
