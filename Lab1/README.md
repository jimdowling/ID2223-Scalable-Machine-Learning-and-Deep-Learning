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
- We dropped the non-relevant features, filled the missing values and encoded the 'type' feature.
- `wine-training-pipeline.ipynb` involved creating a Feature View of the Group and training an XGBoost algorithm with default features, saving the model in Hopsworks.
- To understand if classification algorithms performed better we also implemented some of them in `wine-training-pipeline-classification-ipynb`.
- In `wine-feature-pipeline-daily.py`, we developed a function generating synthetic wines based on EDA insights, incorporating historical knowledge indicating survival likelihood.
- The prediction of the quality for synthetic wines was carried out in `wine-batch-inference-pipeline.py`.

### Huggingface Spaces
Additionally, there are two Huggingface Spaces:
...

These spaces host prediction models and applications for both Iris and Wine datasets.
