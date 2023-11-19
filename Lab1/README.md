# Lab 1

In our first lab session, we implemented a serverless machine-learning system, tackling two distinct machine-learning problems. Every notebook presents comments and data visualization for clarity and depth.

## Iris Dataset Prediction

The first machine learning problem addressed is the classical Iris dataset prediction.
- We stored the features in Hopsworks FeatureStore.
- We trained the serverless K-nearest neighbors algorithm using Modal
- Finally, we hosted a Gradio application on Huggingface Spaces to show the User Interface.


## Wine Dataset Prediction

For the wine dataset:
- A Jupyter Notebook was dedicated to Exploratory Data Analysis (EDA).
-  We dropped the non-relevant features and the duplicate rows, filled the missing values, and encoded the 'type' feature.
-  Extensive analysis was performed on the dataset to analyze trends and correlations, highlighting and visualizing relevant information later used for prediction.
- `wine-training-pipeline_regression.ipynb` involved creating a Feature View of the Group and training an XGBoost algorithm with default features, saving the model in Hopsworks.
- Given that the regression algorithm wasn't accurate enough, we did the same procedure using the Random Forest Classification algorithm in `wine-training-pipeline_classification.ipynb`, which resulted in higher performance. This result is probably due to the adjustments performed by the class weighting.
- Another Random Forest Classifier was trained on an augmented dataset, via ADASYN, to solve the class balancement issues and generate more samples for the minority classes. This model shows greatly improved performance and is the one used in the finalized application.
- In `wine-feature-pipeline-daily.py`, we developed a function generating synthetic wines based on EDA insights, incorporating historical knowledge.
- The prediction of the quality for synthetic wines was carried out in `wine-batch-inference-pipeline.py`.

### Huggingface Spaces
Additionally, there are two Huggingface Spaces for each task:  
Iris classification task:
- The first one is called 'iris'. It allows the user to input iris features and then predicts the type based on the user input. Access it [here](https://laura000-iris.hf.space).
- The second one is called iris-monitor. It displays the latest predicted and actual flower, the last 4 predictions in a dataframe, and a confusion matrix. Access it [here](https://laura000-iris-monitor.hf.space).

Wine classification task:
- The first one is called 'wine'. It allows the user to input wine features and then predicts the wine quality based on the user input. Access it [here](https://laura000-wine.hf.space/).
- The second one is called wine-monitor. It displays the latest predicted and actual quality, the last 4 predictions in a dataframe, and a confusion matrix. Access it [here](https://laura000-wine-monitor.hf.space).
