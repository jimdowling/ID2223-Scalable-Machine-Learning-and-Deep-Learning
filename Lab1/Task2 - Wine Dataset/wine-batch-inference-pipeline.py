import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_batch")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.2.2","dataframe-image", "xgboost"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(minutes=5), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model_upsampled", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model_upsampled.pkl")
    
    feature_view = fs.get_feature_view(name="wine_reduced_new", version=1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    print(y_pred)
    offset = 1
    quality = y_pred[y_pred.size-offset]
    print("Quality predicted: " + str(quality))
    with open("./latest_quality.txt", "w") as file:
        file.write(str(quality))
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_quality.txt", "Resources/qualities", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine_reduced_new", version=1)
    df = wine_fg.read() 
    print(df)
    label = df.iloc[-offset]["quality"]
    print("Actual quality: " + str(label))
    with open("./actual_quality.txt", "w") as file:
        file.write(str(label))
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./actual_quality.txt", "Resources/qualities", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_class.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_class.png", "Resources/qualities", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Create the confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    cm = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(3, 10), yticklabels=range(3, 10))
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix.png")
    dataset_api.upload("./confusion_matrix.png", "Resources/qualities", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_batch")
        with stub.run():
            f()