import os
import random

import modal
import pandas as pd
import hopsworks

LOCAL=False #to run it in modal we have to set LOCAL to false --> to run it into terminal: modal run ... + modal deploy to make it run everyday

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"])

   @stub.function(image=image, schedule=modal.Period(minutes=5), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

project = hopsworks.login()
fs = project.get_feature_store()

wine_fg = fs.get_feature_group(name="wine_reduced_new", version=1)
wine_df = wine_fg.read()

def generate_wine_sample(wine_data):
    """
    Returns a single row of wine data by randomly sampling from the existing wine dataset
    """
    print(wine_df)
    # Randomly pick between white (0) or red (1)
    wine_type = random.choice([0, 1])

    # Filter the dataset based on the randomly chosen wine type
    filtered_data = wine_data[wine_data['type'] == wine_type]
    print(filtered_data)

    # Sample other features from the filtered dataset
    sample = {}
    sample['type'] = wine_type
    for column in filtered_data.columns:
        if column != 'type':  # Exclude the 'type' column
            sample[column] = random.choice(filtered_data[column].tolist())

    return pd.DataFrame([sample])

# Example: Generate a random wine sample
random_wine_sample = generate_wine_sample(wine_df)
print(random_wine_sample)

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    new_wine = generate_wine_sample(wine_df)

    wine_fg = fs.get_feature_group(name="wine_reduced_new",version=1)
    wine_fg.insert(new_wine)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()