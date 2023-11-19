import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model_upsampled1", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model_upsampled1.pkl")
print("Model downloaded")


def wine(key, type, volatile_acidity, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, ph, sulphates, alcohol):
    print("Calling function")
    df = pd.DataFrame([[key, type, volatile_acidity, chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, ph, sulphates, alcohol]],
                      columns=['key', 'type', 'volatile_acidity', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                        'density', 'ph', 'sulphates', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #     print("Res: {0}").format(res)
    print(res)
    return res


demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with wine features to predict quality of wine.",
    allow_flagging="never",
    inputs=[
        gr.Number(value=1, label="Key", visible=False),
        gr.Number(value=0, label="Wine type (0 = white, 1 = red)"),
        gr.Number(value=1.0, label="Volatile Acidity"),
        gr.Number(value=2.0, label="Chlorides"),
        gr.Number(value=1.0, label="Free Sulfur Dioxide"),
        gr.Number(value=1.0, label="Total Sulfur Dioxide"),
        gr.Number(value=1.0, label="Density"),
        gr.Number(value=1.0, label="pH"),
        gr.Number(value=1.0, label="Sulphates"),
        gr.Number(value=1.0, label="Alcohol"),
    ],
    outputs=gr.Number(value = 2, label = "quality"))

demo.launch(debug=True)

