import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/qualities/latest_quality.txt", overwrite = True)
dataset_api.download("Resources/qualities/actual_quality.txt", overwrite = True)
dataset_api.download("Resources/qualities/df_recent_class.png", overwrite = True)
dataset_api.download("Resources/qualities/confusion_matrix.png", overwrite = True)

with open("actual_quality.txt", "r") as file:
    actual_quality = file.read()

with open("latest_quality.txt", "r") as file:
    latest_quality = file.read()

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Quality")
          gr.Textbox(latest_quality, label="predicted quality", lines=1)
      with gr.Column():
          gr.Label("Today's Actual Quality")
          gr.Textbox(actual_quality, label="actual quality", lines=1)
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent_class.png", elem_id="recent-predictions")
      with gr.Column():
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")

demo.launch(share = True)