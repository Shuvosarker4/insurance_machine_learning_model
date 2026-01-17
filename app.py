
import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the Model
with open("insurance_gb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Define the prediction function
def predict_insurance(age, sex, bmi, children, smoker, region,bmi_cat):
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region,bmi_cat
    ]],
    columns=[
        "age", "sex", "bmi", "children", "smoker", "region","bmi_cat"
    ])

    # Predict
    pred = model.predict(input_df)[0]

    # Return the predicted insurance charge
    return f"Predicted Insurance Cost: ${pred:,.2f}"

# The App Interface
inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["male", "female"], label="Sex"),
    gr.Number(label="BMI", value=30.0),
    gr.Slider(0, 5, step=1, label="Number of Children"),
    gr.Radio(["yes", "no"], label="Smoker"),
    gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
]

outputs = gr.Textbox(label="Prediction")

app = gr.Interface(
    fn=predict_insurance,
    inputs=inputs,
    outputs=outputs,
    title="Insurance Cost Predictor",
    description="Predict medical insurance charges based on personal information."
)

app.launch(share=True)

